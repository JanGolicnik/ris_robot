#! /usr/bin/env python3

import math
import os
import subprocess
import time
from enum import Enum

import cv2
import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
import tf2_geometry_msgs  # noqa: F401  registers the PointStamped transform
import yaml
from action_msgs.msg import GoalStatus
from anomaly_detector import AnomalyDetector
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
from detect_lines import LineDetector
from geometry_msgs.msg import (
    Point,
    PointStamped,  # add to your geometry_msgs import
    PoseStamped,
    PoseWithCovarianceStamped,
    Quaternion,
    Twist,
    TwistStamped,
)
from kittentts import KittenTTS
from nav2_msgs.action import NavigateToPose, Spin
from rclpy.action import ActionClient
from rclpy.duration import Duration as RDuration
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import (
    CameraInfo,
    Image,
    PointCloud2,  # add to your sensor_msgs import
)
from std_msgs.msg import String
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler
from visualization_msgs.msg import Marker

from dis_tutorial3_interfaces.msg import FaceDetection


class Task(Enum):
    EXPLORE = 0
    GOTO_FACE = 1
    CONVERSE = 2
    FIND_RINGS = 3
    FIND_BARRELS = 4
    INSPECT_BELT = 5
    FOLLOW_BLUE = 6
    GOTO_POINT = 7


class TaskResult(Enum):
    UNKNOWN = 0
    SUCCEEDED = 1
    CANCELED = 2
    FAILED = 3


def parse_instruction(text):
    t = (text or "").lower()
    print(f"PARSING INSTRUCTION {t}")
    is_belt = any(k in t for k in ("anomal", "defect", "belt"))
    if is_belt and "red" in t:
        return {"type": Task.INSPECT_BELT, "color": "red"}
    if is_belt and "green" in t:
        return {"type": Task.INSPECT_BELT, "color": "green"}
    if "ring" in t:
        return {"type": Task.FIND_RINGS}
    if "barrel" in t:
        return {"type": Task.FIND_BARRELS}
    return None


class RobotCommander(Node):
    MAX_GREET_ATTEMPTS = 2
    FACE_STANDOFF = 0.5
    USE_WAYPOINT_ORIENTATION = True
    BELT_SPEED = 0.15
    BELT_STEER_GAIN = 0.003
    BELT_LOST_LIMIT = 15
    ANOMALY_COOLDOWN = 1.0

    BELT_LANE_OFFSET = (
        0.0  # m, lateral offset from the line to the driving lane (0 = on the line)
    )
    BELT_SIDE = 1  # +1 / -1, which side of the line the lane sits on (only matters if offset != 0)
    BELT_MAX_RANGE = (
        3.0  # m, drop projected points further than this (shallow rays are noisy)
    )
    BELT_MIN_POINTS = 30  # need at least this many accumulated points to trust a fit
    BELT_MIN_LENGTH = 0.3  # m, reject a fit shorter than this
    BELT_MAX_POINTS = 4000  # cap stored points per color
    BELT_POINT_STRIDE = 20  # subsample stride over the line mask pixels each frame

    BELT_FORWARD_SPEED = 0.75  # m/s while following the line
    BELT_TURN_SPEED = 0.5  # rad/s for the 90 deg turn (negative = right/CW)
    BELT_STRAIGHT_SECS_RED = (
        10.0  # how long to drive straight after the turn (your "amount")
    )
    BELT_STRAIGHT_SECS_GREEN = (
        15.0  # how long to drive straight after the turn (your "amount")
    )
    BELT_COLOR_LOST_FRAMES = 10

    def __init__(self):
        super().__init__("robot_commander")

        self.current_pose = None

        self.job_queue = []
        self.current_job = None
        self._mission_done_logged = False

        self.patrol_waypoints = []
        self._patrol_i = 0

        # line points projected into the map during patrol, fit to a line on demand
        self.belt_points = {"red": [], "green": []}
        self.front_K = None  # (fx, fy, cx, cy) from camera_info
        self.front_header = None  # stamp + optical frame of the latest front image

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.detected_face_candidates = []
        self.detected_faces = []
        self.detected_ring_candidates = []
        self.detected_rings = []
        self.detected_barrel_candidates = []
        self.detected_barrels = []

        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status = None

        self.anomaly_detector = AnomalyDetector()
        self.line_detector = LineDetector()
        self.target_line = "blue"

        self.turning_left = False
        self.turn_start = 0
        self.blue_done = False
        self.blue_line_start = (2.75, -0.35)

        self.bridge = CvBridge()
        self.latest_top_image = None
        self.latest_front_image = None
        self.qr = cv2.QRCodeDetector()

        self.tts = KittenTTS()
        self._wav_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "greeting.wav"
        )

        # waypoints file is exported from rviz, defaults to the workspace root
        self.declare_parameter("waypoints_file", "waypoints.yaml")
        wp_path = (
            self.get_parameter("waypoints_file").get_parameter_value().string_value
        )
        try:
            self.patrol_waypoints = self.load_waypoints(wp_path)
            self.info(f"loaded {len(self.patrol_waypoints)} waypoints from {wp_path}")
        except Exception as e:
            self.error(f"could not load waypoints from {wp_path}: {e}")

        self.nav_to_pose_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.spin_client = ActionClient(self, Spin, "spin")

        self.latest_front_cloud = None  # organized (h, w, 3) in the camera frame
        self.front_cloud_header = None

        self.create_subscription(
            PoseWithCovarianceStamped,
            "amcl_pose",
            self._amclPoseCallback,
            QoSProfile(
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )
        self.create_subscription(
            FaceDetection,
            "/face_detections",
            self._facePosCallback,
            QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT),
        )
        self.create_subscription(
            PoseStamped,
            "/ring_positions",
            self._ringCallback,
            QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT),
        )
        self.create_subscription(
            PoseStamped,
            "/barrel_positions",
            self._barrelCallback,
            QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT),
        )
        self.create_subscription(
            Image, "/top_camera/rgb/preview/image_raw", self._topCameraCallback, 10
        )
        self.create_subscription(
            Image, "/oakd/rgb/preview/image_raw", self._frontCameraCallback, 10
        )
        self.create_subscription(
            PointCloud2, "/oakd/rgb/preview/depth/points", self._frontCloudCallback, 10
        )

        self.face_marker_pub = self.create_publisher(
            Marker, "/detected_face_marker", 10
        )
        self.arm_mover_pub = self.create_publisher(String, "/arm_command", 10)
        self.ring_marker_pub = self.create_publisher(
            Marker, "/detected_ring_marker", 10
        )
        self.barrel_marker_pub = self.create_publisher(
            Marker, "/detected_barrel_marker", 10
        )
        self.belt_points_pub = self.create_publisher(
            Marker, "/detected_belts_marker", 10
        )
        self.goal_marker_pub = self.create_publisher(Marker, "/goal_marker", 10)
        self.tts_pub = self.create_publisher(String, "/speak", 10)
        self.cmd_vel_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)

        self.JOB_START = {
            Task.EXPLORE: self.start_explore,
            Task.GOTO_FACE: self.start_goto_face,
            Task.GOTO_POINT: self.start_goto_point,
            Task.FIND_RINGS: self.start_find_rings,
            Task.FIND_BARRELS: self.start_find_barrels,
            Task.INSPECT_BELT: self.start_inspect_belt,
            Task.FOLLOW_BLUE: self.start_follow_blue,
        }

        self.JOB_UPDATE = {
            Task.EXPLORE: self.update_task_complete,
            Task.GOTO_FACE: self.update_task_complete,
            Task.GOTO_POINT: self.update_task_complete,
            Task.FIND_RINGS: self.update_task_complete,
            Task.FIND_BARRELS: self.update_task_complete,
            Task.INSPECT_BELT: self.update_inspect_belt,
            Task.FOLLOW_BLUE: self.update_follow_blue,
        }

        self.JOB_DONE = {
            Task.EXPLORE: self.done_explore,
            Task.GOTO_FACE: self.done_goto_face,
            Task.GOTO_POINT: self.done_goto_point,
            Task.CONVERSE: self.done_converse,
            Task.FIND_RINGS: self.done_find_rings,
            Task.FIND_BARRELS: self.done_find_barrels,
            Task.INSPECT_BELT: self.done_inspect_belt,
            Task.FOLLOW_BLUE: self.done_follow_blue,
        }

        self.info("Robot commander initialized")

    def load_waypoints(self, path):
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        raw = data.get("waypoints", {})
        keys = sorted(raw, key=lambda k: int("".join(c for c in k if c.isdigit()) or 0))
        wps = []
        for k in keys:
            wp = raw[k]
            pose = wp["pose"]
            x, y = float(pose[0]), float(pose[1])
            orient = wp.get("orientation")
            yaw = None
            if orient is not None:
                qz, qw = float(orient[0]), float(orient[3])
                yaw = 2.0 * math.atan2(qz, qw)
            wps.append({"x": x, "y": y, "yaw": yaw})
        return wps

    def main_loop(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            self.tick()
            # only accumulate belt-line points while we are still patrolling
            # self._accumulate_belt_points()
            self.publish_detection_markers()

    # checks if it needs a new job, othewise runs the jobs update() and its done() if its finished
    def tick(self):
        cv2.waitKey(1)
        if self.current_job is None:
            self.current_job = self.next_job()
            if self.current_job is None:
                if not self._mission_done_logged:
                    self.info("no jobs left; mission idle")
                    self._mission_done_logged = True
                return
            self.info(f"got a new job {self.current_job}")
            self._mission_done_logged = False
            start = self.JOB_START.get(self.current_job["type"])
            if start is not None:
                self.info(f"start job {self.current_job}")
                start(self.current_job)
            return

        job = self.current_job
        update = self.JOB_UPDATE.get(job["type"])
        if update is not None:
            # self.info(f"update job {job}")
            if not update(job):
                return

        self.current_job = None
        done = self.JOB_DONE.get(job["type"])
        if done is not None:
            self.info(f"done job {job}")
            done(job, self.getResult())

    def next_person(self):
        robot_pos = np.array(
            [
                self.current_pose.pose.position.x,
                self.current_pose.pose.position.y,
                0.0,
            ]
        )

        candidates = [
            f
            for f in self.detected_faces
            if not f["greeted"]
            and f["attempts"] < self.MAX_GREET_ATTEMPTS
            and np.linalg.norm(f["position"] - robot_pos) <= 2.5
        ]

        face = min(
            candidates,
            key=lambda f: np.linalg.norm(f["position"] - robot_pos),
            default=None,
        )

        return face

    # get the next queued job, then patrol every waypoint, and only then visit known faces
    def next_job(self):
        if self.job_queue:
            return self.job_queue.pop(0)
        if not self.exploration_done():
            return {"type": Task.EXPLORE}
        face = self.next_person()
        if face is not None:
            return {"type": Task.GOTO_FACE, "face": face}
        # return {"type": Task.INSPECT_BELT, "color": "green"}
        if not getattr(self, "blue_done", False):
            return {"type": Task.FOLLOW_BLUE}
        return None

    def exploration_done(self):
        if len(self.patrol_waypoints) == 0:
            return True
        return self._patrol_i >= len(self.patrol_waypoints)

    def enqueue(self, job):
        self.job_queue.append(job)

    # def _accumulate_belt_points(self):
    #     if self.latest_front_image is None or self.latest_front_cloud is None:
    #         return
    #     img = self.latest_front_image
    #     cloud = self.latest_front_cloud
    #     h, w = cloud.shape[:2]
    #     frame = self.front_cloud_header.frame_id
    #     v_off = int(img.shape[0] * 0.5)  # find_line uses the bottom half as ROI

    #     for color in ("red", "green"):
    #         found, _, _, _, mask, _, _, _ = self.line_detector.find_line(img, color)
    #         if not found or mask is None:
    #             continue
    #         ys, xs = np.nonzero(mask)
    #         if xs.size == 0:
    #             continue
    #         for i in range(0, xs.size, self.BELT_POINT_STRIDE):
    #             u = int(xs[i])
    #             v = int(ys[i]) + v_off
    #             if not (0 <= v < h and 0 <= u < w):
    #                 continue
    #             d = cloud[v, u, :]
    #             if not np.isfinite(d).all() or np.linalg.norm(d) < 0.001:
    #                 continue
    #             if np.linalg.norm(d) > self.BELT_MAX_RANGE:
    #                 continue
    #             pm = self._to_map(d, frame, self.front_cloud_header.stamp)
    #             if pm is None:
    #                 continue
    #             self.belt_points[color].append((pm.point.x, pm.point.y))
    #         if len(self.belt_points[color]) > self.BELT_MAX_POINTS:
    #             self.belt_points[color] = self.belt_points[color][
    #                 -self.BELT_MAX_POINTS :
    #             ]

    # fit a straight line to the accumulated points; returns (lane_start, lane_end, yaw) or None
    def _fit_belt(self, color):
        if color == "red":
            start = np.array([0.35, -4.29])
            end = np.array([0.415, -4.29])
            yaw = math.pi * 1.55
        elif color == "green":
            start = np.array([-4.22, -2.72])
            end = np.array([-4.65, 0.23])
            yaw = math.pi
        else:
            return None

        # yaw = math.atan2(end[1] - start[1], end[0] - start[0])
        return start, end, yaw
        # pts = np.array(self.belt_points.get(color, []), dtype=float)
        # if len(pts) < self.BELT_MIN_POINTS:
        #     self.warn(f"only {len(pts)} {color} points, need {self.BELT_MIN_POINTS}")
        #     return None

        # c = pts.mean(axis=0)
        # _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        # d = vt[0]
        # d = d / np.linalg.norm(d)

        # proj = (pts - c) @ d
        # s0, s1 = float(proj.min()), float(proj.max())
        # if (s1 - s0) < self.BELT_MIN_LENGTH:
        #     self.warn(f"{color} line fit too short ({s1 - s0:.2f} m)")
        #     return None

        # start = c + d * s0
        # end = c + d * s1

        # # drive from whichever end is closer to the robot
        # if self.current_pose is not None:
        #     rp = np.array(
        #         [self.current_pose.pose.position.x, self.current_pose.pose.position.y]
        #     )
        #     if np.linalg.norm(rp - end) < np.linalg.norm(rp - start):
        #         start, end = end, start
        #         d = -d

        # perp = np.array([-d[1], d[0]])
        # off = perp * self.BELT_SIDE * self.BELT_LANE_OFFSET
        # lane_start = start + off
        # lane_end = end + off
        # yaw = math.atan2(d[1], d[0])
        # return lane_start, lane_end, yaw

    # if a task needs to wait for nav / rotation
    def update_task_complete(self, job):
        return self.isTaskComplete()

    # set the next waypoint to visit
    def start_explore(self, job):
        wp = self.patrol_waypoints[self._patrol_i]
        self._patrol_i += 1
        pos = np.array([wp["x"], wp["y"], 0.0])
        if self.USE_WAYPOINT_ORIENTATION and wp["yaw"] is not None:
            yaw = wp["yaw"]
        else:
            # face the direction of travel so we don't spin in place on arrival
            yaw = self._heading_to(pos)
        self.publish_goal_marker(wp["x"], wp["y"])
        self.nav2_pose(self._pose(pos, yaw))

    def done_explore(self, job, result):
        if result != TaskResult.SUCCEEDED:
            self.warn("failed to reach patrol waypoint")

    # go to face position
    def start_goto_face(self, job):
        face = job["face"]
        pos = face["pos"] + face["normal"] * self.FACE_STANDOFF
        direction = face["pos"] - pos
        yaw = math.atan2(direction[1], direction[0]) + 0.25
        self.publish_goal_marker(float(pos[0]), float(pos[1]))
        self.nav2_pose(self._pose(pos, yaw))

    # start conversing if succeeded otherwise mark it as attempted
    def done_goto_face(self, job, result):
        face = job["face"]
        if result == TaskResult.SUCCEEDED:
            self.enqueue({"type": Task.CONVERSE, "face": face})
        else:
            face["attempts"] += 1
            self.warn(f"could not reach face (attempt {face['attempts']})")

    # read qr code, get the next instruction, resopond and start the task
    def done_converse(self, job, result):
        face = job["face"]
        face["greeted"] = True
        self.say(f"Hello {face['name']}, the {face['job']}.")

        task = parse_instruction(self.read_qr())
        if task is None:
            self.say("couldnt understand instruction")
            self.warn("invalid qr")
            return
        self.say(f"OK. I will {self._task_phrase(task)}.")
        self.enqueue(task)

    # navigate to a single point, optional yaw, used for rings / barrels / belt spot
    def start_goto_point(self, job):
        pos = job["pos"]
        # face the direction of travel unless an explicit yaw is given
        yaw = job["yaw"] if job.get("yaw") is not None else self._heading_to(pos)
        self.publish_goal_marker(float(pos[0]), float(pos[1]))
        self.nav2_pose(self._pose(pos, yaw))

    def done_goto_point(self, job, result):
        label = job.get("label", "point")
        if result == TaskResult.SUCCEEDED:
            self.info(f"reached {label}")
        else:
            self.warn(f"could not reach {label}")

    # queue a visit to every ring found during the patrol
    def start_find_rings(self, job):
        for i, ring in enumerate(self.detected_rings):
            self.enqueue(
                {
                    "type": Task.GOTO_POINT,
                    "pos": ring["pos"].copy(),
                    "yaw": None,
                    "label": f"ring {i} ({ring['color']})",
                }
            )

    def done_find_rings(self, job, result):
        self.say(f"I will visit {len(self.detected_rings)} rings.")

    # queue a visit to every barrel found during the patrol
    def start_find_barrels(self, job):
        for i, barrel in enumerate(self.detected_barrels):
            self.enqueue(
                {
                    "type": Task.GOTO_POINT,
                    "pos": barrel["pos"].copy(),
                    "yaw": None,
                    "label": f"barrel {i} ({barrel['color']})",
                }
            )

    def done_find_barrels(self, job, result):
        self.say(f"I will visit {len(self.detected_barrels)} barrels.")

    # fit the belt line from the patrol points, drive to its near end, then along it
    def start_inspect_belt(self, job):
        s = String()
        s.data = "look_at_belt_left"
        self.arm_mover_pub.publish(s)
        # print(self._belt_anomaly())
        # return
        color = job["color"]
        self.target_line = color
        fit = self._fit_belt(color)
        job["fitted"] = fit is not None
        job["anomalies"] = 0
        job["last_anomaly_t"] = 0.0
        if fit is None:
            self.warn(f"could not fit a {color} belt line")
            return
        lane_start, lane_end, yaw = fit
        job["lane_end"] = np.array([lane_end[0], lane_end[1], 0.0])
        job["yaw"] = yaw
        job["phase"] = "goto_start"

        self.publish_goal_marker(float(lane_start[0]), float(lane_start[1]))
        self.nav2_pose(self._pose(np.array([lane_start[0], lane_start[1], 0.0]), yaw))

    def update_inspect_belt(self, job):
        if not job.get("fitted", False):
            return True

        if job["phase"] == "goto_start":
            if not self.isTaskComplete():
                return False
            job["lost"] = 0
            job["final_turn"] = False
            job["phase"] = "forward"
            return False

        if job["phase"] == "forward":
            seen = False
            if self.latest_front_image is not None:
                found, *_ = self.line_detector.find_line(
                    self.latest_front_image, job["color"]
                )
                seen = found
            if seen:
                job["lost"] = 0
            else:
                job["lost"] += 1
            if job["lost"] >= self.BELT_COLOR_LOST_FRAMES:
                self._stop()
                job["phase"] = "turn"
                job["turn_start_yaw"] = self._current_yaw()
                return False
            self._drive(self.BELT_FORWARD_SPEED, 0.0)
            return False

        if job["phase"] == "turn":
            turned = abs(self._angle_diff(self._current_yaw(), job["turn_start_yaw"]))
            if turned >= math.pi * 0.55:
                self._stop()
                job["phase"] = "straight"
                job["straight_start"] = time.time()
                return False
            self._drive(0.0, -self.BELT_TURN_SPEED)
            return False

        if job["phase"] == "straight":
            move_time = 1.0
            turn_rate = 0.0

            if not job["final_turn"]:
                self._belt_check_anomaly(job)
                move_time = (
                    self.BELT_STRAIGHT_SECS_GREEN
                    if job["color"] == "green"
                    else self.BELT_STRAIGHT_SECS_RED
                )
                if self.latest_front_image is not None:
                    found, *_ = self.line_detector.find_line(
                        self.latest_front_image, job["color"]
                    )
                    if found:
                        turn_rate = self.BELT_TURN_SPEED * 0.2

            if time.time() - job["straight_start"] >= move_time:
                self._stop()
                if job["final_turn"]:
                    return True
                else:
                    job["phase"] = "turn"
                    job["turn_start_yaw"] = self._current_yaw()
                    job["final_turn"] = True
                return False

            self._drive(self.BELT_FORWARD_SPEED, turn_rate)
            return False

        return True

    def _belt_check_anomaly(self, job):
        now = time.time()
        if now - job["last_anomaly_t"] > self.ANOMALY_COOLDOWN and self._belt_anomaly():
            job["anomalies"] += 1
            job["last_anomaly_t"] = now
            self.warn(f"anomaly #{job['anomalies']} on {job['color']} belt")

    def _angle_diff(self, a, b):
        d = a - b
        return math.atan2(math.sin(d), math.cos(d))

    def done_inspect_belt(self, job, result):
        self._stop()
        if not job.get("fitted", False):
            self.say(f"I could not find the {job['color']} belt.")
            return
        self.say(
            f"Finished the {job['color']} belt. I found {job['anomalies']} anomalies."
        )

    def start_follow_blue(self, job):
        x, y = self.blue_line_start
        job.setdefault("seen_qrs", set())

        self.nav2_pose(self._pose(np.array([x, y, 0.0]), math.pi * 1.5))

        job["phase"] = "goto_start"
        job["lost_frames"] = 0

    def update_follow_blue(self, job):
        if job["phase"] == "goto_start":
            if not self.isTaskComplete():
                return False

            job["phase"] = "follow"
            job["lost_frames"] = 0
            return False

        if job["phase"] == "spinning":
            if not self.isTaskComplete():
                return False

            job["phase"] = "follow"
            job["lost_frames"] = 0
            return False

        if job["phase"] != "follow":
            return False

        cto = next((f for f in self.detected_faces if "cto" in f["job"].lower()), None)

        if cto is not None:
            self._stop()

            self.enqueue({"type": Task.GOTO_FACE, "face": cto})

            self.info("CTO found")
            return True

        img = self.latest_front_image

        if img is None:
            return False

        found, cx, cy, angle, mask, left, middle, right = self.line_detector.find_line(
            img, "blue"
        )

        blue_pixels = 0 if mask is None else cv2.countNonZero(mask)

        print(f"blue pixels: {blue_pixels}")

        if found and blue_pixels < 1200:
            self._stop()
            time.sleep(0.5)

            qr = self.read_qr()

            if qr and qr not in job["seen_qrs"]:
                job["seen_qrs"].add(qr)

                self.say(qr)

            cto = next(
                (f for f in self.detected_faces if "cto" in f["job"].lower()), None
            )

            if cto is not None:
                self.enqueue({"type": Task.GOTO_FACE, "face": cto})

                self.info("CTO found")
                return True

            self.spin(math.pi)

            job["phase"] = "spinning"
            job["lost_frames"] = 0

            return False

        if not found:
            job["lost_frames"] += 1

            if job["lost_frames"] < 40:
                return False

            self._drive(0.0, 0.2)
            return False

        job["lost_frames"] = 0

        width = img.shape[1]

        path_count = int(left) + int(middle) + int(right)
        junction = path_count >= 2

        if junction:
            if right:
                target_x = width * 0.75
            elif middle:
                target_x = width * 0.50
            else:
                target_x = width * 0.25

        else:
            target_x = cx

        err = target_x - width / 2

        linear_speed = 0.15
        angular_speed = -0.003 * err

        self._drive(linear_speed, angular_speed)

        return False

    def done_follow_blue(self, job, result):  # TODO: actually print the damn pdf
        self._stop()
        self.blue_done = True
        self.say(
            f"I found the CTO. I detected {len(self.detected_rings)} rings and {len(self.detected_barrels)} barrels."
        )

    def _drive(self, vx, wz):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = float(vx)
        msg.twist.angular.z = float(wz)
        self.cmd_vel_pub.publish(msg)

    def _task_phrase(self, task):
        return {
            Task.FIND_RINGS: "search for all the rings",
            Task.FIND_BARRELS: "search for all the barrels",
            Task.INSPECT_BELT: f"inspect the {task.get('color')} belt for anomalies",
        }.get(task["type"], "do that")

    def read_qr(self):
        if self.latest_front_image is None:
            return ""
        data, _, _ = self.qr.detectAndDecode(self.latest_front_image)
        return data

    def _belt_anomaly(self):
        if self.latest_top_image is None:
            return False
        img = self.latest_top_image
        h_full = img.shape[0]
        w_full = img.shape[1]
        img = img[
            int(h_full * 0.25) : int(h_full * 0.8),
            int(w_full * 0.2) : int(w_full * 0.8),
        ]
        cv2.imshow("img", img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return False
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cx, cy = x + w // 2, y + h // 2
        size = int(min(w, h) * 0.8)
        if size < 2:
            return False
        tile = img[cy - size // 2 : cy + size // 2, cx - size // 2 : cx + size // 2]
        if tile.size == 0:
            return False
        tile = cv2.resize(tile, (512, 512))
        cv2.imshow("tile", tile)
        is_anomaly, _, _ = self.anomaly_detector.detect(tile)
        return bool(is_anomaly)

    def nav2_pose(self, pose):
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.info("waiting for NavigateToPose server...")
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        fut = self.nav_to_pose_client.send_goal_async(goal_msg, self._feedbackCallback)
        rclpy.spin_until_future_complete(self, fut)
        self.goal_handle = fut.result()
        if not self.goal_handle.accepted:
            self.error("nav goal rejected")
            self.result_future = None
            return False
        self.result_future = self.goal_handle.get_result_async()
        return True

    def spin(self, target_yaw=2.0 * math.pi, time_allowance=15):
        while not self.spin_client.wait_for_server(timeout_sec=1.0):
            self.info("waiting for Spin server...")
        goal_msg = Spin.Goal()
        goal_msg.target_yaw = float(target_yaw)
        goal_msg.time_allowance = Duration(sec=int(time_allowance))
        fut = self.spin_client.send_goal_async(goal_msg, self._feedbackCallback)
        rclpy.spin_until_future_complete(self, fut)
        self.goal_handle = fut.result()
        if not self.goal_handle.accepted:
            self.error("spin goal rejected")
            self.result_future = None
            return False
        self.result_future = self.goal_handle.get_result_async()
        return True

    def isTaskComplete(self):
        if not self.result_future:
            return True
        rclpy.spin_until_future_complete(self, self.result_future, timeout_sec=0.10)
        result = self.result_future.result()
        if result:
            self.status = result.status
            return True
        return False

    def getResult(self):
        if self.status == GoalStatus.STATUS_SUCCEEDED:
            return TaskResult.SUCCEEDED
        if self.status == GoalStatus.STATUS_ABORTED:
            return TaskResult.FAILED
        if self.status == GoalStatus.STATUS_CANCELED:
            return TaskResult.CANCELED
        return TaskResult.UNKNOWN

    def _stop(self):
        self.cmd_vel_pub.publish(TwistStamped())

    def _feedbackCallback(self, msg):
        self.feedback = msg.feedback

    def say(self, text):
        self.tts.generate_to_file(text, self._wav_path, voice="Jasper", speed=1.0)
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", self._wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def _pose(self, pos, yaw):
        p = PoseStamped()
        p.header.frame_id = "map"
        p.header.stamp = self.get_clock().now().to_msg()
        p.pose.position.x = float(pos[0])
        p.pose.position.y = float(pos[1])
        p.pose.orientation = self.YawToQuaternion(yaw)
        return p

    def YawToQuaternion(self, angle_z=0.0):
        q = quaternion_from_euler(0, 0, angle_z)
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

    def _current_yaw(self):
        if self.current_pose is None:
            return 0.0
        q = self.current_pose.pose.orientation
        return 2.0 * math.atan2(q.z, q.w)

    # heading from the robot's current position toward a target point
    def _heading_to(self, target):
        if self.current_pose is None:
            return 0.0
        p = self.current_pose.pose.position
        return math.atan2(target[1] - p.y, target[0] - p.x)

    def _update_candidate(
        self,
        candidates,
        confirmed,
        pos,
        fields,
        min_count=5,
        window=2.0,
        merge_dist=0.5,
    ):
        now = time.time()
        if any(np.linalg.norm(pos - r["pos"]) < merge_dist for r in confirmed):
            return None

        i = next(
            (
                k
                for k, c in enumerate(candidates)
                if np.linalg.norm(c["pos"] - pos) < merge_dist
            ),
            None,
        )
        if i is None:
            candidates.append({"pos": pos, "times": [now], **fields})
            return None

        c = candidates[i]
        c["times"].append(now)
        c["pos"] = np.mean([c["pos"], pos], axis=0)
        c["times"] = [t for t in c["times"] if now - t < window]
        c.update(fields)

        if len(c["times"]) >= min_count:
            record = {"pos": c["pos"].copy(), **{k: c[k] for k in fields}}
            confirmed.append(record)
            candidates.pop(i)
            return record
        return None

    def _amclPoseCallback(self, msg):
        self.current_pose = msg.pose

    def _facePosCallback(self, msg):
        if self.current_pose is None:
            return

        pos = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ]
        )
        yaw = 2.0 * math.atan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        normal = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        nrm = np.linalg.norm(normal)
        if nrm > 0:
            normal = normal / nrm

        # robot_pos = np.array(
        #     [self.current_pose.pose.position.x, self.current_pose.pose.position.y, 0.0]
        # )
        # if np.linalg.norm(pos - robot_pos) > 2.5:
        #     print("face too far")
        #     return

        for f in self.detected_faces:
            if np.linalg.norm(pos - f["pos"]) < 0.5:
                f["name"] = msg.name
                f["job"] = msg.job
                f["pronouns"] = msg.pronouns
                print("face already in")
                return

        rec = self._update_candidate(
            self.detected_face_candidates,
            self.detected_faces,
            pos,
            {
                "name": msg.name,
                "job": msg.job,
                "pronouns": msg.pronouns,
                "normal": normal,
            },
        )
        if rec is not None:
            rec["greeted"] = False
            rec["attempts"] = 0
            self.info(f"CONFIRMED face at {rec['pos']}")
        else:
            print("face not detected enough")

    def _ringCallback(self, msg):
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        rec = self._update_candidate(
            self.detected_ring_candidates,
            self.detected_rings,
            pos,
            {"color": msg.header.frame_id},
        )
        if rec is not None:
            self.info(f"CONFIRMED ring: {rec['color']}")

    def _barrelCallback(self, msg):
        try:
            color, orientation = msg.header.frame_id.split(":")
        except ValueError:
            return
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        rec = self._update_candidate(
            self.detected_barrel_candidates,
            self.detected_barrels,
            pos,
            {"color": color, "orientation": orientation},
        )
        if rec is not None:
            self.info(f"CONFIRMED barrel: {rec['color']} {rec['orientation']}")

    def _topCameraCallback(self, msg):
        self.latest_top_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def _frontCameraCallback(self, msg):
        self.latest_front_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.front_header = msg.header

    def publish_goal_marker(self, x, y):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.id = 0
        m.scale.x = m.scale.y = m.scale.z = 0.5
        m.color.r = 1.0
        m.color.a = 1.0
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        self.goal_marker_pub.publish(m)

    def publish_detection_markers(self):
        for i, face in enumerate(self.detected_faces):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.id = i
            m.scale.x = m.scale.y = m.scale.z = 0.3
            m.color.g = 1.0
            m.color.a = 1.0
            m.pose.position.x = float(face["pos"][0])
            m.pose.position.y = float(face["pos"][1])
            self.face_marker_pub.publish(m)

        for i, ring in enumerate(self.detected_rings):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.id = i
            m.scale.x = m.scale.y = m.scale.z = 0.3
            m.color.r, m.color.g, m.color.b = self._color_to_rgb(ring["color"])
            m.color.a = 1.0
            m.pose.position.x = float(ring["pos"][0])
            m.pose.position.y = float(ring["pos"][1])
            m.pose.position.z = float(ring["pos"][2])
            self.ring_marker_pub.publish(m)

        for i, barrel in enumerate(self.detected_barrels):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.id = i
            m.scale.x = 0.4
            m.scale.y = 0.4
            m.scale.z = 0.6
            m.color.r, m.color.g, m.color.b = self._color_to_rgb(barrel["color"])
            m.color.a = 1.0
            m.pose.position.x = float(barrel["pos"][0])
            m.pose.position.y = float(barrel["pos"][1])
            m.pose.position.z = 0.3
            self.barrel_marker_pub.publish(m)

        for j, color in enumerate(("red", "green")):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "belt_points"
            m.id = j
            m.type = Marker.POINTS
            m.action = Marker.ADD
            m.scale.x = 0.03  # point size in metres
            m.scale.y = 0.03
            m.color.r, m.color.g, m.color.b = self._color_to_rgb(color)
            m.color.a = 1.0
            m.points = [
                Point(x=float(x), y=float(y), z=0.0) for x, y in self.belt_points[color]
            ]
            self.belt_points_pub.publish(m)

    def _frontCloudCallback(self, msg):
        cloud = pc2.read_points_numpy(msg, field_names=("x", "y", "z"))
        self.latest_front_cloud = cloud.reshape((msg.height, msg.width, 3))
        self.front_cloud_header = msg.header

    def _to_map(self, xyz, frame, stamp):
        ps = PointStamped()
        ps.header.frame_id = frame
        ps.header.stamp = Time().to_msg()
        ps.point.x, ps.point.y, ps.point.z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        try:
            return self.tf_buffer.transform(ps, "map", timeout=RDuration(seconds=0.1))
        except Exception as e:
            self.warn(f"TF to map failed: {e}")
            return None

    def _color_to_rgb(self, name):
        return {
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
            "yellow": (1.0, 1.0, 0.0),
            "purple": (0.5, 0.0, 0.5),
            "orange": (1.0, 0.5, 0.0),
            "brown": (0.4, 0.2, 0.0),
            "black": (0.1, 0.1, 0.1),
        }.get(name, (0.7, 0.7, 0.7))

    def info(self, msg):
        self.get_logger().info(msg)

    def warn(self, msg):
        self.get_logger().warn(msg)

    def error(self, msg):
        self.get_logger().error(msg)


def main(args=None):
    rclpy.init(args=args)
    rc = RobotCommander()
    while rc.current_pose is None and rclpy.ok():
        rclpy.spin_once(rc, timeout_sec=0.5)
    rc.info("Got pose, starting main loop")
    try:
        rc.main_loop()
    except KeyboardInterrupt:
        pass
    rc.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
