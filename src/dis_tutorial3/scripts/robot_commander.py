#! /usr/bin/env python3

import math
import os
import subprocess
import time
from enum import Enum, auto

import cv2
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from anomaly_detector import AnomalyDetector
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Quaternion
from kittentts import KittenTTS
from nav2_msgs.action import NavigateToPose, Spin
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image
from std_msgs.msg import String
from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler
from visualization_msgs.msg import Marker

from dis_tutorial3_interfaces.msg import FaceDetection

from detect_lines import LineDetector

class State(Enum):
    SEARCHING = auto()
    MOVING_TO_FACE = auto()
    CONVERSE = auto()
    SPINNING = auto()


class TaskResult(Enum):
    UNKNOWN = 0
    SUCCEEDED = 1
    CANCELED = 2
    FAILED = 3


class RobotCommander(Node):
    def __init__(self):
        super().__init__("robot_commander")

        self.current_pose = None
        self.state = State.SEARCHING

        self.detected_face_candidates = []
        self.detected_faces = []

        self.detected_ring_candidates = []
        self.detected_rings = []

        self.detected_barrel_candidates = []
        self.detected_barrels = []

        self.anomaly_detector = AnomalyDetector()

        self.bridge = CvBridge()
        self.latest_top_image = None

        self.qr = cv2.QRCodeDetector()

        self.line_detector = LineDetector()
        self.target_line = "yellow" #tuki bomo dal da je red ali green, odvisno kaj oseba rece
        self.latest_rgb_image = None #normal kamera kt pr face pa to

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
            Image, "/top_camera/rgb/preview/image_raw", self.top_camera_callback, 10
        )
        self.create_subscription(
            Image, "/oakd/rgb/preview/image_raw", self.front_camera_callback, 10
        )
        self.create_subscription(
            Image,
            "/oakd/rgb/preview/image_raw",
            self.rgb_camera_callback,
            qos_profile_sensor_data
        )
        self.face_marker_pub = self.create_publisher(
            Marker, "/detected_face_marker", 10
        )
        self.ring_marker_pub = self.create_publisher(
            Marker, "/detected_ring_marker", 10
        )
        self.barrel_marker_pub = self.create_publisher(
            Marker, "/detected_barrel_marker", 10
        )

        self.info("Robot commander initialized (manual mode)")

        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status = None
        self.visited_face_i = 0

        self.nav_to_pose_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.spin_client = ActionClient(self, Spin, "spin")
        self.tts_pub = self.create_publisher(String, "/speak", 10)

        self.goal_marker_pub = self.create_publisher(Marker, "/goal_marker", 10)

    def read_qr(self):
        if self.latest_front_image is None:
            return ""
        data, points, _ = self.qr.detectAndDecode(self.latest_front_image)
        return data

    def main_loop(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.state == State.SEARCHING:
                self.update_search()
            elif self.state == State.MOVING_TO_FACE:
                self.update_moving_to_face()
            elif self.state == State.CONVERSE:
                self.update_converse()
            elif self.state == State.SPINNING:
                self.update_spinning()
            self.publish_detection_markers()
            #self.test_anomaly_detection()
            self.test_line_detection()

    def update_converse(self):
        face = self.current_face
        self.say(f"Hello {face['name']}, the {face['job']}.")
        instruction = face["instruction"]
        next_task = ""
        if "anomalies" in instruction or "defect" in instruction or "belt":
            if "red" in instruction:
                next_task = "red anomalies"
            else:
                next_task = "green anomalies"
        elif "rings" in instruction:
            next_task = "rings"
        elif "barrels":
            next_task = "barrels"

        self.say(f"Ok i will do the {next_task}")
        self.current_face = None
        self.state = State.SEARCHING


    def update_search(self):
        if not self.isTaskComplete():
            return
        if self.visited_face_i < len(self.detected_faces):
            self.state = State.MOVING_TO_FACE
            self.info("going towards a face")
            self.current_face = self.detected_faces[self.visited_face_i]
            pos = self.current_face["pos"] + self.current_face["normal"] * 0.5
            dir = self.current_face["pos"] - pos
            yaw = math.atan2(dir[1], dir[0])
            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.pose.position.x = float(pos[0])
            goal.pose.position.y = float(pos[1])
            goal.pose.orientation = self.YawToQuaternion(yaw)
            self.publish_goal_marker(float(pos[0]), float(pos[1]))
            self.visited_face_i += 1
            self.goToPose(goal)

    def update_moving_to_face(self):
        if not self.isTaskComplete():
            return
        face = self.current_face
        if self.getResult() == TaskResult.SUCCEEDED:
            face["instruction"] = self.read_qr()
            self.state = State.CONVERSE
        else:
            self.state = State.SEARCHING
            self.current_face = None

    def update_spinning(self):
        if not self.isTaskComplete():
            return
        self.state = State.SEARCHING

    def say(self, text):
        model = KittenTTS()
        wav_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "greeting.wav"
        )
        model.generate_to_file(text, wav_path, voice="Jasper", speed=1.0)
        subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # msg = String()
        # msg.data = text
        # self.tts_pub.publish(msg)

    def YawToQuaternion(self, angle_z=0.0):
        q = quaternion_from_euler(0, 0, angle_z)
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

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

    def goToPose(self, pose):
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.info("waiting for NavigateToPose server...")
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        send_goal_future = self.nav_to_pose_client.send_goal_async(
            goal_msg, self._feedbackCallback
        )
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()
        if not self.goal_handle.accepted:
            self.error("nav goal rejected")
            return False
        self.result_future = self.goal_handle.get_result_async()
        return True

    def isTaskComplete(self):
        if not self.result_future:
            return True
        rclpy.spin_until_future_complete(self, self.result_future, timeout_sec=0.10)
        if self.result_future.result():
            self.status = self.result_future.result().status
            return True
        return False
    
    # def test_anomaly_detection(self):

    #     if self.latest_top_image is None:
    #         return

    #     img = self.latest_top_image.copy()

    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #     _, thresh = cv2.threshold(
    #         gray,
    #         140,
    #         255,
    #         cv2.THRESH_BINARY
    #     )

    #     contours, _ = cv2.findContours(
    #         thresh,
    #         cv2.RETR_EXTERNAL,
    #         cv2.CHAIN_APPROX_SIMPLE
    #     )

    #     if not contours:
    #         return

    #     largest = max(contours, key=cv2.contourArea)

    #     x, y, w, h = cv2.boundingRect(largest)

    #     cx = x + w // 2
    #     cy = y + h // 2

    #     size = int(min(w, h) * 0.8)

    #     tile = img[
    #         cy - size//2 : cy + size//2,
    #         cx - size//2 : cx + size//2
    #     ]

    #     tile = cv2.resize(tile, (512, 512))

    #     is_anomaly, mask, blackhat = self.anomaly_detector.detect(tile)

    #     debug = img.copy()

    #     cv2.rectangle(
    #         debug,
    #         (x, y),
    #         (x+w, y+h),
    #         (0, 255, 0),
    #         2
    #     )

    #     if is_anomaly:
    #         print("NOK")
    #     else:
    #         print("OK")

    #     cv2.imshow("top_camera", debug)
    #     #cv2.imshow("threshold", thresh)
    #     #cv2.imshow("tile", tile)
    #     cv2.imshow("mask", mask)
    #     #cv2.imshow("blackhat", blackhat)

    #     cv2.waitKey(1)
    
    def test_line_detection(self):
        if self.latest_rgb_image is None:
            return

        img = self.latest_rgb_image.copy()

        found, cx, cy, angle, mask = \
            self.line_detector.find_line(
                img,
                self.target_line
            )

        if found:
            cv2.circle(
                img,
                (cx, cy),
                10,
                (0,255,0),
                -1
            )
            print("FOUND", cx, cy, angle)

        cv2.imshow("line",img)
        cv2.imshow("line_mask",mask)
        
        cv2.waitKey(1)

    def top_camera_callback(self, msg):
        self.latest_top_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def front_camera_callback(self, msg):
        self.latest_front_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def rgb_camera_callback(self, msg):
        self.latest_rgb_image = self.bridge.imgmsg_to_cv2(
            msg,
            "bgr8"
        )

    def getResult(self):
        if self.status == GoalStatus.STATUS_SUCCEEDED:
            return TaskResult.SUCCEEDED
        return TaskResult.UNKNOWN

    def _feedbackCallback(self, msg):
        self.feedback = msg.feedback

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

    def _amclPoseCallback(self, msg):
        self.current_pose = msg.pose

    def _facePosCallback(self, msg):
        name = msg.name
        pronouns = msg.pronouns
        job = msg.job
        pos = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ]
        )
        yaw = 2.0 * math.atan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        normal = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        now = time.time()

        if self.current_pose is None:
            return
        robot_pos = np.array(
            [self.current_pose.pose.position.x, self.current_pose.pose.position.y, 0.0]
        )
        if np.linalg.norm(pos - robot_pos) > 2.5:
            return

        for f in self.detected_faces:
            if np.linalg.norm(pos - f["pos"]) < 0.5:
                f["name"] = name
                f["job"] = job
                f["pronouns"] = pronouns
                return

        i = next(
            (
                i
                for i, c in enumerate(self.detected_face_candidates)
                if np.linalg.norm(c["pos"] - pos) < 0.5
            ),
            None,
        )

        if i is None:
            self.detected_face_candidates.append(
                {"pos": pos, "normal": normal, "times": [now]}
            )
            return

        c = self.detected_face_candidates[i]
        c["times"].append(now)
        c["pos"] = np.mean([c["pos"], pos], axis=0)
        c["times"] = [t for t in c["times"] if now - t < 2.0]
        if len(c["times"]) >= 5:
            n = c["normal"]
            norm = np.linalg.norm(n)
            n = n / norm
            self.detected_faces.append(
                {
                    "pos": c["pos"].copy(),
                    "normal": normal,
                    "name": name,
                    "job": job,
                    "pronouns": pronouns,
                }
            )
            self.detected_face_candidates.pop(i)
            self.info(f"CONFIRMED face at {c['pos']}")

    def _ringCallback(self, msg):
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        color = msg.header.frame_id
        now = time.time()

        if any(np.linalg.norm(pos - r["pos"]) < 0.5 for r in self.detected_rings):
            return

        i = next(
            (
                i
                for i, c in enumerate(self.detected_ring_candidates)
                if np.linalg.norm(c["pos"] - pos) < 0.5
            ),
            None,
        )

        if i is None:
            self.detected_ring_candidates.append(
                {"pos": pos, "color": color, "times": [now]}
            )
            return

        c = self.detected_ring_candidates[i]
        c["times"].append(now)
        c["pos"] = np.mean([c["pos"], pos], axis=0)
        c["times"] = [t for t in c["times"] if now - t < 2.0]
        if len(c["times"]) >= 5:
            self.detected_rings.append({"pos": c["pos"].copy(), "color": c["color"]})
            self.detected_ring_candidates.pop(i)
            self.info(f"CONFIRMED ring: {color}")

    def _barrelCallback(self, msg):
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        try:
            color, orientation = msg.header.frame_id.split(":")
        except ValueError:
            return
        now = time.time()

        if any(np.linalg.norm(pos - b["pos"]) < 0.5 for b in self.detected_barrels):
            return

        i = next(
            (
                i
                for i, c in enumerate(self.detected_barrel_candidates)
                if np.linalg.norm(c["pos"] - pos) < 0.5
            ),
            None,
        )

        if i is None:
            self.detected_barrel_candidates.append(
                {"pos": pos, "color": color, "orientation": orientation, "times": [now]}
            )
            return

        c = self.detected_barrel_candidates[i]
        c["times"].append(now)
        c["pos"] = np.mean([c["pos"], pos], axis=0)
        c["times"] = [t for t in c["times"] if now - t < 2.0]
        if len(c["times"]) >= 5:
            self.detected_barrels.append(
                {
                    "pos": c["pos"].copy(),
                    "color": c["color"],
                    "orientation": c["orientation"],
                }
            )
            self.detected_barrel_candidates.pop(i)
            self.info(f"CONFIRMED barrel: {c['color']} {c['orientation']}")

    def info(self, msg):
        self.get_logger().info(msg)


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
