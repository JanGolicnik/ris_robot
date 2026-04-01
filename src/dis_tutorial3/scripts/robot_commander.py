#! /usr/bin/env python3
# Mofidied from Samsung Research America
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import os
import subprocess
import time
from enum import Enum, auto

import numpy as np
import rclpy
import yaml
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import (
    PoseStamped,
    PoseWithCovarianceStamped,
    Quaternion,
)
from irobot_create_msgs.action import Dock, Undock
from irobot_create_msgs.msg import DockStatus
from kittentts import KittenTTS
from lifecycle_msgs.srv import GetState
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
from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler
from visualization_msgs.msg import Marker


class TaskResult(Enum):
    UNKNOWN = 0
    SUCCEEDED = 1
    CANCELED = 2
    FAILED = 3


class State(Enum):
    SEARCHING = auto()
    MOVING_TO_FACE = auto()
    # POZDRAVLJANJE = auto()
    KONCAL = auto()


class RobotCommander(Node):
    def __init__(self, node_name="robot_commander", namespace=""):
        super().__init__(node_name=node_name, namespace=namespace)

        self.pose_frame_id = "map"

        # Flags and helper variables
        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status = None
        self.initial_pose_received = False
        self.is_docked = None
        self.roam_positions = [[1.73, 5.21], [-2.74, 7.88], [-1.28, 3.3]]
        self.detected_face_candidates = []
        self.detected_faces = []
        self.visited_face_i = 0
        self.state = State.SEARCHING
        self.pozdrav_start_time = None
        self.detected_ring_candidates = []
        self.detected_rings = []
        self.pending_spin = None

        self.waypoints = []
        self.waypoint_i = 0
        self.spinning = False

        waypoint_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "waypoints.yaml"
        )
        if os.path.exists(waypoint_file):
            with open(waypoint_file) as f:
                data = yaml.safe_load(f)
                self.waypoints = data.get("waypoints", [])
            self.info(f"Loaded {len(self.waypoints)} waypoints")

        self.create_subscription(
            DockStatus, "dock_status", self._dockCallback, qos_profile_sensor_data
        )

        self.localization_pose_sub = self.create_subscription(
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

        self.face_pos_sub = self.create_subscription(
            PoseStamped,
            "/face_positions",
            self._facePosCallback,
            QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT),
        )

        self.ring_sub = self.create_subscription(
            PoseStamped,
            "/ring_positions",
            self._ringCallback,
            QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT),
        )

        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "initialpose", 10
        )

        self.marker_pub = self.create_publisher(Marker, "/goal_marker", 10)
        self.detected_marker_pub = self.create_publisher(
            Marker, "/detected_face_marker", 10
        )

        self.nav_to_pose_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.spin_client = ActionClient(self, Spin, "spin")
        self.undock_action_client = ActionClient(self, Undock, "undock")
        self.dock_action_client = ActionClient(self, Dock, "dock")

        self.get_logger().info(f"Robot commander has been initialized!")

    def init(self):
        self.waitUntilNav2Active()

        while self.is_docked is None:
            rclpy.spin_once(self, timeout_sec=0.5)

        if self.is_docked:
            self.undock()

    def main_loop(self):
        while True:
            rclpy.spin_once(self, timeout_sec=0.1)

            if self.state == State.SEARCHING:
                self.update_search()

            if self.state == State.MOVING_TO_FACE:
                self.update_moving_to_face()

            for i, face in enumerate(self.detected_faces):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.id = i
                marker.scale.x = 0.3
                marker.scale.y = 0.3
                marker.scale.z = 0.3
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                marker.pose.position.x = float(face["pos"][0])
                marker.pose.position.y = float(face["pos"][1])
                marker.pose.position.z = 0.0
                self.detected_marker_pub.publish(marker)

    def update_search(self):
        if not self.isTaskComplete():
            self.info("waiting to detect face")
            return
        if self.pending_spin is not None:
            self.doSpin(self.pending_spin)
            self.pending_spin = None
            self.state = State.SEARCHING

        if self.visited_face_i < len(self.detected_faces):
            self.state = State.MOVING_TO_FACE
            self.info("going towards a face")
            face = self.detected_faces[self.visited_face_i]
            pos = face["pos"] + face["normal"]
            dir = face["pos"] - pos
            yaw = math.atan2(dir[1], dir[0])
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = "map"
            goal_pose.header.stamp = self.get_clock().now().to_msg()

            goal_pose.pose.position.x = float(pos[0])
            goal_pose.pose.position.y = float(pos[1])
            goal_pose.pose.orientation = self.YawToQuaternion(yaw)

            self.publish_goal_marker(float(pos[0]), float(pos[1]))
            self.visited_face_i += 1
            self.goToPose(goal_pose)

        if self.visited_face_i == 3 and len(self.detected_rings) == 2:
            self.state = State.KONCAL
            self.info("Finished checking all! :3")

        if self.waypoint_i < len(self.waypoints):
            wp = self.waypoints[self.waypoint_i]
            self.waypoint_i += 1
            self.info(f"Moving to waypoint {self.waypoint_i}/{len(self.waypoints)}")

            goal_pose = PoseStamped()
            goal_pose.header.frame_id = "map"
            goal_pose.header.stamp = self.get_clock().now().to_msg()
            goal_pose.pose.position.x = float(wp["x"])
            goal_pose.pose.position.y = float(wp["y"])
            goal_pose.pose.orientation = self.YawToQuaternion(float(wp.get("yaw", 0.0)))
            self.publish_goal_marker(float(wp["x"]), float(wp["y"]))
            self.goToPose(goal_pose)

            if wp.get("spin", False):
                self.pending_spin = float(wp.get("spin_angle", 6.28))
            else:
                self.pending_spin = None

    def update_moving_to_face(self):
        if not self.isTaskComplete():
            self.info("waiting to reach face")
            return
        self.pozdravi()
        self.state = State.SEARCHING

    def pozdravi(self):
        model = KittenTTS()
        wav_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "greeting.wav"
        )
        model.generate_to_file("Alo Stari!", wav_path, voice="Jasper", speed=1.0)
        os.system(f"ffplay -nodisp -autoexit {wav_path} 2>/dev/null")

    def say_color(self, color):
        model = KittenTTS()
        wav_path = "/tmp/ring.wav"

        model.generate_to_file(f"{color}", wav_path, voice="Jasper", speed=1.0)

        subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def destroy(self):
        self.nav_to_pose_client.destroy()
        super().destroy_node()

    def goToPose(self, pose, behavior_tree=""):
        """Send a `NavToPose` action request."""
        self.debug("Waiting for 'NavigateToPose' action server")
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.info("'NavigateToPose' action server not available, waiting...")

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        goal_msg.behavior_tree = behavior_tree

        self.info(
            "Navigating to goal: "
            + str(pose.pose.position.x)
            + " "
            + str(pose.pose.position.y)
            + "..."
        )
        send_goal_future = self.nav_to_pose_client.send_goal_async(
            goal_msg, self._feedbackCallback
        )
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error(
                "Goal to "
                + str(pose.pose.position.x)
                + " "
                + str(pose.pose.position.y)
                + " was rejected!"
            )
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True

    def undock(self):
        """Perform Undock action."""
        self.info("Undocking...")
        self.undock_send_goal()

        while not self.isUndockComplete():
            time.sleep(0.1)

    def undock_send_goal(self):
        goal_msg = Undock.Goal()
        self.undock_action_client.wait_for_server()
        goal_future = self.undock_action_client.send_goal_async(goal_msg)

        rclpy.spin_until_future_complete(self, goal_future)

        self.undock_goal_handle = goal_future.result()

        if not self.undock_goal_handle.accepted:
            self.error("Undock goal rejected")
            return

        self.undock_result_future = self.undock_goal_handle.get_result_async()

    def isUndockComplete(self):
        """
        Get status of Undock action.

        :return: ``True`` if undocked, ``False`` otherwise.
        """
        if self.undock_result_future is None or not self.undock_result_future:
            return True

        rclpy.spin_until_future_complete(
            self, self.undock_result_future, timeout_sec=0.1
        )

        if self.undock_result_future.result():
            self.undock_status = self.undock_result_future.result().status
            if self.undock_status != GoalStatus.STATUS_SUCCEEDED:
                self.info(f"Goal with failed with status code: {self.status}")
                return True
        else:
            return False

        self.info("Undock succeeded")
        return True

    def cancelTask(self):
        """Cancel pending task request of any type."""
        self.info("Canceling current task.")
        if self.result_future:
            future = self.goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, future)
        return

    def isTaskComplete(self):
        """Check if the task request of any type is complete yet."""
        if not self.result_future:
            # task was cancelled or completed
            return True
        rclpy.spin_until_future_complete(self, self.result_future, timeout_sec=0.10)
        if self.result_future.result():
            self.status = self.result_future.result().status
            if self.status != GoalStatus.STATUS_SUCCEEDED:
                self.debug(f"Task with failed with status code: {self.status}")
                return True
        else:
            # Timed out, still processing, not complete yet
            return False

        self.debug("Task succeeded!")
        return True

    def getFeedback(self):
        """Get the pending action feedback message."""
        return self.feedback

    def getResult(self):
        """Get the pending action result message."""
        if self.status == GoalStatus.STATUS_SUCCEEDED:
            return TaskResult.SUCCEEDED
        elif self.status == GoalStatus.STATUS_ABORTED:
            return TaskResult.FAILED
        elif self.status == GoalStatus.STATUS_CANCELED:
            return TaskResult.CANCELED
        else:
            return TaskResult.UNKNOWN

    def waitUntilNav2Active(self, navigator="bt_navigator", localizer="amcl"):
        """Block until the full navigation system is up and running."""
        self._waitForNodeToActivate(localizer)
        if not self.initial_pose_received:
            time.sleep(1)
        self._waitForNodeToActivate(navigator)
        self.info("Nav2 is ready for use!")
        return

    def _waitForNodeToActivate(self, node_name):
        # Waits for the node within the tester namespace to become active
        self.debug(f"Waiting for {node_name} to become active..")
        node_service = f"{node_name}/get_state"
        state_client = self.create_client(GetState, node_service)
        while not state_client.wait_for_service(timeout_sec=1.0):
            self.info(f"{node_service} service not available, waiting...")

        req = GetState.Request()
        state = "unknown"
        while state != "active":
            self.debug(f"Getting {node_name} state...")
            future = state_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                state = future.result().current_state.label
                self.debug(f"Result of get_state: {state}")
            time.sleep(2)
        return

    def YawToQuaternion(self, angle_z=0.0):
        quat_tf = quaternion_from_euler(0, 0, angle_z)

        # Convert a list to geometry_msgs.msg.Quaternion
        quat_msg = Quaternion(x=quat_tf[0], y=quat_tf[1], z=quat_tf[2], w=quat_tf[3])
        return quat_msg

    def _amclPoseCallback(self, msg):
        self.debug("Received amcl pose")
        self.initial_pose_received = True
        self.current_pose = msg.pose
        return

    def _facePosCallback(self, msg):
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        normal = np.array(
            [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]
        )
        now = time.time()

        self.info("faceposcallback called")

        if any(np.linalg.norm(pos - f["pos"]) < 0.5 for f in self.detected_faces):
            self.info("this person has already been detected")
            return  # already detected

        i = next(
            (
                i
                for i, c in enumerate(self.detected_face_candidates)
                if np.linalg.norm(c["pos"] - pos) < 0.5
            ),
            None,
        )

        if i is None:
            self.info("first time this candidate was seen")
            self.detected_face_candidates.append(
                {"pos": pos, "normal": normal, "times": [now]}
            )
            return

        candidate = self.detected_face_candidates[i]

        self.info("candidate with similar position found")

        candidate["times"].append(now)
        candidate["pos"] = np.mean([candidate["pos"], pos], axis=0)
        candidate["times"] = [t for t in candidate["times"] if now - t < 2.0]

        if len(candidate["times"]) >= 5:
            self.detected_faces.append(
                {"pos": candidate["pos"].copy(), "normal": candidate["normal"]}
            )
            self.detected_face_candidates.pop(i)
            self.info(f"detected a new face at: {candidate['pos']}!")
            return

        self.info("candidate hasnt been detected enough yet")

    def _ringCallback(self, msg):
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        color = msg.header.frame_id
        now = time.time()

        self.info(f"Ring detected: {color}")

        # already confirmed -> ignore

        if any(np.linalg.norm(pos - r["pos"]) < 0.5 for r in self.detected_rings):
            self.info("this ring has already been detected")
            return

        # find existing candidate
        i = next(
            (
                i
                for i, c in enumerate(self.detected_ring_candidates)
                if np.linalg.norm(c["pos"] - pos) < 0.5
            ),
            None,
        )

        if i is None:
            self.info("first time ring candidate was seen")
            self.detected_ring_candidates.append(
                {"pos": pos, "color": color, "times": [now]}
            )
            return

        candidate = self.detected_ring_candidates[i]

        candidate["times"].append(now)
        candidate["pos"] = np.mean([candidate["pos"], pos], axis=0)
        candidate["times"] = [t for t in candidate["times"] if now - t < 2.0]

        if len(candidate["times"]) >= 5:
            self.detected_rings.append(
                {"pos": candidate["pos"].copy(), "color": candidate["color"]}
            )
            self.detected_ring_candidates.pop()
            self.info(f"CONFIRMED ring: {color}")
            self.say_color(f"{color}")

    def _feedbackCallback(self, msg):
        self.debug("Received action feedback message")
        self.feedback = msg.feedback
        return

    def _dockCallback(self, msg: DockStatus):
        self.is_docked = msg.is_docked

    def info(self, msg):
        self.get_logger().info(msg)
        return

    def warn(self, msg):
        self.get_logger().warn(msg)
        return

    def error(self, msg):
        self.get_logger().error(msg)
        return

    def debug(self, msg):
        self.get_logger().debug(msg)
        return

    def publish_goal_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.id = 0
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    rc = RobotCommander()
    rc.init()
    rc.main_loop()
    rc.destroy()


if __name__ == "__main__":
    main()
