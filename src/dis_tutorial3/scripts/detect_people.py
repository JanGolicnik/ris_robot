#!/usr/bin/env python3

import math

import cv2
import numpy as np
import rclpy
import tf2_geometry_msgs
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from rclpy.qos import (
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Duration, Time
from sensor_msgs.msg import CameraInfo, Image
from tf_transformations import quaternion_from_euler
from ultralytics import YOLO
from visualization_msgs.msg import Marker


class detect_faces(Node):
    def __init__(self):
        super().__init__("detect_faces")

        self.declare_parameters(namespace="", parameters=[("device", "")])

        self.BASELINE = 7
        self.APPROACH_DIST = 0.6
        self.DETECTION_COLOR = (0, 0, 255)

        self.device = self.get_parameter("device").get_parameter_value().string_value

        self.rgb_data = None

        self.bridge = CvBridge()

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.rgb_sub = self.create_subscription(
            Image,
            "/robot_rgb_image",
            self.rgb_callback,
            qos,
        )

        self.depth_sub = self.create_subscription(
            Image,
            "/robot_depth_image",
            self.depth_callback,
            qos,
        )

        self.create_subscription(
            CameraInfo,
            "/gemini/color/camera_info",
            self.camera_info_callback,
            10,
        )

        self.model = YOLO("yolov8n.pt")

        self.marker_pub = self.create_publisher(Marker, "/people_marker", 10)
        self.face_pos_pub = self.create_publisher(PoseStamped, "/face_positions", 10)
        self.face_img_pub = self.create_publisher(Image, "/face_image", 10)

        self.tf_buffer = tf2_ros.Buffer(
            cache_time=rclpy.duration.Duration(seconds=10.0)
        )

        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False
        )

    def camera_info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def depth_callback(self, data):
        if (self.fx is None) or (self.rgb_data is None):
            return

        self.do_detection(self.rgb_data, data)

    def rgb_callback(self, data):
        self.rgb_data = data

    def do_detection(self, rgb_msg, depth_msg):
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except CvBridgeError as e:
            self.get_logger().warn(f"cv_bridge: {e}")
            return

        res = self.model.predict(
            rgb_image,
            imgsz=(256, 320),
            show=False,
            verbose=False,
            classes=[0],
            device=self.device,
        )

        faces = []
        for x in res:
            bboxes = x.boxes.xyxy
            if bboxes.nelement() == 0:
                continue
            for bbox in bboxes:
                x1, y1, x2, y2 = (int(bbox[i]) for i in range(4))
                x = (x1 + x2) // 2
                y = (y1 + y2) // 2
                faces.append((x, y, x1, y1, x2, y2))

                cv2.rectangle(rgb_image, (x1, y1), (x2, y2), self.DETECTION_COLOR, 3)
                cv2.circle(rgb_image, (x, y), 5, self.DETECTION_COLOR, -1)

        for idx, (u, v, x1, y1, x2, y2) in enumerate(faces):
            self.publish_face_position(
                depth_image, u, v, x1, y1, x2, y2, depth_msg.header, rgb_image, idx
            )

        cv2.imshow("faces", rgb_image)
        cv2.waitKey(1)

    def publish_face_position(
        self, depth_image, u, v, x1, y1, x2, y2, header, rgb_image, idx
    ):
        z_c = self.get_depth(depth_image, u, v)
        z_l = self.get_depth(depth_image, u - self.BASELINE, v)
        z_r = self.get_depth(depth_image, u + self.BASELINE, v)
        if z_c is None or z_l is None or z_r is None:
            return

        if abs(z_l - z_c) > 0.10 or abs(z_r - z_c) > 0.10:
            return

        p_c_cam = self.unproject(u, v, z_c)
        p_l_cam = self.unproject(u - self.BASELINE, v, z_l)
        p_r_cam = self.unproject(u + self.BASELINE, v, z_r)

        p_c = self.point_to_map(p_c_cam, header)
        p_l = self.point_to_map(p_l_cam, header)
        p_r = self.point_to_map(p_r_cam, header)
        if p_c is None or p_l is None or p_r is None:
            return

        if p_c[2] > 0.35:
            return

        v_horiz = p_r - p_l
        nh = np.linalg.norm(v_horiz)
        if nh < 1e-6:
            return

        v_horiz /= nh

        down = np.array([0.0, 0.0, -1.0])
        normal = np.cross(down, v_horiz)
        nl = np.linalg.norm(normal)
        if nl < 1e-6:
            return

        normal /= nl

        try:
            tf = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                Time.from_msg(header.stamp),
                timeout=Duration(seconds=0.1),
            )
            face_to_robot = np.array(
                [
                    tf.transform.translation.x - p_c[0],
                    tf.transform.translation.y - p_c[1],
                ]
            )
            if np.dot(normal[:2], face_to_robot) < 0:
                normal = -normal
        except Exception as e:
            self.get_logger().warn(f"robot TF failed: {e}")

        yaw = math.atan2(normal[1], normal[0])
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)

        face_pose = PoseStamped()
        face_pose.header.frame_id = "map"
        face_pose.header.stamp = header.stamp
        face_pose.pose.position.x = float(p_c[0])
        face_pose.pose.position.y = float(p_c[1])
        face_pose.pose.position.z = float(p_c[2])
        face_pose.pose.orientation.x = qx
        face_pose.pose.orientation.y = qy
        face_pose.pose.orientation.z = qz
        face_pose.pose.orientation.w = qw
        self.face_pos_pub.publish(face_pose)

        self.get_logger().info(f"face @ {p_c.round(2)} normal {normal.round(2)}")

        self._publish_marker(idx * 2, p_c, 0.10, (1.0, 1.0, 1.0), header.stamp)
        approach_pos = np.array(
            [
                p_c[0] + normal[0] * self.APPROACH_DIST,
                p_c[1] + normal[1] * self.APPROACH_DIST,
                p_c[2],
            ]
        )
        self._publish_marker(
            idx * 2 + 1, approach_pos, 0.15, (1.0, 0.0, 0.0), header.stamp
        )

        try:
            face_crop = rgb_image[y1:y2, x1:x2]
            if face_crop.size > 0:
                img_msg = self.bridge.cv2_to_imgmsg(face_crop, encoding="bgr8")
                img_msg.header.stamp = header.stamp
                img_msg.header.frame_id = header.frame_id
                self.face_img_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f"face image publish failed: {e}")

    def _publish_marker(self, mid, pos, scale, rgb, stamp):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = stamp
        m.ns = "faces"
        m.id = mid
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = scale
        m.color.r, m.color.g, m.color.b = rgb
        m.color.a = 1.0
        m.pose.position.x = float(pos[0])
        m.pose.position.y = float(pos[1])
        m.pose.position.z = float(pos[2])
        m.pose.orientation.w = 1.0
        self.marker_pub.publish(m)

    def get_depth(self, depth, u, v, r=2):
        h, w = depth.shape
        if not (0 <= u < w and 0 <= v < h):
            return None
        u0, u1 = max(u - r, 0), min(u + r + 1, w)
        v0, v1 = max(v - r, 0), min(v + r + 1, h)
        patch = depth[v0:v1, u0:u1].astype(np.float32)
        valid = patch[patch > 0]
        if valid.size < 3:
            return None
        z = float(np.median(valid)) / 1000.0
        return z if 0.1 < z < 10.0 else None

    def unproject(self, u, v, z):
        return np.array(
            [
                (u - self.cx) * z / self.fx,
                (v - self.cy) * z / self.fy,
                z,
            ],
            dtype=np.float32,
        )

    def point_to_map(self, p_cam, header):
        ps = PointStamped()
        ps.header = header
        ps.point.x = float(p_cam[0])
        ps.point.y = float(p_cam[1])
        ps.point.z = float(p_cam[2])
        try:
            ps_map = self.tf_buffer.transform(
                ps, "map", timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except Exception as e:
            self.get_logger().warn(f"TF failed: {e}")
            return None
        return np.array([ps_map.point.x, ps_map.point.y, ps_map.point.z])


def main():
    rclpy.init(args=None)
    node = detect_faces()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
