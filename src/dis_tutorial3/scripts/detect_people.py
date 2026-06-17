#!/usr/bin/env python3

import math
import os
import re

import cv2
import numpy as np
import rclpy
import tf2_geometry_msgs
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, PoseStamped
from insightface.app import FaceAnalysis
from rclpy.node import Node
from rclpy.qos import QoSReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from visualization_msgs.msg import Marker

from dis_tutorial3_interfaces.msg import FaceDetection

CAMERA_FRAME = "oakd_rgb_camera_optical_frame"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def load_face_app(device: str, det_size: int = 320) -> FaceAnalysis:
    use_gpu = device.lower() in ("gpu", "cuda")
    providers = ["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=providers,
    )
    app.prepare(ctx_id=0 if use_gpu else -1, det_size=(det_size, det_size))
    return app


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def parse_reference_filename(fname: str):
    stem = os.path.splitext(fname)[0]
    stem = re.sub(r"_\d+$", "", stem)
    parts = stem.split("_")
    name = parts[0] if parts else stem
    pronouns = f"{parts[1]}/{parts[2]}" if len(parts) >= 3 else ""
    job = " ".join(parts[3:]) if len(parts) >= 4 else ""
    return name, pronouns, job


def build_reference_db(app: FaceAnalysis, faces_dir: str, logger=None):
    embs_by_name: dict[str, list[np.ndarray]] = {}
    meta: dict[str, dict] = {}

    if not faces_dir or not os.path.isdir(faces_dir):
        return [], np.zeros((0, 512), np.float32), {}

    for fname in sorted(os.listdir(faces_dir)):
        if not fname.lower().endswith(IMAGE_EXTS):
            continue
        img = cv2.imread(os.path.join(faces_dir, fname))
        if img is None:
            continue
        detected = app.get(img)
        if not detected:
            continue
        f = max(
            detected, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])
        )

        name, pronouns, job = parse_reference_filename(fname)
        embs_by_name.setdefault(name, []).append(f.normed_embedding)
        new_meta = {"pronouns": pronouns, "job": job}
        if name in meta and meta[name] != new_meta and logger is not None:
            logger.warn(f"Conflicting metadata for '{name}'; keeping first.")
        meta.setdefault(name, new_meta)

    names = list(embs_by_name.keys())
    if not names:
        return [], np.zeros((0, 512), np.float32), {}

    ref = np.stack([_normalize(np.mean(embs_by_name[n], axis=0)) for n in names])
    return names, ref.astype(np.float32), meta


def match_face(emb: np.ndarray, names, ref: np.ndarray, threshold: float) -> str:
    if ref.shape[0] == 0:
        return "unknown"
    sims = ref @ emb
    i = int(np.argmax(sims))
    return names[i] if sims[i] >= threshold else "unknown"


def yaw_to_quaternion(yaw: float):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def label_text(name: str, meta: dict) -> str:
    m = meta.get(name)
    if not m:
        return name
    extra = " ".join(x for x in (m.get("pronouns"), m.get("job")) if x)
    return f"{name} ({extra})" if extra else name


class detect_faces(Node):
    def __init__(self):
        super().__init__("detect_faces")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("device", "gpu"),
                ("faces_dir", "./faces"),
                ("match_threshold", 0.40),
                ("det_score_min", 0.50),
                ("show_window", True),
                ("rgb_topic", "/oakd/rgb/preview/image_raw"),
                ("pointcloud_topic", "/oakd/rgb/preview/depth/points"),
            ],
        )
        gp = self.get_parameter
        self.device = gp("device").get_parameter_value().string_value
        self.faces_dir = gp("faces_dir").get_parameter_value().string_value
        self.match_threshold = gp("match_threshold").get_parameter_value().double_value
        self.det_score_min = gp("det_score_min").get_parameter_value().double_value
        self.show_window = gp("show_window").get_parameter_value().bool_value
        rgb_topic = gp("rgb_topic").get_parameter_value().string_value
        pc_topic = gp("pointcloud_topic").get_parameter_value().string_value

        self.detection_color = (0, 0, 255)
        self.bridge = CvBridge()
        self.cv_image = None
        self.faces = []

        self.get_logger().info("Loading insightface (detection + recognition)...")
        self.app = load_face_app(self.device)
        self.names, self.ref_embs, self.meta = build_reference_db(
            self.app, self.faces_dir, self.get_logger()
        )
        if self.ref_embs.shape[0] == 0:
            self.get_logger().warn(
                f"No reference faces loaded from '{self.faces_dir}'. "
                "Every face will be 'unknown'."
            )
        else:
            self.get_logger().info(f"Loaded identities: {self.names}")

        self.rgb_image_sub = self.create_subscription(
            Image, rgb_topic, self.rgb_callback, qos_profile_sensor_data
        )
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, pc_topic, self.pointcloud_callback, qos_profile_sensor_data
        )
        self.face_pub = self.create_publisher(FaceDetection, "/face_detections", 10)
        self.face_pos_pub = self.create_publisher(
            PoseStamped, "/face_positions", QoSReliabilityPolicy.BEST_EFFORT
        )
        self.face_img_pub = self.create_publisher(Image, "/face_image", 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info("Node initialized.")

    def rgb_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"cv_bridge: {e}")
            return

        self.cv_image = cv_image
        results = []

        for f in self.app.get(cv_image):
            if f.det_score < self.det_score_min:
                continue
            x1, y1, x2, y2 = f.bbox.astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            name = match_face(
                f.normed_embedding, self.names, self.ref_embs, self.match_threshold
            )
            results.append({"bbox": (x1, y1, x2, y2), "center": (cx, cy), "name": name})
            self.get_logger().info(f"Face: {label_text(name, self.meta)}")

            if self.show_window:
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), self.detection_color, 2)
                cv2.circle(cv_image, (cx, cy), 4, self.detection_color, -1)
                cv2.putText(
                    cv_image,
                    label_text(name, self.meta),
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.detection_color,
                    1,
                )

        self.faces = results

        if self.show_window:
            cv2.imshow("faces", cv_image)
            if cv2.waitKey(1) == 27:  # Esc
                rclpy.shutdown()

    def pointcloud_callback(self, data):
        faces = self.faces
        if not faces:
            return

        height, width = data.height, data.width
        cloud = pc2.read_points_numpy(data, field_names=("x", "y", "z"))
        cloud = cloud.reshape((height, width, 3))

        for idx, face in enumerate(faces):
            cx, cy = face["center"]
            if not (0 <= cy < height and 0 <= cx < width):
                continue

            d = cloud[cy, cx, :]
            oy, ox = max(cy - 7, 0), max(cx - 7, 0)
            d2 = cloud[oy, ox, :]

            if not np.isfinite(d).all() or np.linalg.norm(d) < 0.001:
                continue

            p1_map = self._to_map(d, data.header.stamp)
            if p1_map is None:
                continue
            p1 = np.array([p1_map.point.x, p1_map.point.y, p1_map.point.z])

            normal = np.zeros(3)
            if np.isfinite(d2).all():
                p2_map = self._to_map(d2, data.header.stamp)
                if p2_map is not None:
                    p2 = np.array([p2_map.point.x, p2_map.point.y, p2_map.point.z])
                    v = p1 - p2
                    nv = np.linalg.norm(v)
                    if nv > 1e-6:
                        v /= nv
                        normal = np.cross(np.array([0.0, 0.0, -1.0]), v) * 0.5

            self._publish_face(face["name"], p1, normal)
            self._publish_face_crop(face, data.header.stamp)

    def _to_map(self, xyz, stamp):
        ps = PointStamped()
        ps.header.frame_id = CAMERA_FRAME
        ps.header.stamp = Time().to_msg()  # 0 -> "latest available", no extrapolation
        ps.point.x, ps.point.y, ps.point.z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        try:
            return self.tf_buffer.transform(
                ps, "map", timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except Exception as e:
            self.get_logger().warn(f"TF to map failed: {e}")
            return None

    def _publish_face(self, name, p1, normal):
        yaw = (
            math.atan2(normal[1], normal[0])
            if np.linalg.norm(normal[:2]) > 1e-6
            else 0.0
        )
        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        m = self.meta.get(name, {})

        fd = FaceDetection()
        fd.name = name
        fd.pronouns = m.get("pronouns", "")
        fd.job = m.get("job", "")
        fd.pose.header.frame_id = "map"
        fd.pose.header.stamp = self.get_clock().now().to_msg()
        fd.pose.pose.position.x = float(p1[0])
        fd.pose.pose.position.y = float(p1[1])
        fd.pose.pose.position.z = float(p1[2])
        fd.pose.pose.orientation.x = qx
        fd.pose.pose.orientation.y = qy
        fd.pose.pose.orientation.z = qz
        fd.pose.pose.orientation.w = qw
        self.face_pub.publish(fd)

    def _publish_face_crop(self, face, stamp):
        if self.cv_image is None:
            return
        x1, y1, x2, y2 = face["bbox"]
        h, w = self.cv_image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return
        crop = self.cv_image[y1:y2, x1:x2]
        try:
            msg = self.bridge.cv2_to_imgmsg(crop, encoding="bgr8")
            msg.header.stamp = stamp
            msg.header.frame_id = CAMERA_FRAME
            self.face_img_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"Face image publish failed: {e}")


def main():
    rclpy.init(args=None)
    node = detect_faces()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
