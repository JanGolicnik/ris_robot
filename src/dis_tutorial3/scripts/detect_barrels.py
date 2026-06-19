#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
import tf2_geometry_msgs
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import (
    Point,
    PointStamped,
    PoseStamped,
    PoseWithCovarianceStamped,
    Quaternion,
    Twist,
    TwistStamped,
)
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from visualization_msgs.msg import Marker

# malo na vecje zaradi hi, lo spodaj, zaradi red
COLOR_RANGES = {
    "red": [
        (np.array([0, 100, 50]), np.array([10, 255, 255])),
        (np.array([170, 100, 50]), np.array([180, 255, 255])),
    ],
    "green": [(np.array([40, 50, 50]), np.array([90, 255, 255]))],
    "blue": [(np.array([95, 100, 80]), np.array([130, 255, 255]))],
    "yellow": [(np.array([18, 80, 80]), np.array([35, 255, 255]))],
    "black": [(np.array([0, 0, 0]), np.array([180, 255, 50]))],
}


def detect_colored_regions(
    img_bgr,
    min_fill_ratio=0.55,
    inner_crop=0.6,
    min_contour_area=800,
    morph_kernel=(5, 5),
):

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    detections = []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)

    for cname, ranges in COLOR_RANGES.items():
        mask = None
        for lo, hi in ranges:
            m = cv2.inRange(hsv, lo, hi)
            mask = m if mask is None else cv2.bitwise_or(mask, m)

        if mask is None:
            continue

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            img_h, img_w = img_bgr.shape[:2]

            margin = 15

            if x < margin or y < margin:
                continue

            if x + w > img_w - margin:
                continue

            if y + h > img_h - margin:
                continue

            if w == 0 or h == 0:
                continue

            if w > 500 or h > 500:
                continue

            fill = area / (w * h)
            if fill < min_fill_ratio:
                continue

            aspect_wh = float(w) / float(h)

            if aspect_wh > 8 or aspect_wh < 0.125:
                continue

            if min(w, h) < 20:
                continue

            cx0 = int(w * (1 - inner_crop) / 2)
            cy0 = int(h * (1 - inner_crop) / 2)
            cw = max(1, int(w * inner_crop))
            ch = max(1, int(h * inner_crop))

            rect_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(rect_mask, [cnt - [x, y]], -1, 255, -1)
            center_crop = rect_mask[cy0 : cy0 + ch, cx0 : cx0 + cw]
            center_fill = np.count_nonzero(center_crop) / (cw * ch + 1e-9)
            if center_fill < 0.75:
                continue

            orientation = (
                "vertical"
                if h > 1.3 * w
                else ("horizontal" if w > 1.25 * h else "unknown")
            )
            if cname == "black" and orientation == "horizontal":
                continue
            if orientation == "unknown":
                continue

            roi = hsv[y : y + h, x : x + w]
            roi_mask = rect_mask
            if cname != "black":
                hue_pixels = roi[:, :, 0][roi_mask == 255]
                if len(hue_pixels) < 30:
                    continue
                if np.std(hue_pixels) > 25:
                    continue

            detections.append(
                {
                    "color": cname,
                    "bbox": (x, y, w, h),
                    "area": float(area),
                    "fill_ratio": float(fill),
                    "center_fill": float(center_fill),
                    "orientation": orientation,
                }
            )

    return detections


class BarrelDetector(Node):
    MAX_DEPTH = 4.0  # ignore detections farther than this (m)
    CENTER_PATCH = 5  # half-size of the patch sampled around the bbox center

    def __init__(self):
        super().__init__("detect_barrels")

        self.bridge = CvBridge()
        self.cv_image = None
        self.candidates_in_image = []
        self.current_pose = None  # FIX: initialize before any callback can read it

        self.image_sub = self.create_subscription(
            Image,
            "/oakd/rgb/preview/image_raw",
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.pc_sub = self.create_subscription(
            PointCloud2,
            "/oakd/rgb/preview/depth/points",
            self.pointcloud_callback,
            qos_profile_sensor_data,
        )

        qos = QoSProfile(depth=10)
        qos.reliability = QoSReliabilityPolicy.BEST_EFFORT

        self.barrel_pub = self.create_publisher(PoseStamped, "/barrel_positions", qos)
        self.marker_pub = self.create_publisher(Marker, "/barrel_marker", 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
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

        self.get_logger().info("Barrel detector started")

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        detections = detect_colored_regions(
            cv_image,
            min_fill_ratio=0.55,
            inner_crop=0.6,
            min_contour_area=800,
            morph_kernel=(5, 5),
        )

        for det in detections:

            fill = det["fill_ratio"]
            x, y, w, h = det["bbox"]

            aspect = w / float(h)

            det["spill"] = (
                det["orientation"] == "horizontal"
                and (
                    fill < 0.72
                    or aspect > 2.8
                )
            )

        candidates = []
        for det in detections:
            x, y, w, h = det["bbox"]
            cx = x + w // 2
            cy = y + h // 2
            color = det["color"]
            orientation = det["orientation"]
            spill = det["spill"]

            candidates.append((cx, cy, color, orientation))

            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                cv_image,
                f"{color} {orientation} {'SPILL' if spill else ''}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # FIX: publish the list atomically so the pointcloud callback never sees
        # a half-built list (matters under a multithreaded executor).
        self.candidates_in_image = candidates

        cv2.imshow("Barrels", cv_image)
        cv2.waitKey(1)

    def _closest_point_in_patch(self, points, cx, cy, h, w):
        # FIX: sample a small window around the center and take the closest valid
        # point, rather than trusting a single center pixel that may hit the
        # background through a gap.
        r = self.CENTER_PATCH
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        patch = points[y0:y1, x0:x1].reshape(-1, 3)

        finite = patch[np.isfinite(patch).all(axis=1)]
        if finite.shape[0] == 0:
            return None
        norms = np.linalg.norm(finite, axis=1)
        finite = finite[norms > 0.001]
        norms = norms[norms > 0.001]
        if finite.shape[0] == 0:
            return None
        return finite[int(np.argmin(norms))]

    def pointcloud_callback(self, data):
        candidates = self.candidates_in_image
        if not candidates:
            return

        h, w = data.height, data.width
        points = pc2.read_points_numpy(data, field_names=("x", "y", "z"))
        points = points.reshape((h, w, 3))

        for cx, cy, color, orientation in candidates:
            if not (0 <= cy < h and 0 <= cx < w):
                continue

            d = self._closest_point_in_patch(points, cx, cy, h, w)
            if d is None:
                continue

            depth = np.linalg.norm(d)
            if depth > self.MAX_DEPTH:
                continue

            p_cam = PointStamped()
            p_cam.header.frame_id = "oakd_rgb_camera_optical_frame"
            p_cam.header.stamp = Time().to_msg()  # latest available, no extrapolation
            p_cam.point.x = float(d[0])
            p_cam.point.y = float(d[1])
            p_cam.point.z = float(d[2])

            try:
                p_map = self.tf_buffer.transform(
                    p_cam, "map", timeout=rclpy.duration.Duration(seconds=0.5)
                )
            except Exception as e:
                self.get_logger().warn(f"TF transform failed: {e}")
                continue

            pose = PoseStamped()
            pose.header.frame_id = f"{color}:{orientation}"
            # FIX: real stamp instead of zero time
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = p_map.point.x
            pose.pose.position.y = p_map.point.y
            pose.pose.position.z = p_map.point.z
            self.barrel_pub.publish(pose)

            self.get_logger().info(
                f"Barrel: {color} {orientation} at "
                f"({p_map.point.x:.2f}, {p_map.point.y:.2f}, {p_map.point.z:.2f})"
            )

    def _amclPoseCallback(self, msg):
        self.current_pose = msg.pose


def main():
    rclpy.init()
    node = BarrelDetector()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
