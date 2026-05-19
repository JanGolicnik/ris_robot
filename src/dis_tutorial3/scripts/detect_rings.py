#!/usr/bin/python3

from collections import Counter

import cv2
import numpy as np
import rclpy
import tf2_geometry_msgs  # noqa: F401
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    HistoryPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time  # kept for backward compat with prior file
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import (
    point_cloud2 as pc2,  # noqa: F401  (kept; pointcloud path is dormant)
)
from visualization_msgs.msg import Marker

# Unused at module level; kept per request.
qos_profile = QoSProfile(
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class RingDetector(Node):
    def __init__(self):
        super().__init__("ring_detector")

        self.ecc_thr = 100
        self.ratio_thr = 2
        self.center_thr = 8

        # Initialise all latched state up front so the first detection
        # before any callback has fired can't trip an AttributeError.
        self.depth_image = None
        self.camera_info = None
        self.pointcloud = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.bridge = CvBridge()

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Match the relay's publish QoS, otherwise the subscription silently drops.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.image_sub = self.create_subscription(
            Image, "/robot_rgb_image", self.image_callback, sensor_qos
        )
        self.depth_sub = self.create_subscription(
            Image, "/robot_depth_image", self.depth_callback, sensor_qos
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "/gemini/color/camera_info",
            self.camera_info_callback,
            10,
        )
        # Pointcloud path is no longer used by get_3d_position but kept wired
        # so the subscription doesn't need re-adding if it ever comes back.
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            "/oakd/rgb/preview/depth/points",
            self.pointcloud_callback,
            qos_profile_sensor_data,
        )

        self.ring_pub = self.create_publisher(
            PoseStamped,
            "/ring_positions",
            QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT),
        )
        self.marker_pub = self.create_publisher(Marker, "/ring_markers", 10)
        self.marker_id = 0

        cv2.namedWindow("Detected contours", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected rings", cv2.WINDOW_NORMAL)

    def camera_info_callback(self, msg):
        self.camera_info = msg
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def remove_pole(self, image, depth):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        pole_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 60, 150]))

        kernel = np.ones((5, 5), np.uint8)
        pole_mask = cv2.morphologyEx(pole_mask, cv2.MORPH_CLOSE, kernel)
        pole_mask = cv2.morphologyEx(pole_mask, cv2.MORPH_OPEN, kernel)

        hsv_inpainted = cv2.inpaint(
            hsv, pole_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA
        )
        result = cv2.cvtColor(hsv_inpainted, cv2.COLOR_HSV2BGR)

        depth_u8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_inpainted_u8 = cv2.inpaint(
            depth_u8, pole_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA
        )

        d_min, d_max = np.nanmin(depth), np.nanmax(depth)
        depth_inpainted = (
            depth_inpainted_u8.astype(np.float32) / 255.0 * (d_max - d_min) + d_min
        )

        return result, depth_inpainted

    def get_ring_color(self, cv_image, candidate):
        le, se = candidate
        mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, le, 255, -1)
        cv2.ellipse(mask, se, 0, -1)
        debug = cv_image.copy()
        debug[mask > 0] = (0, 255, 255)

        if cv2.countNonZero(mask) == 0:
            return "unknown"

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        pixels = hsv[mask > 0]  # shape (N, 3)

        # OpenCV hue range is 0-179. Red wraps around 0 (and 179).
        # Low-saturation high-value pixels (specular highlights) are skipped
        # so they don't dilute the colour vote.
        colors = []
        for h, s, v in pixels:
            if s < 100:
                if v < 100:
                    colors.append("black")
            elif h < 50 or h > 160:
                colors.append("red")
            elif h < 85:
                colors.append("green")
            else:
                colors.append("blue")

        if not colors:
            return "black"

        return Counter(colors).most_common(1)[0][0]

    def get_ring_depth(self, depth_image, ellipse):
        cx, cy = int(ellipse[0][0]), int(ellipse[0][1])
        a, b = ellipse[1][0] / 2, ellipse[1][1] / 2
        angle = np.deg2rad(ellipse[2])

        depths = []
        for t in np.linspace(0, 2 * np.pi, 36):
            x = a * np.cos(t)
            y = b * np.sin(t)
            px = int(cx + x * np.cos(angle) - y * np.sin(angle))
            py = int(cy + x * np.sin(angle) + y * np.cos(angle))

            if 0 <= px < depth_image.shape[1] and 0 <= py < depth_image.shape[0]:
                d = depth_image[py, px]
                if d > 0 and not np.isnan(d) and not np.isinf(d):
                    depths.append(d)

        if len(depths) == 0:
            return None

        z = float(np.median(depths))
        return z if 0.1 < z < 10.0 else None

    def get_3d_position(self, ellipse, header):
        """Localise the ring centre using depth + intrinsics, mirroring the
        face detector's pattern. `header` should be the RGB image header
        whose stamp drives the TF lookup."""
        if self.fx is None or self.depth_image is None:
            return None

        u = int(ellipse[0][0])
        v = int(ellipse[0][1])

        # Sample depth along the ring perimeter, not at the (hollow) centre.
        z = self.get_ring_depth(self.depth_image, ellipse)
        if z is None:
            return None

        p_cam = self._unproject(u, v, z)
        return self._point_to_map(p_cam, header)

    def _unproject(self, u, v, z):
        return np.array(
            [
                (u - self.cx) * z / self.fx,
                (v - self.cy) * z / self.fy,
                z,
            ],
            dtype=np.float32,
        )

    def _point_to_map(self, p_cam, header):
        ps = PointStamped()
        ps.header = header
        ps.point.x = float(p_cam[0])
        ps.point.y = float(p_cam[1])
        ps.point.z = float(p_cam[2])
        try:
            ps_map = self.tf_buffer.transform(ps, "map", timeout=Duration(seconds=0.1))
        except Exception as e:
            self.get_logger().warn(f"TF transform failed: {e}")
            return None
        return np.array([ps_map.point.x, ps_map.point.y, ps_map.point.z])

    def publish_ring(self, pos, color, stamp):
        # state machine message - color packed into frame_id
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = color
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        self.ring_pub.publish(msg)

        # rviz marker
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = stamp
        marker.ns = "rings"
        marker.id = self.marker_id
        self.marker_id += 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(pos[0])
        marker.pose.position.y = float(pos[1])
        marker.pose.position.z = float(pos[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        marker.color.a = 1.0

        color_map = {
            "red": (1.0, 0.0, 0.0),
            "orange": (1.0, 0.5, 0.0),
            "yellow": (1.0, 1.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
            "purple": (0.5, 0.0, 0.5),
            "white": (1.0, 1.0, 1.0),
            "black": (0.1, 0.1, 0.1),
            "gray": (0.5, 0.5, 0.5),
        }
        r, g, b = color_map.get(color, (1.0, 1.0, 1.0))
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.lifetime.sec = 30
        self.marker_pub.publish(marker)

    def get_contours(self, gray, depth):
        # gray[depth == 0] = 178
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10
        )
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        thresh = cv2.dilate(thresh, kernel, iterations=3)
        _, global_thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_and(thresh, global_thresh)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        return contours

        # cv2.imshow("aada", thresh)
        # depth[depth == 0] = np.nan
        # d_min, d_max = np.nanmin(depth), np.nanmax(depth)
        # depth_u8 = np.zeros(depth.shape, dtype=np.uint8)
        # if d_max > d_min:
        #     depth_u8 = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        # depth_u8 = np.nan_to_num(depth_u8, nan=0).astype(np.uint8)

        # thresh_depth = cv2.adaptiveThreshold(
        #     depth_u8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10
        # )
        # thresh_depth[depth_u8 == 0] = 0
        # thresh_depth = cv2.erode(thresh_depth, kernel, iterations=1)
        # thresh_depth = cv2.dilate(thresh_depth, kernel, iterations=2)
        # cnts, _ = cv2.findContours(thresh_depth, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # return contours + cnts

    def fit_ellipses(self, contours):
        elps = []
        for cnt in contours:
            if cnt.shape[0] >= 20:
                ellipse = cv2.fitEllipse(cnt)
            elif cnt.shape[0] >= 10:
                ellipse = cv2.fitEllipseAMS(cnt)
            else:
                continue

            axes = ellipse[1]
            a, b = max(axes), min(axes)
            b = max(b, 0.0001)
            ratio = a / b
            if ratio <= self.ratio_thr and a < self.ecc_thr and b < self.ecc_thr:
                elps.append(ellipse)

        return elps

    def find_ring_candidates(self, elps):
        candidates = []
        for n in range(len(elps)):
            for m in range(n + 1, len(elps)):
                e1, e2 = elps[n], elps[m]

                dist = np.hypot(e1[0][0] - e2[0][0], e1[0][1] - e2[0][1])
                if dist >= self.center_thr:
                    self.get_logger().debug("center thr")
                    continue

                if e1[1][0] >= e2[1][0] and e1[1][1] >= e2[1][1]:
                    le, se = e1, e2
                elif e2[1][0] >= e1[1][0] and e2[1][1] >= e1[1][1]:
                    le, se = e2, e1
                else:
                    self.get_logger().debug("ecc")
                    continue

                major_ratio = se[1][1] / le[1][1]
                minor_ratio = se[1][0] / le[1][0]

                if abs(major_ratio - minor_ratio) > 0.3:
                    self.get_logger().debug("ratio")
                    continue

                if not (0.4 < major_ratio < 0.85):
                    self.get_logger().debug("ratio2")
                    continue

                candidates.append((le, se))

        return candidates

    def check_hollow(self, candidate):
        le, se = candidate
        cx, cy = int(se[0][0]), int(se[0][1])
        rx = int(se[1][0] * 0.1)
        ry = int(se[1][1] * 0.1)

        cy1 = max(0, cy - ry)
        cy2 = min(self.depth_image.shape[0], cy + ry)
        cx1 = max(0, cx - rx)
        cx2 = min(self.depth_image.shape[1], cx + rx)

        inner_depth = self.depth_image[cy1:cy2, cx1:cx2]
        valid = inner_depth[inner_depth > 0]

        if len(valid) == 0:
            return True

        ring_depth = self.get_ring_depth(self.depth_image, le)
        if ring_depth is None:
            return False

        return abs(float(np.mean(valid)) - ring_depth) > 0.1

    def image_callback(self, data):
        if self.depth_image is None:
            return

        depth = self.depth_image.copy()

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        # cv_image, depth = self.remove_pole(cv_image, depth)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        contours = self.get_contours(gray, depth)
        cv2.drawContours(gray, contours, -1, (255, 0, 0), 1)
        cv2.imshow("Detected contours", gray)
        # cv2.imshow("Detected contours", gray)

        elps = self.fit_ellipses(contours)

        candidates = self.find_ring_candidates(elps)
        candidates = [c for c in candidates if self.check_hollow(c)]

        self.get_logger().info(f"Found {len(candidates)} ring candidates")

        for le, se in candidates:
            color = self.get_ring_color(cv_image, (le, se))
            cx, cy = int(le[0][0]), int(le[0][1])
            pos = self.get_3d_position(le, data.header)

            if pos is not None:
                self.publish_ring(pos, color, data.header.stamp)

            cv2.ellipse(cv_image, le, (0, 255, 0), 2)
            cv2.ellipse(cv_image, se, (0, 255, 0), 2)
            cv2.putText(
                cv_image,
                color,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

        for e in elps:
            cv2.ellipse(cv_image, e, (255, 0, 0), 1)

        # cv2.imshow("Detected rings", cv_image)
        cv2.waitKey(1)

    def depth_callback(self, data):
        try:
            raw = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        # Normalise to float32 metres regardless of incoming encoding,
        # so downstream code is unit-agnostic to the underlying camera.
        if raw.dtype == np.uint16:
            depth_image = raw.astype(np.float32) / 1000.0
        else:
            depth_image = raw.astype(np.float32, copy=False)

        depth_image[~np.isfinite(depth_image)] = 0
        self.depth_image = depth_image

    def pointcloud_callback(self, data):
        self.pointcloud = data


def main():
    rclpy.init(args=None)
    rd_node = RingDetector()
    rclpy.spin(rd_node)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
