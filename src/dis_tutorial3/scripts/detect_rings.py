#!/usr/bin/python3

from collections import Counter

import cv2
import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
import tf2_geometry_msgs
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import Marker

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

        self.depth_image = None
        self.pointcloud = None
        self.pointcloud_height = None
        self.pointcloud_width = None
        self.bridge = CvBridge()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.image_sub = self.create_subscription(
            Image, "/oakd/rgb/preview/image_raw", self.image_callback, 1
        )
        self.depth_sub = self.create_subscription(
            Image, "/oakd/rgb/preview/depth", self.depth_callback, 1
        )
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, 1
        )

        self.ring_pub = self.create_publisher(
            PoseStamped,
            "/ring_positions",
            QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT),
        )
        self.marker_pub = self.create_publisher(Marker, "/ring_markers", 10)
        self.marker_id = 0

        # cv2.namedWindow("Detected contours", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("thresh_gray", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("thresh_sat", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("thresh_depth", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected rings", cv2.WINDOW_NORMAL)

    def pointcloud_callback(self, msg):
        self.pointcloud = msg
        self.pointcloud_height = msg.height
        self.pointcloud_width = msg.width

    def remove_pole(self, image, depth):
        return image, depth
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        pole_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 60, 150]))

        kernel = np.ones((5, 5), np.uint8)
        pole_mask = cv2.morphologyEx(pole_mask, cv2.MORPH_CLOSE, kernel)
        pole_mask = cv2.morphologyEx(pole_mask, cv2.MORPH_OPEN, kernel)
        # cv2.imshow("pole mask", pole_mask)
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

        if cv2.countNonZero(mask) == 0:
            return "unknown"

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        pixels = hsv[mask > 0]

        debug = cv_image.copy()
        debug[mask > 0] = (0, 255, 255)

        colors = []
        for h, s, v in pixels:
            if s < 100:
                # colors.append("black")
                colors.append("unknown")
            if h < 10 or h > 160:
                colors.append("red")
            elif h < 25:
                # colors.append("orange")
                colors.append("unknown")
            elif h < 35:
                colors.append("yellow")
            elif h < 85:
                colors.append("green")
            elif h < 130:
                colors.append("blue")
            elif h < 160:
                # colors.append("purple")
                colors.append("unknown")
            else:
                colors.append("unknown")

        if not colors:
            return "unknown"

        return Counter(colors).most_common(1)[0][0]

    def get_3d_position(self, cx, cy, ellipse):
        if self.pointcloud is None:
            return None

        a = pc2.read_points_numpy(self.pointcloud, field_names=("x", "y", "z"))
        a = a.reshape((self.pointcloud_height, self.pointcloud_width, 3))

        mask = np.zeros((self.pointcloud_height, self.pointcloud_width), dtype=np.uint8)
        cv2.ellipse(mask, ellipse, 255, 2)

        ys, xs = np.where(mask > 0)
        points = []
        for px, py in zip(xs, ys):
            d = a[py, px]
            if np.isfinite(d).all() and np.linalg.norm(d) > 0.001:
                points.append(d)

        if len(points) == 0:
            return None

        pt = np.mean(points, axis=0)

        p = PointStamped()
        p.header.frame_id = "oakd_rgb_camera_optical_frame"
        p.header.stamp = Time().to_msg()
        p.point.x = float(pt[0])
        p.point.y = float(pt[1])
        p.point.z = float(pt[2])

        try:
            point_map = self.tf_buffer.transform(p, "map")
            return np.array([point_map.point.x, point_map.point.y, point_map.point.z])
        except Exception as e:
            self.get_logger().warn(f"TF transform failed: {e}")
            return None

    def publish_ring(self, pos, color, stamp):
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = color
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        self.ring_pub.publish(msg)

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

    def get_contours(self, cv_image, depth):
        kernel = np.ones((3, 3), np.uint8)

        # gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # thresh_gray = cv2.adaptiveThreshold(
        #     gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 7
        # )
        # # thresh_gray = cv2.erode(thresh_gray, kernel, iterations=1)
        # # thresh_gray = cv2.dilate(thresh_gray, kernel, iterations=1)
        # gray_contours, _ = cv2.findContours(
        #     thresh_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        # )

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        kernel = np.ones((5, 5), np.uint8)
        sat = cv2.morphologyEx(sat, cv2.MORPH_CLOSE, kernel)
        # sat = cv2.morphologyEx(sat, cv2.MORPH_OPEN, kernel)
        # cv2.imshow("sat", sat)
        thresh_sat = cv2.adaptiveThreshold(
            sat, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 7
        )

        # thresh_sat = cv2.erode(thresh_sat, kernel, iterations=1)
        # thresh_sat = cv2.dilate(thresh_sat, kernel, iterations=1)
        sat_contours, _ = cv2.findContours(
            thresh_sat, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        depth[depth == 0] = np.nan
        d_min, d_max = np.nanmin(depth), np.nanmax(depth)
        depth_u8 = np.zeros(depth.shape, dtype=np.uint8)
        if d_max > d_min:
            depth_u8 = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        depth_u8 = np.nan_to_num(depth_u8, nan=0).astype(np.uint8)
        thresh_depth = cv2.adaptiveThreshold(
            depth_u8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10
        )
        thresh_depth[depth_u8 == 0] = 0
        thresh_depth = cv2.erode(thresh_depth, kernel, iterations=1)
        thresh_depth = cv2.dilate(thresh_depth, kernel, iterations=2)
        depth_contours, _ = cv2.findContours(
            thresh_depth, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        contours = depth_contours + sat_contours
        # contours = gray_contours + depth_contours + sat_contours

        # cv2.imshow("thresh_gray", thresh_gray)
        # cv2.imshow("thresh_sat", thresh_sat)
        # cv2.imshow("thresh_depth", thresh_depth)

        debug = cv_image.copy()
        cv2.drawContours(debug, contours, -1, (255, 0, 0), 1)
        # cv2.imshow("Detected contours", debug)

        return contours

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
                    continue

                if e1[1][0] >= e2[1][0] and e1[1][1] >= e2[1][1]:
                    le, se = e1, e2
                elif e2[1][0] >= e1[1][0] and e2[1][1] >= e1[1][1]:
                    le, se = e2, e1
                else:
                    continue

                major_ratio = se[1][1] / le[1][1]
                minor_ratio = se[1][0] / le[1][0]

                if abs(major_ratio - minor_ratio) > 0.3:
                    continue

                if not (0.4 < major_ratio < 0.85):
                    continue

                candidates.append((le, se))

        return candidates

    def check_hollow(self, candidate, debug_img=None):
        le, se = candidate

        ex, ey = int(le[0][0]), int(le[0][1])
        half_a, half_b = le[1][0] / 2, le[1][1] / 2
        angle = np.deg2rad(le[2])

        rim_depths = []
        for t in np.linspace(0, 2 * np.pi, 36):
            x = half_a * np.cos(t)
            y = half_b * np.sin(t)
            px = int(ex + x * np.cos(angle) - y * np.sin(angle))
            py = int(ey + x * np.sin(angle) + y * np.cos(angle))

            if (
                0 <= px < self.depth_image.shape[1]
                and 0 <= py < self.depth_image.shape[0]
            ):
                d = self.depth_image[py, px]
                if d > 0 and np.isfinite(d):
                    rim_depths.append(d)
                    # if debug_img is not None:
                    #     cv2.circle(debug_img, (px, py), 2, (0, 255, 0), -1)

        if len(rim_depths) == 0:
            return False

        ring_depth = float(np.median(rim_depths))

        cx, cy = int(se[0][0]), int(se[0][1])
        rx = int(se[1][0] * 0.3)
        ry = int(se[1][1] * 0.3)
        cy1 = max(0, cy - ry)
        cy2 = min(self.depth_image.shape[0], cy + ry)
        cx1 = max(0, cx - rx)
        cx2 = min(self.depth_image.shape[1], cx + rx)

        # if debug_img is not None:
        #     cv2.rectangle(debug_img, (cx1, cy1), (cx2, cy2), (0, 0, 255), 1)

        inner_depth = self.depth_image[cy1:cy2, cx1:cx2]
        valid = inner_depth[inner_depth > 0]

        if len(valid) == 0:
            return True

        return float(np.mean(valid)) >= ring_depth + 0.1

    def image_callback(self, data):
        if self.depth_image is None:
            return

        depth = self.depth_image.copy()

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        cv_image, depth = self.remove_pole(cv_image, depth)

        contours = self.get_contours(cv_image, depth)

        elps = self.fit_ellipses(contours)
        # for e in elps:
        #     cv2.ellipse(cv_image, e, (255, 0, 0), 1)

        candidates = self.find_ring_candidates(elps)

        candidates = [c for c in candidates if self.check_hollow(c, cv_image)]

        self.get_logger().info(f"Found {len(candidates)} ring candidates")

        ring_color_img = cv_image.copy()
        for le, se in candidates:
            color = self.get_ring_color(ring_color_img, (le, se))
            if color == "unknown":
                continue
            cx, cy = int(le[0][0]), int(le[0][1])
            pos = self.get_3d_position(cx, cy, le)

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

        cv2.imshow("Detected rings", cv_image)
        cv2.waitKey(1)

    def depth_callback(self, data):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        depth_image[depth_image == np.inf] = 0
        self.depth_image = depth_image


def main():
    rclpy.init(args=None)
    rd_node = RingDetector()
    rclpy.spin(rd_node)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
