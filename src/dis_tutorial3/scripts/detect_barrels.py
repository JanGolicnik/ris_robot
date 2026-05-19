#!/usr/bin/env python3

import numpy as np
import rclpy
import tf2_ros
import tf2_geometry_msgs
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from visualization_msgs.msg import Marker


# HSV ranges for barrel colors. Tune these by checking actual colors in sim.
COLOR_RANGES = {
    "red":    [(np.array([0, 100, 50]),   np.array([10, 255, 255])),
               (np.array([170, 100, 50]), np.array([180, 255, 255]))],
    "green":  [(np.array([40, 80, 50]),   np.array([80, 255, 255]))],
    "blue":   [(np.array([100, 100, 50]), np.array([130, 255, 255]))],
    "yellow": [(np.array([20, 100, 100]), np.array([35, 255, 255]))],
    "purple": [(np.array([130, 50, 50]),  np.array([160, 255, 255]))],
    "orange": [(np.array([10, 150, 100]), np.array([20, 255, 255]))],
    "brown":  [(np.array([5, 80, 30]),    np.array([20, 200, 150]))],
}


class BarrelDetector(Node):
    def __init__(self):
        super().__init__("detect_barrels")

        self.bridge = CvBridge()
        self.cv_image = None
        self.candidates_in_image = []  # list of (cx, cy, color, orientation)

        self.image_sub = self.create_subscription(
            Image, "/oakd/rgb/preview/image_raw",
            self.image_callback, qos_profile_sensor_data
        )
        self.pc_sub = self.create_subscription(
            PointCloud2, "/oakd/rgb/preview/depth/points",
            self.pointcloud_callback, qos_profile_sensor_data
        )

        self.barrel_pub = self.create_publisher(
            PoseStamped, "/barrel_positions", QoSReliabilityPolicy.BEST_EFFORT
        )
        self.marker_pub = self.create_publisher(Marker, "/barrel_marker", 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info("Barrel detector started")

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        self.candidates_in_image = []

        for color, ranges in COLOR_RANGES.items():
            mask = None
            for lo, hi in ranges:
                m = cv2.inRange(hsv, lo, hi)
                mask = m if mask is None else cv2.bitwise_or(mask, m)

            # cleanup noise
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 800:  # too small, probably noise
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                aspect = h / w if w > 0 else 0

                # vertical barrel: tall and narrow (aspect > 1.3)
                # horizontal barrel: wide and short (aspect < 0.8)
                # everything in between is probably partial view or false positive
                if aspect > 1.3:
                    orientation = "vertical"
                elif aspect < 0.8:
                    orientation = "horizontal"
                else:
                    continue  # ambiguous, skip

                cx = x + w // 2
                cy = y + h // 2
                self.candidates_in_image.append((cx, cy, color, orientation))

                # draw for debugging
                cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(cv_image, f"{color} {orientation}",
                            (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Barrels", cv_image)
        cv2.waitKey(1)

    def pointcloud_callback(self, data):
        if not self.candidates_in_image:
            return

        h, w = data.height, data.width
        points = pc2.read_points_numpy(data, field_names=("x", "y", "z"))
        points = points.reshape((h, w, 3))

        for cx, cy, color, orientation in self.candidates_in_image:
            if not (0 <= cy < h and 0 <= cx < w):
                continue

            d = points[cy, cx]
            if not np.isfinite(d).all() or np.linalg.norm(d) < 0.001:
                continue

            p_cam = PointStamped()
            p_cam.header.frame_id = "oakd_rgb_camera_optical_frame"
            p_cam.header.stamp = data.header.stamp
            p_cam.point.x = float(d[0])
            p_cam.point.y = float(d[1])
            p_cam.point.z = float(d[2])

            try:
                p_map = self.tf_buffer.transform(
                    p_cam, "map", timeout=rclpy.duration.Duration(seconds=0.1)
                )
            except Exception as e:
                self.get_logger().warn(f"TF transform failed: {e}")
                continue

            # publish position, abuse header.frame_id to encode color + orientation
            pose = PoseStamped()
            pose.header.frame_id = f"{color}:{orientation}"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = p_map.point.x
            pose.pose.position.y = p_map.point.y
            pose.pose.position.z = p_map.point.z
            self.barrel_pub.publish(pose)

            self.get_logger().info(
                f"Barrel: {color} {orientation} at "
                f"({p_map.point.x:.2f}, {p_map.point.y:.2f})"
            )


def main():
    rclpy.init()
    node = BarrelDetector()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()