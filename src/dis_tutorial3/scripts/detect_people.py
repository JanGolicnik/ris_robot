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
from rclpy.qos import QoSReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from ultralytics import YOLO
from visualization_msgs.msg import Marker

# from rclpy.parameter import Parameter
# from rcl_interfaces.msg import SetParametersResult


class detect_faces(Node):
    def __init__(self):
        super().__init__("detect_faces")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("device", ""),
            ],
        )

        self.set_parameters(
            [rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True)]
        )

        marker_topic = "/people_marker"
        face_topic = "/face_positions"

        self.detection_color = (0, 0, 255)
        self.device = self.get_parameter("device").get_parameter_value().string_value

        self.bridge = CvBridge()
        self.scan = None

        self.rgb_image_sub = self.create_subscription(
            Image,
            "/oakd/rgb/preview/image_raw",
            self.rgb_callback,
            qos_profile_sensor_data,
        )
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            "/oakd/rgb/preview/depth/points",
            self.pointcloud_callback,
            qos_profile_sensor_data,
        )

        self.marker_pub = self.create_publisher(
            Marker, marker_topic, QoSReliabilityPolicy.BEST_EFFORT
        )
        self.face_pos_pub = self.create_publisher(
            PoseStamped, face_topic, QoSReliabilityPolicy.BEST_EFFORT
        )
        self.face_img_pub = self.create_publisher(Image, "/face_image", 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.model = YOLO("yolov8n.pt")

        self.faces = []

        self.cv_image = None

        self.get_logger().info(
            f"Node has been initialized! Will publish face markers to {marker_topic}."
        )

    def rgb_callback(self, data):

        self.faces = []

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.cv_image = cv_image

            self.get_logger().info(f"Running inference on image...")

            # run inference
            res = self.model.predict(
                cv_image,
                imgsz=(256, 320),
                show=False,
                verbose=False,
                classes=[0],
                device=self.device,
            )

            # iterate over results
            for x in res:
                bbox = x.boxes.xyxy
                if bbox.nelement() == 0:  # skip if empty
                    continue

                self.get_logger().info(f"Person has been detected!")

                bbox = bbox[0]

                # draw rectangle
                cv_image = cv2.rectangle(
                    cv_image,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    self.detection_color,
                    3,
                )

                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)

                # draw the center of bounding box
                cv_image = cv2.circle(cv_image, (cx, cy), 5, self.detection_color, -1)

                self.faces.append((cx, cy))

            cv2.imshow("image", cv_image)
            key = cv2.waitKey(1)
            if key == 27:
                print("exiting")
                exit()

        except CvBridgeError as e:
            print(e)

    def pointcloud_callback(self, data):
        height = data.height
        width = data.width

        a = pc2.read_points_numpy(data, field_names=("x", "y", "z"))
        a = a.reshape((height, width, 3))

        print("POINTCLOUD")
        for x, y in self.faces:
            d = a[y, x, :]
            d2 = a[max(y - 7, 0), max(x - 7, 0)]

            if not np.isfinite(d).all() or np.linalg.norm(d) < 0.001:
                print("WASNT ALRIGHT")
                continue

            p1_base = PointStamped()
            p1_base.header.frame_id = "oakd_rgb_camera_optical_frame"
            p1_base.header.stamp = Time().to_msg()
            p1_base.point.x = float(d[0])
            p1_base.point.y = float(d[1])
            p1_base.point.z = float(d[2])

            p2_base = PointStamped()
            p2_base.header.frame_id = "oakd_rgb_camera_optical_frame"
            p2_base.header.stamp = Time().to_msg()
            p2_base.point.x = float(d2[0])
            p2_base.point.y = float(d2[1])
            p2_base.point.z = float(d2[2])

            try:
                p1_map = self.tf_buffer.transform(
                    p1_base, "map", timeout=rclpy.duration.Duration(seconds=0.1)
                )
                p2_map = self.tf_buffer.transform(
                    p2_base, "map", timeout=rclpy.duration.Duration(seconds=0.1)
                )
                # tf = self.tf_buffer.lookup_transform("map", "base_link", Time())
            except Exception as e:
                print(f"ERROR {e}")
                continue

            p1 = np.array(
                [
                    p1_map.point.x,
                    p1_map.point.y,
                    p1_map.point.z,
                ]
            )

            p2 = np.array(
                [
                    p2_map.point.x,
                    p2_map.point.y,
                    p2_map.point.z,
                ]
            )

            down = np.array([0.0, 0.0, -1.0])
            v = p1 - p2
            v = v / np.linalg.norm(v)
            normal = np.cross(down, v) * 0.5

            # normal = point_pos - robot_pos
            # normal = normal / np.linalg.norm(normal)

            face_pose = PoseStamped()
            face_pose.header.frame_id = "map"
            face_pose.header.stamp = Time().to_msg()
            face_pose.pose.position.x = p1[0]
            face_pose.pose.position.y = p1[1]
            face_pose.pose.position.z = p1[2]
            face_pose.pose.orientation.x = normal[0]
            face_pose.pose.orientation.y = normal[1]
            face_pose.pose.orientation.z = normal[2]
            self.face_pos_pub.publish(face_pose)
            print(f"PUBLISHED FACE POS {p1} {p2} {v} {normal}")

            # create marker
            marker = Marker()
            marker.header.frame_id = "/base_link"
            marker.header.stamp = data.header.stamp
            marker.type = 2
            marker.id = 0
            scale = 0.1
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker.pose.position.x = float(d[0])
            marker.pose.position.y = float(d[1])
            marker.pose.position.z = float(d[2])
            self.marker_pub.publish(marker)
            marker.header.frame_id = "/map"
            marker.id = 1
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.pose.position.x = float(p1[0] + normal[0])
            marker.pose.position.y = float(p1[1] + normal[1])
            marker.pose.position.z = float(p1[2] + normal[2])
            self.marker_pub.publish(marker)

            if self.cv_image is not None:
                # use face bounding box - you need the rect, not just center
                # assuming self.faces stores (x, y) center, crop a fixed region around it
                pad = 50
                h, w = self.cv_image.shape[:2]
                x1 = max(0, x - pad)
                x2 = min(w, x + pad)
                y1 = max(0, y - pad)
                y2 = min(h, y + pad)
                face_crop = self.cv_image[y1:y2, x1:x2]
                try:
                    img_msg = self.bridge.cv2_to_imgmsg(face_crop, encoding="bgr8")
                    img_msg.header.stamp = data.header.stamp
                    img_msg.header.frame_id = "oakd_rgb_camera_optical_frame"
                    self.face_img_pub.publish(img_msg)
                except Exception as e:
                    self.get_logger().warn(f"Face image publish failed: {e}")


def main():
    print("Face detection node starting.")

    rclpy.init(args=None)
    node = detect_faces()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
