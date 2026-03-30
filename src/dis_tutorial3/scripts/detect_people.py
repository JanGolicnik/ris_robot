#!/usr/bin/env python3

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

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.model = YOLO("yolov8n.pt")

        self.faces = []

        self.get_logger().info(
            f"Node has been initialized! Will publish face markers to {marker_topic}."
        )

    def rgb_callback(self, data):

        self.faces = []

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

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

            if not np.isfinite(d).all() or np.linalg.norm(d) < 0.001:
                print("WASNT ALRIGHT")
                continue

            point_base = PointStamped()
            point_base.header.frame_id = "oakd_rgb_camera_optical_frame"
            point_base.header.stamp = Time().to_msg()
            point_base.point.x = float(d[0])
            point_base.point.y = float(d[1])
            point_base.point.z = float(d[2])

            try:
                point_map = self.tf_buffer.transform(
                    point_base, "map", timeout=rclpy.duration.Duration(seconds=0.1)
                )
                tf = self.tf_buffer.lookup_transform("map", "base_link", Time())
            except Exception as e:
                print(f"ERROR {e}")
                continue

            robot_pos = np.array(
                [
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z,
                ]
            )

            point_pos = np.array(
                [
                    point_map.point.x,
                    point_map.point.y,
                    point_map.point.z,
                ]
            )

            normal = point_pos - robot_pos
            normal = normal / np.linalg.norm(normal)

            face_pose = PoseStamped()
            face_pose.header.frame_id = "map"
            face_pose.header.stamp = Time().to_msg()
            face_pose.pose.position.x = point_map.point.x
            face_pose.pose.position.y = point_map.point.y
            face_pose.pose.position.z = point_map.point.z
            face_pose.pose.orientation.x = normal[0]
            face_pose.pose.orientation.y = normal[1]
            face_pose.pose.orientation.z = normal[2]
            self.face_pos_pub.publish(face_pose)
            print("PUBLISHED FACE POS")

            # create marker
            marker = Marker()

            marker.header.frame_id = "/base_link"
            marker.header.stamp = data.header.stamp

            marker.type = 2
            marker.id = 0

            # Set the scale of the marker
            scale = 0.1
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale

            # Set the color
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0

            # Set the pose of the marker
            marker.pose.position.x = float(d[0])
            marker.pose.position.y = float(d[1])
            marker.pose.position.z = float(d[2])

            self.marker_pub.publish(marker)


def main():
    print("Face detection node starting.")

    rclpy.init(args=None)
    node = detect_faces()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
