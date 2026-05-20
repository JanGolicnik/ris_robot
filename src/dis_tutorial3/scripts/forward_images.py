#!/usr/bin/env python3

import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import (
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image


class detect_faces(Node):
    def __init__(self):
        super().__init__("detect_faces")

        self.declare_parameters(namespace="", parameters=[("device", "")])

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sync = ApproximateTimeSynchronizer(
            [
                Subscriber(
                    self,
                    Image,
                    "/gemini/color/image_raw",
                    qos,
                ),
                Subscriber(
                    self,
                    Image,
                    "/gemini/depth/image_raw",
                    qos,
                ),
            ],
            queue_size=10,
            slop=2.0,
        )

        self.sync.registerCallback(self.images_callback)
        self.n_images = 0
        self.create_timer(5.0, self.rate_callback)

        self.rgb_pub = self.create_publisher(Image, "/robot_rgb_image", qos)
        self.depth_pub = self.create_publisher(Image, "/robot_depth_image", qos)

    def images_callback(self, rgb_msg: Image, depth_msg: Image):
        self.get_logger().info("got images")
        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)
        self.n_images += 1

    def rate_callback(self):
        self.get_logger().info(f"rate: {self.n_images / 5.0}")
        self.n_images = 0


def main():
    rclpy.init(args=None)
    node = detect_faces()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
