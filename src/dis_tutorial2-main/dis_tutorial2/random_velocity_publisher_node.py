#!/usr/bin/env python3

import rclpy
import random

from rclpy.node import Node
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import SetParametersResult, ParameterDescriptor, FloatingPointRange

class VelocityPublisher(Node):

    def __init__(self, nodename='velocity_publisher', frequency=5):
        super().__init__(nodename)

        self.timer_period = 1/frequency
        self.counter = 0

        self.lin_scale = 5.0
        self.ang_scale = 5.0

        desc = ParameterDescriptor(
            description="Linear speed multiplier for teleop",
            floating_point_range=[FloatingPointRange(from_value=0.0, to_value=10.0, step=0.01)]
        )

        self.declare_parameter('scale_linear', self.lin_scale, desc)
        self.declare_parameter('scale_angular', self.ang_scale, desc)
        
        self.publisher = self.create_publisher(Twist, "cmd_vel", 10)

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.add_on_set_parameters_callback(self._on_params)

    def _on_params(self, params):
        for p in params:
            if p.name == 'scale_linear':
                self.lin_scale = float(p.value)
                self.get_logger().info(f"Updated speed -> {self.lin_scale}")
            elif p.name == 'scale_angular':
                self.ang_scale = float(p.value)
                self.get_logger().info(f"Updated speed -> {self.ang_scale}")

        return SetParametersResult(successful=True)

    def publish_vel(self):
        scale_linear = self.lin_scale 
        scale_angular = self.ang_scale

        cmd_msg = Twist()
        cmd_msg.linear.x = scale_linear * random.random()
        cmd_msg.angular.z = scale_angular * (random.random() - 0.5)
        self.publisher.publish(cmd_msg)
        self.get_logger().info(f"The parameters are: scale_linear {scale_linear}, scale_angular:{scale_angular}")
        self.get_logger().info(f"I published a Twist command lin:{cmd_msg.linear.x}, ang:{cmd_msg.angular.z}")

    def timer_callback(self):
        self.publish_vel()
        self.counter += 1

def main(args=None):
    rclpy.init(args=args)

    vp = VelocityPublisher("velocity_publisher")
    rclpy.spin(vp)

    rclpy.shutdown()


if __name__=="__main__":
    main()