#!/usr/bin/env python3

import rclpy
import numpy as np
import random

from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose as TurtlePose
from rcl_interfaces.msg import SetParametersResult, ParameterDescriptor, FloatingPointRange
from dis_tutorial1.srv import Oblika
import time
import random

class TurtleMover(Node):

    def __init__(self, nodename='turtle_mover', frequency=10):
        super().__init__(nodename)

        self.timer_period = 1/frequency
        self.oblika = ""
        self.trajanje = 0

        self.msg = Twist()

        self.current_pose = None

        self.triangle_timer = 0
        
        self.srv = self.create_service(Oblika, 'oblikuj', self.oblikuj_callback)
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)

    def oblikuj_callback(self, oblikamsg, response):
        self.oblika = oblikamsg.oblika
        trajanje = float(oblikamsg.trajanje)
        zacetek = time.time()        
        while time.time() - zacetek < trajanje:
            if self.oblika == "circle":
                self.msg.linear.x = 1.0
                self.msg.angular.z = 1.0
                self.publisher.publish(self.msg)
            elif self.oblika == "triangle":
                self.msg.linear.x = 1.0
                self.msg.angular.z = 0.0
                self.publisher.publish(self.msg)
                time.sleep(1)

                self.msg.linear.x = 0.0
                self.msg.angular.z = 2.09
                self.publisher.publish(self.msg)
                time.sleep(1)
            elif self.oblika == "rectangle":
                self.msg.linear.x = 2.0
                self.msg.angular.z = 0.0
                self.publisher.publish(self.msg)
                time.sleep(1)

                self.msg.linear.x = 0.0
                self.msg.angular.z = 1.57
                self.publisher.publish(self.msg)
                time.sleep(1)

                self.msg.linear.x = 1.0
                self.msg.angular.z = 0.0
                self.publisher.publish(self.msg)
                time.sleep(1)

                self.msg.linear.x = 0.0
                self.msg.angular.z = 1.57
                self.publisher.publish(self.msg)
                time.sleep(1)
            elif self.oblika == "random":
                self.msg.linear.x = random.uniform(0.5, 3.0)
                self.msg.angular.z = random.uniform(-20.0, 20.0)
                self.publisher.publish(self.msg)
                
        self.msg.linear.x = 0.0
        self.msg.angular.z = 0.0
        self.publisher.publish(self.msg)

        response.oblika_nazaj = self.oblika
        return response


def main(args=None):
    rclpy.init(args=args)

    tm = TurtleMover("TurtleMover")
    rclpy.spin(tm)

    rclpy.shutdown()


if __name__=="__main__":
    main()