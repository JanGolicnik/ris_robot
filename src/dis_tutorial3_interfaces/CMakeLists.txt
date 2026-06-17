cmake_minimum_required(VERSION 3.8)
project(dis_tutorial3_interfaces)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FaceDetection.msg"
  DEPENDENCIES geometry_msgs
)

ament_package()
