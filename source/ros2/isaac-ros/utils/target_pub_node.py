#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist, Point, Quaternion, Vector3
from builtin_interfaces.msg import Time
import numpy as np
from tf2_ros import TransformBroadcaster, TransformStamped
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.parameter import Parameter
from .functions import to_ros_time_from_seconds
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs  # 변환 연산 지원
import numpy as np
from tf_transformations import quaternion_matrix, translation_matrix, concatenate_matrices, inverse_matrix, quaternion_from_matrix
class TargetTFPublisher(Node):
    def __init__(self):
        super().__init__('tf_pub',
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

    def publish_tf(self, target_pos, target_quat, simtime):

        # Publish the transform
        t = TransformStamped()
        t.header.stamp = to_ros_time_from_seconds(simtime)
        t.header.frame_id = "odom"
        t.child_frame_id = "target_frame"
        t.transform.translation.x = float(target_pos[0])
        t.transform.translation.y = float(target_pos[1])
        t.transform.translation.z = float(target_pos[2])
        t.transform.rotation.x = float(target_quat[1])
        t.transform.rotation.y = float(target_quat[2])
        t.transform.rotation.z = float(target_quat[3])
        t.transform.rotation.w = float(target_quat[0])

        self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = TargetTFPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()