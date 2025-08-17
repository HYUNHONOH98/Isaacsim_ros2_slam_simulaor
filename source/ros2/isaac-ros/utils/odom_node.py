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
class OdomPublisher(Node):
    def __init__(self):
        super().__init__('odom_publisher',
        # parameter_overrides=[
        #     Parameter('use_sim_time', Parameter.Type.BOOL, True)
        # ]
        )
        self.odom_pub = self.create_publisher(Odometry, '/g1/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def publish_odometry(self, pos, quat, lin_vel, ang_vel, simtime):
        """
        pos:      tuple/list (x, y, z)
        quat:     tuple/list (x, y, z, w)
        lin_vel:  tuple/list (vx, vy, vz)
        ang_vel:  tuple/list (wx, wy, wz)
        """
        msg = Odometry()
        # now = self.get_clock().now().to_msg()

        # Header
        # msg.header.stamp = now
        msg.header.stamp = to_ros_time_from_seconds(simtime)
        msg.header.frame_id = "odom"
        msg.child_frame_id = "pelvis"

        msg.pose.pose.position.x = float(pos[0])
        msg.pose.pose.position.y = float(pos[1])
        msg.pose.pose.position.z = float(pos[2])
        # Set the orientation
        msg.pose.pose.orientation.w = float(quat[0])
        msg.pose.pose.orientation.x = float(quat[1])
        msg.pose.pose.orientation.y = float(quat[2])
        msg.pose.pose.orientation.z = float(quat[3])
        # Set the linear velocity
        msg.twist.twist.linear.x = float(lin_vel[0])
        msg.twist.twist.linear.y = float(lin_vel[1])
        msg.twist.twist.linear.z = float(lin_vel[2])
        # Set the angular velocity
        msg.twist.twist.angular.x = float(ang_vel[0])
        msg.twist.twist.angular.y = float(ang_vel[1])
        msg.twist.twist.angular.z = float(ang_vel[2])

        self.odom_pub.publish(msg)
        # self.get_logger().info("Published pelvis odometry.")

        self.publish_tf(pos, quat, simtime)
        # self.get_logger().info("Published odom tf.")

    def publish_tf(self, pos, quat, simtime):
        t = TransformStamped()
        # t.header.stamp = self.get_clock().now().to_msg()
        t.header.stamp = to_ros_time_from_seconds(simtime)
        t.header.frame_id = "odom"
        t.child_frame_id = "pelvis"

        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = float(pos[2])
        t.transform.rotation.w = float(quat[0])
        t.transform.rotation.x = float(quat[1])
        t.transform.rotation.y = float(quat[2])
        t.transform.rotation.z = float(quat[3])

        self.tf_broadcaster.sendTransform(t)

def main():
    rclpy.init()
    node = OdomPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()