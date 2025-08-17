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
class LidarTFPublisher(Node):
    def __init__(self):
        super().__init__('lidar_tf_publisher',
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

    def publish_tf(self, lidar_pos, lidar_quat, base_pos, base_quat, simtime):
        try:
            gt_tf : TransformStamped = self.tf_buffer.lookup_transform(
                    "mid360_link", "pelvis", rclpy.time.Time()
                )
        except:
            self.get_logger().info("No mid360 <> pelvis")
            return
        try:
            slam_tf : TransformStamped = self.tf_buffer.lookup_transform(
                    "odom", "body", rclpy.time.Time()
                )
        except:
            self.get_logger().info("No odom <> body")
            return
        # Compute the transformation from "odom" to "estimated_pelvis"
        # "body" is the estimated pose of "mid360_link"
        # So, we want: odom -> body -> mid360_link -> pelvis
        # Compose the transforms: odom->body (slam_tf), body->mid360_link (identity), mid360_link->pelvis (gt_tf)

        # Convert ROS quaternions to transformation matrices
        def transform_to_matrix(transform):
            trans = transform.transform.translation
            rot = transform.transform.rotation
            t = translation_matrix([trans.x, trans.y, trans.z])
            q = [rot.x, rot.y, rot.z, rot.w]
            r = quaternion_matrix(q)
            return concatenate_matrices(t, r)

        slam_mat = transform_to_matrix(slam_tf)
        gt_mat = transform_to_matrix(gt_tf)

        # The chain is: odom->body->mid360_link->pelvis
        # body->mid360_link is identity (since body is estimated mid360_link)
        # mid360_link->pelvis is gt_tf
        # So, odom->estimated_pelvis = slam_mat @ gt_mat

        odom_to_estimated_pelvis_mat = np.dot(slam_mat, gt_mat)

        # Extract translation and quaternion
        trans = odom_to_estimated_pelvis_mat[:3, 3]
        quat = quaternion_from_matrix(odom_to_estimated_pelvis_mat)

        # Publish the transform
        t = TransformStamped()
        t.header.stamp = to_ros_time_from_seconds(simtime)
        t.header.frame_id = "odom"
        t.child_frame_id = "est_pelvis"
        t.transform.translation.x = trans[0]
        t.transform.translation.y = trans[1]
        t.transform.translation.z = trans[2]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = LidarTFPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()