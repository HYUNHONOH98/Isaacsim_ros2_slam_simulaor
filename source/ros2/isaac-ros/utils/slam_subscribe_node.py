#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import math
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster, TransformStamped
from rclpy.parameter import Parameter
import numpy as np
class SlamSubscriber(Node):
    def __init__(self):
        super().__init__('slam_subscribe_node',
        # parameter_overrides=[
        #     Parameter('use_sim_time', Parameter.Type.BOOL, True)
        # ]
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

    
    def subscribe_est_base_pose(self):
        # Cancel the timer so this callback runs only one time.
        try:
            est_tf : TransformStamped = self.tf_buffer.lookup_transform(
                "odom", "est_pelvis", rclpy.time.Time()
            )
        except:
            self.get_logger().info("SLAM is not initialized yet.")
            return None, None
        est_xyz = np.array([   
                est_tf.transform.translation.x,
                est_tf.transform.translation.y,
                est_tf.transform.translation.z,
            ])
        est_quat = np.array([
                est_tf.transform.rotation.w,
                est_tf.transform.rotation.x,
                est_tf.transform.rotation.y,
                est_tf.transform.rotation.z,
            ])
        return est_xyz, est_quat
    
    def subscribe_est_midsole_pose(self):
        # Cancel the timer so this callback runs only one time.
        try:
            est_tf : TransformStamped = self.tf_buffer.lookup_transform(
                "odom", "est_midsole", rclpy.time.Time()
            )
        except:
            self.get_logger().info("SLAM is not initialized yet.")
            return None, None
        est_xyz = np.array([   
                est_tf.transform.translation.x,
                est_tf.transform.translation.y,
                est_tf.transform.translation.z,
            ])
        est_quat = np.array([
                est_tf.transform.rotation.w,
                est_tf.transform.rotation.x,
                est_tf.transform.rotation.y,
                est_tf.transform.rotation.z,
            ])
        return est_xyz, est_quat

    def subscribe_est_lidar_pose(self):
        # Cancel the timer so this callback runs only one time.
        try:
            est_tf : TransformStamped = self.tf_buffer.lookup_transform(
                "odom", "body", rclpy.time.Time()
            )
        except:
            self.get_logger().info("SLAM is not initialized yet.")
            return None, None
        est_xyz = np.array([   
                est_tf.transform.translation.x,
                est_tf.transform.translation.y,
                est_tf.transform.translation.z,
            ])
        est_quat = np.array([
                est_tf.transform.rotation.w,
                est_tf.transform.rotation.x,
                est_tf.transform.rotation.y,
                est_tf.transform.rotation.z,
            ])
        return est_xyz, est_quat

def main():
    rclpy.init()
    node = SlamSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
