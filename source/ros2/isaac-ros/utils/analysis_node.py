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
class AnalysisNode(Node):
    def __init__(self):
        super().__init__('static_tf_publisher',
        parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True)
        ]
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.tf_timer = self.create_timer(0.05, self.analyze_error)

        self.gt_xyz = []
        self.gt_quat = []
        self.est_xyz = []
        self.est_quat = []
    
    def analyze_error(self):
        # Cancel the timer so this callback runs only one time.
        try:
            gt_tf : TransformStamped = self.tf_buffer.lookup_transform(
                "odom", "mid360_link_frame", rclpy.time.Time()
            )
            est_tf : TransformStamped = self.tf_buffer.lookup_transform(
                "odom", "body", rclpy.time.Time()
            )
        except:
            self.get_logger().info("SLAM is not initialized yet.")
            return

        self.gt_xyz.append(
            np.array([   
                gt_tf.transform.translation.x,
                gt_tf.transform.translation.y,
                gt_tf.transform.translation.z,
            ])
        )
        self.est_xyz.append(
            np.array([   
                est_tf.transform.translation.x,
                est_tf.transform.translation.y,
                est_tf.transform.translation.z,
            ])
        )
        self.gt_quat.append(
            np.array([
                gt_tf.transform.rotation.x,
                gt_tf.transform.rotation.y,
                gt_tf.transform.rotation.z,
                gt_tf.transform.rotation.w,
            ])
        )
        self.est_quat.append(
            np.array([
                est_tf.transform.rotation.x,
                est_tf.transform.rotation.y,
                est_tf.transform.rotation.z,
                est_tf.transform.rotation.w,
            ])
        )

        if len(self.gt_xyz) > 240 and len(self.est_xyz) > 240:
            gt_arr = np.array(self.gt_xyz)
            est_arr = np.array(self.est_xyz)
            # Calculate RMSE on xy axis
            diff_xy = gt_arr[240:, :2] - est_arr[240:, :2]

            rmse_xy = np.sqrt(np.mean(np.sum(diff_xy ** 2, axis=1)))
            self.get_logger().info(f"ATE (RMSE on xy): {rmse_xy:.6f}")
            mean_xy = np.mean(np.linalg.norm(diff_xy, axis=1))
            std_xy = np.std(np.linalg.norm(diff_xy, axis=1))
            self.get_logger().info(f"XY error mean: {mean_xy:.6f}, std: {std_xy:.6f}")

            # if len(self.gt_xyz) == 780:
            if len(self.gt_xyz) == 1000:
                import matplotlib.pyplot as plt
                xy_error_norm = np.linalg.norm(diff_xy, axis=1)
                plt.figure()
                plt.plot(xy_error_norm)
                plt.title("XY Error Norm Over Time")
                plt.xlabel("Frame")
                plt.ylabel("XY Error Norm (meters)")
                plt.grid(True)

                plt.savefig(f"g1_data/exp_log/slam_error.png")
                print("FIGURE SAVED")

def main():
    rclpy.init()
    node = AnalysisNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
