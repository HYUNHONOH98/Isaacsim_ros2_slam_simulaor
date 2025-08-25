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

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from utils import math_utils

yaw_quat = math_utils.as_np(
    math_utils.yaw_quat
)
from tf_transformations import quaternion_matrix, translation_matrix, concatenate_matrices, inverse_matrix, quaternion_from_matrix
class MidsoleTFPublisher(Node):
    def __init__(self):
        super().__init__('midsole_tf_publisher',
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

    def publish_tf(self, midsole_pos_w, midsole_quat_w, base_pos_w, base_quat_w, simtime):
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
        def pos_quat_to_matrix(pos, quat):
            t = translation_matrix([pos[0], pos[1], pos[2]])
            q = [quat[1], quat[2], quat[3], quat[0]]
            r = quaternion_matrix(q)
            return concatenate_matrices(t, r)
        def transform_to_matrix(transform, yaw_only=False):
            trans = transform.transform.translation
            rot = transform.transform.rotation
            t = translation_matrix([trans.x, trans.y, trans.z])
            q = [rot.x, rot.y, rot.z, rot.w]
            if yaw_only:
                q = yaw_quat(np.array(q))  # Convert to yaw-only quaternion
            r = quaternion_matrix(q)
            return concatenate_matrices(t, r)
        
        # Convert world poses to transformation matrices
        midsole_mat_w = pos_quat_to_matrix(midsole_pos_w, midsole_quat_w)
        base_mat_w = pos_quat_to_matrix(base_pos_w, base_quat_w)

        # Compute base->midsole transform: T_base^-1 * T_midsole
        base_mat_w_inv = inverse_matrix(base_mat_w)
        midsole_mat_b = np.dot(base_mat_w_inv, midsole_mat_w)


        slam_mat = transform_to_matrix(slam_tf, yaw_only=True)
        gt_mat = transform_to_matrix(gt_tf, yaw_only=True)
        # midsole_mat = pos_quat_to_matrix(midsole_pos_b, midsole_quat_b)

        # The chain is: odom->body->mid360_link->pelvis
        # body->mid360_link is identity (since body is estimated mid360_link)
        # mid360_link->pelvis is gt_tf
        # So, odom->estimated_pelvis = slam_mat @ gt_mat

        odom_to_estimated_pelvis_mat = np.dot(slam_mat, gt_mat)
        odom_to_estimated_midsole_mat = np.dot(odom_to_estimated_pelvis_mat, midsole_mat_b)

        # Extract translation and quaternion
        trans = odom_to_estimated_midsole_mat[:3, 3]
        quat = quaternion_from_matrix(odom_to_estimated_midsole_mat)

        # Publish the transform
        t = TransformStamped()
        t.header.stamp = to_ros_time_from_seconds(simtime)
        t.header.frame_id = "odom"
        t.child_frame_id = "est_midsole"
        t.transform.translation.x = trans[0]
        t.transform.translation.y = trans[1]
        t.transform.translation.z = 0.
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)


    def publish_tf_from_feet(self, l_pos_w, l_quat_w, r_pos_w, r_quat_w, base_pos_w, base_quat_w, simtime):
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
        def pos_quat_to_matrix(pos, quat):
            """
            quat : wxyz
            """
            t = translation_matrix([pos[0], pos[1], pos[2]])
            q = [quat[1], quat[2], quat[3], quat[0]] # wxyz -> xyzw
            r = quaternion_matrix(q) # xyzw
            return concatenate_matrices(t, r)
        def transform_to_matrix(transform, yaw_only=False, yaw_pitch_only = False):
            trans = transform.transform.translation
            rot = transform.transform.rotation
            t = translation_matrix([trans.x, trans.y, trans.z])
            q = [rot.w, rot.x, rot.y, rot.z] # wxyz
            if yaw_only:
                q = yaw_quat(np.array(q))  # Convert to yaw-only quaternion

            q = [q[1], q[2], q[3], q[0]]
            if yaw_pitch_only:
                # Convert to yaw-pitch-only quaternion
                r = R.from_quat(q)
                euler = r.as_euler('xyz', degrees=False)
                euler[0] = 0.0  # Zero out roll
                r_yaw_pitch = R.from_euler('xyz', euler, degrees=False)
                q = r_yaw_pitch.as_quat()  # xy
            r = quaternion_matrix(q) # xyzw
            return concatenate_matrices(t, r)
        
        # Convert world poses to transformation matrices
        l_mat_w = pos_quat_to_matrix(l_pos_w, l_quat_w)
        r_mat_w = pos_quat_to_matrix(r_pos_w, r_quat_w)
        base_mat_w = pos_quat_to_matrix(base_pos_w, base_quat_w)

        midsole_pos_w = (l_mat_w[:3, 3] + r_mat_w[:3, 3]) / 2.0

        mid_sole_quat_w = np.roll(Slerp([0,1],R.from_quat([np.roll(l_quat_w,-1), np.roll(r_quat_w,-1)]))(0.5).as_quat(),1)
        mid_sole_quat_w = yaw_quat(mid_sole_quat_w) # wxyz

        mid_sole_mat_w = pos_quat_to_matrix(midsole_pos_w, mid_sole_quat_w)

        # Compute base->midsole transform: T_base^-1 * T_midsole
        base_mat_w_inv = inverse_matrix(base_mat_w)
        mid_sole_mat_b = np.dot(base_mat_w_inv, mid_sole_mat_w)
        
        # slam_mat = transform_to_matrix(slam_tf, yaw_pitch_only=True)
        slam_mat = transform_to_matrix(slam_tf)
        gt_mat = transform_to_matrix(gt_tf)
        # midsole_mat_b = transform_to_matrix(mid_sole_mat_b)

        # The chain is: odom->body->mid360_link->pelvis
        # body->mid360_link is identity (since body is estimated mid360_link)
        # mid360_link->pelvis is gt_tf
        # So, odom->estimated_pelvis = slam_mat @ gt_mat

        odom_to_estimated_pelvis_mat = np.dot(slam_mat, gt_mat)
        odom_to_estimated_midsole_mat = np.dot(odom_to_estimated_pelvis_mat, mid_sole_mat_b)

        # Extract translation and quaternion
        trans = odom_to_estimated_midsole_mat[:3, 3]
        quat = quaternion_from_matrix(odom_to_estimated_midsole_mat)

        # Publish the transform
        t = TransformStamped()
        t.header.stamp = to_ros_time_from_seconds(simtime)
        t.header.frame_id = "odom"
        t.child_frame_id = "est_midsole"
        t.transform.translation.x = trans[0]
        t.transform.translation.y = trans[1]
        t.transform.translation.z = 0.
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)


        trans = odom_to_estimated_pelvis_mat[:3, 3]
        quat = quaternion_from_matrix(odom_to_estimated_pelvis_mat)

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

        l_mat_b = np.dot(base_mat_w_inv, l_mat_w)
        r_mat_b = np.dot(base_mat_w_inv, r_mat_w)
        odom_to_estimated_pelvis_mat = np.dot(slam_mat, gt_mat)
        odom_to_estimated_l_mat = np.dot(odom_to_estimated_pelvis_mat, l_mat_b)
        odom_to_estimated_r_mat = np.dot(odom_to_estimated_pelvis_mat, r_mat_b)

        l_trans = odom_to_estimated_l_mat[:3, 3]
        l_quat = quaternion_from_matrix(odom_to_estimated_l_mat)

        r_trans = odom_to_estimated_r_mat[:3, 3]
        r_quat = quaternion_from_matrix(odom_to_estimated_r_mat)


        t = TransformStamped()
        t.header.stamp = to_ros_time_from_seconds(simtime)
        t.header.frame_id = "odom"
        t.child_frame_id = "est_left_foot"
        t.transform.translation.x = l_trans[0]
        t.transform.translation.y = l_trans[1]
        t.transform.translation.z = 0.
        t.transform.rotation.x = l_quat[0]
        t.transform.rotation.y = l_quat[1]
        t.transform.rotation.z = l_quat[2]
        t.transform.rotation.w = l_quat[3]

        self.tf_broadcaster.sendTransform(t)


        t = TransformStamped()
        t.header.stamp = to_ros_time_from_seconds(simtime)
        t.header.frame_id = "odom"
        t.child_frame_id = "est_right_foot"
        t.transform.translation.x = r_trans[0]
        t.transform.translation.y = r_trans[1]
        t.transform.translation.z = 0.
        t.transform.rotation.x = r_quat[0]
        t.transform.rotation.y = r_quat[1]
        t.transform.rotation.z = r_quat[2]
        t.transform.rotation.w = r_quat[3]

        self.tf_broadcaster.sendTransform(t)


