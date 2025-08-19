# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import NormalVelocityCommandCfg, UniformVelocityCommandCfg


class UniformVelocityCommand(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """

    cfg: UniformVelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.env = env

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, heading
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["action_acceleration"] = torch.zeros(self.num_envs, device=self.device)
        self.step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.metrics["error_vel_xy_log"] = torch.zeros(self.num_envs, 351, device=self.device)
        self.metrics["error_vel_x_log"] = torch.zeros(self.num_envs, 351, device=self.device)
        self.metrics["error_vel_y_log"] = torch.zeros(self.num_envs, 351, device=self.device)
        self.metrics["error_vel_yaw_log"] = torch.zeros(self.num_envs, 351, device=self.device)

        # -- similarity measure
        self.metrics["similarity_index_step_distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["similarity_index_air_time"] = torch.zeros(self.num_envs, device=self.device)
        # -- head stability
        self.metrics["lidar_lin_acc"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["lidar_ang_acc"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["lidar_lin_jerk"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["lidar_ang_jerk"] = torch.zeros(self.num_envs, device=self.device)
        self.lidar_lin_acc_buf = torch.zeros((self.num_envs, 1), device=self.device)
        self.lidar_ang_acc_buf = torch.zeros((self.num_envs, 1), device=self.device)

        self.last_lidar_lin_acc = None
        self.last_lidar_ang_acc = None

        self.lidar_lin_jerk_square_sum = torch.zeros(self.num_envs, device=self.device)
        self.lidar_ang_jerk_square_sum = torch.zeros(self.num_envs, device=self.device)

        self.env_counter = torch.zeros(self.num_envs, device=self.device)

        # -- walking motion
        self.metrics["knee_actuation"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["air_time_diff"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["both_feet_contact_ratio"] = torch.zeros(self.num_envs, device=self.device)
        self.both_feet_contact_counter = torch.zeros(self.num_envs, device=self.device)

        self.step_distance = torch.ones(self.num_envs, 2, device=self.device)

        self.sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
        self.sensor_cfg.resolve(self.env.scene)

        # -- deployability
        # self.metrics["action_acceleration"] = torch.zeros(self.num_envs, device=self.device)
        # self.metrics["dof_pos_jitter"] = torch.zeros(self.num_envs, device=self.device)

        # self.vel_cmds = torch.zeros(1, self.num_envs, 3, device=self.device)
        # self.vel_states = torch.zeros(1, self.num_envs, 3, device=self.device)

        self.total_counter = 0

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        self.env_counter += 1
        self.total_counter += 1

        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] = torch.norm(
            self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2],
            dim=-1,
        )
        self.metrics["error_vel_yaw"] = torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        row = torch.arange(self.num_envs, device=self.device)
        col = self.step_counter.to(torch.long).clamp(max=self.metrics["error_vel_x_log"].shape[1] - 1) # why?
        self.metrics["error_vel_x_log"][row, col] = torch.abs(self.vel_command_b[:, 0] - self.robot.data.root_lin_vel_b[:, 0])
        self.metrics["error_vel_y_log"][row, col] = torch.abs(self.vel_command_b[:, 1] - self.robot.data.root_lin_vel_b[:, 1])
        # self.metrics["error_vel_xy_log"][row, col] = self.metrics["error_vel_xy"].clone()
        self.metrics["error_vel_yaw_log"][row, col] = self.metrics["error_vel_yaw"].clone()
        # print(torch.mean(self.metrics["error_vel_xy_log"]))
        self.step_counter += 1
        ########################## Walking motion ##########################
        self.metrics["pelvis_height"] = torch.abs(self.robot.data.root_pos_w[:, 2])

        left_knee_id = self.robot.find_joints("left_knee_joint")[0][0]
        right_knee_id = self.robot.find_joints("right_knee_joint")[0][0]
        knee_joint_bound = (
            self.robot.data.soft_joint_pos_limits[:, left_knee_id, 1]
            - self.robot.data.soft_joint_pos_limits[:, left_knee_id, 0]
        )
        left_knee_actuation = (
            self.robot.data.joint_pos[:, left_knee_id] - self.robot.data.soft_joint_pos_limits[:, left_knee_id, 0]
        ) / knee_joint_bound
        right_knee_actuation = (
            self.robot.data.joint_pos[:, right_knee_id] - self.robot.data.soft_joint_pos_limits[:, right_knee_id, 0]
        ) / knee_joint_bound
        # from icecream import ic
        # ic(left_knee_actuation)
        # ic(right_knee_actuation)
        # ic()
        self.metrics["knee_actuation"] += ((left_knee_actuation + right_knee_actuation) / 2) / max_command_step

        contact_sensor = self.env.scene.sensors[self.sensor_cfg.name]
        last_contact_feet_pose_w = contact_sensor.data.last_contact_pose_w[:, self.sensor_cfg.body_ids, :]
        current_feet_pose_w = contact_sensor.data.pose_w[:, self.sensor_cfg.body_ids, :]

        feet_forces = torch.abs(contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids, 2])
        feet_contact_mask = feet_forces > 0.5

        self.both_feet_contact_counter += (torch.sum(feet_contact_mask, dim=-1) == 2).float()
        self.metrics["both_feet_contact_ratio"] = self.both_feet_contact_counter / max_command_step

        ########################## Symmetricity ##########################
        # Symmetric locomotion related metrics

        last_contact_l_pose_w = last_contact_feet_pose_w[:, 0, :]
        last_contact_r_pose_w = last_contact_feet_pose_w[:, 1, :]
        current_contact_l_pose_w = current_feet_pose_w[:, 0, :]
        current_contact_r_pose_w = current_feet_pose_w[:, 1, :]
        l_step_distance = torch.norm(last_contact_l_pose_w - current_contact_l_pose_w, dim=1)  # (N,)
        r_step_distance = torch.norm(last_contact_r_pose_w - current_contact_r_pose_w, dim=1)  # (N,)

        step_distance = torch.stack([l_step_distance, r_step_distance], dim=1)  # (N,2)
        self.step_distance = torch.where(feet_contact_mask, step_distance, self.step_distance)
        step_diff = torch.abs(self.step_distance[:, 0] - self.step_distance[:, 1])
        step_mean = torch.mean(self.step_distance, dim=1)
        self.metrics["similarity_index_step_distance"] = torch.where(
            step_mean == 0, torch.zeros_like(step_diff), torch.div(step_diff, step_mean)
        )

        last_air_time = contact_sensor.data.last_air_time[:, self.sensor_cfg.body_ids]  # (N,2)
        air_diff = torch.abs(last_air_time[:, 0] - last_air_time[:, 1])
        air_mean = torch.mean(last_air_time, dim=1)
        self.metrics["similarity_index_air_time"] = torch.where(
            air_mean == 0, torch.zeros_like(air_diff), torch.div(air_diff, air_mean)
        )

        ########################## Head link stability ##########################
        # Calculate head link stability metrics.
        # mean over envs ( mean over timestamps ( measurements ) )
        if self.total_counter > 100:
            lidar_link_id = self.robot.find_bodies("mid360_link_frame")[0][0]
            # LiDAR unit's angular, linear acceleration
            curr_lidar_ang_acc = self.robot.data.body_ang_acc_w[:, lidar_link_id, :]
            ang_acc_norm = torch.norm(curr_lidar_ang_acc, dim=-1)
            self.lidar_ang_acc_buf = torch.cat([self.lidar_ang_acc_buf, ang_acc_norm.unsqueeze(-1)], dim=-1)
            self.metrics["lidar_ang_acc"] = torch.mean(self.lidar_ang_acc_buf, dim=1)

            curr_lidar_lin_acc = self.robot.data.body_lin_acc_w[:, lidar_link_id, :]
            lin_acc_norm = torch.norm(curr_lidar_lin_acc, dim=-1)
            self.lidar_lin_acc_buf = torch.cat([self.lidar_lin_acc_buf, lin_acc_norm.unsqueeze(-1)], dim=-1)
            self.metrics["lidar_lin_acc"] = torch.mean(self.lidar_lin_acc_buf, dim=1)


            # LiDAR unit's angular, linear jerk
            if self.last_lidar_lin_acc is None or self.last_lidar_ang_acc is None:
                self.last_lidar_lin_acc = curr_lidar_lin_acc.clone()
                self.last_lidar_ang_acc = curr_lidar_ang_acc.clone()
            else:
                self.lidar_lin_jerk_square_sum += torch.norm(curr_lidar_lin_acc - self.last_lidar_lin_acc, dim=-1) ** 2
                self.lidar_ang_jerk_square_sum += torch.norm(curr_lidar_ang_acc - self.last_lidar_ang_acc, dim=-1) ** 2
                self.metrics["lidar_lin_jerk"] = (
                    torch.sqrt(self.lidar_lin_jerk_square_sum / self.env_counter) / self._env.step_dt
                )
                self.metrics["lidar_ang_jerk"] = (
                    torch.sqrt(self.lidar_ang_jerk_square_sum / self.env_counter) / self._env.step_dt
                )

                self.last_lidar_lin_acc = curr_lidar_lin_acc.clone()
                self.last_lidar_ang_acc = curr_lidar_ang_acc.clone()
        
        ########################## Action smoothness ##########################
        self.metrics["action_acceleration"] = torch.norm(
            self.env.action_manager.action - 2* self.env.action_manager.prev_action + self.env.action_manager.prev_prev_action,
            dim=-1
        ) ** 2

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset the command generator and log metrics.

        This function resets the command counter and resamples the command. It should be called
        at the beginning of each episode.

        Args:
            env_ids: The list of environment IDs to reset. Defaults to None.

        Returns:
            A dictionary containing the information to log under the "{name}" key.
        """
        # resolve the environment IDs
        if env_ids is None:
            env_ids = slice(None)
        # set the command counter to zero
        self.command_counter[env_ids] = 0
        # resample the command
        self.step_counter[env_ids] = 0
        self._resample(env_ids)
        # add logging metrics
        extras = {}
        for metric_name, metric_value in self.metrics.items():
            # compute the mean metric value
            extras[metric_name] = torch.mean(metric_value[env_ids]).item()
            extras[f"per_env/{metric_name}"] = metric_value.clone()
            # reset the metric value
            metric_value[env_ids] = 0.0

        self.both_feet_contact_counter[env_ids] = 0
        self.env_counter[env_ids] = 0

        self.lidar_lin_acc_buf = torch.zeros((self.num_envs, 1), device=self.device)
        self.lidar_ang_acc_buf = torch.zeros((self.num_envs, 1), device=self.device)
        self.last_lidar_lin_acc = None
        self.last_lidar_ang_acc = None

        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

        if self.cfg.test_type == "walking_alive" :
            self.is_standing_env[env_ids] |= r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
            early_mask = self.step_counter[env_ids] <= (1.0 / 0.2)   # 5 스텝
            early_ids  = env_ids[early_mask]
            self.is_standing_env[early_ids] = False

            old_mask = self.step_counter[env_ids] > (4.5 / 0.2)      # 22.5 스텝
            old_ids  = env_ids[old_mask]
            self.is_standing_env[old_ids] = True

        if self.cfg.test_type == "standing_alive":
            self.is_standing_env[env_ids] = True

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        # Enforce standing (i.e., zero velocity command) for standing envs
        # TODO: check if conversion is needed
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


class NormalVelocityCommand(UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from a normal distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    The command is sampled from a normal distribution with mean and standard deviation specified in
    the configuration. With equal probability, the sign of the individual components is flipped.
    """

    cfg: NormalVelocityCommandCfg
    """The command generator configuration."""

    def __init__(self, cfg: NormalVelocityCommandCfg, env: ManagerBasedEnv):
        """Initializes the command generator.

        Args:
            cfg: The command generator configuration.
            env: The environment.
        """
        super().__init__(cfg, env)
        # create buffers for zero commands envs
        self.is_zero_vel_x_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_zero_vel_y_env = torch.zeros_like(self.is_zero_vel_x_env)
        self.is_zero_vel_yaw_env = torch.zeros_like(self.is_zero_vel_x_env)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "NormalVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    def _resample_command(self, env_ids):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.normal_(mean=self.cfg.ranges.mean_vel[0], std=self.cfg.ranges.std_vel[0])
        self.vel_command_b[env_ids, 0] *= torch.where(r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.normal_(mean=self.cfg.ranges.mean_vel[1], std=self.cfg.ranges.std_vel[1])
        self.vel_command_b[env_ids, 1] *= torch.where(r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0)
        # -- angular velocity - yaw direction
        self.vel_command_b[env_ids, 2] = r.normal_(mean=self.cfg.ranges.mean_vel[2], std=self.cfg.ranges.std_vel[2])
        self.vel_command_b[env_ids, 2] *= torch.where(r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0)

        # update element wise zero velocity command
        # TODO what is zero prob ?
        self.is_zero_vel_x_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[0]
        self.is_zero_vel_y_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[1]
        self.is_zero_vel_yaw_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[2]

        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Sets velocity command to zero for standing envs."""
        # Enforce standing (i.e., zero velocity command) for standing envs
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()  # TODO check if conversion is needed
        self.vel_command_b[standing_env_ids, :] = 0.0

        # Enforce zero velocity for individual elements
        # TODO: check if conversion is needed
        zero_vel_x_env_ids = self.is_zero_vel_x_env.nonzero(as_tuple=False).flatten()
        zero_vel_y_env_ids = self.is_zero_vel_y_env.nonzero(as_tuple=False).flatten()
        zero_vel_yaw_env_ids = self.is_zero_vel_yaw_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[zero_vel_x_env_ids, 0] = 0.0
        self.vel_command_b[zero_vel_y_env_ids, 1] = 0.0
        self.vel_command_b[zero_vel_yaw_env_ids, 2] = 0.0
