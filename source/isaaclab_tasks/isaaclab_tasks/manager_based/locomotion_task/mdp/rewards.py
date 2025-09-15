# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch as th
from typing import TYPE_CHECKING, Union, List

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.envs import mdp
import isaaclab.utils.math as math_utils
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.assets import Articulation, RigidObject
from icecream import ic
from . import zmp_rwd_computation_helper as zmp
import numpy as np
import math


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> th.Tensor:
    #Penalize joint torques applied on the articulation using L2 squared kernel

    #NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return th.sum(th.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
"""


def action_limits(env, action_name: str, command_name: str, limits: List[float] = [-0.5, 0.5]):
    action = env.action_manager.get_term(action_name)
    # compute out of limits constraints
    out_of_limits = -(action.raw_actions - limits[0]).clip(max=0.0)
    out_of_limits += (action.raw_actions - limits[1]).clip(min=0.0)

    rew = th.sum(out_of_limits, dim=1)
    # command \
    #     = env.command_manager.get_term(command_name)
    # # Zero-mask reward for the moving command envs
    # moving_env_ids = (~command.is_standing_env).nonzero(as_tuple=False).flatten()
    # rew[moving_env_ids] = 0.

    return rew


def track_lin_vel_xy_yaw_frame_exp_v2(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> th.Tensor:
    """Reward tracking of base linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = th.sum(th.square(command.vel_command_b[:, :2] - vel_yaw[:, :2]), dim=1)
    rew = th.exp(-lin_vel_error / std**2)

    # Zero-mask reward for the standing command envs
    standing_env_ids = command.is_standing_env.nonzero(as_tuple=False).flatten()
    rew[standing_env_ids] = 0.0

    return rew


def track_ang_vel_z_world_exp_v2(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> th.Tensor:
    """Reward tracking of base angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)
    ang_vel_error = th.square(command.vel_command_b[:, 2] - asset.data.root_ang_vel_w[:, 2])

    rew = th.exp(-ang_vel_error / std**2)
    # Zero-mask reward for the standing command envs
    standing_env_ids = command.is_standing_env.nonzero(as_tuple=False).flatten()
    rew[standing_env_ids] = 0.0

    return rew


def joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> th.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    rew = th.sum(th.abs(angle), dim=1)

    command = env.command_manager.get_term(command_name)
    # Zero-mask reward for the standing command envs
    standing_env_ids = command.is_standing_env.nonzero(as_tuple=False).flatten()
    rew[standing_env_ids] = 0.0

    return rew


def joint_deviation_from_initial_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> th.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    # from icecream import ic

    # ic(asset.data.initial_pos[:, asset_cfg.joint_ids].shape)
    # ic(asset.data.joint_pos[:, asset_cfg.joint_ids].shape)
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.initial_pos[:, asset_cfg.joint_ids]
    rew = th.sum(th.abs(angle), dim=1)

    # print(len(asset_cfg.joint_ids))

    return rew


def joint_deviation_arm(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> th.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    default_arm_pos = th.tensor(
        [
            0.3500,
            0.3500,
            0.1600,
            -0.1600,
            0.0000,
            0.0000,
            0.8700,
            0.8700,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
        ],
        device=env.device,
    )
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - default_arm_pos
    rew = th.sum(th.abs(angle), dim=1)

    command = env.command_manager.get_term(command_name)
    # Zero-mask reward for the standing command envs
    standing_env_ids = command.is_standing_env.nonzero(as_tuple=False).flatten()
    rew[standing_env_ids] = 0.0

    return rew


def action_norm_l2(env, action_name: str):
    action = env.action_manager.get_term(action_name)

    if "Residual_action_norm" not in env.reward_manager.episode_stat_sums.keys():
        env.reward_manager.episode_stat_sums["Raw_Residual_action_norm"] = th.zeros(
            env.num_envs, dtype=th.float, device=env.device
        )
        env.reward_manager.episode_stat_sums["Residual_action_norm"] = th.zeros(
            env.num_envs, dtype=th.float, device=env.device
        )

    env.reward_manager.episode_stat_sums["Residual_action_norm"] += th.norm(action.processed_actions, dim=-1)

    env.reward_manager.episode_stat_sums["Raw_Residual_action_norm"] += th.norm(action.raw_actions, dim=-1)

    # return th.sum(th.square(action.processed_actions), dim=-1)
    return th.sum(th.square(action.raw_actions), dim=-1)


def rel_joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> th.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    rel_torque = asset.data.applied_torque / asset.root_physx_view.get_dof_max_forces().clone().to(env.device)
    rew = th.sum(th.square(rel_torque[:, asset_cfg.joint_ids]), dim=1)
    # extract the used quantities (to enable type-hinting)
    return rew


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> th.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = th.sum((last_air_time - threshold) * first_contact, dim=1)
    # print()
    # ic(first_contact)
    # ic(last_air_time)
    # ic(reward)
    # no reward for zero command
    reward *= th.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > 0.05
    # reward *= (~env.command_manager.get_term(command_name).is_standing_env).float()

    return reward


def air_time_v4(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.4,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
) -> th.Tensor:

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = th.where(in_contact, contact_time, air_time)
    # print()
    # ic(contact_time)
    # ic(air_time)
    # ic(in_mode_time)
    single_stance = th.sum(in_contact.int(), dim=1) == 1
    rew = th.min(th.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]

    rew = th.where(rew < threshold, rew, 2 * threshold - rew)
    rew = th.clamp(rew, min=0.0)

    rew *= th.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.05

    return rew


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> th.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # checking if contacts exist
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # retrieve velocity of feet
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    feet_links = ["left_ankle_roll_link", "right_ankle_roll_link"]
    feet_ids = asset.find_bodies(feet_links)[0]
    body_vel = asset.data.body_lin_vel_w[:, feet_ids, :2]

    # compute the reward as: velocity norm if there is a contact, otherwise zero.
    reward = th.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> th.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = th.sum(
        th.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]),
        dim=1,
    )
    return th.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> th.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = th.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return th.exp(-ang_vel_error / std**2)


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> th.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    energy = th.clip(
        asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids],
        min=0,
    )

    return th.sum(energy, dim=-1)


def action_rate_l2(env: ManagerBasedRLEnv) -> th.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return th.sum(
        th.square(env.action_manager.action - env.action_manager.prev_action),
        dim=1,
    )


def standing_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
) -> th.Tensor:
    """
    Penalizes zero stance or single stance for stationaly command
    """

    command = env.command_manager.get_term(command_name)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    double_stance = th.sum(in_contact.int(), dim=1) == 2
    rew = (~double_stance).float()

    # Zero-mask reward for the moving command envs
    # Gets zero for moving environment; otherwise, you get 0 rwd if both feet are on the ground,
    #   1 if only one foot is ground
    moving_env_ids = (~command.is_standing_env).nonzero(as_tuple=False).flatten()
    rew[moving_env_ids] = 0.0

    return rew


def standing_still_four_contact_points(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    left_foot_sensor_cfg: SceneEntityCfg,
    right_foot_sensor_cfg: SceneEntityCfg,
) -> th.Tensor:

    command = env.command_manager.get_term(command_name)

    asset: Articulation = env.scene[asset_cfg.name]
    left_foot_sensor: ContactSensorExtra = env.scene.sensors[left_foot_sensor_cfg.name]
    right_foot_sensor: ContactSensorExtra = env.scene.sensors[right_foot_sensor_cfg.name]

    left_forces, left_points, left_masks = zmp.process_contact_data(
        left_foot_sensor.data.c_force,
        left_foot_sensor.data.c_normal,
        left_foot_sensor.data.c_point,
        left_foot_sensor.data.c_idx,
        left_foot_sensor.data.c_num,
    )

    right_forces, right_points, right_masks = zmp.process_contact_data(
        right_foot_sensor.data.c_force,
        right_foot_sensor.data.c_normal,
        right_foot_sensor.data.c_point,
        right_foot_sensor.data.c_idx,
        right_foot_sensor.data.c_num,
    )

    left_num_contacts = left_masks.sum(dim=-1)
    right_num_contacts = right_masks.sum(dim=-1)
    total_num_contacts = left_num_contacts + right_num_contacts
    total_num_sensors = 8

    # return 1 for indicating not all feet are in contact
    not_all_feet_in_contacts = total_num_contacts != total_num_sensors
    rew = not_all_feet_in_contacts.float()

    # zero reward for moving envs
    moving_env_ids = (~command.is_standing_env).nonzero(as_tuple=False).flatten()
    rew[moving_env_ids] = 0.0
    return rew


def action_smoothness_l2(env: ManagerBasedRLEnv) -> th.Tensor:
    "Penalize the discrepancy between consecutive actions."
    action_smoothness = (
        env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action
    )
    rew = th.sum(th.square(action_smoothness), dim=1)
    return rew


def com_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> th.Tensor:
    """
    rewards for diffrence between target and current com height
    """
    asset: Articulation = env.scene[asset_cfg.name]

    command = env.command_manager.get_term(command_name)

    command_name = f"get_eetrack_height"
    func = getattr(command, command_name, None)
    target_height = func()

    com_height = zmp.compute_com(asset, env.device)[:, 2]

    height_diff = th.abs(com_height - target_height) - 0.2
    rew = th.clamp(height_diff, min=0.0)
    moving_env_ids = (~command.is_standing_env).nonzero(as_tuple=False).flatten()
    rew[moving_env_ids] = 0.0
    return rew


def foot_outward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> th.Tensor:
    asset = env.scene[asset_cfg.name]
    l_body_name: str = "left_ankle_roll_link"
    r_body_name: str = "right_ankle_roll_link"
    l_body_idx = asset.find_bodies(l_body_name)[0][0]
    r_body_idx = asset.find_bodies(r_body_name)[0][0]
    l_yaw_w = math_utils.yaw_quat(asset.data.body_quat_w[:, l_body_idx, :])
    r_yaw_w = math_utils.yaw_quat(asset.data.body_quat_w[:, r_body_idx, :])
    torso_yaw_w = math_utils.yaw_quat(asset.data.root_quat_w)
    rew = th.norm(th.clamp(torso_yaw_w - l_yaw_w, min=0.0), dim=-1) + th.norm(
        th.clamp(r_yaw_w - torso_yaw_w, min=0.0), dim=-1
    )
    # ic(rew)
    return rew


def ankle_parallel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> th.Tensor:
    """
    Reward by the z_pos variance of ankle keypoints.
    Following HoST paper (https://arxiv.org/pdf/2502.08378)
    """

    def construct_keypoints(pos, quat):
        # keypoints in body frame
        kpts_b = (
            th.tensor([[-0.05, 0.025, -0.03], [-0.05, -0.025, -0.03], [0.12, 0.03, -0.03], [0.12, -0.03, -0.03]])
            .to(pos.device)
            .repeat(pos.shape[0], 1, 1)
        )

        kpts_w = math_utils.transform_points(kpts_b, pos, quat)

        return kpts_w

    asset = env.scene[asset_cfg.name]
    feet_links = ["left_ankle_roll_link", "right_ankle_roll_link"]
    feet_ids = asset.find_bodies(feet_links)[0]

    feet_pos = asset.data.body_pos_w[:, feet_ids]
    feet_quat = asset.data.body_quat_w[:, feet_ids]

    left_feet_kpts = construct_keypoints(feet_pos[:, 0], feet_quat[:, 0])
    right_feet_kpts = construct_keypoints(feet_pos[:, 1], feet_quat[:, 1])

    z_var = left_feet_kpts[..., 2].var(dim=1) + right_feet_kpts[..., 2].var(dim=1)

    return z_var


def keep_torso_upright(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> th.Tensor:
    """Penalize non-flat torso orientation using L2 squared kernel.
    This is computed by penalizing the xy-components of the projected gravity vector.
    Referenced from mdp.flat_orientation_l2
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    torso_link_idx = asset.find_bodies("torso_link")[0][0]
    torso_quat_w = asset.data.body_quat_w[:, torso_link_idx, :]

    torso_projected_gravity_b = math_utils.quat_rotate_inverse(torso_quat_w, asset.data.GRAVITY_VEC_W)
    return th.sum(th.square(torso_projected_gravity_b[:, :2]), dim=1)


def follow_command_vel_z(env: ManagerBasedRLEnv, command_name: str) -> th.Tensor:
    """Reward tracking of linear velocity commands (z axis) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    command_term = env.command_manager.get_term(command_name)
    command_vel_z_w = command_term.command[..., 0]

    asset = env.scene["robot"]
    pelvis_vel_z_w = asset.data.root_lin_vel_w[:, 2]
    return th.square((command_vel_z_w - pelvis_vel_z_w) / command_term.cfg.max_velocity)


class is_terminated_term(ManagerTermBase):
    """Penalize termination for specific terms that don't correspond to episodic timeouts.
    Migrated here from isaaclab.envs.mdp.rewards because it is used in our mdp

    The parameters are as follows:

    * attr:`term_keys`: The termination terms to penalize. This can be a string, a list of strings
      or regular expressions. Default is ".*" which penalizes all terminations.

    The reward is computed as the sum of the termination terms that are not episodic timeouts.
    This means that the reward is 0 if the episode is terminated due to an episodic timeout. Otherwise,
    if two termination terms are active, the reward is 2.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        term_keys = cfg.params.get("term_keys", ".*")
        self._term_names = env.termination_manager.find_terms(term_keys)

    def __call__(self, env: ManagerBasedRLEnv, term_keys: str | list[str] = ".*") -> th.Tensor:
        # Return the unweighted reward for the termination terms
        reset_buf = th.zeros(env.num_envs, device=env.device)
        for term in self._term_names:
            # Sums over terminations term values to account for multiple terminations in the same step
            reset_buf += env.termination_manager.get_term(term)

        return (reset_buf * (~env.termination_manager.time_outs)).float()


def standing_still_four_contact_points_v2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    left_foot_sensor_cfg: SceneEntityCfg,
    right_foot_sensor_cfg: SceneEntityCfg,
) -> th.Tensor:
    """Discrete version of standing_still_four_contact_points which is binary."""

    command = env.command_manager.get_term(command_name)

    asset: Articulation = env.scene[asset_cfg.name]
    left_foot_sensor: ContactSensorExtra = env.scene.sensors[left_foot_sensor_cfg.name]
    right_foot_sensor: ContactSensorExtra = env.scene.sensors[right_foot_sensor_cfg.name]

    left_forces, left_points, left_masks = zmp.process_contact_data(
        left_foot_sensor.data.c_force,
        left_foot_sensor.data.c_normal,
        left_foot_sensor.data.c_point,
        left_foot_sensor.data.c_idx,
        left_foot_sensor.data.c_num,
    )

    right_forces, right_points, right_masks = zmp.process_contact_data(
        right_foot_sensor.data.c_force,
        right_foot_sensor.data.c_normal,
        right_foot_sensor.data.c_point,
        right_foot_sensor.data.c_idx,
        right_foot_sensor.data.c_num,
    )

    left_num_contacts = left_masks.sum(dim=-1)
    right_num_contacts = right_masks.sum(dim=-1)
    total_num_contacts = left_num_contacts + right_num_contacts
    total_num_sensors = 8

    rew = (total_num_contacts >= total_num_sensors).float()

    # zero reward for moving envs
    moving_env_ids = (~command.is_standing_env).nonzero(as_tuple=False).flatten()
    rew[moving_env_ids] = 0.0
    return rew


def penalize_command_efficient_joint_power(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> th.Tensor:
    """Penalize the normalized joint power of the robot."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # compute the normalized joint power
    joint_power = th.sum(
        th.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
        * th.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )

    command_denominator = th.clip(th.sum(th.square(command[:, :2]), dim=-1) + 0.2 * th.square(command[:, 2]), min=0.1)

    rew = joint_power / command_denominator

    return rew


def penalize_torque_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> th.Tensor:
    """Penalize the joint torques that exceed the soft torque limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = (
        th.abs(asset.data.applied_torque[:, asset_cfg.joint_ids])
        - asset.data.soft_torque_limits[:, asset_cfg.joint_ids]
    ).clip(min=0.0)
    return th.sum(out_of_limits, dim=1)


def penalize_feet_xy_vel_on_step(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> th.Tensor:
    """Penalize feet slip using the contact forces sensor."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    # compute the reward as the norm of the contact forces
    feet_contact_mask = th.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]) > 1.0

    # compute the feet xy velocity
    feet_xy_vel = th.norm(asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :2], dim=-1)

    rew = th.sum(feet_xy_vel * feet_contact_mask, dim=-1)

    return rew


def penalize_feet_xy_force_on_step(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    force_ratio: float = 3.0,
) -> th.Tensor:
    """Penalize feet slip using the contact forces sensor."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # compute the feet z contact force
    feet_z_force = th.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])

    # compute the feet xy contact force
    feet_xy_force = th.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=-1)

    rew = th.any(feet_xy_force > force_ratio * feet_z_force, dim=1).float()

    return rew


def avoid_both_foot_off_ground(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    force_threshold: float = 0.5,
    curriculum_weight: float = 1.0,
) -> th.Tensor:
    """This is a binary reward that is 1 when one foot is off the ground, and 0 otherwise."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    command = env.command_manager.get_command(command_name)

    # compute the reward as the norm of the contact forces
    feet_forces = th.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    feet_contact_mask = feet_forces > force_threshold

    # Ensure the reward manager has a buffer for feet forces
    if not hasattr(env.reward_manager, "feet_forces_buffer"):
        env.reward_manager.feet_forces_buffer = th.zeros((1000, env.num_envs, feet_forces.shape[1]), device=env.device)

    # Shift buffer and store current feet forces
    env.reward_manager.feet_forces_buffer = th.roll(env.reward_manager.feet_forces_buffer, shifts=-1, dims=0)
    env.reward_manager.feet_forces_buffer[-1] = feet_forces

    # ic(env.reward_manager.feet_forces_buffer[-10:,...])

    single_contact = th.sum(feet_contact_mask, dim=1) == 1
    double_contact = th.sum(feet_contact_mask, dim=1) == 2

    # rew = single_contact.float()
    # full reward for zero command
    # rew = th.max(rew, 1.0 * th.norm(command[:, :3], dim=-1) < 0.05)
    rew = th.where(th.norm(command[:, :3], dim=-1) > 0.05, single_contact.float(), double_contact.float())
    # rew = th.where(~env.command_manager.get_term(command_name).is_standing_env, single_contact.float(), double_contact.float())
    # rew = th.max(rew, 1.0 * env.command_manager.get_term(command_name).is_standing_env.float())

    return curriculum_weight * rew


def standing_still_four_contact_points_v3(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    left_foot_sensor_cfg: SceneEntityCfg,
    right_foot_sensor_cfg: SceneEntityCfg,
) -> th.Tensor:
    """Standing still with four contact points, but with a more relaxed condition."""
    command = env.command_manager.get_term(command_name)

    asset: Articulation = env.scene[asset_cfg.name]
    left_foot_sensor: ContactSensorExtra = env.scene.sensors[left_foot_sensor_cfg.name]
    right_foot_sensor: ContactSensorExtra = env.scene.sensors[right_foot_sensor_cfg.name]

    left_forces, left_points, left_masks = zmp.process_contact_data(
        left_foot_sensor.data.c_force,
        left_foot_sensor.data.c_normal,
        left_foot_sensor.data.c_point,
        left_foot_sensor.data.c_idx,
        left_foot_sensor.data.c_num,
    )

    right_forces, right_points, right_masks = zmp.process_contact_data(
        right_foot_sensor.data.c_force,
        right_foot_sensor.data.c_normal,
        right_foot_sensor.data.c_point,
        right_foot_sensor.data.c_idx,
        right_foot_sensor.data.c_num,
    )

    left_num_contacts = left_masks.sum(dim=-1)
    right_num_contacts = right_masks.sum(dim=-1)
    total_num_contacts = left_num_contacts + right_num_contacts
    total_num_sensors = 8

    rew = (total_num_contacts >= total_num_sensors - 2).float()

    rew *= th.norm(command.command[:, :3], dim=-1) < 0.05  # works only for zero command
    # rew *= command.is_standing_env.float()
    # rew *= asset.data.body_pos_w[:, asset.find_bodies("pelvis")[0][0], 2] >= 0.735  # pelvis is above a threshold

    return rew


def feet_swing_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    feet_height: float = 0.08,
) -> th.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    contact = contact_time > 0.0

    asset: Articulation = env.scene[asset_cfg.name]
    pos_error = th.square(asset.data.body_state_w[..., asset_cfg.body_ids, 2] - feet_height) * ~contact

    return th.sum(pos_error, dim=1)


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> th.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = th.sum(th.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1)
    return th.exp(-lin_vel_error / std**2)


def body_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> th.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        Currently, it assumes a flat terrain, i.e. the target height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # TODO: Fix this for rough-terrain.
    ic("body_height_l2")
    ic(asset.data.body_pos_w[:, asset_cfg.body_ids, 2])
    return th.sum(th.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height), dim=1)


def body_flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> th.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    projected_gravity_b = math_utils.quat_rotate_inverse(body_quat_w, asset.data.GRAVITY_VEC_W)

    return th.sum(th.square(projected_gravity_b[:, :2]), dim=1)


def keep_body_frame_y_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_threshold: float = 0.2,
    left_body_name: str = "left_ankle_roll_link",
    right_body_name: str = "right_ankle_roll_link",
) -> th.Tensor:
    """Reward for keeping the distance between the feet within a certain range."""
    asset: Articulation = env.scene[asset_cfg.name]

    left_body_ids = asset.find_bodies(left_body_name)[0][0]
    right_body_ids = asset.find_bodies(right_body_name)[0][0]
    body_pos_left_b, _ = math_utils.subtract_frame_transforms(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        asset.data.body_state_w[:, left_body_ids, :3],
        asset.data.body_state_w[:, left_body_ids, 3:7],
    )
    body_pos_right_b, _ = math_utils.subtract_frame_transforms(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        asset.data.body_state_w[:, right_body_ids, :3],
        asset.data.body_state_w[:, right_body_ids, 3:7],
    )
    # ic(th.abs(body_pos_left_b[:, 1] - body_pos_right_b[:, 1]))
    body_y_distance = distance_threshold - th.abs(body_pos_left_b[:, 1] - body_pos_right_b[:, 1])
    # return th.clamp(body_y_distance, min=0.0)  # penalize only if the distance is greater than the threshold
    return th.abs(body_y_distance)  # penalize only if the distance is greater than the threshold


def encourage_symmetric_step(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
) -> th.Tensor:
    asset = env.scene[asset_cfg.name]

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    last_contact_feet_pose_w = contact_sensor.data.last_contact_pose_w[:, sensor_cfg.body_ids, :]
    current_feet_pose_w = contact_sensor.data.pose_w[:, sensor_cfg.body_ids, :]

    feet_forces = th.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    feet_contact_mask = feet_forces > 0.5

    last_contact_l_pose_w = last_contact_feet_pose_w[:, 0, :]
    last_contact_r_pose_w = last_contact_feet_pose_w[:, 1, :]
    current_contact_l_pose_w = current_feet_pose_w[:, 0, :]
    current_contact_r_pose_w = current_feet_pose_w[:, 1, :]

    def compute_body_pos(contact_pose_w):
        """Compute the body position in the body frame."""
        body_pos_b, _ = math_utils.subtract_frame_transforms(
            asset.data.root_pos_w,
            asset.data.root_quat_w,
            contact_pose_w[:, :3],
            contact_pose_w[:, 3:7],
        )
        return body_pos_b

    last_body_pos_l_b = compute_body_pos(last_contact_l_pose_w)
    last_body_pos_r_b = compute_body_pos(last_contact_r_pose_w)
    current_body_pos_l_b = compute_body_pos(current_contact_l_pose_w)
    current_body_pos_r_b = compute_body_pos(current_contact_r_pose_w)

    l_state = (last_body_pos_l_b[:, 0] * current_body_pos_l_b[:, 0]) < 0
    r_state = (last_body_pos_r_b[:, 0] * current_body_pos_r_b[:, 0]) < 0

    return th.sum(th.stack([l_state, r_state], dim=1) * feet_contact_mask, dim=1)


def feet_contact_rhythm_from_gait_phase(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    command_name: str = "gait_phase",
    stance_ratio: float = 0.7,
):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_forces = th.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    feet_contact_mask = feet_forces > 1.0  # (N, 2)

    phase_command = env.command_manager.get_term(command_name)
    phase = phase_command.phase

    # stance_ratio = th.tensor(stance_ratio, requires_grad=False, device=env.device)
    c = np.cos(math.pi * stance_ratio)
    I_l = th.tanh(10 * (th.cos(2 * math.pi * (phase + 0.5)) - c))
    I_r = th.tanh(10 * (th.cos(2 * math.pi * phase) - c))
    gait_rew = th.stack([I_l, I_r], dim=-1).float() * feet_contact_mask
    rew = th.sum(gait_rew, dim=-1)
    # rew *= (th.norm(env.command_manager.get_command("base_velocity")[:, :3], dim=-1) > 0.05).float()

    return rew


def penalize_joint_velocity_on_standing(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
):
    asset = env.scene[asset_cfg.name]
    joint_vels = asset.data.joint_vel[:, asset_cfg.joint_ids]
    rew = th.sum(th.abs(joint_vels), dim=1)

    command = env.command_manager.get_term(command_name)
    # Zero-mask reward for the standing command envs
    standing = th.norm(command.command[:, :3], dim=-1) < 0.05
    rew *= standing.float()

    return rew


def encourage_both_feet_contact_on_standing(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    command_name: str = "base_velocity",
):
    command = env.command_manager.get_term(command_name)
    # Zero-mask reward for the standing command envs
    standing = th.norm(command.command[:, :3], dim=-1) < 0.05

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_forces = th.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    double_contact = th.sum(feet_forces > 0.5, dim=-1) == 2

    return double_contact.float() * standing


def encourage_low_lidar_lin_acc(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lidar_frame_name: str = "mid360_link_frame",
    std: float = 2,
):
    asset = env.scene[asset_cfg.name]
    lidar_frame_id = asset.find_bodies(lidar_frame_name)[0][0]
    lidar_lin_acc = asset.data.body_lin_acc_w[:, lidar_frame_id, :]

    return th.exp(-th.norm(lidar_lin_acc, dim=-1) / std**2)


def encourage_low_lidar_ang_acc(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lidar_frame_name: str = "mid360_link_frame",
    std: float = 2,
):
    asset = env.scene[asset_cfg.name]
    lidar_frame_id = asset.find_bodies(lidar_frame_name)[0][0]
    lidar_ang_acc = asset.data.body_ang_acc_w[:, lidar_frame_id, :]

    return th.exp(-th.norm(lidar_ang_acc, dim=-1) / std**2)


def body_lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> th.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    body_lin_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :].squeeze(1)

    body_lin_vel_b = math_utils.quat_rotate_inverse(body_quat_w, body_lin_vel_w)
    return th.square(body_lin_vel_b[:, 2])


def body_ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> th.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    body_ang_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :].squeeze(1)

    body_ang_vel_b = math_utils.quat_rotate_inverse(body_quat_w, body_ang_vel_w)
    return th.sum(th.square(body_ang_vel_b[:, :2]), dim=1)


def penalize_foot_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    force_threshold: float = 400.0,
):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_forces = th.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    forces_over_threshold_mask = feet_forces > force_threshold  # (N, 2)

    rew = th.sum(forces_over_threshold_mask.float(), dim=-1)
    return rew


def penalize_torso_ang_acc_z_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> th.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    torso_id = asset.find_bodies("torso_link")[0][0]
    quat_w = asset.data.body_quat_w[:, torso_id, :]
    ang_acc_w = asset.data.body_ang_acc_w[:, torso_id, :]
    ang_acc_b = quat_rotate_inverse(quat_w, ang_acc_w)

    # ic(th.sum(th.square(ang_acc_b[:, 2:]), dim=1))
    return th.sum(th.square(ang_acc_b[:, 2:]), dim=1)


def penalize_feet_z_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    threshold: float = 400.0,
):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_forces = th.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] - threshold)

    # ic(feet_forces)
    feet_forces_over_threshold = th.clamp(feet_forces, min=0.0)
    
    return th.sum(feet_forces_over_threshold, dim=-1)

# def straight_support_leg(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
# ):
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     feet_forces = th.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
#     feet_contact_mask = feet_forces > 1.0  # (N, 2)

#     # (Pdb) contact_sensor.find_bodies("left_ankle_roll_link")
#     # ([7], ['left_ankle_roll_link'])
#     # (Pdb) contact_sensor.find_bodies("right_ankle_roll_link")
#     # ([14], ['right_ankle_roll_link'])

#     asset = env.scene[asset_cfg.name]
#     l_knee_id = asset.find_joints("left_knee_joint")[0][0]
#     r_knee_id = asset.find_joints("right_knee_joint")[0][0]
#     knee_joints = asset.data.joint_pos[:, [l_knee_id, r_knee_id]]


#     return


@configclass
class G1LocomotionRewards:
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    ### Regualization terms
    # dof vel >> mdp.joint_vel_l2
    # dof vel limit >> mdp.joint_vel_limits
    # dof acc >> mdp.joint_acc_l2
    # dof pos limit >> mdp.joint_pos_limits
    # torques >> rel_joint_torques_l2
    # joint power >> penalize_command_efficient_joint_power
    # action rate >> action_rate_l2
    # action smoothness >> action_smoothness_l2
    # orientation >> flat_orientation_l2
    # torque limit >> penalize_torque_limits

    rel_torques_l2 = RewTerm(
        func=rel_joint_torques_l2,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist_.*_joint",  # NOTE(HH) include or not?
                    ".*_ankle_roll_joint",
                    ".*_ankle_pitch_joint",
                    ".*_hip_pitch_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_yaw_joint",
                    ".*_knee_joint",
                ],
            )
        },
    )

    energy = RewTerm(
        func=energy,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist_.*_joint",  # NOTE(HH) include or not?
                    ".*_ankle_roll_joint",
                    ".*_ankle_pitch_joint",
                    ".*_hip_pitch_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_yaw_joint",
                    ".*_knee_joint",
                ],
            )
        },
        weight=-0.0002,
        # weight=0.
    )

    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.0e-8,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist_.*_joint",  # NOTE(HH) include or not?
                    ".*_ankle_roll_joint",
                    ".*_ankle_pitch_joint",
                    ".*_hip_pitch_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_yaw_joint",
                    ".*_knee_joint",
                ],
            )
        },
    )

    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist_.*_joint",  # NOTE(HH) include or not?
                    ".*_ankle_roll_joint",
                    ".*_ankle_pitch_joint",
                    ".*_hip_pitch_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_yaw_joint",
                    ".*_knee_joint",
                ],
            )
        },
    )
    action_rate_l2 = RewTerm(func=action_rate_l2, weight=-0.005)
    action_smoothness_l2 = RewTerm(func=action_smoothness_l2, weight=-0.005)

    ang_vel_xy_l2 = RewTerm(
        func=body_ang_vel_xy_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )

    lin_vel_z_l2 = RewTerm(
        func=body_lin_vel_z_l2,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )

    orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
        # params={"asset_cfg": SceneEntityCfg("robot")},
        # params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )

    torso_orientation_l2 = RewTerm(
        func=body_flat_orientation_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )

    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        # weight=0.,
        weight=-50.0,
        params={
            # "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "target_height": 0.75,
        },
    )

    ### Style rewards
    # - Feet ground parallel >> ankle_parallel
    # - Feet air time >> feet_air_time
    # - Feet slip >> penalize_feet_xy_vel_on_step
    # - Feet stumble >> penalize_feet_xy_force_on_step
    # - No fly >> avoid_both_foot_off_ground

    # NOTE(hh / 0702) : feet swing height added.

    termination_penalty = RewTerm(
        func=is_terminated_term,
        params={"term_keys": ["pelvis_below_minimum", "bad_pelvis_ori"]},
        weight=-50.0,
    )

    ankle_parallel = RewTerm(
        func=ankle_parallel,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    feet_air_time = RewTerm(
        func=feet_air_time,  # not working?
        # func=air_time_v4,
        weight=3.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

    feet_swing_height = RewTerm(
        func=feet_swing_height,
        weight=-20.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "feet_height": 0.08,
        },
    )

    penalize_feet_xy_vel_on_step = RewTerm(
        func=penalize_feet_xy_vel_on_step,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    penalize_feet_xy_force_on_step = RewTerm(
        func=penalize_feet_xy_force_on_step,
        weight=-1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "force_ratio": 3.0,
        },
    )
    avoid_both_foot_off_ground = RewTerm(
        func=avoid_both_foot_off_ground,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "force_threshold": 0.5,
            "curriculum_weight": 1.0,
        },
    )
    feet_contact_rhythm_from_gait_phase = RewTerm(
        func=feet_contact_rhythm_from_gait_phase,
        weight=0.75,
        params={
            "command_name": "gait_phase",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )

    lateral_foot_distance = RewTerm(
        func=keep_body_frame_y_distance,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "distance_threshold": 0.2,
            "left_body_name": "left_ankle_roll_link",
            "right_body_name": "right_ankle_roll_link",
        },
    )

    # lateral_knee_distance = RewTerm(
    #     func=keep_body_frame_y_distance,
    #     weight=0.5,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "distance_threshold": 0.2,
    #         "left_body_name": "left_knee_link",
    #         "right_body_name": "right_knee_link",
    #     },
    # )

    penalize_joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    # "waist_.*_joint",  # NOTE(HH) include or not?
                    ".*_ankle_roll_joint",
                    ".*_ankle_pitch_joint",
                    # ".*_hip_pitch_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_yaw_joint",
                    ".*_knee_joint",
                ],
            )
        },
    )

    joint_deviation_hip_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint"])},
    )

    joint_deviation_hip_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])},
    )

    joint_deviation_waist_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_roll_joint"])},
    )

    joint_deviation_waist_pitch = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_pitch_joint"])},
    )

    joint_deviation_waist_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint"])},
    )
    ### Task Rewards
    # - Base xy vel >> track_lin_vel_xy_yaw_frame_exp
    # - Base ang vel z >> track_ang_vel_z_world_exp
    # - Standing still >> standing_still_four_contact_points_v3

    track_lin_vel_xy_exp = RewTerm(
        func=track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,
        params={
            # "std": 0.25,
            "std": 0.5,
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={
            # "std": 0.1,
            "std": 0.5,
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # standing_still_four_contact_points_v3 = RewTerm(
    #     func=standing_still_four_contact_points_v3,
    #     weight=0.1,
    #     params={
    #         "command_name": "base_velocity",
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "left_foot_sensor_cfg": SceneEntityCfg("contact_forces", body_names="left_ankle_roll_link"),
    #         "right_foot_sensor_cfg": SceneEntityCfg("contact_forces", body_names="right_ankle_roll_link"),
    #     },
    # )

    encourage_low_lidar_lin_acc = RewTerm(
        func=encourage_low_lidar_lin_acc,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "lidar_frame_name": "mid360_link_frame",
            "std": 2,
        },
    )

    encourage_low_lidar_ang_acc = RewTerm(
        func=encourage_low_lidar_ang_acc,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "lidar_frame_name": "mid360_link_frame",
            "std": 2,
        },
    )

    penalize_foot_force = RewTerm(
        func=penalize_foot_force,
        weight= -2.5e-4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "force_threshold": 400.0,
        },
    )


    penalize_torso_ang_acc_z_l2 = RewTerm(
        func=penalize_torso_ang_acc_z_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )

    penalize_feet_z_force = RewTerm(
        func=penalize_feet_z_force,
        weight=-2.5e-4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 400.0,
        },
    )