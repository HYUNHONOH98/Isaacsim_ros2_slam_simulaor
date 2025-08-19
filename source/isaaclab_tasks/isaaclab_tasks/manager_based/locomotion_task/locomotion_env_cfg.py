import os
import torch as th
import math

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from dataclasses import MISSING

import isaaclab_tasks.manager_based.locomotion_task.mdp as mdp
from .g1_spawn_info import *


def prev_pelvis_height(env, asset_cfg=SceneEntityCfg("robot")) -> th.Tensor:
    asset = env.scene[asset_cfg.name]

    if asset.data._prev_root_state_w is not None:
        prev_pelvis_height = asset.data._prev_root_state_w[..., 2:3]
    else:
        prev_pelvis_height = asset.data.root_state_w[..., 2:3]
    return prev_pelvis_height


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # contact sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, track_pose=True
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # noise scale refereced from HOMIE, https://arxiv.org/pdf/2502.13013
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.02, n_max=0.02))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-2.0, n_max=2.0))
        velocity_cmd = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        phase_cmd = ObsTerm(func=mdp.generated_commands, params={"command_name": "gait_phase"})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PolicyCfg = PolicyCfg()


@configclass
class LocomotionCommandsCfg:
    # Refereced from Isaac Lab locomotion task's LocomotionVelocityRoughEnvCfg. https://github.com/isaac-sim/IsaacLab/
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=False,
        resampling_time_range=(5.0, 5.0),
        heading_control_stiffness=0.0,
        rel_standing_envs=0.01,
        rel_heading_envs=0.0,
        debug_vis=True,
        # Referenced from HOMIE. https://arxiv.org/pdf/2502.13013
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            # lin_vel_x=(0.0, 1.0),  # m/s
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.5, 0.5),  # m/s
            # ang_vel_z=(-0.5, 0.5),  # rad/s
            ang_vel_z=(-1.0, 1.0),  # rad/s
            heading=(-math.pi / 6, math.pi / 6),  # rad
        ),
    )

    gait_phase = mdp.PhaseCommandCfg(
        class_type=mdp.PhaseCommand,
        resampling_time_range=(10.0, 10.0),
        period=1.0,
        offset=0.5,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    pelvis_below_minimum = DoneTerm(func=mdp.eetrack_pelvis_height_below_minimum, params={"minimum_height": 0.2})
    bad_pelvis_ori = DoneTerm(func=mdp.eetrack_bad_pelvis_ori, params={"limit_euler_angle": [0.9, 1.0]})


@configclass
class CurriculumCfg:
    delay_curriculum = CurrTerm(
        func=mdp.delay_curriculum, params={"status_checker": mdp.is_velocity_tracking_error_under_threshold}
    )
    perturbation_curriculum = CurrTerm(
        func=mdp.perturbation_curriculum, params={"status_checker": mdp.is_velocity_tracking_error_under_threshold}
    )
    linvel_cmd_curriculum = CurrTerm(func=mdp.linvel_cmd_curriculum)
    angvel_cmd_curriculum = CurrTerm(func=mdp.angvel_cmd_curriculum)
    walking_phase_curriculum = CurrTerm(func=mdp.walking_phase_curriculum)


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "waist_.*_joint",  # NOTE(HH) include or not?
            ".*_ankle_roll_joint",
            ".*_ankle_pitch_joint",
            ".*_hip_pitch_joint",
            ".*_hip_roll_joint",
            ".*_hip_yaw_joint",
            ".*_knee_joint",
        ],
        scale=0.5,
        use_default_offset=False,
        randomize_torque_rfi=True,
        rfi_lim_scale=0.1,  # Scale factor for the random torque R
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            # "position_range": (-0.2, 0.2),
            "position_range": (-0.15, 0.15),
            # "position_range": (-0.05, 0.05),
            "velocity_range": (-0.1, 0.1),
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
            ),
        },
    )

    # Domain randomization according to https://arxiv.org/pdf/2410.03654 and https://arxiv.org/pdf/2406.08858
    dr_robot_link_physics_parameters = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        # min_step_count_between_reset=201,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "static_friction_range": (0.1, 2.0),
            "dynamic_friction_range": (0.1, 2.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 500,  # Various number of buckets
            "make_consistent": True,  # ensure dynamic friction is always less than static friction
        },
    )

    dr_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        min_step_count_between_reset=500,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    dr_hand_payload = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        min_step_count_between_reset=500,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["*._hand_palm_link"]),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    """
        For implicit actuators, this function uses CPU tensors to assign the actuator gains into the simulation.
        In such cases, it is recommended to use this function only during the initialization of the environment.
    """
    dr_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        min_step_count_between_reset=500,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # external force torque during responing.
    dr_torso_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        # mode="interval",
        # interval_range_s=(0.1, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (-20.0, 20.0),
            "torque_range": (-5.0, 5.0),
        },
    )

    # # Slightly push robot while handling task
    dr_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 6.0),
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
    )


@configclass
class G1LocomotionEnvCfg(ManagerBasedRLEnvCfg):
    initial_state_buffer_path: str = ""
    rewards: mdp.G1LocomotionRewards = mdp.G1LocomotionRewards()
    commands: LocomotionCommandsCfg = LocomotionCommandsCfg()
    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # post init of parent
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s / 2, self.episode_length_s / 2)

        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # Scene
        # self.scene.robot = G1_LIDAR_CFG.replace(
        #     prim_path="{ENV_REGEX_NS}/Robot",
        # )
        self.scene.robot = G1_LIDAR_NO_MERGE_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )

        self.scene.robot.actuators["G1"].min_delay = 0
        self.scene.robot.actuators["G1"].max_delay = 6

        # Domain Randomization
        self.disable_all_dr_terms_except([
            "dr_joint_stiffness_and_damping",
            "dr_link_mass",
            "dr_robot_link_physics_parameters",
            "dr_push_robot",
            # "dr_hand_payload",
        ])

        # curriculum
        # self.curriculum.delay_curriculum = None
        # self.curriculum.perturbation_curriculum = None

        # Observation noise
        self.observations.policy.enable_corruption = True

        # Asymmetric policy <> critic
        self.observations.policy.base_lin_vel = None

        # Contact sensors
        # self.scene.contact_left_foot = mdp.ContactSensorExtraCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        #     filter_prim_paths_expr=["/World/ground/GroundPlane/CollisionPlane"],
        #     update_period=0.0,
        #     history_length=6,
        #     debug_vis=True,
        #     max_contact_data_count=8 * 4096,
        # )
        # self.scene.contact_right_foot = mdp.ContactSensorExtraCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        #     filter_prim_paths_expr=["/World/ground/GroundPlane/CollisionPlane"],
        #     update_period=0.0,
        #     history_length=6,
        #     debug_vis=True,
        #     max_contact_data_count=8 * 4096,
        # )

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

    def disable_all_dr_terms_except(self, exception_names):
        reset_term_names = self.events.to_dict().keys()
        for term in exception_names:
            assert term in reset_term_names

        for term_name in reset_term_names:
            if term_name in exception_names or not ("dr_" in term_name):
                continue
            print("Disabling " + term_name)
            setattr(self.events, term_name, None)


class G1LocomotionEnvCfg_PLAY(G1LocomotionEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        self.curriculum.delay_curriculum = None
        self.curriculum.perturbation_curriculum = None
        self.curriculum.walking_phase_curriculum = None
        # Domain Randomization
        self.disable_all_dr_terms_except([
            # "dr_joint_stiffness_and_damping",
            # "dr_link_mass",
            # "dr_robot_link_physics_parameters",
            # "dr_push_robot",
            # "dr_hand_payload",
        ])

        self.scene.robot.actuators["G1"].min_delay = 0
        self.scene.robot.actuators["G1"].max_delay = 0
        self.observations.policy.enable_corruption = False



        self.episode_length_s = 10.0
        self.commands.base_velocity.resampling_time_range = (0.2, 0.2)
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # set command for play.

        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.3)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.3, 0.3)
        # self.commands.base_velocity.ranges.ang_vel_z = (1., 1.)
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # To keep the heading direction
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 1.0
        self.commands.base_velocity.ranges.heading = (0, 0)
        self.commands.base_velocity.rel_heading_envs = 1.0

        # Fix reset direction
        self.events.reset_base.params["pose_range"] = {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-0.0, 0.0)}

        # fix cam
        self.viewer.eye = (0.0, 3.0, 0.4)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
