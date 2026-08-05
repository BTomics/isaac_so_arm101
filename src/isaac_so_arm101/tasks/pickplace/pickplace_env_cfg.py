# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils

# from . import mdp
import isaac_so_arm101.tasks.pickplace.mdp as mdp
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObjectCfg,
    RigidObjectCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# from isaaclab.utils.offset import OffsetCfg
# from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
# from isaaclab.utils.visualizer import FRAME_MARKER_CFG
# from isaaclab.utils.assets import RigidBodyPropertiesCfg


##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.1, 0.1),
            pos_y=(-0.3, -0.1),
            # PickPlace: target is ON THE TABLE (cube resting center ~0.015 for the
            # 3 cm cube), not airborne. VERIFY this z in sim/play. Was (0.2, 0.35).
            pos_z=(0.015, 0.020),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.2, 0.2), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )

    # Clear the anti-slide "was-lifted" latch at the start of every episode.
    reset_lifted_latch = EventTerm(func=mdp.reset_lifted_latch, mode="reset")


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.05}, weight=1.0)

    # Reward a top-down gripper approach (weighted by proximity to the cube). This
    # is the ROOT fix for the contorted/sideways grasp — inherited from lift, where
    # nothing constrains the arm's configuration — that stops the cube being set
    # down flat or released. Watch object_orientation_error: if it goes UP, flip
    # _GRIPPER_APPROACH_LOCAL's sign in rewards.py.
    grasp_top_down = RewTerm(
        func=mdp.grasp_top_down,
        params={"std": 0.5, "near_std": 0.1},
        weight=5.0,
    )

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.025}, weight=15.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.025, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.025, "command_name": "object_pose"},
        weight=5.0,
    )

    # --- place / release / at-rest (Increment 1 authorship) ---
    # NOTE: weights below are STARTING POINTS to tune. The invariant: these place
    # terms must DOMINATE the retained airborne terms above (object_goal_tracking
    # 16, lifting_object 15) or the arm hovers the cube instead of setting it down.
    # `lift_height` (the anti-slide gate: cube must clear this height to unlock any
    # place reward) must be the SAME across all place terms + the success terms.
    # z_std deliberately WIDE (0.08): rewards the cube getting lower toward the
    # table continuously from several cm up, so the descent has a monotonic
    # gradient instead of a reward valley at the airborne-term z-gate (0.025).
    # This term also carries the cube toward the target XY (lifted × z × xy).
    # place_on_table is now DELIBERATELY smaller than `released`: it pays whether
    # the gripper is open or closed, so a high weight makes "hold the cube on the
    # spot" as good as letting go. Keep it as descent shaping, let `released` win.
    place_on_table = RewTerm(
        func=mdp.object_at_target_on_table,
        params={"xy_std": 0.05, "z_std": 0.08, "command_name": "object_pose", "lift_height": 0.06},
        weight=12.0,
    )

    # Reward a clean upright placement — attacks the tilted/under-the-cube grasp
    # (high object_orientation_error) that leaves the cube unstable to release.
    place_orientation = RewTerm(
        func=mdp.object_orientation_to_target,
        params={"std": 0.5, "command_name": "object_pose", "lift_height": 0.06},
        weight=8.0,
    )

    released = RewTerm(
        func=mdp.object_released,
        params={
            "command_name": "object_pose",
            "lin_vel_thresh": 0.02,
            "xy_std": 0.05,
            "z_std": 0.01,
            "gripper_open_thresh": 0.25,
            "lift_height": 0.06,
        },
        weight=30.0,
    )

    at_rest = RewTerm(
        func=mdp.object_at_rest,
        params={
            "lin_vel_thresh": 0.02,
            "ang_vel_thresh": 0.5,
            "command_name": "object_pose",
            "xy_std": 0.05,
            "z_std": 0.01,
            "lift_height": 0.06,
        },
        weight=5.0,
    )

    # One-time bonus when the place is genuinely complete (same condition as the
    # place_success termination). Must OUT-VALUE the dense place reward the agent
    # gives up by ending the episode early, or it avoids success to keep farming.
    # TUNE upward if success-rate stays low while time_out stays ~1.
    place_success_bonus = RewTerm(
        func=mdp.place_success_bonus,
        params={
            "command_name": "object_pose",
            "xy_threshold": 0.02,
            "z_tol": 0.01,
            "lin_vel_thresh": 0.02,
            "ang_vel_thresh": 0.05,
            "gripper_open_thresh": 0.25,
            "ee_clearance": 0.05,
            "lift_height": 0.06,
        },
        weight=50.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )

    # Success = cube placed at target, at rest, gripper open, EE withdrawn.
    # Thresholds are STARTING POINTS to tune (see terminations.place_success).
    place_success = DoneTerm(
        func=mdp.place_success,
        params={
            "command_name": "object_pose",
            "xy_threshold": 0.02,
            "z_tol": 0.01,
            "lin_vel_thresh": 0.02,
            "ang_vel_thresh": 0.05,
            "gripper_open_thresh": 0.25,
            "ee_clearance": 0.05,
            "lift_height": 0.06,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )

    # Switch OFF the airborne hover-over-target reward once the pick is learned,
    # so the policy is forced to lower the cube onto the table for place reward
    # instead of parking it in the air. lifting_object is kept (it bootstraps the
    # pick). num_steps is the key knob: too early and the pick isn't learned yet;
    # too late and it wastes iterations hovering. Tune from the run.
    decay_goal_tracking = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "object_goal_tracking", "weight": 0.0, "num_steps": 12000},
    )

    decay_goal_tracking_fine = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "object_goal_tracking_fine_grained", "weight": 0.0, "num_steps": 12000},
    )


##
# Environment configuration
##


@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the pick-and-place environment.

    Cloned verbatim from the lift task (SO-ARM101-Lift-Cube-v0) as the starting
    baseline for SO-ARM101-PickPlace-v0. Trains identically to lift until the
    place-specific rewards/terminations are added here + in ``mdp/``.
    See ``SOARMRL/docs/pickplace_contract.md`` for the intended diff.
    """

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
