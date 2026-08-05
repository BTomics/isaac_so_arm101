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

    # Reward holding the cube top-down, weighted by proximity and gated on the cube
    # actually being off the table (min_height) so it cannot be farmed by hovering
    # open-handed over a grounded cube. Targets the contorted/sideways grasp
    # inherited from lift, where nothing constrains the arm's configuration. The
    # approach-axis sign in rewards.py is settled — do not re-derive it.
    grasp_top_down = RewTerm(
        func=mdp.grasp_top_down,
        params={"std": 0.5, "near_std": 0.1, "min_height": 0.025},
        weight=3.0,
    )

    # Weight 15 and NOT decayed. Cutting this to 5 (+decay to 1) killed the pick
    # outright — lifting_object never left 1e-7 and nothing downstream ever fired,
    # while runs that lifted reliably all had 15 under identical action penalties.
    # The frozen-hold this was meant to fix was a PATH problem (the gradient pointed
    # backward mid-descent), and latch-gating object_goal_tracking already fixes it:
    # holding aloft pays ~30, placed pays ~70, and the path between is monotonic.
    # Suppressing the lift signal was never part of that fix.
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.025}, weight=15.0)

    # Latch-gated (see mdp.object_goal_distance): pays continuously from pick all
    # the way down to the cube resting on the target, so the descent is monotonic.
    # NOT decayed — unlike the airborne version, this term no longer fights the
    # place. lift_height must match every other place term.
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "lift_height": 0.06, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "lift_height": 0.06, "command_name": "object_pose"},
        weight=5.0,
    )

    # --- place / release / at-rest (Increment 1 authorship) ---
    # NOTE: weights below are STARTING POINTS to tune. `lift_height` (the anti-slide
    # gate: cube must clear this height to unlock any place reward) must be the SAME
    # across all place terms + the success terms.
    # Now that object_goal_tracking is latch-gated it already provides the dense
    # pull all the way down to the target, so this term largely duplicates it —
    # hence 12 -> 6, to avoid double-counting descent. It stays as the term that
    # specifically shapes "at table height" (wide z_std 0.08) rather than just
    # "near the goal point". It is DELIBERATELY smaller than `released`: it pays
    # whether the gripper is open or closed, so a high weight would make "hold the
    # cube on the spot" as good as letting go.
    place_on_table = RewTerm(
        func=mdp.object_at_target_on_table,
        params={"xy_std": 0.05, "z_std": 0.08, "command_name": "object_pose", "lift_height": 0.06},
        weight=6.0,
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

    # Pushed out from 10000 (iteration ~417) to 60000 (~2500), i.e. past the end of
    # a 1500-iteration run, so the penalties stay at their -1e-4 base throughout.
    # Reason: these two are a smoothness polish, but they were firing BEFORE the
    # grasp was ever discovered and then preventing it. reaching_object peaked at
    # 0.85 (EE 7.6 mm from the cube centre) and fell to ~0.64 (19 mm) exactly at
    # iteration ~420 when they jumped to -1e-1; a gripper 2 cm off a 3 cm cube
    # cannot close on it, so lifting_object stayed at exactly 0.
    # The pick must be learned first - re-tighten these only once it is reliable.
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 60000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 60000}
    )

    # No reward decays here on purpose. Every decay tried on this task fired before
    # the behaviour it was fading had actually been learned, and killed it:
    #   - decaying object_goal_tracking to 0 removed the only pull toward the target
    #     and turned lifting_object into a hold-forever annuity (the frozen pose);
    #   - decaying lifting_object to 1 at iteration ~460 destroyed the pick entirely.
    # Latch-gated tracking makes the whole trajectory monotonic, so nothing needs
    # switching off to force the place.
    #
    # WATCH THE UNITS if a decay is ever reintroduced: modify_reward_weight's
    # num_steps counts ENVIRONMENT steps, ~24 per training iteration. num_steps=12000
    # fired at iteration ~500, not 12000 — a third of the way into a 1500-iter run.


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
