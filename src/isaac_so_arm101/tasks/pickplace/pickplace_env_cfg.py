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
        # Goal box, in the ROBOT BASE frame (SO-101: +x forward).
        #
        # The inherited box x[-0.1,0.1] y[-0.3,-0.1] was the SO-100 convention
        # (-y forward) and was never re-derived for this arm. Sampling the SO-101
        # fingertip workspace off the URDF puts it at only ~68-72% reachable at
        # ANY height: roughly a third of commanded goals were impossible, which is
        # why the goal reads as a weak knob that the policy learned to ignore.
        #
        # This box sits in front of the arm where the cube actually is, and is
        # >=93% reachable across its whole z range. Lowered from z(0.2,0.35)
        # toward the table, but deliberately NOT to table height: object_goal_
        # distance still gates on `object_z > minimal_height`, so a goal at
        # 0.015 would switch the reward off exactly when the cube arrives. 0.06
        # is the lowest goal that stays clear of that gate.
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.10, 0.30),
            pos_y=(-0.20, 0.20),
            pos_z=(0.06, 0.20),
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

    # NOTE for Increment 1: `reset_lifted_latch` was removed with the place terms.
    # Any term that gates on the was-lifted latch MUST come back together with
    # `reset_lifted_latch = EventTerm(func=mdp.reset_lifted_latch, mode="reset")`,
    # or the latch never clears and every episode after the first starts "lifted".

    # Spawn box, as an offset from the object's init pos [0.2, 0.0, 0.015], so
    # x in [0.10, 0.30], y in [-0.25, 0.25] in the robot base frame.
    #
    # y widened from +-0.20. Sampling the SO-101 fingertip workspace off the URDF
    # (top-down capable, fingertips at the cube's centre height) gives a y reach of
    # +-0.267 at x=0.30 and +-0.35 nearer the base, so every corner of this box is
    # inside the envelope with >=1.7 cm of margin.
    #
    # x is left alone on purpose: the envelope narrows fast with distance, so
    # pushing the far edge past 0.30 puts the far CORNERS out of reach even though
    # the far centre is fine. Widening x means shaping the region (or sampling in
    # polar coords), not stretching the rectangle.
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    STEP 0 RESET: this is the lift task's reward set, verbatim and unweighted-
    changed. Eleven runs of place-reward surgery on top of it never produced a
    pick, so the place terms are unwired (the functions stay in ``mdp/`` for
    Increment 1) and the only variables moved are the spawn and goal boxes.

    Do not add a term back without a run that shows the pick surviving first.
    """

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.05}, weight=1.0)

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

    # place_success is unwired for the step-0 reset — there is no place to succeed
    # at while the goal is airborne. mdp.place_success is unchanged, ready for
    # Increment 1.


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # Back to the lift baseline's 10000 for the step-0 reset. That value is what
    # the working lift policy was trained under, so keeping it means the boxes are
    # the only variable in this run.
    #
    # Pre-registered prediction, since this fires at iteration ~417: if
    # reaching_object climbs and then falls back around there, these are the cause
    # (that is exactly what happened in run 10) and the wider boxes have made the
    # pick harder to find before the penalties land. Push num_steps to 60000 and
    # rerun — that is a clean single-variable follow-up, not a thing to pre-empt.
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )

    # No reward decays here on purpose. Every decay tried on this task fired before
    # the behaviour it was fading had actually been learned, and killed it:
    #   - decaying object_goal_tracking to 0 removed the only pull toward the target
    #     and turned lifting_object into a hold-forever annuity (the frozen pose);
    #   - decaying lifting_object to 1 at iteration ~460 destroyed the pick entirely.
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

    STEP 0 RESET. This is the lift task's MDP — same rewards, same weights, same
    terminations, same penalty curriculum — with exactly two things changed:

      * the spawn box is wider in y (+-0.25 from +-0.20);
      * the goal box moved in front of the arm and down (x[0.10,0.30],
        y[-0.20,0.20], z[0.06,0.20], from x[-0.1,0.1] y[-0.3,-0.1] z[0.2,0.35]).

    Both boxes were sized against the SO-101 fingertip workspace sampled off the
    URDF, not guessed. The old goal box was ~68-72% reachable on this arm — it was
    the SO-100 (-y forward) convention, inherited through the clone and never
    re-derived.

    The place rewards and the success termination are written and unwired; see
    ``RewardsCfg`` and ``SOARMRL/docs/pickplace_contract.md``. Re-wire them only
    after a run shows the pick surviving the wider boxes.
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
