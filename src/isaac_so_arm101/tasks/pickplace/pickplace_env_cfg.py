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

    object_pose = mdp.ObjectAwarePoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        # The goal is on the table now, in the same plane the cube spawns in, so a
        # plain uniform draw can land it on top of the cube. The latch keeps that
        # un-farmable but it makes those episodes trivial (pick up, put down). Draw
        # again until the goal is 12 cm away in XY, so every episode is a transport.
        min_separation=0.12,
        # Goal box, in the ROBOT BASE frame (SO-101: +x forward).
        #
        # The inherited box x[-0.1,0.1] y[-0.3,-0.1] was the SO-100 convention
        # (-y forward) and was never re-derived for this arm. Sampling the SO-101
        # fingertip workspace off the URDF puts it at only ~68-72% reachable at
        # ANY height: roughly a third of commanded goals were impossible, which is
        # why the goal reads as a weak knob that the policy learned to ignore.
        #
        # z is the cube's resting centre height: the goal is ON THE TABLE. This is
        # only safe because object_goal_tracking now uses the LATCHED variant — the
        # airborne-gated one would switch off exactly as the cube arrives, which is
        # the contradiction that wrecked every earlier attempt at this task.
        #
        # x starts at 0.15, not 0.10. Goals close to the base are reachable but
        # CRAMPED, and raw reachability hides it: sampling arm poses that land in
        # each x band, x[0.10,0.12) has only 1.95% of configurations versus 6.20%
        # at x[0.24,0.26), and needs mean |shoulder_lift| 1.06 rad against 0.61.
        # Few available configurations is what forces the folded, contorted pose —
        # the geometry picks it, not the policy. Moving the floor to 0.15 also
        # takes whole-box reachability from 92.4% to 99.6%.
        #
        # This reduces the pressure toward contortion but does not remove it:
        # nothing in this reward set constrains posture. That is grasp_top_down's
        # job (axis now corrected), in Increment 1.
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.30),
            pos_y=(-0.20, 0.20),
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

    # Required by the latched goal tracking: without this the latch never clears
    # and every episode after the first starts already "lifted".
    reset_lifted_latch = EventTerm(func=mdp.reset_lifted_latch, mode="reset")

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

    Built on the step-0 lift reward set, which earned the right to be extended:
    two runs (4000 and 8000 iterations) picked the cube reliably and carried it to
    the goal region. The bar set here — "do not add a term back without a run that
    shows the pick surviving first" — is met, so Increment 1b wires the first
    place term.

    INCREMENT 1b — the descent. Run A (8000 iters) measured the policy refusing to
    set the cube down, and the reward weights explain why. Holding the cube above
    2.5 cm pays a flat 15/step. Setting it down at t=3s of a 5s episode forfeits
    that for 40% of the episode (-6.0) and buys back at most +1.7 coarse tracking
    and +1.85 fine tracking. Completing the place was a NET LOSS of ~2.5, and the
    run behaved accordingly: between iteration 4000 and 8000 lift duty rose
    62% -> 70% while fine tracking FELL 0.183 -> 0.162 and orientation_error rose
    1.68 -> 1.80. PPO found the profitable side of that trade, which is hovering.

    Two changes flipped the sign:
      * ``object_at_target_on_table`` below — until then NO term paid for the cube
        resting at the goal, the one state the task is actually about;
      * ``lifting_object`` decays 15 -> 3 in ``CurriculumCfg``, so the bootstrap
        stops outbidding the placement once the pick is established.

    1b worked: object_at_target reached 2.38 and was still climbing at 4000, fine
    tracking went 0.162 -> 1.311 (cube ~4 cm from the goal, from ~9), position_error
    0.208 -> 0.159.

    INCREMENT 1c — the controlled release. 1b's policy reaches the goal and then
    DROPS the cube: reaching_object fell 0.590 -> 0.266 (EE 2.0 cm -> 4.7 cm from
    the cube) and lift duty fell 70% -> 13%. Nothing objected, because
    object_at_target scores a dropped cube and a placed cube identically once both
    are at rest, and the joint_vel penalty pays the arm to let go and go still.

    Three changes, all about the release:
      * ``object_at_rest`` and ``object_released`` wired below — velocity-gated, so
        they pay only once the cube has genuinely settled;
      * ``reaching_object`` switches off after the pick, because it was paying the
        arm to hold on to the cube it is supposed to release.

    Still unwired for Increment 1d: ``place_success`` + ``place_success_bonus`` (the
    bonus has a sizing constraint — it must out-value the dense reward forgone by
    ending the episode early — that is best not mixed with the release terms), and
    ``grasp_top_down`` (the contorted carry; orientation_error is up at 2.40).
    """

    # Switches off once the cube is picked. An always-on reach term pays the arm to
    # keep hold of the cube it is meant to release, and directly contradicts the
    # success condition (place_complete wants the EE >5 cm clear). 1b measured it
    # at 0.266 = 4.7 cm, right on that boundary. lift_height 0.04 matches the other
    # latched terms.
    reaching_object = RewTerm(
        func=mdp.object_ee_distance_before_lift,
        params={"std": 0.05, "lift_height": 0.04},
        weight=1.0,
    )

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.025}, weight=15.0)

    # LATCHED, not height-gated. The goal is on the table, so tracking has to keep
    # paying as the cube descends onto it; the airborne gate would switch off at
    # exactly the moment of arrival. Once the cube has been genuinely picked this
    # episode the latch stays set, so lift -> transport -> descent -> placed is
    # monotonic and hovering scores strictly worse than setting down.
    #
    # lift_height 0.04 (not the old 0.06): the latch exists only to forbid sliding,
    # and at 0.04 the cube's underside is 2.5 cm clear of the table, which no slide
    # produces. 0.06 left a dead band above lifting_object's 0.025 where a higher
    # lift bought nothing. Keep this value identical across every latched term.
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance_latched,
        params={"std": 0.3, "lift_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance_latched,
        params={"std": 0.05, "lift_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    # The state the task is about: cube at the commanded XY *and* down at table
    # height. The two tracking terms above are 3D distance to the goal, so they
    # are already near-maximal for a cube hovering a few cm above the target —
    # they do not distinguish "held over the spot" from "placed on the spot".
    # This one multiplies an XY kernel by a Z kernel, so the last few cm of
    # descent are the steepest part of it.
    #
    # z_std 0.02, not 0.05: the whole point is resolving the final descent, and
    # the cube's resting centre is only 1.5 cm off the table. A loose z kernel
    # would pay nearly full value for a cube still in the gripper.
    #
    # weight 12 vs lifting_object's decayed 3: putting the cube down must beat
    # holding it up by a clear margin, not by a coin flip.
    #
    # lift_height 0.04 matches the latched terms above, as their comment requires
    # — this function is also a latch updater, and a second threshold would let
    # one term set the latch that another still considers unset.
    object_at_target = RewTerm(
        func=mdp.object_at_target_on_table,
        params={"xy_std": 0.05, "z_std": 0.02, "lift_height": 0.04, "command_name": "object_pose"},
        weight=12.0,
    )

    # INCREMENT 1c — the controlled release.
    #
    # Both terms multiply the at-target gate by velocity thresholds, and that
    # product is what separates a PLACE from a DROP. object_at_target alone cannot:
    # a cube released 5 cm up that lands on the goal ends in the same state as one
    # set down gently, so 1b's policy learned to drop (reaching_object 0.59 -> 0.266
    # as the EE withdrew, lift duty 70% -> 13%).
    #
    # A dropped cube is in flight, then bounces and rolls; at 0.02 m/s and
    # 0.05 rad/s neither term pays a cent until it has actually settled. A placed
    # cube pays from the moment it touches down. Over an episode that is the
    # settling time plus the accuracy a bounce costs — a real gradient toward
    # setting the cube down, though an indirect one.
    #
    # Honest limitation: this rewards the settled END STATE, it does not penalise
    # impact speed. If 1c still drops, the next lever is an impact-velocity penalty
    # as its own term (contract §Increment 1 trap catalogue), not more weight here.
    #
    # Thresholds are copied from place_complete deliberately — the shaping should
    # aim at the exact predicate the 1d success termination will test, or the policy
    # learns to sit just outside it.
    object_at_rest = RewTerm(
        func=mdp.object_at_rest,
        params={
            "lin_vel_thresh": 0.02,
            "ang_vel_thresh": 0.05,
            "xy_std": 0.05,
            "z_std": 0.02,
            "lift_height": 0.04,
            "command_name": "object_pose",
        },
        weight=5.0,
    )

    # Weighted above object_at_rest on purpose. at_rest is satisfied by a cube held
    # perfectly still in a closed gripper at the goal; released additionally demands
    # the gripper be OPEN. Keeping released the larger of the two makes letting go
    # strictly better than holding on, which is the entire point of the increment.
    # robot_cfg MUST be passed here, not left as the function's signature default.
    # The managers resolve only the SceneEntityCfg objects they find in params, and
    # an unresolved cfg keeps joint_ids = slice(None) — which crashed on step one
    # with "TypeError: 'slice' object is not subscriptable". This is the first term
    # in the task to read a JOINT rather than root_state_w, which is why nothing
    # caught it earlier.
    object_released = RewTerm(
        func=mdp.object_released,
        params={
            "lin_vel_thresh": 0.02,
            "xy_std": 0.05,
            "z_std": 0.02,
            "gripper_open_thresh": 0.25,
            "lift_height": 0.04,
            "command_name": "object_pose",
            "robot_cfg": SceneEntityCfg("robot", joint_names=["gripper"]),
        },
        weight=8.0,
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

    # The one reward decay on this task. Read the history before touching it:
    #   - decaying object_goal_tracking to 0 removed the only pull toward the target
    #     and turned lifting_object into a hold-forever annuity (the frozen pose);
    #   - decaying lifting_object to 1 at iteration ~460 destroyed the pick entirely.
    #
    # Both failed for the SAME reason — they fired before the behaviour they were
    # fading had been learned. That is a statement about TIMING, not about decays.
    # Run A dates the pick precisely: lifting_object leaves zero at iteration ~600
    # and is still climbing at 8000. So iteration ~460 was squarely before the pick
    # existed, and the lesson is "decay after the behaviour is established", not
    # "never decay".
    #
    # This one fires at iteration ~1500 — 900 iterations after the pick appears —
    # and lands on 3, not 1. lifting_object stays the strongest single term through
    # the whole pick-learning phase; it only stops out-bidding object_at_target
    # (weight 12) once the arm can already pick reliably.
    #
    # Prediction, so this is falsifiable: at ~1500 expect lifting_object's logged
    # value to drop ~5x on the weight change alone (15 -> 3 is arithmetic, not
    # behaviour) while object_at_target and the fine tracking term start to climb.
    # If instead the PICK degrades — reaching_object falling, lifting duty
    # collapsing toward zero — the decay is still too early even here, and the fix
    # is num_steps 72000 (iteration ~3000), not abandoning the decay.
    #
    # WATCH THE UNITS: modify_reward_weight's num_steps counts ENVIRONMENT steps,
    # ~24 per training iteration. This has bitten the project twice. 36000 / 24 =
    # iteration ~1500. num_steps=12000 would fire at ~500, not 12000.
    lifting_object = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "lifting_object", "weight": 3.0, "num_steps": 36000}
    )


##
# Environment configuration
##


@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the pick-and-place environment.

    INCREMENT 1c. The lineage: step 0 reset the MDP to the lift task's rewards and
    moved only the spawn and goal boxes; 1a put the goal on the table and added the
    goal/cube separation constraint; 1b made finishing the place pay more than
    hovering over it; 1c (here) turns the resulting drop into a controlled release.
    See ``RewardsCfg`` for the measurements behind each and
    ``SOARMRL/docs/pickplace_contract.md`` for the increment plan.

    The boxes, unchanged since 1a:

      * spawn x[0.10,0.30], y[-0.25,0.25] on the table;
      * goal x[0.15,0.30], y[-0.20,0.20], z[0.015,0.020] — on the table, at the
        cube's resting centre height, at least 0.12 from the cube in XY.

    Both were sized against the SO-101 fingertip workspace sampled off the URDF,
    not guessed. The original goal box was ~68-72% reachable on this arm — it was
    the SO-100 (-y forward) convention, inherited through the clone and never
    re-derived.

    Still unwired: ``place_success`` + ``place_success_bonus``, and
    ``grasp_top_down``. Increment 1d.
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
