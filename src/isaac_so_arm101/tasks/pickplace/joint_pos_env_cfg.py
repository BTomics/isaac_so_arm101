# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp
from isaaclab.assets import RigidObjectCfg

# from isaaclab.managers NotImplementedError
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# `mdp` above is the UPSTREAM lift mdp, which has no pickplace terms. The eval
# variants below need the local one, so it is imported under its own name rather
# than shadowing an alias the rest of this file depends on.
import isaac_so_arm101.tasks.pickplace.mdp as pickplace_mdp
from isaac_so_arm101.robots import SO_ARM100_CFG, SO_ARM101_CFG  # noqa: F401
from isaac_so_arm101.tasks.pickplace.pickplace_env_cfg import PickPlaceEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


@configclass
class SoArm100PickPlaceEnvCfg(PickPlaceEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set so arm as robot
        self.scene.robot = SO_ARM100_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            open_command_expr={"gripper": 0.5},
            close_command_expr={"gripper": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = ["gripper"]

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0.0, 0.015], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.5, 0.5, 0.5),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, -0.09, 0.01],
                    ),
                ),
            ],
        )


@configclass
class SoArm100PickPlaceEnvCfg_PLAY(SoArm100PickPlaceEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class SoArm101PickPlaceEnvCfg(PickPlaceEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set so arm as robot
        self.scene.robot = SO_ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            open_command_expr={"gripper": 0.5},
            close_command_expr={"gripper": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = ["gripper_link"]

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0.0, 0.015], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.5, 0.5, 0.5),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.01, 0.0, -0.09],
                    ),
                ),
            ],
        )


@configclass
class SoArm101PickPlaceEnvCfg_PLAY(SoArm101PickPlaceEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class SoArm101PickPlaceEnvCfg_NOISE(SoArm101PickPlaceEnvCfg):
    """Evaluation only: perturb object_position to stand in for real perception.

    The trained policy reads the cube pose from
    mdp.object_position_in_robot_root_frame - simulator ground truth, exact to
    machine precision, and it has never seen anything else: enable_corruption is
    True but not one ObsTerm carries a noise model, and EventCfg holds only
    resets. On hardware that input becomes a camera and a pose estimator with
    millimetres of error.

    So the question this answers is not "does it work with noise" but "HOW
    ACCURATE DOES MY PERCEPTION HAVE TO BE" - run the registered levels and find
    where place_success falls off. That number is the spec for the camera.

    NOT for training. Subclasses set NOISE_M in metres; only object_position is
    perturbed, so the result is attributable to perception error alone rather
    than to a bundle of physics changes. Joint encoder noise, cube friction/mass
    randomisation and actuator latency are the other three transfer axes and are
    deliberately NOT in here - add them one at a time or the answer is
    uninterpretable, which is the mistake increment 1d already paid for.
    """

    NOISE_M = 0.005

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.observations.policy.object_position.noise = Unoise(
            n_min=-self.NOISE_M, n_max=self.NOISE_M
        )
        # enable_corruption is already True on PolicyCfg; without it the noise
        # model above is attached but never applied.
        self.observations.policy.enable_corruption = True


@configclass
class SoArm101PickPlaceEnvCfg_BIAS(SoArm101PickPlaceEnvCfg):
    """Evaluation only: a PER-EPISODE CONSTANT offset on the cube position.

    The _NOISE variants above resample every step, so the policy averages them
    out over 250 steps and shrugs off +-10 mm. A miscalibrated camera does not
    behave that way - it is wrong in the same direction for the whole episode,
    and there is nothing to average. This is the harder and far more realistic
    test, and where place_success falls off across the levels is the CALIBRATION
    BUDGET for the camera mount.

    Swaps the observation for the biased variant and adds the reset event that
    redraws the bias. BOTH are required: the observation without the event holds
    one draw forever and silently measures something else.
    """

    BIAS_M = 0.005

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.observations.policy.object_position = ObsTerm(
            func=pickplace_mdp.object_position_biased, params={"max_bias": self.BIAS_M}
        )
        self.events.reset_object_position_bias = EventTerm(
            func=pickplace_mdp.reset_object_position_bias,
            mode="reset",
            params={"max_bias": self.BIAS_M},
        )


@configclass
class SoArm101PickPlaceEnvCfg_BIAS2(SoArm101PickPlaceEnvCfg_BIAS):
    BIAS_M = 0.002  # a carefully calibrated mount


@configclass
class SoArm101PickPlaceEnvCfg_BIAS5(SoArm101PickPlaceEnvCfg_BIAS):
    BIAS_M = 0.005  # a realistic hand-eye calibration


@configclass
class SoArm101PickPlaceEnvCfg_BIAS10(SoArm101PickPlaceEnvCfg_BIAS):
    BIAS_M = 0.010  # half of place_complete's 0.02 xy threshold, spent before the arm moves


@configclass
class SoArm101PickPlaceEnvCfg_NOISE2(SoArm101PickPlaceEnvCfg_NOISE):
    NOISE_M = 0.002  # a good depth camera, cube well inside the frame


@configclass
class SoArm101PickPlaceEnvCfg_NOISE5(SoArm101PickPlaceEnvCfg_NOISE):
    NOISE_M = 0.005  # realistic for a 3 cm cube at working distance


@configclass
class SoArm101PickPlaceEnvCfg_NOISE10(SoArm101PickPlaceEnvCfg_NOISE):
    NOISE_M = 0.010  # a third of the cube's width; place_complete's xy threshold is 0.02


@configclass
class SoArm101PickPlaceEnvCfg_RESUME(SoArm101PickPlaceEnvCfg):
    """Use this for --resume. Never for a fresh run.

    The curriculum counter lives on the env, and --resume builds a FRESH env, so
    every modify_reward_weight term restarts and re-fires. That silently rewinds
    the reward function underneath a trained policy:

      - lifting_object snaps back to 15, which is the regime where hovering over
        the target pays 27.2/step against 21.0 for completing the place. The
        policy is actively pushed back toward not setting the cube down, for the
        first ~1500 iterations of the resumed run.
      - the penalty ramp restarts at -1e-4 and takes ~2500 iterations to climb
        back, long enough for raw action_rate_l2 to grow again (it reached ~500
        unconstrained, against ~24 under the full ramp).

    This variant pins every curriculum'd weight to its converged value and clears
    the curriculum, so the resumed reward function is identical to the one the
    checkpoint was last trained under. Used by hand as a config edit before; this
    makes it a task id instead, because the edit had to be reverted afterwards
    and a fresh run at lifting_object=3 never bootstraps the pick.
    """

    # Converged endpoints. Must match the last stage of each curriculum in
    # PickPlaceEnvCfg.CurriculumCfg.
    PINNED = {"lifting_object": 3.0, "action_rate": -1e-1, "joint_vel": -1e-1}

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Clear the curriculum, refusing to guess about any term that does not
        # target a weight pinned above - a new term would otherwise be silently
        # dropped here and resume with whatever the config's base weight happens
        # to be, which is exactly the failure this class exists to prevent.
        for name, term in list(self.curriculum.__dict__.items()):
            if term is None:
                continue
            target = term.params.get("term_name") if term.params else None
            if target not in self.PINNED:
                raise ValueError(
                    f"Curriculum term {name!r} targets {target!r}, which is not in "
                    f"SoArm101PickPlaceEnvCfg_RESUME.PINNED. Add its converged weight there "
                    f"before resuming, or the resumed run silently uses the base weight."
                )
            setattr(self.curriculum, name, None)

        for term_name, weight in self.PINNED.items():
            getattr(self.rewards, term_name).weight = weight


# ---------------------------------------------------------------------------
# Deployment-path evaluation variants. EVALUATION ONLY - never train on these.
#
# The _NOISE/_BIAS variants above perturb what the policy SEES. These perturb
# what its actions DO, which is the axis nobody has tested. The trained policy's
# action becomes a joint target directly at 50 Hz; the deployed policy's action
# goes through a 30 Hz loop, a 0.1 slow-blend and a 0.03 rad delta clamp before
# it reaches a servo. Five things differ between training and deployment at once
# (rate, blend, clamp, zeroed velocities, dead-reckoned cube) and the hardware
# run changed all five together.
#
# Replay the 12000-iteration checkpoint through each one. The behaviour to match
# is the 2026-08-20 run, taken with a goal INSIDE the trained box (the earlier
# "parks 65 mm short" attractor turned out to be an out-of-distribution target,
# not a policy failure):
#
#   - the arm does not track its own commands while carrying. Measured sag of
#     0.085-0.098 rad on shoulder_lift and +0.042 on elbow_flex, held for
#     hundreds of ticks, in the direction gravity pulls a loaded arm.
#   - demand runs far ahead of delivery: elbow want +2.27 against sent +0.85,
#     and by the time the arm arrives the policy has reversed. That is a
#     phase-lagged oscillation, and the cube's distance to goal swings
#     289 -> 355 -> 312 -> 347 mm rather than closing.
#   - the release fires at 347 mm on a gripper channel that is swinging +-12,
#     i.e. on noise rather than intent.
#   - afterwards every joint goes static. PHASE_RELEASED freezes the tracked
#     cube, so a memoryless policy fed a frozen observation emits a constant
#     action forever. In training the post-release state is always near the
#     goal; at 347 mm it is off-distribution and has no learned behaviour.
#
# If _DEPLOYED reproduces that, the failure has moved from a 12 second bench run
# into a 4096-env simulator. If it does NOT, the actuation-path thesis is wrong
# and the retrain should not be built on it.
# ---------------------------------------------------------------------------


@configclass
class SoArm101PickPlaceEnvCfg_DEPLOY30(SoArm101PickPlaceEnvCfg):
    """Evaluation only: the 30 Hz control rate the bridge actually runs.

    PickPlace trains at 50 Hz (sim.dt 0.01 x decimation 2). scripts/grasp/
    pickplace_live.py runs it at HZ = 30.0, inherited from the reach bridge where
    30 Hz was correct because reach trained at sim dt 1/60 x decimation 2. Nobody
    chose 30 for this policy - it came along with the file.

    Episode length is held at 5 s of WALL time, so the policy gets 150 steps here
    against 250 in training. That is deliberate: the arm does not get extra
    seconds because the loop is slower, and step count is not what the policy
    perceives anyway (the network is memoryless).
    """

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 1.0 / 60.0
        self.decimation = 2
        self.sim.render_interval = self.decimation


@configclass
class SoArm101PickPlaceEnvCfg_BLEND(SoArm101PickPlaceEnvCfg):
    """Evaluation only: the bridge's slow-blend and delta clamp in the loop.

    Defaults mirror pickplace_live.py (slow 0.1, max_delta 0.03 rad). At 30 Hz
    the blend alone is a ~0.32 s first-order lag, against a discount horizon of
    1/(1-gamma) = 50 steps = 1.0 s at the trained rate.

    The throttle has been tested on hardware as a binary - "opening it up makes
    the arm dive, not carry" - and both settings are wrong. Throttled gives the
    stuck fixed point; unthrottled gives a policy that never paid a smoothness
    penalty driving a real arm at full tilt. There is no correct setting, which
    is the argument for training inside the rate limit rather than tuning around
    it.
    """

    SLOW = 0.1
    MAX_DELTA = 0.03

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action = pickplace_mdp.DeploymentJointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
            scale=0.5,
            use_default_offset=True,
            slow=self.SLOW,
            max_delta=self.MAX_DELTA,
        )


@configclass
class SoArm101PickPlaceEnvCfg_ZEROVEL(SoArm101PickPlaceEnvCfg):
    """Evaluation only: the joint velocity block zeroed, as the bridge sends it.

    grasp_bridge.build_grasp_obs zeroes obs[6:12] by default - six of 28
    dimensions, 21% of the observation - because a finite difference of noisy,
    USB-lagged encoder readings drove a sustained limit cycle on hardware.

    This is the CORRECT-BASELINE comparison that has never been run. The hardware
    test compared zeroed velocities against finite-difference velocities and
    found "the same attractor to within a centimetre", which was read as
    exonerating the velocity block. Both of those are wrong relative to sim's
    true physics velocity; two wrong answers agreeing says nothing about the
    right one. This variant asks the question that test did not.

    scale=0.0 rather than deleting the term, so the observation stays 28-dim and
    the result is attributable to the information loss alone, not to a changed
    contract.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.joint_vel.scale = 0.0


@configclass
class SoArm101PickPlaceEnvCfg_DEPLOYED(SoArm101PickPlaceEnvCfg_BLEND):
    """Evaluation only: rate, blend, clamp and zeroed velocities together.

    The composition of the three variants above. This is the one to replay first;
    the singles exist to attribute whatever it shows.

    NOT YET FAITHFUL IN ONE RESPECT, and it is the largest one: object_position
    here is still simulator ground truth. On hardware it comes from a three-phase
    dead-reckoning estimator (grasp_bridge.object_position_for_tick) - a
    hard-coded constant while seeking, forward kinematics of the jaw tip while
    held, then FROZEN at the release point with z pinned to the table. Its errors
    are structured, not Gaussian, so the _NOISE/_BIAS variants do not stand in
    for it. Wiring that estimator into an ObsTerm is the remaining piece, and it
    is the same term the retrain wants anyway (train on the estimator's output,
    not on ground truth). Until it exists, read a negative result here as "not
    reproduced BY THESE FOUR", never as "not reproduced".
    """

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 1.0 / 60.0
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.observations.policy.joint_vel.scale = 0.0


# ---------------------------------------------------------------------------
# Run A - the aligned baseline. THIS ONE IS FOR TRAINING.
# ---------------------------------------------------------------------------


@configclass
class SoArm101PickPlaceEnvCfg_ALIGNED(SoArm101PickPlaceEnvCfg):
    """Train here: sim's actuation path made the same path the bridge deploys.

    Every change below serves one invariant - the plant the policy learns on and
    the plant it is deployed into are the same plant. They ship together because
    they are not independently meaningful: a rate limit at the wrong control rate
    is not a smaller version of this change, it is an incoherent one.

      1. 30 Hz control (sim.dt 1/60, decimation 2), matching the bridge exactly.
         PickPlace trained at 50 Hz and has always been deployed at 30.
      2. A hard per-step rate limit on the joint target, so clamp_delta at
         deployment becomes a NO-OP instead of a distortion. The policy cannot
         ask for more than the arm delivers, so it never builds a plan that
         depends on doing so.
      3. action_rate and joint_vel penalties DELETED, along with all six of their
         curriculum stages. The rate limit enforces smoothness structurally, so
         the penalties are paying twice for it. This removes the single largest
         source of training instability in this task's history - three failed
         runs, an entropy collapse, and a ~1000-iteration re-absorption tax on
         every resume - and it is what livekit's so-frame does for the same
         reason ("no smoothness penalties", with a per-step motion cap).
      4. joint_vel OBSERVATION scaled to zero. The bridge zeroes those six dims
         because a finite difference of lagged encoder readings drove a limit
         cycle; training on true physics velocity means 6 of 28 inputs are
         fiction at deployment. Kept at 28 dims so the bridge needs no change.
      5. episode_length_s 5 -> 8. Four phases under a rate cap does not fit in
         5 s; 8 s at 30 Hz is 240 steps, close to the old step budget. The
         command resampling window moves with it or the goal changes mid-episode.

    gamma 0.98 -> 0.99 is the sixth change and lives in the agent cfg
    (PickPlaceAlignedPPORunnerCfg), because it is an algorithm parameter. At
    50 Hz with gamma 0.98 the effective horizon was 1/(1-g) = 50 steps = ONE
    SECOND against a five-second task. At 30 Hz with 0.99 it is 100 steps = 3.3 s.

    WATCH FOR, in order of likelihood:
      - the approach becoming too slow to reach the cube inside the episode.
        That looks exactly like a broken reach reward. Check lift duty before
        touching any weight.
      - chatter within the cap. The rate limit bounds target SPEED, not
        direction changes; a policy can still dither +-max_delta every step.
        so-frame reports this is fine in practice, but if the renders look
        buzzy that is what it is, and the answer is a smaller max_delta, not
        the deleted penalties coming back.
      - posture regressing after release. The penalties were incidentally
        damping the post-release flail. The lever for that is joint_deviation.

    Every reward number in docs/training_journey.md becomes incomparable across
    this boundary: gamma and episode length both rescale returns. Say so in the
    log rather than comparing across it.
    """

    MAX_DELTA = 0.03  # rad per 30 Hz step = 0.9 rad/s; see the cfg's docstring
    EPISODE_S = 8.0
    ACTION_L2 = -0.01

    def __post_init__(self):
        super().__post_init__()

        # 30 Hz CONTROL at 90 Hz PHYSICS. The first version of this ran dt 1/60 x
        # decimation 2, which is also 30 Hz control - but it silently dropped
        # physics from 100 Hz to 60 Hz, and this task's solver iteration counts and
        # contact thresholds were tuned at 100. Only the control rate needed to
        # change. 1.5x the physics cost of the base task, which is the price of not
        # re-tuning contact for a grasp.
        self.sim.dt = 1.0 / 90.0
        self.decimation = 3
        self.sim.render_interval = self.decimation

        self.actions.arm_action = pickplace_mdp.RateLimitedJointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
            scale=0.5,
            use_default_offset=True,
            max_delta=self.MAX_DELTA,
        )

        # The penalties and their whole curriculum. Note the name collision:
        # rewards.joint_vel is the penalty term, observations.policy.joint_vel is
        # the six observation dims. Both change here, for unrelated reasons.
        self.rewards.action_rate = None
        self.rewards.joint_vel = None
        for name, term in list(self.curriculum.__dict__.items()):
            if term is None or not term.params:
                continue
            if term.params.get("term_name") in ("action_rate", "joint_vel"):
                setattr(self.curriculum, name, None)

        # ...but NOT without an action-MAGNITUDE cost. Deleting the rate penalties
        # in front of a saturating rate clamp left nothing bounding the policy
        # output, and the first attempt at this config diverged at iteration 215:
        # mean_noise_std climbing 1.007 -> 1.55 from step zero, value loss 1e31,
        # then inf -> NaN. action_rate penalised how fast the action CHANGES;
        # action_l2 penalises how BIG it is, which is the force that was missing.
        #
        # Arithmetic at -0.01: six actions at |a| ~ 1 gives action_l2 ~ 6, so
        # -0.06/step against a measured ~3.4/step reward - under 2%, small enough
        # not to suppress exploration. At |a| ~ 10 it is -6/step and bites hard,
        # which is the point.
        #
        # NO CURRICULUM, deliberately. Its job is to bound from step zero, and a
        # ramp would reintroduce exactly the discontinuity Run A exists to delete.
        self.rewards.action_l2 = RewTerm(func=pickplace_mdp.action_l2,
                                         weight=self.ACTION_L2)

        self.observations.policy.joint_vel.scale = 0.0

        # last_action is 6 of the 28 observation dims, and mdp.last_action reads
        # env.action_manager.action - the UNCLAMPED policy output, not the value
        # the action term clamps internally. So the critic can be fed a runaway
        # even when the plant cannot. This is the bound that protects the critic.
        self.observations.policy.actions.clip = (-10.0, 10.0)

        self.episode_length_s = self.EPISODE_S
        self.commands.object_pose.resampling_time_range = (self.EPISODE_S, self.EPISODE_S)


@configclass
class SoArm101PickPlaceEnvCfg_ALIGNED_RESUME(SoArm101PickPlaceEnvCfg_ALIGNED):
    """--resume for the aligned task. Never for a fresh run.

    Far less to pin than the original RESUME variant, because Run A deleted the
    penalty curriculum - only the lifting_object decay is left. That is the point
    of deleting it: a resume no longer rewinds the reward function underneath a
    trained policy, so it no longer costs ~1000 iterations to re-absorb.

    lifting_object at 3.0 from step zero still never bootstraps a pick, so this
    remains resume-only.
    """

    PINNED = {"lifting_object": 3.0}

    def __post_init__(self):
        super().__post_init__()
        for name, term in list(self.curriculum.__dict__.items()):
            if term is None:
                continue
            target = term.params.get("term_name") if term.params else None
            if target not in self.PINNED:
                raise ValueError(
                    f"Curriculum term {name!r} targets {target!r}, which is not in "
                    f"SoArm101PickPlaceEnvCfg_ALIGNED_RESUME.PINNED. Add its converged "
                    f"weight there before resuming, or the resumed run silently uses "
                    f"the base weight."
                )
            setattr(self.curriculum, name, None)

        for term_name, weight in self.PINNED.items():
            getattr(self.rewards, term_name).weight = weight


@configclass
class SoArm101PickPlaceEnvCfg_DR(SoArm101PickPlaceEnvCfg_ALIGNED):
    """Run B: the aligned plant, randomized around the numbers measured on the arm.

    Run A made sim's actuation path the SAME path the bridge deploys, and it
    worked - 74.4% per-episode place success (CI 70.5-78.0, 512 episodes) against
    increment 2's 56.4% (52.1-60.7). But "the same path" is still ONE plant, and
    the real arm is a different one every time it warms up. Run B trains on a
    distribution instead of a point.

    Every range below comes from scripts/sysid/ on 2026-08-21, not from a guess:

      lag        101-202 ms across the six joints  -> delay_steps (1, 2) at 10 Hz
      settling   300-390 ms                        -> the reason for 10 Hz
      droop      up to 3.32 deg                    -> joint_offset 0.03 rad
      deadband   0.6-3.6 deg                       -+

    THE CHANGES

    1. CONTROL RATE 30 -> 10 Hz (sim.dt 1/90, decimation 9). Physics stays at
       90 Hz, so contact is untouched. Two independent reasons:

       - the measurement. Settling takes 300-390 ms. At 30 Hz the policy issues
         nine to twelve new targets inside one settle, none of which the servo
         ever reaches. It is steering a plant that cannot hear it.
       - the field. Both SO-101 pure-RL place results train at 10 Hz. livekit's
         so-frame moved DOWN from 50 deliberately, and pairs it with a per-step
         motion cap and no smoothness penalties - which is, term for term, the
         action space Run A already arrived at independently. Squint (arXiv
         2602.21203) trains at 10 Hz on a PD joint-position-delta controller.
         Two groups, two simulators, same number.

    2. MAX_DELTA 0.03 -> 0.09 rad/step. The rate cap is a SPEED, and the speed is
       held fixed at 0.9 rad/s across the change: 0.03 rad per 1/30 s and 0.09 rad
       per 1/10 s are the same arm. Rescaling is not optional - keeping 0.03 would
       have silently cut the arm to a third of its speed and made Run B a
       comparison of two things at once.

       Still under velocity_limit_sim (1.5 rad/s), so this term stays the binding
       constraint rather than PhysX.

    3. TRANSPORT DELAY, 1-2 steps, per episode. See RateLimitedJointPositionActionCfg.

    4. JOINT OFFSET, +-0.03 rad, per episode. Droop and deadband, applied to the
       target because they are plant errors and the encoder is honest.

    5. ACTUATOR GAINS x[0.7, 1.4], per episode. The stiffness a warm servo holds
       is not the one in the config, and the policy should not be able to tell.

    6. ENCODER NOISE +-0.0015 rad on joint_pos - one STS3215 encoder step
       (0.088 deg, 4096/rev). This one IS per-step, correctly: quantization
       really is fresh on every read. It is deliberately tiny; it is a floor on
       sensing, not a randomization lever.

    7. CUBE MASS AND FRICTION. The cube had NO DECLARED MASS - it inherited
       whatever density the Nucleus DexCube USD ships, which nobody has read. So
       mass is declared first (0.025 kg, the real cube on the bench) and only
       then randomized x[0.6, 1.6]. Randomizing around an unknown nominal is not
       domain randomization, it is a wider unknown.

    8. INITIAL ARM POSE +-0.05 rad. reset_scene_to_default meant every episode of
       every run to date started from the identical pose. That is a
       generalization limit before it is a transfer gap.

    NOT DONE, deliberately: the load-dependent position error from the plan. The
    1.8-2.6 deg figure is published at 1.5 kg; the cube is 25 g, and the measured
    sag under it was 0.16 deg. Modelling it would inject an error sixteen times
    larger than the one the hardware actually has.

    WATCH FOR: the number going DOWN relative to Run A's 74.4%, and that being
    correct. A policy that holds 65% across a distribution of arms is worth more
    than one that holds 74.4% on exactly one. The comparison that matters is the
    hardware run; the sim comparison to make is Run B's policy against Run A's,
    BOTH evaluated on this task.
    """

    MAX_DELTA = 0.09  # rad per 10 Hz step = 0.9 rad/s, the same speed as Run A
    DELAY_STEPS = (1, 2)  # 100-200 ms at 10 Hz; measured lag is 101-202 ms
    JOINT_OFFSET = 0.03  # rad, ~1.7 deg: droop 3.32 deg, deadband 0.6-3.6 deg
    CUBE_MASS = 0.025  # kg, the cube on the bench

    def __post_init__(self):
        super().__post_init__()

        # 10 Hz control, 90 Hz physics. decimation is the only thing that moves;
        # sim.dt stays where Run A put it so contact behaviour is unchanged.
        self.decimation = 9
        self.sim.render_interval = self.decimation

        self.actions.arm_action = pickplace_mdp.RateLimitedJointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
            scale=0.5,
            use_default_offset=True,
            max_delta=self.MAX_DELTA,
            delay_steps=self.DELAY_STEPS,
            joint_offset=self.JOINT_OFFSET,
        )

        # One encoder step. Per-step, unlike everything else here, because
        # quantization genuinely is redrawn on every read.
        self.observations.policy.joint_pos.noise = Unoise(n_min=-0.0015, n_max=0.0015)

        # Declare the cube's mass before randomizing it. See point 7.
        self.scene.object.spawn.mass_props = MassPropertiesCfg(mass=self.CUBE_MASS)

        # Events run in declaration order within a mode, and these must all land
        # AFTER reset_scene_to_default - otherwise the default reset overwrites
        # them and the run silently trains with no randomization at all. setattr
        # on a fresh name appends, which is why they go here and not in EventCfg.
        self.events.randomize_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.7, 1.4),
                "damping_distribution_params": (0.7, 1.4),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )
        self.events.randomize_cube_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "mass_distribution_params": (0.6, 1.6),
                "operation": "scale",
                "distribution": "uniform",
                "recompute_inertia": True,
            },
        )
        self.events.randomize_cube_friction = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "static_friction_range": (0.5, 1.2),
                "dynamic_friction_range": (0.4, 1.0),
                "restitution_range": (0.0, 0.1),
                "num_buckets": 64,
            },
        )
        self.events.randomize_start_pose = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "position_range": (-0.05, 0.05),
                "velocity_range": (0.0, 0.0),
            },
        )


@configclass
class SoArm101PickPlaceEnvCfg_DR_RESUME(SoArm101PickPlaceEnvCfg_DR):
    """--resume for Run B. Never for a fresh run.

    Same contract as SoArm101PickPlaceEnvCfg_ALIGNED_RESUME and for the same
    reason: lifting_object at 3.0 from step zero never bootstraps a pick, so its
    curriculum has to be pinned rather than replayed under a trained policy.
    """

    # Spelled out rather than read off SoArm101PickPlaceEnvCfg_ALIGNED_RESUME:
    # @configclass rebuilds the class and unannotated attributes are not
    # readable from outside it, so referencing it there raises AttributeError
    # at import. Keep the two in step by hand.
    PINNED = {"lifting_object": 3.0}

    def __post_init__(self):
        super().__post_init__()
        for name, term in list(self.curriculum.__dict__.items()):
            if term is None:
                continue
            target = term.params.get("term_name") if term.params else None
            if target not in self.PINNED:
                raise ValueError(
                    f"Curriculum term {name!r} targets {target!r}, which is not in "
                    f"SoArm101PickPlaceEnvCfg_DR_RESUME.PINNED. Add its converged "
                    f"weight there before resuming, or the resumed run silently uses "
                    f"the base weight."
                )
            setattr(self.curriculum, name, None)

        for term_name, weight in self.PINNED.items():
            getattr(self.rewards, term_name).weight = weight


@configclass
class SoArm101PickPlaceEnvCfg_DR_NOMINAL(SoArm101PickPlaceEnvCfg_DR):
    """EVALUATION ONLY: Run B's plant with every randomization switched off.

    WHY THIS EXISTS. The first three evals of Run B's policy could not answer the
    question they were run to answer, because the Aligned task and the DR task
    differ in TWO things, not one: randomization, and control rate (30 Hz /
    max_delta 0.03 against 10 Hz / 0.09). So:

      Run A on DR      1.2%   - mostly a rate mismatch, not a robustness result
      Run B on Aligned 17.6%  - likewise, in the other direction
      Run B on DR      21.1%  - the only clean number of the three

    This task holds the rate fixed and removes only the randomization, which is
    the comparison that was wanted. Run B's policy scored here against its 21.1%
    on the DR task is the cost of randomization AT TEST TIME, with nothing else
    moving.

    It is also the closer predictor of HARDWARE. The real arm is one arm: a
    single draw from the distribution, and a calibrated one. The DR task averages
    over draws including the worst corner of every range at once - +-0.03 rad of
    offset on every joint, 2-step delay, gains at 0.7x and a heavy cube - which is
    a machine that does not exist. A policy can be worth deploying and still score
    poorly against that average.

    Reading it:
      NOMINAL >> DR   the ranges are too wide; narrow them before retraining.
      NOMINAL ~= DR   the policy is genuinely rate-limited or under-trained, and
                      the action saturation at |a| ~ 5.14 is the first suspect.

    Everything the DR cfg declared for the PLANT rather than for randomization
    stays: 10 Hz, max_delta 0.09, and the cube's declared 25 g mass. Only the
    draws are removed.
    """

    DELAY_STEPS = (0, 0)
    JOINT_OFFSET = 0.0

    def __post_init__(self):
        super().__post_init__()

        # Rebuild the action term at the parent's own settings so the two tasks
        # cannot drift apart: same scale, same max_delta, only the per-episode
        # draws removed.
        self.actions.arm_action = pickplace_mdp.RateLimitedJointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
            scale=0.5,
            use_default_offset=True,
            max_delta=self.MAX_DELTA,
            delay_steps=self.DELAY_STEPS,
            joint_offset=self.JOINT_OFFSET,
        )

        self.observations.policy.joint_pos.noise = None

        self.events.randomize_actuator_gains = None
        self.events.randomize_cube_mass = None
        self.events.randomize_cube_friction = None
        self.events.randomize_start_pose = None


@configclass
class SoArm101PickPlaceEnvCfg_DR_C(SoArm101PickPlaceEnvCfg_DR):
    """Run C: ONE change from Run B - the action-magnitude penalty actually bites.

    THE MEASUREMENT. Run B ended with action_l2 at -1.5853, which puts the RMS
    policy output at about 5.14 per dimension against a max_raw_action clamp of
    5.0. The term reads the UNCLAMPED action, so that means the policy spends
    most steps asking for more than the plant will accept, on essentially every
    dimension. Run A ended the same way at about 5.8.

    WHY IT MATTERS MORE AT 10 Hz. A saturated policy can only move in units of
    max_delta, because every step it asks for the maximum:

        Run A   max_delta 0.03 rad  ->  1.72 deg per step
        Run B   max_delta 0.09 rad  ->  5.16 deg per step

    Both runs saturated, but Run B's saturation is three times COARSER. The
    success predicate wants the cube inside 20 mm, and a policy whose finest
    available move is five degrees on every joint cannot land that reliably. So
    max_delta 0.09 was right as a SPEED - it holds 0.9 rad/s, the same arm - and
    wrong in combination with saturation, where it stops being a ceiling and
    becomes the step size.

    The failure breakdown agrees. Of Run B's at-target kernel mass, 78% has the
    gripper open and the cube still, and 73% is at rest - both legitimate
    conditionals, since object_released and object_at_rest are object_at_target
    multiplied by indicator gates. The sequence works: it picks, carries, sets
    down and lets go. Strict place_success is 8.1% of episode time. Run B is not
    failing at the structure of the task, it is missing the tolerance.

    WHY -0.01 NEVER BOUND ANYTHING. It stayed at about 2% of the reward stack for
    the entire run - 2% at iteration 939 with mean reward 50, still 1.6% at 11999
    with mean reward 100. The penalty grew in step with the reward instead of
    biting as actions grew, so it was never a constraint, only a tax.

    -0.05 separates the two regimes:

        |a| ~ 1   6 x 1 x 0.05  = -0.3/step   against ~100   0.3%
        |a| ~ 5   6 x 25 x 0.05 = -7.5/step   against ~100   7.5%

    Negligible while exploring, decisive once saturating. Note this does NOT
    forbid a large action; the hard clamp already does that. It supplies the
    GRADIENT toward small ones that a saturating action space cannot provide by
    itself - which is the same argument that put action_l2 in Run A, just at a
    weight that survives a reward stack thirty times larger than the one it was
    sized against.

    NOTHING ELSE MOVES. Not max_delta, not the rate, not the DR ranges. If Run C
    beats Run B the credit is attributable to one number, and if it does not, the
    next suspect is already named: 80 steps for five phases is 16 steps per
    phase, which may simply be too tight at 10 Hz.
    """

    ACTION_L2 = -0.05

    def __post_init__(self):
        super().__post_init__()
        # The entire run rests on this one number reaching the reward term.
        # @configclass strips unannotated class attributes from the CLASS object
        # (that is what broke SoArm101PickPlaceEnvCfg_DR_C_RESUME.PINNED), so an
        # override that silently resolved to the parent's -0.01 would produce a
        # rerun of Run B under Run C's name and cost 11 h to discover. Fail loudly
        # at construction instead.
        if self.rewards.action_l2.weight != -0.05:
            raise ValueError(
                f"Run C requires action_l2 weight -0.05, resolved "
                f"{self.rewards.action_l2.weight}. The ACTION_L2 override did not "
                f"reach the reward term, so this run would repeat Run B."
            )


@configclass
class SoArm101PickPlaceEnvCfg_DR_C_RESUME(SoArm101PickPlaceEnvCfg_DR_C):
    """--resume for Run C. This is the CHEAP experiment - use this one first.

    Resuming Run B's checkpoint under the heavier penalty tests the hypothesis in
    about 3 h instead of 11.6. Two numbers answer it:

      action_l2 should climb from -1.5853 toward -0.3 as the RMS output drops
      below the clamp, and place_success should rise from 0.0809.

    If the output comes down and place_success does not follow, saturation was
    not the binding constraint and the episode budget is next.

    Changing a reward weight under a trained policy normally costs a re-absorption
    period - that is exactly what deleting the penalty curriculum in Run A was
    meant to stop. Expect a dip before any gain, and do not read the first ~500
    iterations.
    """

    PINNED = {"lifting_object": 3.0}

    def __post_init__(self):
        super().__post_init__()
        for name, term in list(self.curriculum.__dict__.items()):
            if term is None:
                continue
            target = term.params.get("term_name") if term.params else None
            if target not in self.PINNED:
                raise ValueError(
                    f"Curriculum term {name!r} targets {target!r}, which is not in "
                    f"SoArm101PickPlaceEnvCfg_DR_C_RESUME.PINNED. Add its converged "
                    f"weight there before resuming, or the resumed run silently uses "
                    f"the base weight."
                )
            setattr(self.curriculum, name, None)

        for term_name, weight in self.PINNED.items():
            getattr(self.rewards, term_name).weight = weight


@configclass
class SoArm101PickPlaceEnvCfg_DR_D(SoArm101PickPlaceEnvCfg_DR_C):
    """Run D: transport is HORIZONTAL. One change - the crane, not the drag.

    From watching play: the arm should pick up, lift, turn and lower. It picks up
    and then contorts down to shuffle the cube along at table height. The reward
    explains it exactly - object_goal_tracking measured a 3D norm to a goal on
    the table, so lifting increased the distance the largest term in the stack
    was minimising. Carrying 10 cm up cost 9.1/step against lifting_object's 3.0.
    Net -6.1/step to stay low. See object_goal_distance_xy_latched.

    Both tracking scales move to horizontal distance. object_at_target keeps its
    xy x z kernel and becomes the only term that pays for the descent, which is
    what makes the two jobs disjoint rather than opposed:

        tracking (16) + fine (5)   get the cube OVER the goal
        object_at_target (12)      bring it DOWN onto the goal

    Nothing changes about the anti-slide latch, the thresholds, or any weight.

    WATCH FOR:
      - lift duty rising off 13.9%. That is the direct prediction. If it does not
        move, this diagnosis is wrong and the trajectory is limited by something
        other than the reward gradient.
      - grasp_top_down finally becoming readable. It is gated on z > 0.03 and has
        been structurally suppressed at 0.02 for three runs; a real carry height
        is the first chance to see what the grasp angle actually is.
      - the cube being carried high and DROPPED rather than placed. object_at_rest
        and object_released both gate on object_at_target's z kernel, so a drop
        from height pays nothing until it settles - but it is the failure mode
        this change opens the door to, and it is what to look for in play.

    NOT changed, deliberately: object_at_target's xy_std, which is 0.05 against a
    20 mm success threshold and is the next lever. Shipping both would repeat
    increment 1d's mistake of two changes in one run.
    """

    def __post_init__(self):
        super().__post_init__()

        self.rewards.object_goal_tracking.func = pickplace_mdp.object_goal_distance_xy_latched
        self.rewards.object_goal_tracking_fine_grained.func = (
            pickplace_mdp.object_goal_distance_xy_latched
        )

        # Same guard as Run C's: the swap must actually reach the terms, or this
        # is an 11 h rerun of Run C under a different name.
        for name in ("object_goal_tracking", "object_goal_tracking_fine_grained"):
            func = getattr(self.rewards, name).func
            if func is not pickplace_mdp.object_goal_distance_xy_latched:
                raise ValueError(
                    f"Run D requires {name} to use object_goal_distance_xy_latched, "
                    f"resolved {func}. Without the swap this run repeats Run C."
                )


@configclass
class SoArm101PickPlaceEnvCfg_DR_D_RESUME(SoArm101PickPlaceEnvCfg_DR_D):
    """--resume for Run D, from Run C's checkpoint.

    Resuming is the right call here even though the reward CHANGES shape: the
    pick, the grasp and the release are all intact in Run C's policy and none of
    them are touched. Only the transport gradient is. Expect a dip while the
    carry re-forms, and do not read the first ~500 iterations.
    """

    PINNED = {"lifting_object": 3.0}

    def __post_init__(self):
        super().__post_init__()
        for name, term in list(self.curriculum.__dict__.items()):
            if term is None:
                continue
            target = term.params.get("term_name") if term.params else None
            if target not in self.PINNED:
                raise ValueError(
                    f"Curriculum term {name!r} targets {target!r}, which is not in "
                    f"SoArm101PickPlaceEnvCfg_DR_D_RESUME.PINNED. Add its converged "
                    f"weight there before resuming, or the resumed run silently uses "
                    f"the base weight."
                )
            setattr(self.curriculum, name, None)

        for term_name, weight in self.PINNED.items():
            getattr(self.rewards, term_name).weight = weight
