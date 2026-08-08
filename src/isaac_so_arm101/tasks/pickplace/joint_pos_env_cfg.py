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
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
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
