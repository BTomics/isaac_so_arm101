# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_error_magnitude, quat_mul

from .place import object_was_lifted

# The gripper's APPROACH direction in the gripper's local frame (normalized), i.e.
# the axis that must point world-down for a clean top-down grasp. This is the
# ee_frame offset (gripper_link -> fingertips) NEGATED: with the raw offset sign
# the policy converged to holding the cube overhead, gripper pointing up, so the
# frame's local axis runs opposite to the offset. Sign verified in sim, do not
# re-derive it from the offset.
_GRIPPER_APPROACH_LOCAL = (-0.01, 0.0, 0.09)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    lift_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward tracking the cube to the goal pose (tanh-kernel), gated on the
    was-lifted LATCH rather than the cube's current height.

    Diverges deliberately from the lift-task original, which gated on
    ``object_z > minimal_height`` — i.e. it only paid while the cube was
    AIRBORNE. That is a lift assumption, and it is the root cause of the traps
    this task kept hitting: the term fights the descent (the cube must come down
    to be placed, which switches the reward off), so it had to be curriculum-
    decayed to zero, which in turn deleted the only pull toward the target and
    promoted ``lifting_object`` into a risk-free hold-forever annuity.

    Gating on the latch instead means: once the cube has been genuinely picked
    this episode, tracking pays CONTINUOUSLY — lift, transport, descent, and
    placed-on-the-table — because the goal itself is on the table. That makes the
    whole trajectory monotonic (hovering scores strictly worse than descending)
    and needs no curriculum decay.
    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # anti-slide gate: only credit tracking once the cube was genuinely picked
    lifted = object_was_lifted(env, lift_height, object_cfg, update=True)
    return lifted * (1 - torch.tanh(distance / std))


def object_ee_distance_and_lifted(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Combined reward for reaching the object AND lifting it."""
    # Get reaching reward
    reach_reward = object_ee_distance(env, std, object_cfg, ee_frame_cfg)
    # Get lifting reward
    lift_reward = object_is_lifted(env, minimal_height, object_cfg)
    # Combine rewards multiplicatively
    return reach_reward * lift_reward


# ---------------------------------------------------------------------------
# PickPlace authorship — the place / release / at-rest terms.
# These are the reward functions that turn "lift" into "place". They are STUBS:
# fill them in. Spec + trap catalogue: SOARMRL/docs/pickplace_contract.md
# (Increment 1). Weight them so place/release DOMINATE the surviving reach
# term, or the arm hovers the cube forever instead of setting it down.
# ---------------------------------------------------------------------------


def object_at_target_on_table(
    env: ManagerBasedRLEnv,
    xy_std: float,
    z_std: float,
    command_name: str,
    lift_height: float = 0.06,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the cube being at the commanded XY *and* down at table height.

    This is the term that replaces the airborne ``object_goal_distance`` gate —
    it must stay alive as the cube DESCENDS to the table, otherwise the policy
    is punished for finishing the place (the core blocker, contract §"The
    blocker you must design around").

    What to compute (reuse the ``object_goal_distance`` pattern above):
      - Desired pos in world frame: transform ``command[:, :3]`` by the robot
        root state (``combine_frame_transforms``), exactly like
        ``object_goal_distance`` lines 65-68.
      - XY reward: tanh-kernel on the *planar* distance to the target
        (``des_pos_w[:, :2]`` vs ``object.data.root_pos_w[:, :2]``), std ``xy_std``.
      - Z gate: the cube is at table height, i.e. ``|object_z - target_z| < z_tol``.
        Do NOT gate on ``> minimal_height`` — that is the airborne gate you are
        replacing. The place target z is the cube's resting center (~0.015 for
        the 3 cm cube; verify in sim).

    Invariant: this reward must be MAXIMAL when the cube is correctly placed at
    rest on the table, not zero. Watch the play render for the yank-back-up
    symptom that means the airborne term is still winning.
    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    
    # XY reward: tanh-kernel on the *planar* distance to the target
    distance_xy = torch.norm(des_pos_w[:, :2] - object.data.root_pos_w[:, :2], dim=1)
    xy_reward = 1.0 - torch.tanh(distance_xy / xy_std)
    
    # Z reward: the cube is at table height
    distance_z = torch.abs(object.data.root_pos_w[:, 2] - des_pos_w[:, 2])
    z_reward = 1.0 - torch.tanh(distance_z / z_std)

    # Anti-slide gate: only credit a placement if the cube was genuinely PICKED
    # (lifted clear of the table) at some point this episode. Pure sliding never
    # sets the latch, so it earns nothing here. This term runs every step, so it
    # is the latch's updater (update=True); object_released / object_at_rest
    # inherit the gate by calling this function; place_complete reads it.
    lifted = object_was_lifted(env, lift_height, object_cfg, update=True)

    return lifted * z_reward * xy_reward


def object_released(
    env: ManagerBasedRLEnv,
    command_name: str,
    lin_vel_thresh: float,
    xy_std: float,
    z_std: float,
    gripper_open_thresh: float,
    lift_height: float = 0.06,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward opening the gripper once the cube is placed and nearly still.

    The point of this term: the task is not done until the gripper LETS GO.
    Gate release on low object velocity so the policy cannot farm it by
    dropping the cube from height (the "early drop" trap).

    What to compute:
      - Gripper openness: read the gripper joint position from the robot
        articulation. ``robot_cfg`` already selects the gripper joint — get its
        index via ``robot_cfg.joint_ids`` and read
        ``robot.data.joint_pos[:, gripper_idx]``. Open ⇔ position past
        ``gripper_open_thresh`` (binary action commands open=0.5/close=0.0, so a
        threshold near the mid-point works; confirm the sign on your arm).
      - At target: cube within ``xy_std``/``z_tol`` of the commanded spot
        (same as ``object_at_target_on_table`` — you can call it or inline it).
      - Nearly still: ``torch.norm(object.data.root_lin_vel_w, dim=1) < vel_thresh``.
      - Reward = the AND of (open) AND (at target) AND (still). Multiplicative
        gating keeps it un-farmable.

    Trap: if the policy releases early to grab this reward, tighten ``vel_thresh``
    or add an impact-speed penalty (a separate term). Contract §Increment 1.
    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    
    # Gripper openness
    gripper_idx = robot_cfg.joint_ids[0]  # assuming single gripper joint
    gripper_pos = robot.data.joint_pos[:, gripper_idx]
    gripper_open = (gripper_pos > gripper_open_thresh).float()
    
    # At target (inherits the anti-slide latch gate)
    at_target = object_at_target_on_table(
        env=env,
        xy_std=xy_std,
        z_std=z_std,
        command_name=command_name,
        lift_height=lift_height,
        robot_cfg=robot_cfg,
        object_cfg=object_cfg,
    )
    
    # Nearly still
    linear_vel = torch.norm(object.data.root_lin_vel_w, dim=1)
    still_check = (linear_vel < lin_vel_thresh).float()
    
    # Reward = AND of all three
    return gripper_open * at_target * still_check

def object_at_rest(
    env: ManagerBasedRLEnv,
    lin_vel_thresh: float,
    ang_vel_thresh: float,
    command_name: str,
    xy_std: float,
    z_std: float,
    lift_height: float = 0.06,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the cube being at rest — low linear AND angular velocity.

    Distinguishes "placed and settled" from "still being held / dragged / in
    flight". Use both ``object.data.root_lin_vel_w`` and
    ``object.data.root_ang_vel_w`` (a spun/toppling cube is not placed).

    Keep this a gentle shaping term, not a dominant one — on its own it rewards
    the arm for simply not touching the cube. It earns its weight only in
    combination with the at-target term (a still cube in the WRONG place should
    not score well), so consider multiplying by the at-target gate rather than
    summing this in raw.
    """
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    
    # Linear velocity
    linear_vel = torch.norm(object.data.root_lin_vel_w, dim=1)
    linear_vel_check = (linear_vel < lin_vel_thresh).float()
    
    # Angular velocity
    angular_vel = torch.norm(object.data.root_ang_vel_w, dim=1)
    angular_vel_check = (angular_vel < ang_vel_thresh).float()
    
    object_at_target = object_at_target_on_table(
        env=env,
        xy_std=xy_std,
        z_std=z_std,
        command_name=command_name,
        lift_height=lift_height,
        robot_cfg=robot_cfg,
        object_cfg=object_cfg,
    )
    # Reward = AND of both (object_at_target already carries the anti-slide gate)
    return linear_vel_check * angular_vel_check * object_at_target


def object_orientation_to_target(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    lift_height: float = 0.06,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the cube's orientation matching the commanded (upright) orientation.

    Nothing else constrains *how* the cube ends up at the target, so the policy
    can hit the target position with a tilted / under-the-cube grasp (seen as a
    large ``Metrics/object_orientation_error``). A badly rotated, hand-held cube
    is not resting flat, so opening the gripper drops it — which is why the
    policy won't release. This term pushes a clean, upright placement so the set-
    down is stable and releasing becomes safe.

    Tanh-kernel on the quaternion angle between the cube and the commanded
    orientation (command rotation is in the robot base frame → composed into
    world). Use a generous ``std`` so it punishes large tilts, not small yaw.
    Gated on the anti-slide latch (read-only) so an untouched upright cube at
    spawn earns nothing.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    # desired orientation in the world frame: robot_root_quat ⊗ command_quat
    des_quat_w = quat_mul(robot.data.root_state_w[:, 3:7], command[:, 3:7])
    orientation_error = quat_error_magnitude(object.data.root_quat_w, des_quat_w)
    reward = 1.0 - torch.tanh(orientation_error / std)

    lifted = object_was_lifted(env, lift_height, object_cfg, update=False)
    return lifted * reward


def grasp_top_down(
    env: ManagerBasedRLEnv,
    std: float,
    near_std: float = 0.1,
    min_height: float = 0.03,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward holding the cube top-down *while it is lifted off the table*.

    The base lift reward (``object_ee_distance``) only pulls the EE *point* to the
    cube — it never constrains the arm's configuration, so with a redundant arm +
    self-collisions the policy settles into a folded, sideways/under grasp. That
    contorted hold can't set the cube down flat or release it (inherited straight
    into the place task). This term rewards the gripper's approach axis (the
    ee-frame offset direction, ~gripper-local −Z) pointing world-down, weighted by
    proximity to the cube.

    Un-farmable gate: it only pays while the cube is actually OFF the table
    (``root_z > min_height``). On the table it is exactly zero, so the policy
    cannot park open-handed over a grounded cube farming posture (the observed
    dead-end where ``lifting_object`` collapsed to 0). The only way to collect it
    is to lift — and while lifting, ``lifting_object`` is already driving the cube
    up — so this term shapes the *lift* to stay top-down instead of competing with
    it.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    quat = ee_frame.data.target_quat_w[..., 0, :]  # (N, 4) gripper world orientation
    approach_local = torch.tensor(_GRIPPER_APPROACH_LOCAL, device=quat.device, dtype=quat.dtype)
    approach_local = (approach_local / torch.norm(approach_local)).expand(quat.shape[0], 3)
    approach_world = quat_apply(quat, approach_local)

    # cos angle with world-down (0, 0, -1); = 1 when the gripper points straight down
    cos_down = -approach_world[:, 2]
    down_reward = 1.0 - torch.tanh((1.0 - cos_down) / std)

    # only shape the grasp: weight by how close the EE already is to the cube
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    dist = torch.norm(ee_pos - obj.data.root_pos_w[:, :3], dim=1)
    near = 1.0 - torch.tanh(dist / near_std)

    # un-farmable: zero unless the cube is genuinely lifted off the table
    lifted_now = (obj.data.root_pos_w[:, 2] > min_height).float()

    return lifted_now * near * down_reward
