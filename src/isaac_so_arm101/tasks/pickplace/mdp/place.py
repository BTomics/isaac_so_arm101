# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared pick-and-place logic: the was-lifted latch (anti-slide gate) and the
single "cube is genuinely placed" predicate used by both the success
termination and the success-bonus reward.

Why a latch: the place target sits on the table at the cube's resting height, so
``object_at_target_on_table`` and friends are otherwise satisfiable by *sliding*
the cube — the policy never learns to pick. The latch records whether the cube
was ever lifted clear of the table this episode; the place terms gate on it, so
sliding earns nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gripper_joint_pos(robot: RigidObject, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Gripper joint position, (num_envs,), from a cfg that selects exactly one joint.

    ``SceneEntityCfg.joint_ids`` defaults to ``slice(None)`` and only becomes a list
    of indices when ``.resolve(scene)`` turns ``joint_names`` into ``joint_ids``. The
    managers resolve only the ``SceneEntityCfg`` objects they find in a term's
    ``params`` dict — a cfg left as a function-signature default is never resolved.
    So ``robot_cfg.joint_ids[0]`` raises ``TypeError: 'slice' object is not
    subscriptable`` on the first step unless the term passes ``robot_cfg`` in params.

    Every term before Increment 1c used ``robot_cfg`` only for ``root_state_w``,
    which needs no resolution, so nothing caught this until a term first read a
    joint.

    Raising beats coping here. The tempting fix — index with ``joint_ids`` and mean
    over whatever comes back — turns an unresolved cfg into the mean of ALL SIX joint
    angles, which is arm posture, not gripper aperture. That trains happily and
    silently rewards the wrong thing.
    """
    joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    if joint_pos.ndim != 2 or joint_pos.shape[-1] != 1:
        n = joint_pos.shape[-1] if joint_pos.ndim == 2 else "all"
        raise ValueError(
            f"Expected a SceneEntityCfg selecting exactly one gripper joint, got {n} joints "
            f"(joint_ids={robot_cfg.joint_ids!r}, joint_names={robot_cfg.joint_names!r}). "
            "Pass robot_cfg=SceneEntityCfg('robot', joint_names=['gripper']) in the term's "
            "params dict so the manager resolves it — a signature default is not resolved."
        )
    return joint_pos.squeeze(-1)


def object_was_lifted(
    env: ManagerBasedRLEnv,
    lift_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    update: bool = True,
) -> torch.Tensor:
    """Per-env latch as a float (1.0 / 0.0): has the cube cleared ``lift_height``
    at any point since the last reset.

    Forces a genuine PICK — the place rewards/termination gate on this, so a
    policy that only slides the cube along the table never unlocks them. State
    lives on ``env._cube_lifted_latch`` and is cleared per-episode by the
    :func:`reset_lifted_latch` event.

    ``update=True`` (the per-step place reward) OR-accumulates the current
    above-height test into the latch; read-only callers (the success predicate)
    pass ``update=False``. OR-accumulation is idempotent, so being called by
    several terms in one step is harmless.
    """
    if not hasattr(env, "_cube_lifted_latch"):
        env._cube_lifted_latch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if update:
        obj: RigidObject = env.scene[object_cfg.name]
        env._cube_lifted_latch |= obj.data.root_pos_w[:, 2] > lift_height
    return env._cube_lifted_latch.float()


def reset_lifted_latch(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Event (mode="reset"): clear the was-lifted latch for the resetting envs."""
    if hasattr(env, "_cube_lifted_latch"):
        env._cube_lifted_latch[env_ids] = False


def place_complete(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    xy_threshold: float = 0.02,
    z_tol: float = 0.01,
    lin_vel_thresh: float = 0.02,
    ang_vel_thresh: float = 0.05,
    gripper_open_thresh: float = 0.25,
    ee_clearance: float = 0.05,
    lift_height: float = 0.06,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Bool per-env: the cube is *genuinely* placed. Single source of truth for
    both the success termination and the success-bonus reward.

    ALL must hold: the cube was lifted this episode (anti-slide gate), is within
    ``xy_threshold`` of the commanded XY, at table height (``z_tol``), at rest
    (linear AND angular velocity), the gripper is open, and the end-effector has
    withdrawn ``ee_clearance`` from the cube.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    command = env.command_manager.get_command(command_name)

    # desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)

    # cube at commanded XY and at table height
    at_xy = torch.norm(des_pos_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1) < xy_threshold
    at_z = torch.abs(obj.data.root_pos_w[:, 2] - des_pos_w[:, 2]) < z_tol

    # cube at rest
    at_rest = (torch.norm(obj.data.root_lin_vel_w, dim=1) < lin_vel_thresh) & (
        torch.norm(obj.data.root_ang_vel_w, dim=1) < ang_vel_thresh
    )

    # gripper open
    gripper_open = gripper_joint_pos(robot, robot_cfg) > gripper_open_thresh

    # end-effector withdrawn from the cube
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    withdrawn = torch.norm(ee_pos - obj.data.root_pos_w[:, :3], dim=1) > ee_clearance

    # genuine pick (anti-slide) — read-only; the place reward keeps the latch current
    lifted = object_was_lifted(env, lift_height, object_cfg, update=False) > 0.5

    return at_xy & at_z & at_rest & gripper_open & withdrawn & lifted


def place_success_bonus(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    xy_threshold: float = 0.02,
    z_tol: float = 0.01,
    lin_vel_thresh: float = 0.02,
    ang_vel_thresh: float = 0.05,
    gripper_open_thresh: float = 0.25,
    ee_clearance: float = 0.05,
    lift_height: float = 0.06,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward: a one-time bonus (1.0) on the step ``place_complete`` holds.

    Because the ``place_success`` termination ends the episode on the same
    condition, this bonus is collected on the terminal step. Its weight MUST
    out-value the dense place reward the agent forgoes by ending early —
    otherwise the policy avoids success to keep farming (the observed
    ``time_out ≈ 1`` / decaying success-rate failure mode). Full explicit
    signature (not ``**kwargs``) so the manager resolves the ``SceneEntityCfg``
    defaults for this term.
    """
    return place_complete(
        env,
        command_name=command_name,
        xy_threshold=xy_threshold,
        z_tol=z_tol,
        lin_vel_thresh=lin_vel_thresh,
        ang_vel_thresh=ang_vel_thresh,
        gripper_open_thresh=gripper_open_thresh,
        ee_clearance=ee_clearance,
        lift_height=lift_height,
        robot_cfg=robot_cfg,
        object_cfg=object_cfg,
        ee_frame_cfg=ee_frame_cfg,
    ).float()
