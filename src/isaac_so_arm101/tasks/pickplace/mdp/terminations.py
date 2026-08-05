# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

from .place import place_complete

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

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

    # rewarded if the object is lifted above the threshold
    return distance < threshold


def place_success(
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
    """Success termination for pick-and-place — the AND of every place condition.

    A real "placed" state, not just "near the spot": the cube was lifted this
    episode (anti-slide gate), is within ``xy_threshold`` of the commanded XY, at
    table height (``z_tol``), at rest (lin AND ang velocity), the gripper is
    open, and the EE has withdrawn ``ee_clearance`` from the cube. The condition
    lives in :func:`place.place_complete` so the success-bonus reward uses the
    exact same definition. Keep ``object_dropping`` as the failure counterpart.
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
    )
