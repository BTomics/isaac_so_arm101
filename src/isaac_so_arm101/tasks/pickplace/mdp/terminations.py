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
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Success termination for pick-and-place — the AND of every place condition.

    A real "placed" state, not just "near the spot". Terminate (success) only
    when ALL hold (elementwise, return a bool tensor of shape (num_envs,)):

      1. cube within ``xy_threshold`` of the commanded XY (transform the command
         to world frame like ``object_reached_goal`` above),
      2. cube at table height: ``|object_z - target_z| < z_tol``,
      3. cube at rest: lin AND ang velocity below ``vel_thresh``,
      4. gripper OPEN: gripper joint pos past ``gripper_open_thresh``
         (index via ``robot_cfg.joint_ids``),
      5. EE WITHDRAWN: end-effector at least ``ee_clearance`` from the cube
         (``ee_frame.data.target_pos_w[..., 0, :]`` vs ``object.data.root_pos_w``)
         — without this the gripper camps on the placed cube (contract trap).

    Keep the existing ``object_dropping`` term as a failure termination; this is
    the *success* counterpart. Spec: pickplace_contract.md Increment 2.
    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    
    # 1. Cube within xy_threshold of commanded XY
    distance_xy = torch.norm(des_pos_w[:, :2] - object.data.root_pos_w[:, :2], dim=1)
    xy_cond = (distance_xy < xy_threshold).float()
    
    # 2. Cube at table height
    target_z = des_pos_w[:, 2]  # the height of the command
    z_cond = (torch.abs(object.data.root_pos_w[:, 2] - target_z) < z_tol).float()
    
    # 3. Cube at rest (linear AND angular velocity below vel_thresh)
    linear_vel = torch.norm(object.data.root_lin_vel_w, dim=1)
    ang_vel = torch.norm(object.data.root_ang_vel_w, dim=1)
    rest_cond = ((linear_vel < lin_vel_thresh) & (ang_vel < ang_vel_thresh)).float()
    
    # 4. Gripper OPEN
    gripper_idx = robot_cfg.joint_ids[0]  # assuming single gripper joint
    gripper_pos = robot.data.joint_pos[:, gripper_idx]
    gripper_open_cond = (gripper_pos > gripper_open_thresh).float()
    
    # 5. EE WITHDRAWN
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    dist_ee_cube = torch.norm(ee_pos - object.data.root_pos_w[:, :3], dim=1)
    withdrawn_cond = (dist_ee_cube > ee_clearance).float()
    
    # Success = ALL conditions hold (elementwise AND)
    success_tensor = (xy_cond * z_cond * rest_cond * gripper_open_cond * withdrawn_cond)
    
    return success_tensor > 0.5  # return bool tensor
