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
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


def object_position_biased(
    env: ManagerBasedRLEnv,
    max_bias: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """EVALUATION ONLY: cube position with a PER-EPISODE CONSTANT offset.

    This is the perception error that actually matters. An AdditiveUniformNoise
    model on the plain observation resamples every step, so across a 250-step
    episode at 50 Hz the policy sees 250 independent draws around the true value
    and low-pass-filters them away almost for free - the trained policy shrugs off
    +-10 mm of it. A real camera does not fail that way: an extrinsic calibration
    error reports the cube off in the SAME direction every frame, and there is
    nothing to average out. The policy simply believes the cube is somewhere it
    is not.

    So the bias is drawn once per episode, uniform in [-max_bias, max_bias] per
    axis, and held until reset. Where place_success falls off across a few levels
    is the CALIBRATION BUDGET for the camera mount.

    State lives on ``env._object_pos_bias`` and is resampled per-episode by the
    :func:`reset_object_position_bias` event - the same pattern the was-lifted
    latch uses. Registering this observation WITHOUT that event leaves the bias
    at its initial draw forever, which silently measures something else.
    """
    true_pos_b = object_position_in_robot_root_frame(env, robot_cfg, object_cfg)
    if not hasattr(env, "_object_pos_bias"):
        env._object_pos_bias = torch.zeros_like(true_pos_b).uniform_(-max_bias, max_bias)
    return true_pos_b + env._object_pos_bias


def reset_object_position_bias(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor, max_bias: float
) -> None:
    """Event (mode="reset"): draw a fresh per-episode perception bias."""
    if not hasattr(env, "_object_pos_bias"):
        env._object_pos_bias = torch.zeros(env.num_envs, 3, device=env.device)
    env._object_pos_bias[env_ids] = torch.empty(
        len(env_ids), 3, device=env.device
    ).uniform_(-max_bias, max_bias)
