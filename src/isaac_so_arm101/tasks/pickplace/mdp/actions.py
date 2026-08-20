# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The deployment actuation path, reproduced in simulation.

WHY: the trained policy's action becomes a joint target directly. The DEPLOYED
policy's action goes through two more stages first, in SOARMRL's bridge:

    blended = prev_sent + slow * (target - prev_sent)       # bridge.slow_blend
    sent    = clamp(blended, prev_sent +- max_delta)        # bridge.clamp_delta

with `slow = 0.1` and `max_delta = 0.03` rad in scripts/grasp/pickplace_live.py.
At the deployed 30 Hz that blend is a first-order lag with a time constant of
about 0.32 s -- roughly ten ticks -- and the clamp caps joint speed at 0.9 rad/s
against the simulator's 1.5. Neither exists in training.

That is enough on its own to explain a policy which settles into a stable fixed
point on hardware instead of carrying the cube: it is driving a plant an order of
magnitude slower than the one it learned on, with a discount horizon
(gamma 0.98 at 50 Hz) of about one second.

USE FOR EVALUATION FIRST. Replaying the existing checkpoint through this is what
tells you whether the actuation path explains the hardware attractor, before
spending a training run on the answer. If it does, the fix is NOT to keep this
term: it is to train against a rate limit the policy can actually plan around
(see the retrain plan's Run A), and then drop the blend at deployment because
the policy no longer needs taming.

NOTE: the blend is applied in `process_actions`, which runs once per POLICY step.
`apply_actions` runs once per SIM step -- decimation=2 -- so doing it there would
apply the blend twice per policy step and halve the time constant being modelled.
"""

from __future__ import annotations

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass


class DeploymentJointPositionAction(JointPositionAction):
    """JointPositionAction plus the bridge's slow-blend and per-tick delta clamp."""

    cfg: DeploymentJointPositionActionCfg

    def __init__(self, cfg: DeploymentJointPositionActionCfg, env):
        super().__init__(cfg, env)
        # Start from where the arm actually is, not from zero: a first tick that
        # blends toward the target from an arbitrary origin is a transient the
        # hardware never has, since the bridge seeds prev_sent from the encoders.
        self._prev_target = self._asset.data.joint_pos[:, self._joint_ids].clone()

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        step = self.cfg.slow * (self._processed_actions - self._prev_target)
        step = step.clamp(-self.cfg.max_delta, self.cfg.max_delta)
        self._processed_actions = self._prev_target + step
        self._prev_target = self._processed_actions.clone()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            self._prev_target[:] = self._asset.data.joint_pos[:, self._joint_ids]
        else:
            self._prev_target[env_ids] = self._asset.data.joint_pos[env_ids][:, self._joint_ids]


@configclass
class DeploymentJointPositionActionCfg(JointPositionActionCfg):
    """Defaults are the values scripts/grasp/pickplace_live.py actually deploys.

    Keep them in sync with that file. If they drift apart, this term measures a
    deployment path that does not exist, which is worse than not measuring one.
    """

    class_type: type = DeploymentJointPositionAction

    slow: float = 0.1
    """Blend factor per policy step; 1.0 disables the blend."""

    max_delta: float = 0.03
    """Per-step joint target change cap, radians."""
