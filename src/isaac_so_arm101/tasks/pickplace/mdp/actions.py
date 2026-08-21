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

TWO FORMS, and the difference is the whole point:

  RateLimitedJointPositionActionCfg   slow=1.0  -- a hard rate limit. TRAIN on this.
  DeploymentJointPositionActionCfg    slow=0.1  -- rate limit + the bridge's lag.
                                                   EVALUATE on this, never train.

A rate limit is a constraint the policy can plan around: it learns that a joint
moves at most `max_delta` per step and paces itself. A blend is a lag that fights
it: the policy asks for a move, gets 10% of it, asks harder, and by the time the
arm arrives the policy has reversed. The 2026-08-20 hardware log is that failure
written out -- elbow demand +2.27 rad against +0.85 sent, reversing to -0.12
before the arm got there.

So the sequence is: evaluate with the blend to confirm the diagnosis, train with
the rate limit, then drop the blend at deployment because a policy that paces
itself no longer needs taming.

NOTE: the blend is applied in `process_actions`, which runs once per POLICY step.
`apply_actions` runs once per SIM step -- decimation=2 -- so doing it there would
apply the blend twice per policy step and halve the time constant being modelled.
"""

from __future__ import annotations

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass


class RateLimitedJointPositionAction(JointPositionAction):
    """JointPositionAction with a per-step cap on how far the target may move.

    With `slow = 1.0` (the default) this is a pure rate limit: a hard ceiling on
    joint speed that a policy can learn to plan around, and the form to TRAIN
    against. With `slow < 1.0` it additionally reproduces the bridge's
    exponential blend, which is a lag that fights the policy rather than a
    constraint it can plan around -- that form is for evaluation only.

    The clamp is against the previous COMMANDED target, not the measured joint
    position, matching bridge.clamp_delta. Clamping against the measurement would
    let a sagging joint drag the target down with it and turn the action space
    into an integrator wrapped around the tracking error.
    """

    cfg: RateLimitedJointPositionActionCfg

    def __init__(self, cfg: RateLimitedJointPositionActionCfg, env):
        super().__init__(cfg, env)
        # Seed from where the arm actually is. The bridge seeds prev_sent from
        # the encoders after its ramp, so starting anywhere else would give the
        # first step a transient the hardware never has.
        self._prev_target = self._asset.data.joint_pos[:, self._joint_ids].clone()

        lo, hi = cfg.delay_steps
        if lo < 0 or hi < lo:
            raise ValueError(f"delay_steps must be a non-negative (lo, hi); got {cfg.delay_steps}")
        # Newest target at index 0, so index `d` is the target from d steps ago.
        self._history = self._prev_target.unsqueeze(1).repeat(1, hi + 1, 1)
        self._delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._bias = torch.zeros_like(self._prev_target)
        self._env_arange = torch.arange(self.num_envs, device=self.device)
        self._resample(self._env_arange)

    def _resample(self, env_ids: torch.Tensor):
        """Draw the per-EPISODE plant parameters: transport delay and joint offset.

        Per episode, not per step, and that is the whole point. This project
        already measured the difference on the cube-position observation: +-10 mm
        resampled every step cost nothing, +-10 mm held constant for an episode
        nearly broke the policy. A servo's backlash and its gravity droop are
        constants within a run; white noise is not a model of either.
        """
        lo, hi = self.cfg.delay_steps
        if hi > 0:
            self._delay[env_ids] = torch.randint(
                lo, hi + 1, (len(env_ids),), device=self.device
            )
        if self.cfg.joint_offset > 0.0:
            self._bias[env_ids] = torch.empty(
                len(env_ids), self._bias.shape[1], device=self.device
            ).uniform_(-self.cfg.joint_offset, self.cfg.joint_offset)

    def process_actions(self, actions: torch.Tensor):
        # Bound the raw action BEFORE the parent maps it to a joint target.
        #
        # WHY (learned the hard way, 2026-08-20, crash at iteration 215): the rate
        # clamp below makes every action beyond the band produce an IDENTICAL
        # target, so there is no gradient opposing action growth. Delete the
        # action-magnitude penalties on top of that and nothing bounds the policy
        # output at all - mean_noise_std climbed 1.007 -> 1.55 from iteration zero
        # and the value loss reached 1e31.
        #
        # +-5 with scale 0.5 is +-2.5 rad of target offset, ~80x max_delta, so this
        # never binds on any useful behaviour. It only removes the runaway.
        actions = actions.clamp(-self.cfg.max_raw_action, self.cfg.max_raw_action)
        super().process_actions(actions)
        step = self.cfg.slow * (self._processed_actions - self._prev_target)
        step = step.clamp(-self.cfg.max_delta, self.cfg.max_delta)
        commanded = self._prev_target + step
        # Clamp against what was COMMANDED, not what the plant received. The
        # bridge clamps against its own last sent value and knows nothing about
        # the servo's internal lag, so the rate limit the policy learns has to be
        # measured from the same place.
        self._prev_target = commanded.clone()

        # Transport delay, then the standing joint offset. Order matters: the
        # delay is the servo taking time to hear the command, the offset is where
        # it settles once it has. Applying the offset first would delay it too,
        # which would make gravity droop appear only after the lag.
        self._history = torch.roll(self._history, shifts=1, dims=1)
        self._history[:, 0] = commanded
        self._processed_actions = self._history[self._env_arange, self._delay] + self._bias

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = self._env_arange
            self._prev_target[:] = self._asset.data.joint_pos[:, self._joint_ids]
        else:
            self._prev_target[env_ids] = self._asset.data.joint_pos[env_ids][:, self._joint_ids]
        # Flush the delay line to the new pose, or the first steps of a fresh
        # episode replay targets from the previous one.
        self._history[env_ids] = self._prev_target[env_ids].unsqueeze(1)
        self._resample(env_ids)


@configclass
class RateLimitedJointPositionActionCfg(JointPositionActionCfg):
    """Training form: a hard rate limit, no blend."""

    class_type: type = RateLimitedJointPositionAction

    slow: float = 1.0
    """Blend factor per policy step. 1.0 = no blend, a pure rate limit."""

    max_delta: float = 0.03
    """Per-step joint target change cap, radians.

    0.03 rad at 30 Hz is 0.9 rad/s. That is the one joint speed with hardware
    evidence behind it: the deployed clamp sits at exactly this value, the
    2026-08-20 log shows it saturated through the whole aggressive phase, and
    the arm tracked it with bounded error -- so the servos demonstrably deliver
    at least this much under load. It is a floor, not a measured ceiling; the
    chirp and step profiles in SOARMRL's scripts/sysid/ are what turn it into a
    real number.

    Keep it below the actuator's velocity_limit_sim of 1.5 rad/s so this term,
    not PhysX, is the binding constraint - otherwise the rate limit the policy
    learns is not the one the config states. The ceiling is rate-dependent:
    0.05 rad/step at 30 Hz, 0.15 at 10 Hz. This cap is a SPEED, so it must be
    rescaled whenever the control rate moves, or the arm silently gets faster or
    slower and a run that was meant to test one change tests two.
    """

    delay_steps: tuple[int, int] = (0, 0)
    """Per-episode transport delay, in POLICY STEPS, drawn uniformly per env.

    The measured hardware lag is 101-202 ms across the six joints (scripts/sysid/,
    2026-08-21). At the 10 Hz Run B trains at that is 1-2 steps; at 30 Hz it was
    3-6. State the range in steps rather than seconds so it stays honest when the
    control rate moves - a delay expressed in seconds silently changes meaning.

    (0, 0) is a no-op and keeps Run A's plant exactly as it was.
    """

    joint_offset: float = 0.0
    """Per-episode constant offset added to the joint TARGET, radians.

    This models the two largest measured discrepancies at once, because from the
    policy's side they are indistinguishable: gravity droop (up to 3.32 deg, the
    arm settling below where it was told) and servo deadband (0.6-3.6 deg, the
    arm not moving until the error exceeds it). Both make the achieved pose sit a
    fixed distance from the commanded one, in a direction that is constant for a
    given arm and pose.

    Applied to the TARGET, not to the observation: the encoder reports where the
    joint really is, so this is a plant error, not a sensor error. Modelling it as
    observation noise would teach the policy to distrust a reading that is in fact
    correct.

    0.0 is a no-op. Run B uses 0.03 rad = 1.7 deg, the middle of the measured band.
    """

    max_raw_action: float = 5.0
    """Hard bound on the raw policy output, before scaling.

    A saturating action space has no gradient opposing action growth, so SOMETHING
    has to bound it. This is the hard guarantee; the action_l2 reward term is the
    soft one that supplies an actual gradient toward small actions. Both are
    needed and they act at different points: this clamp bounds what reaches the
    PLANT, action_l2 penalises env.action_manager.action, which is the unclamped
    policy output, and the observation's own `clip` bounds what reaches the CRITIC.
    """


@configclass
class DeploymentJointPositionActionCfg(RateLimitedJointPositionActionCfg):
    """Evaluation form: the rate limit PLUS the bridge's slow-blend lag.

    Defaults are the values scripts/grasp/pickplace_live.py actually deploys.
    Keep them in sync with that file -- if they drift apart, this models a
    deployment path that does not exist, which is worse than not modelling one.
    """

    slow: float = 0.1
