# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PickPlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "pickplace"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PickPlaceAlignedPPORunnerCfg(PickPlacePPORunnerCfg):
    """Run A's agent. The only change from the above is gamma.

    gamma 0.98 gives an effective horizon of 1/(1-gamma) = 50 steps. At the 50 Hz
    this task used to train at, that is ONE SECOND of lookahead on a task that
    takes five seconds to complete: pick, carry, set down, release, withdraw. The
    value function could barely see the release from the grasp.

    0.99 is 100 steps, and at Run A's 30 Hz that is 3.3 s against an 8 s episode.
    Still not the whole task, but the same order as it.

    The algorithm block is restated in full rather than mutated in __post_init__,
    so that reading this file tells you what the run used without having to
    resolve an inheritance chain -- the same reason read_terms.py exists.

    Everything else is deliberately unchanged. num_steps_per_env stays at 24 so
    the batch size is comparable to every previous run; gamma is the horizon
    lever and shipping two at once makes the result unattributable.
    """

    experiment_name = "pickplace_aligned"
    max_iterations = 24000

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
