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


@configclass
class PickPlaceDRPPORunnerCfg(PickPlaceAlignedPPORunnerCfg):
    """Run B's agent. Identical to Run A's except for the experiment name.

    That is deliberate. Run B changes the PLANT - control rate, delay, offsets,
    randomized gains and cube. Changing the algorithm in the same run would make
    the result unattributable, and this project has paid for that mistake before
    (SLOW and ACTION_SCALE shipped together, ~4x aggressiveness, one run wasted).

    gamma stays 0.99 and it means MORE at 10 Hz, not less: 1/(1-gamma) = 100
    steps is 10 s of lookahead against an 8 s episode, so for the first time the
    value function can see the whole task from the first frame. At Run A's 30 Hz
    the same gamma bought 3.3 s of an 8 s task.

    num_steps_per_env stays 24, which is now 2.4 s of episode rather than 0.8 s.
    The batch size in samples is unchanged; the batch size in TASK TIME tripled.

    WALL CLOCK: decimation 9 against Run A's 3 means three times the physics per
    iteration, so expect roughly 4 s/iteration against Run A's 1.3 - about 13 h
    for the full 12000 rather than 4.2. Per unit of simulated task time it is the
    same cost; it is the iteration counter that changed meaning, not the price.
    Do not compare Run A and Run B curves on the iteration axis.
    """

    experiment_name = "pickplace_dr"
    max_iterations = 12000


@configclass
class PickPlaceDRCPPORunnerCfg(PickPlaceDRPPORunnerCfg):
    """Run C's agent. Only the experiment name differs from Run B's.

    The change under test is ACTION_L2, which lives in the ENV cfg. Touching the
    algorithm here as well would make the result unattributable - the mistake
    increment 1d made and paid a whole run for.
    """

    experiment_name = "pickplace_dr_c"


@configclass
class PickPlaceDRDPPORunnerCfg(PickPlaceDRCPPORunnerCfg):
    """Run D's agent. Experiment name only; the change is in the env cfg."""

    experiment_name = "pickplace_dr_d"


@configclass
class PickPlaceAPrimePPORunnerCfg(PickPlaceDRDPPORunnerCfg):
    """Run A's agent, restated for the re-baseline. Nothing in the algorithm moves.

    gamma 0.99, num_steps_per_env 24, batch and epochs exactly as Runs A-D had
    them, so the result is attributable to the plant and the two reward fixes the
    env cfg carries. The chain from PickPlaceAlignedPPORunnerCfg down to here has
    only ever changed experiment_name and max_iterations, and this run keeps it
    that way.

    max_iterations 24000 because Run A never spent its budget: it was configured
    for 24000 and its last checkpoint is model_11999. The metric was still
    climbing there, so the second half is untested rather than known-useless.

    WALL CLOCK: back at decimation 3, so roughly 1.3 s/iteration as Run A ran -
    about 4.5 h to the 12000 gate and 9 h to the end, against Run B/C/D's ~13 h
    for 12000 at decimation 9. Do not compare any of these curves to the 10 Hz
    runs on the iteration axis; the iteration means a third of the task time.
    """

    experiment_name = "pickplace_aprime"
    max_iterations = 24000


@configclass
class PickPlaceAPrime2PPORunnerCfg(PickPlaceAPrimePPORunnerCfg):
    """A' retrained on the plant that enforces its own joint limits.

    WHY A NEW NAME FOR AN IDENTICAL CONFIG. `pickplace_aprime` now means two
    different plants: the one where wrist_flex spent 23.9% of steps up to 1.073
    rad past its hard stop, and the one after armature, 32 solver position
    iterations and a 1.0 m/s depenetration cap made the constraint hold. Same
    task id, same log directory, incomparable results. Every expensive mistake on
    this project has been two things sharing one name, so they get two.

    The gap is not cosmetic: replayed on the fixed plant, A's own checkpoint drops
    from 79.9% to ZERO placements, with 12 lifted steps in 15360. The exploit was
    not a contributor to that score, it was the policy.

    max_iterations 12000, not 24000. A' scored 79.7% at 12000 and 79.9% at 24000 -
    the second 4.5 hours bought 0.2 points on the rate, while the place_success
    TIME-FRACTION rose 0.415 -> 0.445. It learned to place sooner, not more often.
    Resume past 12000 only if the eval is still moving; the -Resume-v0 twin is
    there for that and makes continuing a deliberate act rather than a default.

    Nothing else moves. Same env cfg class, same rewards, same gamma, same batch.
    The plant fix IS the change under test.
    """

    experiment_name = "pickplace_aprime2"
    max_iterations = 12000
