# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-episode place success rate. The number `play` does not give you.

WHY THIS EXISTS: `Episode_Reward/place_success` in the training log is the
FRACTION OF EPISODE TIME the cube spends fully placed, not the fraction of
episodes that succeed. It conflates "how often does it work" with "how fast does
it get there", and it moves when episode_length_s changes even if behaviour does
not -- which makes the 5 s runs and the 8 s runs incomparable.

This counts episodes: an episode succeeds if place_complete was EVER true during
it. That is a rate, it is comparable across every config, and it is what belongs
in a writeup.

    uv run python -m isaac_so_arm101.scripts.rsl_rl.eval_success \
        --task Isaac-SO-ARM101-PickPlace-Aligned-v0 \
        --checkpoint logs/rsl_rl/pickplace_aligned/<run>/model_11999.pt \
        --num_envs 256 --episodes 512 --headless

The success criterion is not redefined here. It calls the task's own
place_success reward term with the task's own params, so this measures exactly
what the policy was trained against -- 20 mm horizontal, 10 mm vertical, at rest,
gripper open, end-effector withdrawn 50 mm, and the lift latch set.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Measure per-episode place success rate.")
parser.add_argument("--num_envs", type=int, default=256, help="Environments to run in parallel.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--episodes", type=int, default=512,
                    help="Episodes to score before reporting. More is tighter; "
                         "512 gives about +/-4 percentage points at 95%%.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401
import isaac_so_arm101.tasks.pickplace.mdp as pickplace_mdp
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def wilson_interval(successes: int, n: int, z: float = 1.96):
    """95% confidence interval on a proportion.

    Wilson rather than the textbook normal interval: with a few hundred episodes
    and a rate that may sit near 0 or 1, the normal interval runs off the end of
    [0, 1] and reports impossible bounds.
    """
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Use the TASK's own success term and params. Restating the criterion here
    # would let this script and the reward drift apart, and then the number would
    # be measuring something the policy was never trained to do.
    term = env.unwrapped.cfg.rewards.place_success
    params = dict(term.params)
    print(f"[INFO]: scoring with {term.func.__name__}({params})")

    n_envs = env.unwrapped.num_envs
    ever = torch.zeros(n_envs, dtype=torch.bool, device=env.unwrapped.device)
    successes = 0
    episodes = 0

    obs, _ = env.get_observations()
    while episodes < args_cli.episodes and simulation_app.is_running():
        # Read the CURRENT state before stepping. After a step the done envs have
        # already been reset, so a post-step read would score the fresh episode.
        ever |= pickplace_mdp.place_success_bonus(env.unwrapped, **params) > 0.5

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        done_idx = dones.nonzero(as_tuple=False).flatten()
        if len(done_idx):
            successes += int(ever[done_idx].sum().item())
            episodes += len(done_idx)
            ever[done_idx] = False
            rate = successes / episodes
            lo, hi = wilson_interval(successes, episodes)
            print(f"  {episodes:5d} episodes   success {rate:6.1%}   "
                  f"95% CI [{lo:.1%}, {hi:.1%}]", flush=True)

    lo, hi = wilson_interval(successes, episodes)
    print("\n" + "=" * 60)
    print(f"task            {args_cli.task}")
    print(f"checkpoint      {resume_path}")
    print(f"episodes        {episodes}")
    print(f"SUCCESS RATE    {successes / episodes:.1%}  (95% CI {lo:.1%} - {hi:.1%})")
    print("=" * 60)
    print("This is per-EPISODE success, comparable across episode lengths and\n"
          "control rates. Episode_Reward/place_success in the training log is not.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
