# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dump a policy rollout to .npz: joint angles, commanded targets, grasp geometry.

WHY THIS EXISTS: two questions about A' could only be answered by inference from
weighted reward averages, and both answers mattered enough to measure.

  1. IS THE WRIST PINNED? wrist_flex's home is 1.57 against a URDF limit of
     1.65806 - 0.088 rad of travel one way and 3.23 the other, while every other
     joint sits mid-range. If the policy spends its carry against that limit, the
     folded posture is a kinematic consequence and no reward weight will fix it.

  2. WHAT IS cos_down, REALLY? Episode_Reward/grasp_top_down is the product of a
     proximity kernel and an orientation kernel, so inverting it gives a LOWER
     BOUND on cos_down and nothing more - it cannot tell "40 degrees off vertical"
     from "fully inverted". This records the angle itself.

It records the STATE BEFORE EACH STEP, the same convention eval_success.py uses:
after env.step the done envs have already been reset, so a post-step read would
attribute a fresh episode's first frame to the finished one.

    uv run python -m isaac_so_arm101.scripts.rsl_rl.dump_rollout \
        --task Isaac-SO-ARM101-PickPlace-APrime-v0 \
        --checkpoint logs/rsl_rl/pickplace_aprime/<run>/model_23999.pt \
        --num_envs 32 --steps 480 --headless

Then read it with analyze_rollout.py, which needs no Isaac Lab.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Dump a rollout's joint and grasp geometry to .npz.")
parser.add_argument("--num_envs", type=int, default=32, help="Environments to run in parallel.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--steps", type=int, default=480,
                    help="Policy steps to record. 480 is two 8 s episodes at 30 Hz.")
parser.add_argument("--out", type=str, default=None,
                    help="Output .npz path. Defaults to the checkpoint's directory.")
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
import numpy as np
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401
import isaac_so_arm101.tasks.pickplace.mdp as pickplace_mdp
# The reward's own approach axis, not a copy of it. Restating the constant here
# is how this script and grasp_top_down would drift apart, and then the angle
# reported would not be the angle the policy was paid for.
from isaac_so_arm101.tasks.pickplace.mdp.rewards import _GRIPPER_APPROACH_LOCAL
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def commanded_targets(env, fallback: torch.Tensor) -> torch.Tensor:
    """The joint target the action term actually sent, after scale/offset/clamp.

    RateLimitedJointPositionAction does its arithmetic in process_actions, so the
    post-clamp target lives on the term. Isaac Lab has renamed this attribute
    before, and a silently wrong target column would make every clipping number
    in the analysis a fiction - so this returns NaN and says so rather than
    guessing.
    """
    term = env.unwrapped.action_manager.get_term("arm_action")
    for name in ("processed_actions", "_processed_actions", "_clamped_targets"):
        value = getattr(term, name, None)
        if value is not None:
            return value.clone()
    return torch.full_like(fallback, float("nan"))


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

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    ee_frame = unwrapped.scene["ee_frame"]
    obj = unwrapped.scene["object"]
    joint_idx = [robot.joint_names.index(n) for n in JOINTS]

    approach_local = torch.tensor(_GRIPPER_APPROACH_LOCAL, device=unwrapped.device, dtype=torch.float32)
    approach_local = approach_local / torch.norm(approach_local)

    # The success predicate's own params, for the same reason the approach axis is
    # imported rather than restated.
    success_params = dict(unwrapped.cfg.rewards.place_success.params)

    rec = {k: [] for k in ("joint_pos", "target", "cos_down", "ee_pos", "cube_pos",
                           "grasp_dist", "placed", "done")}

    obs, _ = env.get_observations()
    for step in range(args_cli.steps):
        if not simulation_app.is_running():
            break

        joint_pos = robot.data.joint_pos[:, joint_idx]
        quat = ee_frame.data.target_quat_w[..., 0, :]
        approach_world = quat_apply(quat, approach_local.expand(quat.shape[0], 3))
        cos_down = -approach_world[:, 2]  # +1 = straight down, the reward's convention
        ee_pos = ee_frame.data.target_pos_w[..., 0, :]
        cube_pos = obj.data.root_pos_w[:, :3]

        rec["joint_pos"].append(joint_pos.cpu().numpy())
        rec["target"].append(commanded_targets(unwrapped, joint_pos).cpu().numpy())
        rec["cos_down"].append(cos_down.cpu().numpy())
        rec["ee_pos"].append(ee_pos.cpu().numpy())
        rec["cube_pos"].append(cube_pos.cpu().numpy())
        rec["grasp_dist"].append(torch.norm(ee_pos - cube_pos, dim=1).cpu().numpy())
        rec["placed"].append(
            (pickplace_mdp.place_success_bonus(unwrapped, **success_params) > 0.5).cpu().numpy()
        )

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        rec["done"].append(dones.cpu().numpy().astype(bool))

        if (step + 1) % 60 == 0:
            print(f"  {step + 1}/{args_cli.steps} steps", flush=True)

    out = args_cli.out or os.path.join(os.path.dirname(resume_path), "rollout.npz")
    arrays = {k: np.asarray(v) for k, v in rec.items()}
    np.savez_compressed(
        out,
        joint_names=np.array(JOINTS),
        task=np.array(args_cli.task),
        checkpoint=np.array(resume_path),
        **arrays,
    )
    n_nan = int(np.isnan(arrays["target"]).all(axis=(0, 2)).sum()) if arrays["target"].ndim == 3 else 0
    if n_nan:
        print("[WARN] commanded targets unavailable on the action term - the clipping "
              "columns in analyze_rollout will be empty. Fix commanded_targets() "
              "rather than reading zeros as evidence.")
    print(f"\nwrote {out}   {arrays['joint_pos'].shape[0]} steps x {arrays['joint_pos'].shape[1]} envs")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
