# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Does a joint limit actually hold? Drive each joint into its stop and measure.

WHY THIS EXISTS: a rollout of the A' policy put wrist_flex 1.073 rad (61 deg) past
its hard limit on 23.9% of steps, while 0.0% of commanded targets were below that
limit. The drive was not asking for it, so a contact was forcing the joint through
its stop and PhysX was not holding the constraint. A policy trained against that
plant is exploiting travel the real arm does not have.

This is the unit test for the fix. It commands each joint PAST its limit and
reports how far the joint actually gets, with no policy involved - so an armature,
solver-iteration or depenetration change can be judged in a minute instead of
inferred from a nine-hour run.

    uv run python -m isaac_so_arm101.scripts.limit_probe \
        --task Isaac-SO-ARM101-PickPlace-APrime-v0 --headless
    uv run python -m isaac_so_arm101.scripts.limit_probe \
        --task Isaac-SO-ARM101-PickPlace-APrime-v0 --hold-cube --headless

WHAT IT CANNOT DO, and the reason to read a PASS carefully: it reproduces the
DRIVE-versus-limit case and, with --hold-cube, a static load. The violation seen
in the rollout happened in a folded carry pose where self-collision between
capsule approximations is a candidate cause, and this probe does not fold the arm.
A clean result here means "the limit holds under a drive commanding through it",
never "the limit cannot be violated". The rollout is still the authority - re-run
dump_rollout on the same checkpoint and read the VIOLATED column.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Drive each joint into its stop and measure the overshoot.")
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-PickPlace-APrime-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--overshoot", type=float, default=0.4,
                    help="rad past the limit to command. Larger than any clamp the action term applies.")
parser.add_argument("--hold_steps", type=int, default=180,
                    help="sim steps to hold each commanded target. 180 at 100 Hz is 1.8 s, well past settling.")
parser.add_argument("--tol", type=float, default=0.01,
                    help="rad of overshoot tolerated before a joint is reported FAIL.")
parser.add_argument("--hold-cube", dest="hold_cube", action="store_true",
                    help="teleport the cube into the jaw and close the gripper first, to probe under load.")
parser.add_argument("--joints", type=str, default="all",
                    help="comma-separated joint names, or 'all'.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def probe(env, robot, scene, sim, j: int, name: str, lo: float, hi: float, args) -> list[tuple]:
    """Command past each stop in turn; return (direction, commanded, worst position)."""
    dt = sim.get_physics_dt()
    default_q = robot.data.default_joint_pos.clone()
    out = []

    for direction, limit in (("lower", lo), ("upper", hi)):
        sign = -1.0 if direction == "lower" else 1.0
        commanded = limit + sign * args.overshoot

        robot.write_joint_state_to_sim(default_q, torch.zeros_like(default_q))
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)

        target = default_q.clone()
        target[:, j] = commanded
        worst = default_q[0, j].item()

        for _ in range(args.hold_steps):
            robot.set_joint_position_target(target)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
            pos = robot.data.joint_pos[:, j]
            reached = pos.min().item() if sign < 0 else pos.max().item()
            worst = min(worst, reached) if sign < 0 else max(worst, reached)

        out.append((direction, commanded, worst, (limit - worst) if sign < 0 else (worst - limit)))

    return out


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    unwrapped = env.unwrapped
    scene, sim = unwrapped.scene, unwrapped.sim
    robot = scene["robot"]

    limits = robot.data.joint_pos_limits[0].cpu()
    names = list(robot.joint_names)
    wanted = names if args_cli.joints == "all" else [n.strip() for n in args_cli.joints.split(",")]
    unknown = [n for n in wanted if n not in names]
    if unknown:
        raise SystemExit(f"no such joint: {unknown}. Available: {names}")

    if args_cli.hold_cube:
        # A static load case, not the folded carry pose. Teleporting the cube into
        # the jaw is the closest thing to "holding" available without a policy.
        obj = scene["object"]
        ee = scene["ee_frame"].data.target_pos_w[..., 0, :]
        root = obj.data.root_state_w.clone()
        root[:, :3] = ee
        root[:, 7:] = 0.0
        obj.write_root_state_to_sim(root)
        gripper_j = names.index("gripper")
        closed = robot.data.default_joint_pos.clone()
        closed[:, gripper_j] = float(limits[gripper_j, 0])
        for _ in range(60):
            robot.set_joint_position_target(closed)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
        print("[INFO]: cube teleported into the jaw and the gripper closed on it.\n")

    print(f"task        {args_cli.task}")
    print(f"commanding  {args_cli.overshoot} rad past each stop, held {args_cli.hold_steps} sim steps")
    print(f"load        {'cube held in the jaw' if args_cli.hold_cube else 'unloaded'}\n")
    print(f"{'joint':<16}{'dir':>7}{'limit':>10}{'commanded':>12}{'reached':>10}{'overshoot':>12}   verdict")

    failures = 0
    for name in wanted:
        j = names.index(name)
        lo, hi = float(limits[j, 0]), float(limits[j, 1])
        for direction, commanded, reached, over in probe(env, robot, scene, sim, j, name, lo, hi, args_cli):
            verdict = "held" if over <= args_cli.tol else "FAIL"
            failures += verdict == "FAIL"
            limit = lo if direction == "lower" else hi
            print(f"{name:<16}{direction:>7}{limit:>10.3f}{commanded:>12.3f}"
                  f"{reached:>10.3f}{over:>12.4f}   {verdict}")

    print(f"\n{failures} of {2 * len(wanted)} stops let the joint through by more than {args_cli.tol} rad.")
    if not failures:
        print("A clean sweep here means the limit holds under a drive commanding through it.\n"
              "It does NOT mean the limit cannot be violated - see this script's docstring.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
