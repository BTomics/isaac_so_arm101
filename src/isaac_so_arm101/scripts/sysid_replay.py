# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a hardware sysid sweep in sim, so the plant can be calibrated against it.

WHY THIS EXISTS: the fix for wrist_flex leaving its own limit added `armature` to
the actuators, and that number is an ESTIMATE - I_rotor * N^2 at the STS3215's
~1:345 brackets it at 0.012..0.12, a factor of ten. Training 4.5 hours against a
guessed plant is how this project spent its last four days.

The ground truth already exists. scripts/sysid/ measured the real arm on
2026-08-21: lag 101-202 ms, settling 300-390 ms, droop up to 3.32 deg. This script
replays the SAME commands in sim and writes the result in the same format, so
SOARMRL's own analyzer reads both with one metric implementation:

    python scripts/sysid/analyze.py logs/trajectories/sysid_unloaded_*.npz \
        --sim sim_replay_a0.02.npz

`analyze.py --sim` and `sysid.compare_runs` were written for exactly this file and
have been waiting for it. compare_runs REFUSES to compare when the two command
sequences differ, which is why this replays `target_sent` from the hardware log
verbatim rather than regenerating the profiles: identical commands by
construction, not by agreement between two copies of a profile generator.

CALIBRATING THE ARMATURE:

    for a in 0.005 0.01 0.02 0.05 0.1; do
      uv run python -m isaac_so_arm101.scripts.sysid_replay \
          --sweep sysid_unloaded_2026-08-21.npz --armature $a \
          --out sim_replay_a$a.npz --headless
    done

then read `lag sim ms` and the settle figures against the real column and keep the
value that matches. A plant tuned to the measured arm makes every number after it
mean something about hardware; one tuned to arithmetic does not.

WHAT THIS DOES NOT DO: no policy, no task rewards, no cube. It drives joint
targets and records what came back, which is what the sweep did on hardware.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a hardware sysid sweep in sim.")
parser.add_argument("--sweep", type=str, required=True,
                    help="hardware sweep .npz written by SOARMRL scripts/sysid/sweep.py")
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-PickPlace-APrime-v0")
parser.add_argument("--out", type=str, default="sim_replay.npz")
parser.add_argument("--armature", type=float, default=None,
                    help="override the actuator armature for this replay; the point of the script")
parser.add_argument("--hz", type=float, default=None,
                    help="control rate of the replay. Default: the sweep's own meta['hz'].")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import numpy as np
import pathlib
import torch

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# SOARMRL's conversion.ARM_JOINTS, in its order. Restated here because that module
# lives in the other repo and is not importable on the VM - so the width check
# below is load-bearing: it is what catches the restatement going stale.
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def load_sweep(path: pathlib.Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        run = {k: z[k] for k in z.files}
    sidecar = path.with_suffix(".json")
    run["meta"] = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    return run


def column_names(width: int) -> list[str]:
    """Which joint each target_sent column is, or a loud failure."""
    if width == len(ARM_JOINTS):
        return list(ARM_JOINTS)
    if width == len(ARM_JOINTS) + 1:
        return ARM_JOINTS + ["gripper"]
    raise SystemExit(
        f"target_sent has {width} columns; expected {len(ARM_JOINTS)} (arm) or "
        f"{len(ARM_JOINTS) + 1} (arm + gripper). If the sweep's layout changed, "
        f"ARM_JOINTS here is stale - fix it rather than reindexing at the call site."
    )


def main():
    sweep_path = pathlib.Path(args_cli.sweep)
    sweep = load_sweep(sweep_path)
    cmd = np.asarray(sweep["target_sent"], dtype=np.float64)
    if cmd.ndim != 2 or not cmd.size:
        raise SystemExit(f"{sweep_path} has no target_sent column to replay")
    names = column_names(cmd.shape[1])
    hz = float(args_cli.hz or sweep["meta"].get("hz", 30.0))
    n_ticks = cmd.shape[0]

    env_cfg = parse_env_cfg(args_cli.task, num_envs=1)
    if args_cli.armature is not None:
        for actuator in env_cfg.scene.robot.actuators.values():
            actuator.armature = args_cli.armature

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    unwrapped = env.unwrapped
    scene, sim = unwrapped.scene, unwrapped.sim
    robot = scene["robot"]

    dt = sim.get_physics_dt()
    substeps = max(1, int(round(1.0 / (hz * dt))))
    realised_hz = 1.0 / (substeps * dt)
    if abs(realised_hz - hz) > 0.01 * hz:
        print(f"[WARN] the sweep ran at {hz:.2f} Hz; the closest this sim dt ({dt:.5f} s) "
              f"reaches is {realised_hz:.2f} Hz. Every timing number inherits that error.")

    idx = [robot.joint_names.index(n) for n in names]
    print(f"[INFO]: replaying {n_ticks} ticks at {realised_hz:.2f} Hz "
          f"({substeps} physics steps per tick)")
    print(f"[INFO]: columns -> {names}")
    print(f"[INFO]: armature {'overridden to ' + str(args_cli.armature) if args_cli.armature is not None else 'as configured'}")

    # Start from the sweep's first commanded pose rather than the task's home, or
    # the first tick is a step from somewhere the hardware never was.
    q = robot.data.default_joint_pos.clone()
    q[:, idx] = torch.tensor(cmd[0], device=unwrapped.device, dtype=q.dtype)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt)

    encoders = np.zeros_like(cmd)
    target = robot.data.default_joint_pos.clone()
    for t in range(n_ticks):
        target[:, idx] = torch.tensor(cmd[t], device=unwrapped.device, dtype=target.dtype)
        for _ in range(substeps):
            robot.set_joint_position_target(target)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
        encoders[t] = robot.data.joint_pos[0, idx].cpu().numpy()
        if (t + 1) % 2000 == 0:
            print(f"  {t + 1}/{n_ticks} ticks", flush=True)

    out = pathlib.Path(args_cli.out)
    n = n_ticks
    np.savez_compressed(
        out,
        t_wall=np.arange(n, dtype=np.float64) / realised_hz,
        phase=np.asarray(sweep["phase"][:n], dtype=np.str_),
        obs=np.zeros((n, 0), dtype=np.float64),
        action_raw=np.zeros((n, 0), dtype=np.float64),
        target_sent=cmd,
        encoders=encoders,
    )
    meta = dict(sweep["meta"])
    meta.update({
        "label": "sim_replay",
        "source_sweep": str(sweep_path),
        "sim_task": args_cli.task,
        "armature": args_cli.armature,
        "hz": realised_hz,
        "policy_path": None,
        "policy_sha": "nopolicy",
    })
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nwrote {out} and its sidecar   {n} ticks x {len(names)} joints")
    print(f"read it with:  python scripts/sysid/analyze.py {sweep_path.name} --sim {out.name}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
