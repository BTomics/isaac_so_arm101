# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read a dump_rollout .npz and answer the three questions it was recorded for.

Runs WITHOUT Isaac Lab - numpy and the standard library only, so the numbers can
be read on the laptop from a file the VM produced.

    uv run python -m isaac_so_arm101.scripts.rsl_rl.analyze_rollout rollout.npz

WHAT IT REPORTS

  1. JOINT TRAVEL against the limits the SIM ENFORCED, per joint: the range used, and how
     much of the rollout sits inside --margin of a limit. wrist_flex is the one
     to look at: its home is 1.57 against a limit of 1.65806, so it has 0.088 rad
     of travel one way and 3.23 the other while every other joint sits mid-range.
     If the carry lives against that stop, the folded posture is kinematic and a
     reward weight will not move it.

  2. COMMANDED TARGETS outside the limits - what the policy ASKED for versus what
     the joint can do. A target the simulator clamps is a target the bridge will
     also clamp, and a policy steering through a clamp is steering a plant it
     cannot feel.

  3. GRASP GEOMETRY: cos_down and the jaw aperture, split by whether the cube is
     lifted and whether the end-effector is on it. cos_down here is the ANGLE
     ITSELF, not the grasp_top_down reward - that term multiplies the angle by a
     proximity kernel, so inverting the logged average gives a lower bound and
     cannot separate "off vertical" from "inverted".

Every threshold is a flag, and the defaults are the task's own values.
"""

import argparse
import pathlib
import sys
import xml.etree.ElementTree as ET

import numpy as np

URDF_DEFAULT = (pathlib.Path(__file__).resolve().parents[2]
                / "robots" / "trs_so101" / "urdf" / "so_arm101.urdf")


def urdf_limits(path: pathlib.Path) -> dict[str, tuple[float, float]]:
    """{joint name: (lower, upper)} for every revolute joint with a limit."""
    out = {}
    root = ET.parse(path).getroot()
    for joint in root.iter("joint"):
        limit = joint.find("limit")
        name = joint.get("name")
        if limit is None or name is None:
            continue
        lower, upper = limit.get("lower"), limit.get("upper")
        if lower is None or upper is None:
            continue
        out[name] = (float(lower), float(upper))
    return out


def percentiles(x: np.ndarray, qs=(1, 50, 99)) -> list[float]:
    return [float(np.percentile(x, q)) for q in qs] if x.size else [float("nan")] * len(qs)


def main() -> int:
    p = argparse.ArgumentParser(description="Read a dump_rollout .npz.")
    p.add_argument("npz", type=str)
    p.add_argument("--urdf", type=str, default=str(URDF_DEFAULT))
    p.add_argument("--margin", type=float, default=0.02,
                   help="rad from a limit that counts as 'at the stop'. 0.02 is ~1.1 deg.")
    p.add_argument("--lift_height", type=float, default=0.03,
                   help="cube world z above which it counts as lifted; grasp_top_down's min_height.")
    p.add_argument("--grasp_dist", type=float, default=0.05,
                   help="EE-to-cube distance under which the EE counts as ON the cube.")
    p.add_argument("--gripper_open", type=float, default=0.25,
                   help="gripper joint pos above which the jaw is open; place_complete's threshold.")
    args = p.parse_args()

    data = np.load(args.npz, allow_pickle=False)
    names = [str(n) for n in data["joint_names"]]
    q = data["joint_pos"]            # (steps, envs, joints)
    target = data["target"]
    cos_down = data["cos_down"]      # (steps, envs)
    cube_z = data["cube_pos"][..., 2]
    grasp_dist = data["grasp_dist"]
    placed = data["placed"]
    done = data["done"]

    steps, envs = cos_down.shape
    print(f"task        {data['task']}")
    print(f"checkpoint  {data['checkpoint']}")
    print(f"rollout     {steps} steps x {envs} envs, {int(done.sum())} episode boundaries\n")

    # Prefer the limits the simulator ENFORCED, recorded at rollout time. The URDF
    # only declares: the URDF->USD conversion can change a limit, and PhysX solves
    # limits as constraints, so a joint under load can be dragged through one.
    # Measuring travel against a limit that was not the operative one is how a
    # broken plant reads as a policy pathology.
    if "sim_limits" in data.files:
        source = "sim (enforced)"
        limits = {n: (float(lo), float(hi)) for n, (lo, hi) in zip(names, data["sim_limits"])}
        declared = urdf_limits(pathlib.Path(args.urdf))
        for n in names:
            u, e = declared.get(n), limits.get(n)
            if u and e and (abs(u[0] - e[0]) > 1e-3 or abs(u[1] - e[1]) > 1e-3):
                print(f"!! {n}: URDF declares {u[0]:+.4f}..{u[1]:+.4f} but the sim "
                      f"enforces {e[0]:+.4f}..{e[1]:+.4f} - the conversion changed it")
    else:
        source = "URDF (declared; this dump predates sim_limits)"
        limits = urdf_limits(pathlib.Path(args.urdf))
    missing = [n for n in names if n not in limits]
    if missing:
        print(f"!! no limit for {missing} - reported as nan rather than assumed\n")

    print(f"JOINT TRAVEL vs limit, source: {source}")
    print(f"{'joint':<16}{'limit':>18}{'used (p1..p99)':>22}{'at lower':>10}{'at upper':>10}")
    for j, name in enumerate(names):
        col = q[:, :, j].ravel()
        lo, hi = limits.get(name, (float("nan"), float("nan")))
        p1, _, p99 = percentiles(col)
        at_lo = float(np.mean(col <= lo + args.margin)) if np.isfinite(lo) else float("nan")
        at_hi = float(np.mean(col >= hi - args.margin)) if np.isfinite(hi) else float("nan")
        # Outside the limit is not travel, it is the constraint failing. Reported
        # separately because "at the stop" and "through the stop" have different
        # causes and different fixes.
        beyond = float(np.mean((col < lo) | (col > hi))) if np.isfinite(lo) else float("nan")
        flag = "  <<<" if (np.isfinite(at_hi) and max(at_lo, at_hi) > 0.10) else ""
        if np.isfinite(beyond) and beyond > 0.01:
            worst = max(float(lo - col.min()), float(col.max() - hi))
            flag = f"  VIOLATED {beyond:.1%} by up to {worst:.3f}"
        print(f"{name:<16}{lo:>8.3f}..{hi:<8.3f}{p1:>10.3f}..{p99:<10.3f}"
              f"{at_lo:>9.1%}{at_hi:>10.1%}{flag}")

    if np.isnan(target).all():
        print("\nCOMMANDED TARGETS   unavailable in this dump - see the dump_rollout warning")
    else:
        print("\nCOMMANDED TARGETS outside the limit (what the clamp swallowed)")
        print(f"{'joint':<16}{'below lower':>14}{'above upper':>14}{'max overshoot':>16}")
        for j, name in enumerate(names):
            col = target[:, :, j].ravel()
            col = col[np.isfinite(col)]
            lo, hi = limits.get(name, (float("nan"), float("nan")))
            if not col.size or not np.isfinite(lo):
                continue
            below, above = float(np.mean(col < lo)), float(np.mean(col > hi))
            over = max(0.0, lo - float(col.min()), float(col.max()) - hi)
            flag = "  <<<" if max(below, above) > 0.10 else ""
            print(f"{name:<16}{below:>13.1%}{above:>14.1%}{over:>15.3f}{flag}")

    lifted = cube_z > args.lift_height
    on_cube = grasp_dist < args.grasp_dist
    gripper = q[:, :, names.index("gripper")] if "gripper" in names else np.full_like(cos_down, np.nan)
    closed = gripper <= args.gripper_open
    carrying = lifted & on_cube

    print("\nGRASP GEOMETRY")
    print(f"{'window':<28}{'steps':>10}{'cos_down p5':>14}{'p50':>10}{'p95':>10}{'jaw closed':>13}")
    for label, mask in (("all steps", np.ones_like(lifted)),
                        ("cube lifted", lifted),
                        ("lifted AND on the cube", carrying)):
        n = int(mask.sum())
        if not n:
            print(f"{label:<28}{n:>10}{'  - never occurred':>48}")
            continue
        c5, c50, c95 = percentiles(cos_down[mask], (5, 50, 95))
        jaw = float(np.mean(closed[mask]))
        print(f"{label:<28}{n:>10}{c5:>14.3f}{c50:>10.3f}{c95:>10.3f}{jaw:>12.1%}")

    print(f"\n  cos_down +1 = gripper straight down, 0 = horizontal, -1 = inverted."
          f"\n  'jaw closed' = gripper joint <= {args.gripper_open} rad, place_complete's open threshold."
          f"\n  placed on {float(np.mean(placed)):.1%} of steps ({int(placed.sum())} of {placed.size}).")

    if carrying.sum():
        med = float(np.median(cos_down[carrying]))
        if med < 0.3:
            print(f"\n  VERDICT: median cos_down {med:+.2f} while carrying - "
                  f"{np.degrees(np.arccos(np.clip(med, -1, 1))):.0f} deg off vertical. "
                  f"Not a top-down grasp.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
