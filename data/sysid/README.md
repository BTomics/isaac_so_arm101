# Hardware sysid sweeps — the calibration ground truth

Two 12082-tick (407 s) sweeps of the real SO-ARM101, recorded 2026-08-20 at 30 Hz
by `SOARMRL/scripts/sysid/sweep.py`. They live here because the VM needs them and
SOARMRL gitignores `logs/`, so this is also the only copy off the laptop.

| file | load |
|---|---|
| `sysid_unloaded_2026-08-20_18-44-37_nopolicy.npz` | free arm |
| `sysid_loaded_2026-08-20_18-54-31_nopolicy.npz` | holding the 25 g cube |

Both sweep all five arm joints, amplitude 0.3 rad, triangle periods 12/8/4/2 s.
`target_sent` is what went on the wire; `encoders` is what came back. The
difference between those two columns over time IS the actuator gap.

What they measured on this arm: lag **101-202 ms**, settling **300-390 ms**, droop
up to **3.32 deg**, deadband **0.6-3.6 deg**, and a clean negative on the load —
carrying the cube changed the bias by 0.16 deg, which a 25 g cube against a
published 1.5 kg figure predicts.

## Why they are in this repo

`scripts/sysid_replay.py` drives the sim with the `target_sent` column from one of
these and writes its result in the same format, so the plant can be calibrated
against the arm instead of against arithmetic:

    uv run python -m isaac_so_arm101.scripts.sysid_replay \
        --sweep data/sysid/sysid_unloaded_2026-08-20_18-44-37_nopolicy.npz \
        --armature 0.02 --out sim_replay_a0.02.npz --headless

Read the result back in SOARMRL, where the metric implementations live:

    python scripts/sysid/analyze.py \
        logs/trajectories/sysid_unloaded_2026-08-20_18-44-37_nopolicy.npz \
        --sim sim_replay_a0.02.npz

Keep the `--armature` whose lag and settling land inside the measured ranges.

The `.json` sidecars carry `hz`, the joint list and the profile parameters, and
`trajectory_log.load()` expects them next to the npz — copy both or the metadata
silently defaults.
