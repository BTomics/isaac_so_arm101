# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-SO-ARM100-PickPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100PickPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlacePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM100-PickPlace-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100PickPlaceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlacePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-PickPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlacePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-PickPlace-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlacePPORunnerCfg",
    },
    disable_env_checker=True,
)

# Evaluation only, for `play`. Perturbs object_position to stand in for real
# perception error - the policy trains on simulator ground truth and has never
# seen that input perturbed. Run all three and find where place_success falls
# off; that is the accuracy spec for the camera. Never train on these.
#
# Noise* resamples EVERY STEP, so the policy low-pass-filters it over a 250-step
# episode and shrugs off even 10 mm. Bias* draws ONE offset per episode and holds
# it, which is how a miscalibrated camera actually fails - nothing to average
# out. Bias is the real test; where it breaks is the calibration budget.
for _name, _cls in (("Noise2mm", "NOISE2"), ("Noise5mm", "NOISE5"), ("Noise10mm", "NOISE10"),
                    ("Bias2mm", "BIAS2"), ("Bias5mm", "BIAS5"), ("Bias10mm", "BIAS10")):
    gym.register(
        id=f"Isaac-SO-ARM101-PickPlace-{_name}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceEnvCfg_{_cls}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlacePPORunnerCfg",
        },
        disable_env_checker=True,
    )

# Evaluation only, for `play`. Perturbs what the actions DO rather than what the
# policy sees: the 30 Hz control rate the bridge runs, its slow-blend and delta
# clamp, and the zeroed velocity block. Together these are the deployed actuation
# path, which the policy has never trained against.
#
# Run Deployed first and the three singles to attribute whatever it shows. The
# signature to match, from the 2026-08-20 hardware run with an in-distribution
# goal: the arm does not track its own commands while carrying (elbow demand ran
# 1.4 rad ahead of what was sent, and reversed before the arm arrived), the cube
# oscillates instead of approaching, the release fires early on a noisy gripper
# channel, and the arm then freezes with every joint static. Reproducing that in
# sim turns it into a regression test. Never train on these.
for _name, _cls in (("Deploy30", "DEPLOY30"), ("Blend", "BLEND"),
                    ("ZeroVel", "ZEROVEL"), ("Deployed", "DEPLOYED")):
    gym.register(
        id=f"Isaac-SO-ARM101-PickPlace-{_name}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceEnvCfg_{_cls}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlacePPORunnerCfg",
        },
        disable_env_checker=True,
    )

# Run A - the aligned baseline. THIS is the one to train. Sim's actuation path is
# made the same path the bridge deploys: 30 Hz, a hard per-step rate limit
# instead of a lag, no smoothness penalties (the rate limit subsumes them), the
# velocity observation zeroed as the bridge sends it, an 8 s episode, and
# gamma 0.99 so the discount horizon is the same order as the task length.
#
# The -Resume- variant pins lifting_object and clears the curriculum, same
# contract as the original Resume task. It has far less to pin because Run A
# deleted the penalty curriculum, which is precisely why resuming is now cheap.
gym.register(
    id="Isaac-SO-ARM101-PickPlace-Aligned-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceEnvCfg_ALIGNED",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlaceAlignedPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-PickPlace-Aligned-Resume-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceEnvCfg_ALIGNED_RESUME",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlaceAlignedPPORunnerCfg",
    },
    disable_env_checker=True,
)

# For --resume ONLY. A fresh run on this task never bootstraps the pick, because
# lifting_object starts at its decayed weight of 3 instead of 15. See
# SoArm101PickPlaceEnvCfg_RESUME for why resuming the normal task is unsafe.
gym.register(
    id="Isaac-SO-ARM101-PickPlace-Resume-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceEnvCfg_RESUME",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlacePPORunnerCfg",
    },
    disable_env_checker=True,
)


# Run B: Run A's aligned plant, randomized around the sysid measurements, at the
# 10 Hz both published SO-101 RL place results train at. The -Resume- variant
# carries the same lifting_object pin as Aligned-Resume.
gym.register(
    id="Isaac-SO-ARM101-PickPlace-DR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceEnvCfg_DR",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlaceDRPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-PickPlace-DR-Resume-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceEnvCfg_DR_RESUME",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlaceDRPPORunnerCfg",
    },
    disable_env_checker=True,
)
