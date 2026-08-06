# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pose command that keeps the goal a minimum distance from the object.

Once the place target sits on the table it occupies the same plane as the cube's
spawn area, so a plain uniform command can put the goal on top of the cube. The
was-lifted latch stops that being farmable — the cube still has to be picked —
but it makes a large share of episodes trivial: pick straight up, set straight
down. This command rejects those draws so every episode requires real transport.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs.mdp import UniformPoseCommand, UniformPoseCommandCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ObjectAwarePoseCommand(UniformPoseCommand):
    """Uniform pose command, redrawn until the goal is ``min_separation`` from the
    object in the XY plane.

    Separation is measured in XY only, on purpose: the goal and the cube's resting
    pose are both at table height, so a 3D distance would be dominated by the same
    two axes anyway, and an XY threshold says what is actually meant — the cube has
    to be carried somewhere else, not just lifted and replaced.

    Resampling happens on reset, after the ``reset`` events have repositioned the
    cube, so the draw sees the object's new spawn. ``resampling_time_range`` equals
    the episode length in this task, so in practice this runs once per episode.
    """

    cfg: ObjectAwarePoseCommandCfg

    def __init__(self, cfg: ObjectAwarePoseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._object: RigidObject = env.scene[cfg.object_name]
        # resolved here rather than reusing the base class's own handle, so this
        # does not depend on an attribute name internal to UniformPoseCommand
        self._robot: RigidObject = env.scene[cfg.asset_name]

    def _object_xy_b(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Object XY in the robot base frame, for the given envs."""
        object_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[env_ids, :3],
            self._robot.data.root_state_w[env_ids, 3:7],
            self._object.data.root_pos_w[env_ids, :3],
        )
        return object_pos_b[:, :2]

    def _resample_command(self, env_ids: torch.Tensor):
        super()._resample_command(env_ids)
        if self.cfg.min_separation <= 0.0:
            return

        object_xy = self._object_xy_b(env_ids)
        for _ in range(self.cfg.max_resample_tries):
            too_close = torch.norm(self.pose_command_b[env_ids, :2] - object_xy, dim=1) < self.cfg.min_separation
            if not bool(too_close.any()):
                return
            super()._resample_command(env_ids[too_close])

        # Deterministic fallback so a rare unlucky env is never left with a goal on
        # top of the cube: send it to whichever y edge of the sampling range is
        # further from the object. The range spans 40 cm, so this always clears
        # min_separation for any object inside the spawn box.
        too_close = torch.norm(self.pose_command_b[env_ids, :2] - object_xy, dim=1) < self.cfg.min_separation
        if bool(too_close.any()):
            ids = env_ids[too_close]
            y_low, y_high = self.cfg.ranges.pos_y
            self.pose_command_b[ids, 1] = torch.where(
                object_xy[too_close, 1] > 0.0,
                torch.full_like(self.pose_command_b[ids, 1], y_low),
                torch.full_like(self.pose_command_b[ids, 1], y_high),
            )


@configclass
class ObjectAwarePoseCommandCfg(UniformPoseCommandCfg):
    """Configuration for :class:`ObjectAwarePoseCommand`."""

    class_type: type = ObjectAwarePoseCommand

    object_name: str = "object"
    """Scene entity whose position the goal must stay away from."""

    min_separation: float = 0.12
    """Minimum XY distance between the goal and the object, in metres.

    Set to 0.0 to fall back to plain uniform sampling. 0.12 is ~4 cube widths and
    comfortably exceeds the ~6 cm placement error the lift-only policy converged
    to, so hitting the goal requires transport rather than luck.
    """

    max_resample_tries: int = 8
    """Rejection-sampling rounds before the deterministic fallback takes over."""
