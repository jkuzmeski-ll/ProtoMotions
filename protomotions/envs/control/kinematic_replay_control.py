# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Kinematic replay control - plays reference motions without physics."""

from dataclasses import dataclass, field
from typing import Dict, TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.base import ControlComponent, ControlComponentConfig
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    MarkerState,
    VisualizationMarkerConfig,
)
from protomotions.simulator.base_simulator.simulator_state import ResetState

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


@dataclass
class KinematicReplayControlConfig(ControlComponentConfig):
    _target_: str = "protomotions.envs.control.kinematic_replay_control.KinematicReplayControl"
    show_motion_markers: bool = True
    marker_keys: tuple[str, ...] = field(
        default=("marker_targets", "marker_reconstructed")
    )


class KinematicReplayControl(ControlComponent):
    """Plays reference motions kinematically (bypasses physics)."""
    
    config: KinematicReplayControlConfig
    
    def __init__(self, config: KinematicReplayControlConfig, env: "BaseEnv"):
        super().__init__(config, env)
        self._marker_data = self._load_marker_data()
        self._marker_offset = torch.zeros(self.env.num_envs, 3, device=self.env.device)
    
    def reset(self, env_ids: Tensor):
        if self._marker_offset is not None and len(env_ids) > 0:
            self._marker_offset[env_ids] = 0.0

    def _load_marker_data(self) -> dict[str, torch.Tensor]:
        """Load optional marker overlay tensors directly from motion files."""
        if not self.config.show_motion_markers or self.env.motion_lib is None:
            return {}

        motion_files = getattr(self.env.motion_lib, "motion_files", None)
        if not motion_files:
            return {}

        marker_data: dict[str, list[torch.Tensor]] = {
            key: [] for key in self.config.marker_keys
        }
        for motion_file in motion_files:
            motion = torch.load(motion_file, map_location="cpu", weights_only=False)
            for key in self.config.marker_keys:
                if key in motion:
                    marker_data[key].append(
                        motion[key].detach().to(device=self.env.device, dtype=torch.float32)
                    )

        loaded = {}
        for key, tensors in marker_data.items():
            if len(tensors) == len(motion_files):
                loaded[key] = torch.cat(tensors, dim=0)

        if loaded:
            print(
                "Loaded kinematic replay marker overlays: "
                + ", ".join(f"{key}={tuple(value.shape)}" for key, value in loaded.items())
            )
        return loaded

    def _marker_frame_indices(self) -> torch.Tensor:
        motion_ids = self.env.motion_manager.motion_ids
        motion_times = self.env.motion_manager.motion_times
        frame_ids = torch.round(motion_times / self.env.motion_lib.motion_dt[motion_ids]).long()
        frame_ids = torch.minimum(
            frame_ids,
            self.env.motion_lib.motion_num_frames[motion_ids] - 1,
        )
        frame_ids = torch.clamp(frame_ids, min=0)
        return frame_ids + self.env.motion_lib.length_starts[motion_ids]
    
    def step(self):
        # Advance motion time
        sync_motion_dt = self.env.simulator.decimation * 1.0 / self.env.simulator.config.sim.fps
        self.env.motion_manager.motion_times += sync_motion_dt
        
        # Handle done clips
        done_clip = self.env.motion_manager.get_done_tracks()
        if any(done_clip):
            done_env_ids = torch.where(done_clip)[0]
            self.env.motion_manager.sample_motions(done_env_ids)
        
        # Get reference state
        ref_state = self.env.motion_lib.get_motion_state(
            self.env.motion_manager.motion_ids,
            self.env.motion_manager.motion_times,
        )
        
        # Zero velocities for kinematic replay
        ref_state.dof_vel *= 0
        ref_state.rigid_body_vel *= 0
        ref_state.rigid_body_ang_vel *= 0
        ref_reset_state = ResetState.from_robot_state(ref_state)
        
        env_ids = torch.arange(self.env.num_envs, dtype=torch.long, device=self.env.device)
        
        # Get object state
        ref_object_state = self.env.scene_lib.get_scene_pose(
            env_ids,
            self.env.motion_manager.motion_times,
            self.env.config.ref_object_respawn_offset,
        )
        ref_object_state.root_vel = torch.zeros_like(ref_object_state.root_pos)
        ref_object_state.root_ang_vel = torch.zeros_like(ref_object_state.root_pos)
        
        # Apply terrain offset
        offset = self.env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
            ref_reset_state.root_pos[:, None, :], env_ids
        ).squeeze(1)
        ref_reset_state.root_pos += offset
        self._marker_offset[env_ids] = offset
        
        if self.env.scene_lib.num_scenes() > 0:
            ref_object_state.root_pos += offset.unsqueeze(1)
        
        # Set robot state directly
        self.env.simulator.reset_envs(ref_reset_state, ref_object_state, env_ids)
        
        # Prevent double reset
        self.env.progress_buf[env_ids] = 0
        self.env.reset_buf[env_ids] = 0
        self.env.terminate_buf[env_ids] = 0
    
    def populate_context(self, ctx: EnvContext) -> None:
        """Kinematic replay doesn't add any context variables."""
        pass

    def create_visualization_markers(
        self, headless: bool
    ) -> Dict[str, VisualizationMarkerConfig]:
        """Create C3D target/reconstructed marker overlays for kinematic replay."""
        if headless or not self._marker_data:
            return {}

        num_markers = next(iter(self._marker_data.values())).shape[1]
        markers = [MarkerConfig(size="small") for _ in range(num_markers)]
        configs = {}
        if "marker_targets" in self._marker_data:
            configs["marker_targets"] = VisualizationMarkerConfig(
                type="sphere", color=(1.0, 0.45, 0.0), markers=markers
            )
        if "marker_reconstructed" in self._marker_data:
            configs["marker_reconstructed"] = VisualizationMarkerConfig(
                type="sphere", color=(0.0, 0.9, 0.2), markers=markers
            )
        return configs

    def get_markers_state(self) -> Dict[str, MarkerState]:
        """Get C3D target/reconstructed marker positions for rendering."""
        if self.env.simulator.headless or not self._marker_data:
            return {}

        frame_indices = self._marker_frame_indices()
        markers_state = {}
        for key, marker_positions in self._marker_data.items():
            positions = marker_positions[frame_indices].clone()
            positions += self._marker_offset[:, None, :]
            markers_state[key] = MarkerState(
                translation=positions,
                orientation=torch.zeros(
                    self.env.num_envs,
                    positions.shape[1],
                    4,
                    device=self.env.device,
                ),
            )
        return markers_state
