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
from dataclasses import dataclass, field
from typing import Dict, List

from protomotions.components.pose_lib import ControlInfo
from protomotions.robot_configs.base import (
    ControlConfig,
    ControlType,
    RobotAssetConfig,
    RobotConfig,
    SimulatorParams,
)
from protomotions.simulator.isaacgym.config import IsaacGymSimParams
from protomotions.simulator.isaaclab.config import IsaacLabSimParams
from protomotions.simulator.genesis.config import GenesisSimParams
from protomotions.simulator.newton.config import NewtonSimParams
from protomotions.simulator.mujoco.config import MujocoSimParams


@dataclass
class MSHumanLowerRobotConfig(RobotConfig):
    """Simplified MS-Human-700 locomotion skeleton.

    The MJCF referenced here is generated locally from a user-provided
    MS-Human-700 checkout by data/biomechanics_retargeting/scripts/convert_ms_human_lowerbody_to_proto.py.
    It keeps the lower-body skeleton and simple upper torso, removes muscles and
    tendons, replaces the original 6-DOF pelvis sliders with a free root, and
    simplifies each knee to one hinge joint.
    """

    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["talus_l", "calcn_l", "toes_l"],
            "all_right_foot_bodies": ["talus_r", "calcn_r", "toes_r"],
            "all_left_hand_bodies": [],
            "all_right_hand_bodies": [],
            "head_body_name": ["head_neck"],
            "torso_body_name": ["thoracic12"],
        }
    )

    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "pelvis",
            "sacrum",
            "thoracic12",
            "head_neck",
            "femur_l",
            "tibia_l",
            "calcn_l",
            "toes_l",
            "femur_r",
            "tibia_r",
            "calcn_r",
            "toes_r",
        ]
    )

    contact_bodies: List[str] = field(
        default_factory=lambda: ["calcn_l", "toes_l", "calcn_r", "toes_r"]
    )

    default_root_height: float = 0.95
    anchor_body_name: str = "pelvis"

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/ms_human_700/MS-Human-700-Locomotion-Simple.xml",
            usd_asset_file_name=None,
        )
    )

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info={
                ".*hip.*": ControlInfo(
                    stiffness=300,
                    damping=30,
                    armature=0.01,
                    effort_limit=300,
                    velocity_limit=100,
                ),
                ".*knee.*": ControlInfo(
                    stiffness=250,
                    damping=25,
                    armature=0.01,
                    effort_limit=300,
                    velocity_limit=100,
                ),
                ".*(ankle|subtalar|mtp).*": ControlInfo(
                    stiffness=120,
                    damping=12,
                    armature=0.01,
                    effort_limit=150,
                    velocity_limit=100,
                ),
            },
        )
    )

    simulation_params: SimulatorParams = field(
        default_factory=lambda: SimulatorParams(
            isaacgym=IsaacGymSimParams(
                fps=60,
                decimation=2,
                substeps=2,
            ),
            isaaclab=IsaacLabSimParams(
                fps=120,
                decimation=4,
            ),
            genesis=GenesisSimParams(
                fps=60,
                decimation=2,
                substeps=2,
            ),
            newton=NewtonSimParams(
                fps=120,
                decimation=4,
            ),
            mujoco=MujocoSimParams(
                fps=120,
                decimation=4,
            ),
        )
    )
