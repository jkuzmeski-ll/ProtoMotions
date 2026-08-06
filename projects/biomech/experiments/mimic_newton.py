# SPDX-License-Identifier: MIT

"""Mimic-on-Newton experiment for the biomech (Rajagopal2015) gold-standard skeleton.

Milestone M8 (ProtoMotions-style imitation env). Trains a PPO policy to imitate a
gold-standard biomech motion clip on the Newton simulator, using ProtoMotions' existing
Mimic environment. The robot (`--robot-name biomech`) is the fitted skeleton exported by
`projects/biomech` (`biomech.export.protomotions_robot`); its MJCF lives at
`protomotions/data/assets/mjcf/biomech_rajagopal.xml`.

Prerequisites
-------------
1. Write the robot asset (once, or after refitting a subject)::

       python -c "import sys; sys.path.insert(0,'projects'); \
           from biomech.osim import parse_osim; \
           from biomech.export.protomotions_robot import write_biomech_asset; \
           write_biomech_asset(parse_osim('projects/biomech/models/rajagopal_data/Rajagopal2015.osim'))"

2. Build a sim-body-aligned motion clip with
   `biomech.export.protomotions_robot.build_simbody_motion` (its body order matches this
   robot, including the exporter's dummy bodies) and `torch.save` it as a `.motion`.

Usage
-----
::

    python protomotions/train_agent.py \
        --robot-name biomech \
        --simulator newton \
        --experiment-path projects/biomech/experiments/mimic_newton.py \
        --experiment-name biomech_mimic_newton \
        --motion-file path/to/biomech_clip.motion \
        --num-envs 4096 --batch-size 16384 --ngpu 1

The env/agent wiring mirrors `examples/experiments/mimic/mlp.py`; only the robot and
simulator defaults differ. Set ``BIOMECH_FOOT_COLLISION`` to ``none``, ``spheres``, or
``boxes`` (default) to select the collision asset. Unified pipeline bundles can also set
``BIOMECH_ASSET_ROOT`` and ``BIOMECH_ASSET_STEM`` to select a subject-specific asset
without overwriting the repository's default Rajagopal asset. Control gains are a uniform
PD starting point (see `protomotions/robot_configs/biomech.py`) and should be tuned against
measured GRF.
"""

import argparse
import os
import sys

from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.motion_lib import MotionLibConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig


# Foot-collision variants (all share the same skeleton/FK/inertia and one .motion clip;
# they differ only in the colliding foot geoms baked into the MJCF -- see
# projects/biomech/export/foot_collision.py and tools/export_s001_subject.py):
#   none    -- no foot colliders (character falls through the floor; visual only)
#   spheres -- OpenSim-style discrete plantar contact spheres (several per foot body)
#   boxes   -- ProtoMotions-style single AABB box per foot body
_FOOT_COLLISION_SUFFIXES = {
    "none": "",
    "spheres": "_spheres",
    "boxes": "_boxes",
}
_DEFAULT_FOOT_COLLISION = "boxes"
_DEFAULT_ASSET_STEM = "biomech_rajagopal"


def _foot_collision_choice() -> str:
    """Select the foot-collision variant: ``{none, spheres, boxes}``.

    ``train_agent.py`` parses argv with strict ``parse_args``, so it *rejects* an
    experiment-specific ``--foot-collision`` flag before this experiment ever runs. Select
    the variant with the ``BIOMECH_FOOT_COLLISION`` environment variable instead, e.g.::

        set BIOMECH_FOOT_COLLISION=spheres   (Windows)
        export BIOMECH_FOOT_COLLISION=spheres (POSIX)

    For forward-compat (if a future train_agent uses ``parse_known_args``) a
    ``--foot-collision X`` / ``--foot-collision=X`` in ``sys.argv`` is honored too. Falls
    back to ``_DEFAULT_FOOT_COLLISION``.
    """
    choice = os.environ.get("BIOMECH_FOOT_COLLISION", _DEFAULT_FOOT_COLLISION)
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--foot-collision" and i + 1 < len(argv):
            choice = argv[i + 1]
        elif a.startswith("--foot-collision="):
            choice = a.split("=", 1)[1]
    if choice not in _FOOT_COLLISION_SUFFIXES:
        raise ValueError(
            f"foot-collision must be one of {sorted(_FOOT_COLLISION_SUFFIXES)}, got {choice!r}"
        )
    return choice


def _asset_file_name(choice: str) -> str:
    """Asset path for the selected subject bundle and collision variant."""
    stem = os.environ.get("BIOMECH_ASSET_STEM", _DEFAULT_ASSET_STEM)
    if not stem or stem != os.path.basename(stem) or stem.endswith(".xml"):
        raise ValueError(
            "BIOMECH_ASSET_STEM must be a file stem without directories or .xml, "
            f"got {stem!r}"
        )
    return f"mjcf/{stem}{_FOOT_COLLISION_SUFFIXES[choice]}.xml"


def terrain_config(args: argparse.Namespace):
    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    import torch

    from protomotions.envs.action import make_pd_action_config
    from protomotions.envs.component_factories import (
        action_smoothness_factory,
        contact_match_rew_factory,
        max_coords_obs_factory,
        mimic_target_poses_max_coords_factory,
        mimic_tracking_rewards_factory,
        pow_rew_factory,
        previous_actions_factory,
        tracking_error_term_factory,
    )
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig

    control_components = {"mimic": MimicControlConfig(bootstrap_on_episode_end=True)}

    observation_components = {
        "max_coords_obs": max_coords_obs_factory(),
        "previous_actions": previous_actions_factory(history_steps=1),
        "mimic_target_poses": mimic_target_poses_max_coords_factory(with_velocities=True),
    }

    termination_components = {
        "tracking_error": tracking_error_term_factory(threshold=0.5),
    }

    reward_components = {
        "action_smoothness": action_smoothness_factory(weight=-0.02),
        **mimic_tracking_rewards_factory(
            gt_weight=0.5,
            gr_weight=0.3,
            gv_weight=0.1,
            gav_weight=0.2,
            rh_weight=0.2,
            gt_coef=-25.0,
            gr_coef=-5.0,
            gv_coef=-0.5,
            gav_coef=-0.1,
            rh_coef=-100.0,
        ),
        "pow_rew": pow_rew_factory(
            weight=-1e-5,
            min_value=-0.5,
            indices=torch.tensor(
                [
                    i
                    for i, name in enumerate(robot_cfg.kinematic_info.dof_names)
                    if robot_cfg.control.control_info[name].actuated is not False
                ],
                dtype=torch.long,
            ),
        ),
        # Reference contact labels come from the GRF-derived per-body foot contacts baked
        # into the .motion by tools/export_s001_subject.py (foot_contacts_from_clip).
        "contact_match_rew": contact_match_rew_factory(
            weight=-0.1, zero_during_grace_period=True
        ),
    }

    return EnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        num_state_history_steps=2,
        control_components=control_components,
        observation_components=observation_components,
        termination_components=termination_components,
        reward_components=reward_components,
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
        ),
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.common.config import MLPLayerConfig, MLPWithConcatConfig
    from protomotions.agents.evaluators.config import (
        MimicEvaluatorConfig,
        MotionWeightsRulesConfig,
    )
    from protomotions.agents.ppo.config import (
        AdvantageNormalizationConfig,
        PPOActorConfig,
        PPOModelConfig,
    )
    from protomotions.envs.component_factories import (
        gr_error_factory,
        gt_error_factory,
        max_joint_error_factory,
    )

    in_keys = ["max_coords_obs", "mimic_target_poses", "previous_actions"]

    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=in_keys,
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=in_keys,
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(6)],
        ),
    )

    critic_config = MLPWithConcatConfig(
        in_keys=in_keys,
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )

    return PPOAgentConfig(
        model=PPOModelConfig(
            in_keys=in_keys,
            out_keys=["action", "mean_action", "neglogp", "value"],
            actor=actor_config,
            critic=critic_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        evaluator=MimicEvaluatorConfig(
            evaluation_components={
                "gt_error": gt_error_factory(threshold=0.5),
                "gr_error": gr_error_factory(),
                "max_joint_error": max_joint_error_factory(),
            },
            motion_weights_rules=MotionWeightsRulesConfig(
                motion_weights_update_success_discount=0.999,
                motion_weights_update_failure_discount=0,
            ),
        ),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True, shift_mean=True, use_ema=True
        ),
    )


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Select the foot-collision asset variant and add foot contact sensors.

    ``BIOMECH_FOOT_COLLISION`` picks which MJCF the simulator loads. Unified pipeline
    bundles may set ``BIOMECH_ASSET_ROOT`` and ``BIOMECH_ASSET_STEM`` to point at their
    subject-specific assets. All variants share the fitted skeleton's topology/FK/inertia
    and DOF set, so the ``kinematic_info``/``control_info`` already extracted from the
    default asset stay valid -- only geometry and physical parameters differ. Newton's
    MJCF importer routes geoms with ``contype/conaffinity=0`` to visuals and ``=1`` to
    colliders.
    """
    choice = _foot_collision_choice()
    asset_root = os.environ.get("BIOMECH_ASSET_ROOT")
    if asset_root:
        robot_cfg.asset.asset_root = asset_root
    robot_cfg.asset.asset_file_name = _asset_file_name(choice)
    print(
        f"[biomech] foot-collision variant: {choice} -> "
        f"{os.path.join(robot_cfg.asset.asset_root, robot_cfg.asset.asset_file_name)}"
    )
    robot_cfg.update_fields(
        contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"]
    )


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg,
    agent_cfg,
    terrain_cfg: TerrainConfig,
    motion_lib_cfg: MotionLibConfig,
    scene_lib_cfg: SceneLibConfig,
    args: argparse.Namespace,
):
    if hasattr(env_cfg, "termination_components") and env_cfg.termination_components:
        env_cfg.termination_components = {}
    env_cfg.max_episode_length = 1000000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0
