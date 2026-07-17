# SPDX-License-Identifier: MIT

"""M8 bridge tests — biomech fit -> runnable ProtoMotions Newton imitation setup.

Validates ``biomech.export.protomotions_robot``:
- the sim-body motion clip aligns 1:1 (order + count) with ProtoMotions'
  ``extract_kinematic_info`` on the exact exported MJCF (the dummy-body reconciliation),
- a ``RobotConfig`` for the fitted skeleton instantiates and self-validates,
- the sim-body clip loads as a ``RobotState`` with matching body count,
- ``export_protomotions_bundle`` writes the asset (+ motion) files.

Requires mujoco (pinned 3.5.0) and, for the config/RobotState tests, protomotions.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from biomech.osim import parse_osim
from biomech.tests import SkipTest

_ROOT = Path(__file__).resolve().parents[1]
_OSIM = _ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"

_SPEC = None


def _spec():
    global _SPEC
    if _SPEC is None:
        _SPEC = parse_osim(str(_OSIM))
    return _SPEC


def _require_mujoco():
    try:
        import mujoco  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SkipTest(f"mujoco not available: {exc}")


def _require_protomotions():
    try:
        import protomotions  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SkipTest(f"protomotions not importable: {exc}")


def _feasible(spec, rng, n):
    lo = np.array([c.limit_lo for j in spec.joints for c in j.coordinates])
    hi = np.array([c.limit_hi for j in spec.joints for c in j.coordinates])
    locked = np.array([c.locked for j in spec.joints for c in j.coordinates])
    lo = np.where(np.isfinite(lo), lo, -1.0)
    hi = np.where(np.isfinite(hi), hi, 1.0)
    Q = rng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo), size=(n, len(lo)))
    Q[:, locked] = 0.0
    return Q


def test_simbody_motion_aligns_with_kinematic_info():
    _require_mujoco()
    _require_protomotions()
    from protomotions.components.pose_lib import extract_kinematic_info

    from biomech.export.mjcf import export_mjcf
    from biomech.export.protomotions_robot import build_simbody_motion

    spec = _spec()
    Q = _feasible(spec, np.random.default_rng(0), 5)
    xml = export_mjcf(spec, coupled_knee="coupled").xml
    clip = build_simbody_motion(spec, Q, fps=100.0, mjcf_xml=xml)

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "m.xml"
        p.write_text(xml)
        ki = extract_kinematic_info(str(p))

    assert clip.body_names == ki.body_names, (len(clip.body_names), len(ki.body_names))
    assert clip.data["rigid_body_pos"].shape == (5, len(ki.body_names), 3)


def test_simbody_motion_is_upright():
    """Z-up clip has correct anatomy: torso above pelvis, feet below (guards the
    free-root xyzw->wxyz quat order into MuJoCo FK, which otherwise inverts the body)."""
    _require_mujoco()
    from biomech.export.protomotions_robot import build_simbody_motion

    spec = _spec()
    dof = {n: i for i, n in enumerate(spec.dof_names)}
    q = np.zeros(spec.num_dofs)
    q[dof["pelvis_ty"]] = 0.95  # stand the neutral pose up (OpenSim Y)
    clip = build_simbody_motion(spec, q, fps=100.0)
    pos = clip.data["rigid_body_pos"].numpy()[0]
    z = {n: pos[i, 2] for i, n in enumerate(clip.body_names)}
    assert z["torso"] > z["pelvis"], (z["torso"], z["pelvis"])
    assert z["calcn_r"] < z["pelvis"] and z["calcn_l"] < z["pelvis"]
    assert z["pelvis"] > 0.5  # upright, not collapsed


def test_robot_config_instantiates_and_validates():
    _require_mujoco()
    _require_protomotions()
    from biomech.export.protomotions_robot import (
        build_biomech_robot_config,
        write_biomech_asset,
    )

    spec = _spec()
    with tempfile.TemporaryDirectory() as tmp:
        write_biomech_asset(
            spec,
            asset_file_name="mjcf/biomech_test.xml",
            coupled_knee="coupled",
            asset_root="assets",
            repo_root=tmp,
        )
        cfg = build_biomech_robot_config(
            asset_file_name="mjcf/biomech_test.xml",
            asset_root=str(Path(tmp) / "assets"),
        )
    # derived fields populated from the MJCF
    assert cfg.number_of_actions == cfg.kinematic_info.num_dofs
    assert cfg.number_of_actions == 31  # default export: coupled-knee, MTP locked
    # semantic maps resolve to real bodies
    assert "calcn_r" in cfg.common_naming_to_robot_body_names["all_right_foot_bodies"]
    assert cfg.anchor_body_name == "torso"
    assert cfg.anchor_body_index == cfg.kinematic_info.body_names.index("torso")


def test_simbody_clip_matches_robot_body_count():
    _require_mujoco()
    _require_protomotions()
    from biomech.export.protomotions_robot import (
        build_biomech_robot_config,
        build_simbody_motion,
        write_biomech_asset,
    )

    spec = _spec()
    Q = _feasible(spec, np.random.default_rng(1), 4)
    with tempfile.TemporaryDirectory() as tmp:
        write_biomech_asset(
            spec,
            asset_file_name="mjcf/biomech_test.xml",
            coupled_knee="coupled",
            asset_root="assets",
            repo_root=tmp,
        )
        cfg = build_biomech_robot_config(
            asset_file_name="mjcf/biomech_test.xml",
            asset_root=str(Path(tmp) / "assets"),
        )
    clip = build_simbody_motion(spec, Q, fps=100.0)
    assert len(clip.body_names) == len(cfg.kinematic_info.body_names)
    assert clip.body_names == cfg.kinematic_info.body_names


def test_simbody_clip_loads_as_robotstate():
    _require_mujoco()
    _require_protomotions()
    try:
        from protomotions.simulator.base_simulator.simulator_state import (
            RobotState,
            StateConversion,
        )
    except Exception as exc:  # noqa: BLE001
        raise SkipTest(f"RobotState not importable: {exc}")

    from biomech.export.protomotions_robot import build_simbody_motion

    spec = _spec()
    Q = _feasible(spec, np.random.default_rng(2), 4)
    clip = build_simbody_motion(spec, Q, fps=100.0)
    rs = RobotState.from_dict(clip.data, state_conversion=StateConversion.COMMON)
    rs.fps = clip.fps
    assert rs.num_bodies == len(clip.body_names)
    assert rs.motion_num_frames == 4


def test_export_bundle_writes_files():
    _require_mujoco()
    from biomech.export.protomotions_robot import export_protomotions_bundle

    spec = _spec()
    Q = _feasible(spec, np.random.default_rng(3), 3)
    with tempfile.TemporaryDirectory() as tmp:
        bundle = export_protomotions_bundle(
            spec,
            Q,
            fps=100.0,
            asset_file_name="mjcf/biomech_test.xml",
            asset_root="assets",
            repo_root=tmp,
            motion_path=Path(tmp) / "clip.motion",
        )
        assert bundle.asset_path.exists()
        assert bundle.motion_path.exists()
        assert len(bundle.body_names) > 20  # includes dummy bodies


def test_biomech_robot_in_factory():
    _require_mujoco()
    _require_protomotions()
    from protomotions.robot_configs.factory import robot_config

    asset = _ROOT.parents[1] / "protomotions" / "data" / "assets" / "mjcf" / "biomech_rajagopal.xml"
    if not asset.exists():
        raise SkipTest(f"biomech asset not written: {asset}")
    cfg = robot_config("biomech")
    assert cfg.number_of_actions == 33
    assert cfg.anchor_body_name == "torso"
    assert "calcn_l" in cfg.common_naming_to_robot_body_names["all_left_foot_bodies"]


def test_mimic_newton_experiment_builds_configs():
    """The M8 experiment builds robot/env/agent configs end to end (no sim launched)."""
    _require_mujoco()
    _require_protomotions()
    import argparse
    import importlib.util

    from protomotions.robot_configs.factory import robot_config

    asset = _ROOT.parents[1] / "protomotions" / "data" / "assets" / "mjcf" / "biomech_rajagopal.xml"
    if not asset.exists():
        raise SkipTest(f"biomech asset not written: {asset}")

    exp_path = _ROOT / "experiments" / "mimic_newton.py"
    spec_mod = importlib.util.spec_from_file_location("biomech_mimic_newton", exp_path)
    exp = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(exp)

    robot_cfg = robot_config("biomech")
    args = argparse.Namespace(
        motion_file=None, batch_size=1024, training_max_steps=1000
    )
    terrain = exp.terrain_config(args)
    scene = exp.scene_lib_config(args)
    motion = exp.motion_lib_config(args)
    env = exp.env_config(robot_cfg, args)
    agent = exp.agent_config(robot_cfg, env, args)
    exp.configure_robot_and_simulator(robot_cfg, None, args)

    assert terrain is not None and scene is not None and motion is not None
    assert agent.model.actor.num_out == robot_cfg.number_of_actions == 33
    # foot contact sensors resolved to real bodies
    assert "calcn_r" in robot_cfg.contact_bodies and "calcn_l" in robot_cfg.contact_bodies


def test_foot_collision_switch_selects_asset():
    """``--foot-collision`` picks the matching MJCF variant (default boxes)."""
    _require_mujoco()
    _require_protomotions()
    import argparse
    import importlib.util
    import sys

    from protomotions.robot_configs.factory import robot_config

    asset = _ROOT.parents[1] / "protomotions" / "data" / "assets" / "mjcf" / "biomech_rajagopal.xml"
    if not asset.exists():
        raise SkipTest(f"biomech asset not written: {asset}")

    exp_path = _ROOT / "experiments" / "mimic_newton.py"
    spec_mod = importlib.util.spec_from_file_location("biomech_mimic_newton", exp_path)
    exp = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(exp)

    args = argparse.Namespace(motion_file=None, batch_size=1024, training_max_steps=1000)
    saved_argv = sys.argv
    try:
        # explicit selection
        sys.argv = ["train_agent.py", "--foot-collision", "spheres"]
        cfg = robot_config("biomech")
        exp.configure_robot_and_simulator(cfg, None, args)
        assert cfg.asset.asset_file_name == "mjcf/biomech_rajagopal_spheres.xml"

        # =-form
        sys.argv = ["train_agent.py", "--foot-collision=none"]
        cfg = robot_config("biomech")
        exp.configure_robot_and_simulator(cfg, None, args)
        assert cfg.asset.asset_file_name == "mjcf/biomech_rajagopal.xml"

        # default when the flag is absent
        sys.argv = ["train_agent.py"]
        cfg = robot_config("biomech")
        exp.configure_robot_and_simulator(cfg, None, args)
        assert cfg.asset.asset_file_name == "mjcf/biomech_rajagopal_boxes.xml"
    finally:
        sys.argv = saved_argv
