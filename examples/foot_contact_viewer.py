# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Interactive MJCF viewer for the foot-contact sandbox.

This intentionally does not use the full ProtoMotions environment stack. It loads an
MJCF directly with MuJoCo so contact geometry, keyframes, sensors, and solver options
can be inspected quickly while the camera remains free.

Usage:
    python examples/foot_contact_viewer.py

    python examples/foot_contact_viewer.py \
        --model-file projects/humanoid_model/smpl_humanoid_foot_contact.xml \
        --keyframe \
        --paused \
        --show-contacts \
        --camera-distance 1.25

    python examples/foot_contact_viewer.py \
        --model-file projects/feet_models/bones_foot.xml \
        --keyframe rest_pose \
        --paused \
        --show-contacts

Mouse controls are MuJoCo viewer defaults:
    left-drag    rotate camera
    right-drag   zoom / dolly
    middle-drag  pan
    scroll       zoom

Keyboard controls added by this script:
    Space        pause / unpause physics
    .            single-step while paused
    R            reset to current keyframe or default qpos
    N / P        next / previous keyframe
    1-9          jump to keyframe by index
    C            toggle contact point/force visualization
    F            reset free camera
    W / S        move free body +x / -x
    A / D        move free body +y / -y
    E / Q        move free body +z / -z
    X / Z        yaw free body + / -
    Esc          close viewer
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_FILE = (
    PROJECT_ROOT / "projects" / "humanoid_model" / "smpl_humanoid_foot_contact.xml"
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="View and interactively inspect an MJCF foot-contact model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=DEFAULT_MODEL_FILE,
        help="MJCF file to load. Relative paths are resolved from the ProtoMotions root.",
    )
    parser.add_argument(
        "--keyframe",
        type=str,
        nargs="?",
        const="",
        default="auto",
        help="Initial keyframe name or index. Use 'auto' for the first keyframe. If provided without a value, skips keyframe reset.",
    )
    parser.add_argument(
        "--paused",
        action="store_true",
        default=False,
        help="Start paused. Useful for inspecting contact poses without dynamics.",
    )
    parser.add_argument(
        "--show-contacts",
        action="store_true",
        default=False,
        help="Show contact points and contact forces in the viewer at startup.",
    )
    parser.add_argument(
        "--print-contact-summary",
        action="store_true",
        default=False,
        help="Print per-contact-region normal force summaries while running.",
    )
    parser.add_argument(
        "--contact-print-interval",
        type=float,
        default=0.5,
        help="Seconds between contact summary prints.",
    )
    parser.add_argument(
        "--move-step",
        type=float,
        default=0.01,
        help="Meters per keyboard translation step for models with a free joint.",
    )
    parser.add_argument(
        "--yaw-step-deg",
        type=float,
        default=5.0,
        help="Degrees per keyboard yaw step for models with a free joint.",
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=0.75,
        help="Initial free-camera distance.",
    )
    parser.add_argument(
        "--realtime-factor",
        type=float,
        default=1.0,
        help="Target wall-clock playback speed when physics is unpaused.",
    )
    return parser


def resolve_model_file(path: Path) -> Path:
    if path.is_absolute():
        model_file = path
    else:
        model_file = PROJECT_ROOT / path

    model_file = model_file.resolve()
    if not model_file.exists():
        raise FileNotFoundError(f"MJCF file not found: {model_file}")
    return model_file


def list_keyframes(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, key_id) or f"key_{key_id}"
        for key_id in range(model.nkey)
    ]


def parse_keyframe(model: mujoco.MjModel, requested: str) -> int | None:
    if requested == "":
        return None

    if model.nkey == 0:
        print("Model has no keyframes; using default qpos.")
        return None

    if requested.lower() in {"auto", "first"}:
        return 0

    if requested.isdigit():
        key_id = int(requested)
        if key_id < 0 or key_id >= model.nkey:
            raise ValueError(
                f"Keyframe index {key_id} outside range [0, {model.nkey})."
            )
        return key_id

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, requested)
    if key_id < 0:
        names = ", ".join(list_keyframes(model))
        raise ValueError(
            f"Unknown keyframe '{requested}'. Available keyframes: {names}"
        )
    return key_id


def reset_to_keyframe(
    model: mujoco.MjModel, data: mujoco.MjData, key_id: int | None
) -> None:
    if key_id is None:
        mujoco.mj_resetData(model, data)
        key_name = "default qpos"
    else:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        key_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, key_id)
    mujoco.mj_forward(model, data)
    print(f"Reset to {key_name}; ncon={data.ncon}")


def has_free_root(model: mujoco.MjModel) -> bool:
    return model.nq >= 7 and model.nv >= 6


def axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half_angle = 0.5 * angle
    return np.array(
        [
            math.cos(half_angle),
            axis[0] * math.sin(half_angle),
            axis[1] * math.sin(half_angle),
            axis[2] * math.sin(half_angle),
        ],
        dtype=np.float64,
    )


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def move_free_root(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    translation: np.ndarray | None = None,
    yaw: float = 0.0,
) -> None:
    if not has_free_root(model):
        print(
            "This model does not appear to start with a free joint; cannot move root."
        )
        return

    if translation is not None:
        data.qpos[:3] += translation

    if yaw != 0.0:
        dq = axis_angle_quat(np.array([0.0, 0.0, 1.0]), yaw)
        q = quat_mul_wxyz(dq, data.qpos[3:7])
        q /= np.linalg.norm(q)
        data.qpos[3:7] = q

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def reset_free_camera(viewer: Any, camera_distance: float) -> None:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = np.array([0.04, 0.0, 0.02])
    viewer.cam.distance = camera_distance
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -25.0


def set_contact_visualization(viewer: Any, enabled: bool) -> None:
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = enabled
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = enabled


def contact_summary(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    normal_force_by_geom: dict[str, float] = {}
    force = np.zeros(6, dtype=np.float64)

    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

        if geom1 is None or geom2 is None:
            continue

        if geom1 == "floor":
            patch_name = geom2
        elif geom2 == "floor":
            patch_name = geom1
        else:
            patch_name = f"{geom1}/{geom2}"

        mujoco.mj_contactForce(model, data, contact_id, force)
        normal_force_by_geom[patch_name] = normal_force_by_geom.get(
            patch_name, 0.0
        ) + float(force[0])

    return normal_force_by_geom


def print_contact_summary(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    summary = contact_summary(model, data)
    if not summary:
        print(f"t={data.time:7.3f}  ncon={data.ncon:2d}  no contacts")
        return

    parts = [f"{name}:{force:7.2f}N" for name, force in sorted(summary.items())]
    print(f"t={data.time:7.3f}  ncon={data.ncon:2d}  " + "  ".join(parts))


def main() -> None:
    args = create_parser().parse_args()
    model_file = resolve_model_file(args.model_file)

    print("\n=== Foot Contact MJCF Viewer ===")
    print(f"Model file: {model_file}")

    model = mujoco.MjModel.from_xml_path(str(model_file))
    data = mujoco.MjData(model)

    keyframe_names = list_keyframes(model)
    print(
        f"Bodies: {model.nbody}, geoms: {model.ngeom}, sensors: {model.nsensor}, pairs: {model.npair}"
    )
    print(f"Keyframes: {keyframe_names if keyframe_names else 'none'}")

    current_key_id = parse_keyframe(model, args.keyframe)
    reset_to_keyframe(model, data, current_key_id)

    command_lock = threading.Lock()
    commands: list[tuple[str, Any]] = []
    state = {
        "paused": args.paused,
        "step_once": False,
        "show_contacts": args.show_contacts,
    }

    def enqueue(command: str, payload: Any = None) -> None:
        with command_lock:
            commands.append((command, payload))

    def key_callback(key: int) -> None:
        if key == 256:  # Esc
            enqueue("close")
            return
        if key == 32:  # Space
            enqueue("toggle_pause")
            return
        if key == ord("."):
            enqueue("step")
            return

        try:
            char = chr(key).lower()
        except ValueError:
            return

        if char == "r":
            enqueue("reset")
        elif char == "n":
            enqueue("next_keyframe")
        elif char == "p":
            enqueue("previous_keyframe")
        elif char == "c":
            enqueue("toggle_contacts")
        elif char == "f":
            enqueue("reset_camera")
        elif char == "w":
            enqueue("move", np.array([args.move_step, 0.0, 0.0]))
        elif char == "s":
            enqueue("move", np.array([-args.move_step, 0.0, 0.0]))
        elif char == "a":
            enqueue("move", np.array([0.0, args.move_step, 0.0]))
        elif char == "d":
            enqueue("move", np.array([0.0, -args.move_step, 0.0]))
        elif char == "e":
            enqueue("move", np.array([0.0, 0.0, args.move_step]))
        elif char == "q":
            enqueue("move", np.array([0.0, 0.0, -args.move_step]))
        elif char == "x":
            enqueue("yaw", math.radians(args.yaw_step_deg))
        elif char == "z":
            enqueue("yaw", -math.radians(args.yaw_step_deg))
        elif char.isdigit() and char != "0":
            enqueue("set_keyframe", int(char) - 1)

    print("\n=== Controls ===")
    print("Mouse: left rotate, right zoom, middle pan, scroll zoom")
    print("Keyboard: Space pause, . step, R reset, N/P keyframes, C contacts, F camera")
    print("Move root: W/S x, A/D y, E/Q z, X/Z yaw")
    print("Esc closes the viewer")
    print(f"Starting {'paused' if args.paused else 'running'}")

    close_requested = False
    last_contact_print = 0.0

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        with viewer.lock():
            reset_free_camera(viewer, args.camera_distance)
            set_contact_visualization(viewer, state["show_contacts"])

        while viewer.is_running() and not close_requested:
            loop_start = time.time()

            with command_lock:
                pending_commands = list(commands)
                commands.clear()

            with viewer.lock():
                for command, payload in pending_commands:
                    if command == "close":
                        close_requested = True
                    elif command == "toggle_pause":
                        state["paused"] = not state["paused"]
                        print(f"Physics {'paused' if state['paused'] else 'running'}")
                    elif command == "step":
                        state["step_once"] = True
                        state["paused"] = True
                    elif command == "reset":
                        reset_to_keyframe(model, data, current_key_id)
                    elif command == "next_keyframe":
                        if model.nkey > 0:
                            current_key_id = (
                                0
                                if current_key_id is None
                                else (current_key_id + 1) % model.nkey
                            )
                            reset_to_keyframe(model, data, current_key_id)
                    elif command == "previous_keyframe":
                        if model.nkey > 0:
                            current_key_id = (
                                model.nkey - 1
                                if current_key_id is None
                                else (current_key_id - 1) % model.nkey
                            )
                            reset_to_keyframe(model, data, current_key_id)
                    elif command == "set_keyframe":
                        key_id = int(payload)
                        if key_id < model.nkey:
                            current_key_id = key_id
                            reset_to_keyframe(model, data, current_key_id)
                    elif command == "toggle_contacts":
                        state["show_contacts"] = not state["show_contacts"]
                        set_contact_visualization(viewer, state["show_contacts"])
                        print(
                            f"Contact visualization {'on' if state['show_contacts'] else 'off'}"
                        )
                    elif command == "reset_camera":
                        reset_free_camera(viewer, args.camera_distance)
                    elif command == "move":
                        move_free_root(model, data, translation=payload)
                    elif command == "yaw":
                        move_free_root(model, data, yaw=float(payload))

                if state["paused"]:
                    if state["step_once"]:
                        mujoco.mj_step(model, data)
                        state["step_once"] = False
                    else:
                        mujoco.mj_forward(model, data)
                else:
                    mujoco.mj_step(model, data)

                if (
                    args.print_contact_summary
                    and loop_start - last_contact_print >= args.contact_print_interval
                ):
                    print_contact_summary(model, data)
                    last_contact_print = loop_start

            viewer.sync()

            if not state["paused"] and args.realtime_factor > 0.0:
                elapsed = time.time() - loop_start
                target_dt = model.opt.timestep / args.realtime_factor
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)
            else:
                time.sleep(0.01)

    print("Viewer closed.")


if __name__ == "__main__":
    main()
