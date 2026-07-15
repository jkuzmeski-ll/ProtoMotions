# SPDX-License-Identifier: MIT
"""Subject anthropometry priors for biomechanical marker fitting.

The real S001 capture includes a Visual3D/Plug-in-Gait ``.mp`` file with subject
segment lengths in millimetres.  OpenSim/Nimble-style marker fitting should not let
weakly observed scale axes float freely: dynamic cluster markers are noisy and soft-tissue
contaminated, while segment lengths are static subject metadata.  This module converts the
metadata into a vector prior on the Rajagopal group-scale parameters used by
:class:`biomech.fitting.marker_fitter.MarkerFitter`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from biomech.osim.spec import SkeletonSpec
from biomech.skeleton.skeleton import WarpSkeleton


def read_mp(path: str | Path) -> dict[str, float]:
    """Read a Visual3D/Plug-in-Gait ``.mp`` file as SI metres where appropriate.

    Values in S001.mp are millimetres or degrees depending on the key.  This function only
    parses numeric values and leaves unit interpretation to the caller.  Keys are returned
    without the leading ``$``.
    """
    vals: dict[str, float] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or not line.startswith("$") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        try:
            vals[key.strip()[1:]] = float(val.strip().split()[0])
        except Exception:
            continue
    return vals


def _group_index(spec: SkeletonSpec) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, group in enumerate(spec.scale_groups):
        for body in group:
            out[body] = i
    return out


def _marker_offset(spec: SkeletonSpec, name: str) -> np.ndarray:
    return np.asarray(spec.marker(name).offset, dtype=np.float64)


def anthropometric_scale_prior(
    spec: SkeletonSpec,
    mp_path: str | Path,
    *,
    length_weight: float = 20.0,
    width_weight: float = 3.0,
    include_upper_body: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Build ``(target, weights, diagnostics)`` for ``MarkerFitConfig``.

    The target vector has shape ``(3*num_scale_groups,)`` in the same order as
    ``spec.scale_groups``.  Only anatomically measured axes get nonzero extra weight; other
    axes keep the fitter's weak neutral prior.

    Lower-body constraints currently used:
      * femur Y-scale from ``UpperLegLength`` / model hip-to-knee length
      * tibia Y-scale from ``LowerLegLength`` / model knee-to-ankle length
      * calcaneus/foot X-scale from ``FootLength`` / model RCAL-to-RTOE distance

    Optional upper-body constraints use the same idea for humerus and ulna Y-scales.  The
    prior is symmetric by construction: left/right homologous segments get the same target.
    """
    mp = read_mp(mp_path)
    gi = _group_index(spec)
    G3 = 3 * len(spec.scale_groups)
    target = np.ones(G3, dtype=np.float64)
    weights = np.zeros(G3, dtype=np.float64)
    diag: dict[str, float] = {}

    skel = WarpSkeleton(spec, device="cpu")
    world, _ = skel.forward(np.zeros(spec.num_dofs))
    body_names = [b.name for b in spec.bodies]
    pos = {name: np.asarray(world[0, i, :3, 3]) for i, name in enumerate(body_names)}

    def set_axis(body: str, axis: int, scale: float, weight: float, key: str):
        if body not in gi or not np.isfinite(scale) or scale <= 0.0:
            return
        j = 3 * gi[body] + axis
        target[j] = float(scale)
        weights[j] = max(weights[j], float(weight))
        diag[f"{key}:{body}:axis{axis}"] = float(scale)

    # Lower-body segment lengths. Axis 1 is the OpenSim/Rajagopal long axis for femur/tibia.
    if "UpperLegLength" in mp:
        model_femur = float(np.linalg.norm(pos["tibia_r"] - pos["femur_r"]))
        s = (mp["UpperLegLength"] * 1e-3) / model_femur
        for body in ("femur_r", "femur_l"):
            set_axis(body, 1, s, length_weight, "UpperLegLength")
        diag["model_femur_length_m"] = model_femur
        diag["subject_upper_leg_length_m"] = mp["UpperLegLength"] * 1e-3
    if "LowerLegLength" in mp:
        model_tibia = float(np.linalg.norm(pos["talus_r"] - pos["tibia_r"]))
        s = (mp["LowerLegLength"] * 1e-3) / model_tibia
        for body in ("tibia_r", "tibia_l"):
            set_axis(body, 1, s, length_weight, "LowerLegLength")
        diag["model_tibia_length_m"] = model_tibia
        diag["subject_lower_leg_length_m"] = mp["LowerLegLength"] * 1e-3
    if "FootLength" in mp:
        # Rajagopal has RCAL/RTOE marker offsets on calcn; this is a better model analogue
        # for Plug-in-Gait FootLength than the calcn->toes joint-origin distance.
        model_foot = float(np.linalg.norm(_marker_offset(spec, "RTOE") - _marker_offset(spec, "RCAL")))
        s = (mp["FootLength"] * 1e-3) / model_foot
        for body in ("calcn_r", "calcn_l"):
            set_axis(body, 0, s, width_weight, "FootLength")
        diag["model_foot_marker_length_m"] = model_foot
        diag["subject_foot_length_m"] = mp["FootLength"] * 1e-3

    if include_upper_body:
        if "UpperArmLength" in mp:
            model_upper_arm = float(np.linalg.norm(pos["ulna_r"] - pos["humerus_r"]))
            s = (mp["UpperArmLength"] * 1e-3) / model_upper_arm
            for body in ("humerus_r", "humerus_l"):
                set_axis(body, 1, s, width_weight, "UpperArmLength")
            diag["model_upper_arm_length_m"] = model_upper_arm
            diag["subject_upper_arm_length_m"] = mp["UpperArmLength"] * 1e-3
        if "LowerArmLength" in mp:
            model_lower_arm = float(np.linalg.norm(pos["radius_r"] - pos["ulna_r"]))
            if model_lower_arm > 1e-6:
                s = (mp["LowerArmLength"] * 1e-3) / model_lower_arm
                for body in ("ulna_r", "ulna_l"):
                    set_axis(body, 1, s, width_weight, "LowerArmLength")
                diag["model_lower_arm_length_m"] = model_lower_arm
                diag["subject_lower_arm_length_m"] = mp["LowerArmLength"] * 1e-3

    return target, weights, diag
