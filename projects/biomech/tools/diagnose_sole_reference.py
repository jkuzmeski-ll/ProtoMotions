# SPDX-License-Identifier: MIT
"""Decisive diagnostic: does ankle==0 (baked neutral) correspond to sole-horizontal?

Localizes the ~10 deg plantarflexion / "heel never contacts" bias by separating three
frames of reference:

  * the ANKLE DOF value (baked so it reads ~0 at the static standing pose),
  * the CALCN body's world orientation (its +y axis pitch off vertical), and
  * the anchor-derived SOLE plane normal (``_foot_axes(anchors).up``) pitch, both as an
    intrinsic tilt inside the calcn frame and as a world-space tilt per dynamic frame.

If the sole normal is tilted relative to calcn +y (intrinsic), then "ankle==0" and
"plantar sole horizontal" cannot coincide -- a frame-reference bug upstream of the
z-shift ground registration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_BIOMECH = Path(__file__).resolve().parents[1]
_CACHE = _BIOMECH / "docs" / "figures" / "_s001_ik_cache.npz"


def _signed_pitch(nrm_world: np.ndarray) -> float:
    """Signed sagittal pitch (deg) of a world normal off vertical (+Y up, +X fwd).

    Positive = normal tilted forward (toe-down / heel-up plantar surface).
    """
    # project onto the sagittal (X-Y) plane; angle from +Y toward +X
    return float(np.degrees(np.arctan2(nrm_world[0], nrm_world[1])))


def main() -> int:
    from biomech.contact.foot_geometry import calcn_anchors_from_spec, _foot_axes
    from biomech.skeleton.skeleton import WarpSkeleton

    cache = np.load(_CACHE, allow_pickle=True)
    spec = cache["spec_pickle"].item()
    poses = np.asarray(cache["poses"], dtype=np.float64)   # (F, 37) dynamic clip
    scales = np.asarray(cache["scales"], dtype=np.float64)  # (60,)
    dof = spec.dof_index_map()
    ai = dof["ankle_angle_r"]

    # --- intrinsic sole tilt inside the calcn_r frame (frame-invariant) ---
    anchors = calcn_anchors_from_spec(spec, "R", group_scales=scales)
    fwd, lat, up = _foot_axes(anchors)
    # angle of the sole 'up' off the calcn +y axis, signed in the calcn X-Y (sagittal) plane
    intrinsic_pitch = float(np.degrees(np.arctan2(up[0], up[1])))
    tilt_total = float(np.degrees(np.arccos(np.clip(up[1], -1.0, 1.0))))
    print("=== intrinsic sole plane vs calcn frame ===")
    print(f"anchor heel={anchors.heel}  mt5={anchors.mt5}  toe={anchors.toe}")
    print(f"sole up axis (calcn frame) = {up}")
    print(f"  total tilt off +y = {tilt_total:.2f} deg")
    print(f"  sagittal (toe-down +) intrinsic pitch = {intrinsic_pitch:.2f} deg")
    print("  (this is the fixed offset between 'calcn +y vertical' and 'sole flat')")

    # --- FK the dynamic clip (OpenSim Y-up) ---
    skel = WarpSkeleton(spec, device="cpu")
    world, _ = skel.forward(poses, scales)  # (F, B, 4, 4)
    bidx = {b.name: i for i, b in enumerate(spec.bodies)}
    ci = bidx["calcn_r"]
    Rc = np.asarray(world)[:, ci, :3, :3]  # (F,3,3)

    F = poses.shape[0]
    ankle = np.degrees(poses[:, ai])
    calcn_y_pitch = np.empty(F)  # pitch of calcn +y off vertical
    sole_pitch = np.empty(F)     # pitch of sole normal off vertical
    for f in range(F):
        calcn_y_world = Rc[f] @ np.array([0.0, 1.0, 0.0])
        sole_up_world = Rc[f] @ up
        calcn_y_pitch[f] = _signed_pitch(calcn_y_world)
        sole_pitch[f] = _signed_pitch(sole_up_world)

    print("\n=== per-frame (deg): ankle DOF | calcn+y pitch | sole-normal pitch ===")
    print(f"ankle_r:      min/mean/max = {ankle.min():.1f}/{ankle.mean():.1f}/{ankle.max():.1f}")
    print(f"calcn+y pitch: min/mean/max = {calcn_y_pitch.min():.1f}/{calcn_y_pitch.mean():.1f}/{calcn_y_pitch.max():.1f}")
    print(f"sole pitch:   min/mean/max = {sole_pitch.min():.1f}/{sole_pitch.mean():.1f}/{sole_pitch.max():.1f}")
    print("  (sole pitch = calcn+y pitch + intrinsic; positive = heel-up/toe-down)")

    # frame where the SOLE is flattest (closest to horizontal) and what ankle reads there
    f_flat = int(np.argmin(np.abs(sole_pitch)))
    print(f"\nsole flattest at frame {f_flat}: sole_pitch={sole_pitch[f_flat]:.2f} deg, "
          f"ankle={ankle[f_flat]:.2f} deg")
    print(f"  => when the SOLE is horizontal, the ankle DOF reads {ankle[f_flat]:.2f} deg "
          f"(should be ~0 if references were consistent)")

    # frame where the ankle reads ~0 and what the sole does there
    f_ankle0 = int(np.argmin(np.abs(ankle)))
    print(f"ankle~0 at frame {f_ankle0}: ankle={ankle[f_ankle0]:.2f} deg, "
          f"sole_pitch={sole_pitch[f_ankle0]:.2f} deg (should be ~0 if consistent)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
