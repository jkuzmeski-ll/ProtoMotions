# SPDX-License-Identifier: MIT
#
# Milestone M3 — export a fitted OpenSim ``SkeletonSpec`` (+ group scales) to an
# **MJCF** that the Newton MuJoCo solver (``newton.ModelBuilder.add_mjcf``) loads and
# whose forward kinematics reproduce the Warp/Nimble skeleton FK to machine precision.
#
# Mapping (derived + Newton-verified, see tests/test_mjcf_export.py):
#
#   DART relative transform for a joint:  T_rel(q) = Tp · J(q) · Tc⁻¹
#   with  Tp = scaled location/orientation in parent,  Tc = scaled in child,
#   and world[child] = world[parent] · T_rel(q).
#
#   MuJoCo composes a child body as  world[child] = world[parent] · M · Π Gᵢ(qᵢ)
#   where M is the body rest frame (``pos``/``quat``) and each Gᵢ is a joint's local
#   transform (hinge = rotation about anchor pᵢ / axis aᵢ ; slide = translation q·aᵢ),
#   with pᵢ / aᵢ fixed in the child body frame.
#
#   Choosing  M = Tp · J(0) · Tc⁻¹  and, per DOF slot,
#     hinge:  aᵢ = R_child · axisₛ ,  pᵢ = t_child ,  qpos = fₛ(coord)
#     slide:  aᵢ = R_child · e_comp ,             qpos = fₛ(coord)
#   gives  M · Π Gᵢ = Tp · J(0) · Tc⁻¹ · Tc · J(0)⁻¹ · J(q) · Tc⁻¹ = Tp · J(q) · Tc⁻¹.
#   (Requires J(0) = I, which holds for the target models; asserted at export.)
#   Translation slots are emitted **before** rotation slots because DART's
#   CustomJoint transform is [[R, t]] = Trans(t) · Rot(R).
#
# OpenSim ``CustomJoint`` SimmSpline coupling (the walker knee) cannot be expressed in
# MuJoCo joints. It is exported as independent + dependent joints tied by an
# ``<equality><joint>`` quartic ``polycoef`` fitted to the spline; the fit residual is
# reported. A ``coupled_knee="hinge"`` mode instead drops the coupled DOFs (a pure
# flexion hinge) and reports the dropped rotation/translation magnitude.

"""Fitted ``SkeletonSpec`` + group scales -> MJCF for the Newton MuJoCo solver (M3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from biomech.export.bone_geometry import MESHDIR_REL, BoneMesh
from biomech.osim.spec import (
    ConstantFunctionSpec,
    JointSpec,
    LinearFunctionSpec,
    SkeletonSpec,
    TransformAxisSpec,
)

_UNIT = {0: np.array([1.0, 0.0, 0.0]), 1: np.array([0.0, 1.0, 0.0]), 2: np.array([0.0, 0.0, 1.0])}
_UNIT_Z = _UNIT[2]
_UNIT_X = _UNIT[0]
_UNIT_Y = _UNIT[1]


# ---------------------------------------------------------------------------
# One MuJoCo joint (a single DOF) derived from a DART joint slot
# ---------------------------------------------------------------------------


@dataclass
class _MjJoint:
    name: str
    kind: str  # "hinge" | "slide"
    axis: np.ndarray  # (3,) in child-body-local frame
    anchor: np.ndarray  # (3,) child-body-local frame (hinge only; slide ignores)
    coord: Optional[str]  # driving DART coordinate
    independent: bool  # True: qpos == coord value ; False: qpos == func(coord)
    func: object  # coupling function (value(x))
    range: Optional[tuple[float, float]]  # DART coordinate limits (independent only)


@dataclass
class MjcfExportResult:
    xml: str
    body_names: list[str]  # ALL MJCF bodies in Newton body_q row order (incl. dummies)
    real_body_names: list[str]  # OpenSim bodies in spec order
    real_body_row: dict  # OpenSim body name -> row index in body_names / Newton body_q
    joint_names: list[str]  # per-DOF MuJoCo joint names, in qpos order (excl. free root)
    qpos_dim: int  # length of Newton joint_q (free root = 7)
    coupled_report: dict  # per coupled joint: fit/dropped diagnostics


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _is_identity_linear(fn) -> bool:
    return (
        isinstance(fn, LinearFunctionSpec)
        and abs(fn.slope - 1.0) < 1e-12
        and abs(fn.intercept) < 1e-12
    )


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion in MuJoCo (w, x, y, z) order."""
    m = np.asarray(R, dtype=np.float64)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _fmt(vals) -> str:
    return " ".join(repr(float(v)) for v in np.asarray(vals).ravel())


class _ScaleMap:
    def __init__(self, spec: SkeletonSpec, group_scales: Optional[np.ndarray]):
        G = len(spec.scale_groups)
        if group_scales is None:
            group_scales = np.ones(3 * G, dtype=np.float64)
        self.scales = np.asarray(group_scales, dtype=np.float64).reshape(G, 3)
        self.body_group: dict[str, int] = {}
        for gi, group in enumerate(spec.scale_groups):
            for name in group:
                self.body_group[name] = gi

    def of(self, body: Optional[str]) -> np.ndarray:
        if body is None:
            return np.ones(3, dtype=np.float64)
        return self.scales[self.body_group[body]]


def _scaled_frames(joint: JointSpec, sm: _ScaleMap) -> tuple[np.ndarray, np.ndarray]:
    """Return (Tp, Tc) with translations scaled by parent/child group scales."""
    Tp = joint.T_parent.copy()
    if joint.parent_body is not None:
        Tp[:3, 3] = Tp[:3, 3] * sm.of(joint.parent_body)
    Tc = joint.T_child.copy()
    Tc[:3, 3] = Tc[:3, 3] * sm.of(joint.child_body)
    return Tp, Tc


def _joint_transform(joint: JointSpec, q_local: np.ndarray) -> np.ndarray:
    """DART J(q) for a joint (mirror of skeleton.fk_numpy's _joint_transform_np)."""
    from biomech.skeleton import spatial as S

    T = np.eye(4, dtype=np.float64)
    if joint.joint_class == "PinJoint":
        T[:3, :3] = S.rodrigues_np(_UNIT_Z, float(q_local[0]))
        return T
    if joint.joint_class == "UniversalJoint":
        T[:3, :3] = S.rodrigues_np(_UNIT_X, float(q_local[0])) @ S.rodrigues_np(
            _UNIT_Y, float(q_local[1])
        )
        return T
    if joint.joint_class == "WeldJoint":
        return T
    local = {c.name: k for k, c in enumerate(joint.coordinates)}
    rotations = [a for a in joint.transform_axes if a.kind == "rotation"]
    translations = [a for a in joint.transform_axes if a.kind == "translation"]
    e = np.zeros(3, dtype=np.float64)
    for s in range(3):
        a = rotations[s]
        x = q_local[local[a.coordinate]] if a.coordinate is not None else 0.0
        e[s] = joint.flip_axis_map[s] * a.function.value(float(x))
    R = S.euler_to_matrix_np(S.axis_order_code(joint.axis_order), e)
    t = np.zeros(3, dtype=np.float64)
    for a in translations:
        comp = int(np.argmax(np.abs(a.axis)))
        x = q_local[local[a.coordinate]] if a.coordinate is not None else 0.0
        t[comp] = a.function.value(float(x))
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _joint_dofs(
    joint: JointSpec, Rc: np.ndarray, tc: np.ndarray, coupled_knee: str
) -> list[_MjJoint]:
    """MuJoCo DOFs for a joint, in emit order (slides before hinges).

    Locked DART coordinates (``limit_lo == limit_hi``) contribute no DOF: they are
    rigid at their default 0 (J(0) = I), so dropping them is exact and matches DART's
    locked-DOF behaviour. A joint with all coordinates locked becomes a weld.
    """
    dofs: list[_MjJoint] = []
    lim = {c.name: (c.limit_lo, c.limit_hi) for c in joint.coordinates}
    locked = {c.name for c in joint.coordinates if c.locked}

    if joint.joint_class == "PinJoint":
        c = joint.coordinates[0].name
        if c in locked:
            return dofs
        dofs.append(
            _MjJoint(f"{joint.name}__{c}", "hinge", Rc @ _UNIT_Z, tc, c, True, None, lim[c])
        )
        return dofs
    if joint.joint_class == "UniversalJoint":
        for c, ax in zip(
            (joint.coordinates[0].name, joint.coordinates[1].name), (_UNIT_X, _UNIT_Y)
        ):
            if c in locked:
                continue
            dofs.append(
                _MjJoint(f"{joint.name}__{c}", "hinge", Rc @ ax, tc, c, True, None, lim[c])
            )
        return dofs
    if joint.joint_class == "WeldJoint":
        return dofs

    # CustomJoint family (root free joint handled by the caller)
    rotations = [a for a in joint.transform_axes if a.kind == "rotation"]
    translations = [a for a in joint.transform_axes if a.kind == "translation"]

    def emit(a: TransformAxisSpec, kind: str):
        if isinstance(a.function, ConstantFunctionSpec) or a.coordinate is None:
            return  # constant slot -> baked into M (J(0)); no DOF
        if a.coordinate in locked:
            return  # locked coordinate -> rigid; baked into M
        indep = _is_identity_linear(a.function)
        if not indep and coupled_knee == "hinge":
            return  # drop coupled DOFs in hinge mode
        if kind == "hinge":
            axis = Rc @ np.asarray(a.axis, dtype=np.float64)
        else:
            comp = int(np.argmax(np.abs(a.axis)))
            axis = Rc @ _UNIT[comp]
        rng = lim[a.coordinate] if indep else None
        dofs.append(
            _MjJoint(f"{joint.name}__{a.name}", kind, axis, tc, a.coordinate, indep, a.function, rng)
        )

    for a in translations:
        emit(a, "slide")
    for a in rotations:
        emit(a, "hinge")
    return dofs


def _assert_zero_config(joint: JointSpec):
    """Verify J(0) == I so the M-body / delta split is exact."""
    q0 = np.zeros(joint.num_dofs, dtype=np.float64)
    J0 = _joint_transform(joint, q0)
    if not np.allclose(J0, np.eye(4), atol=1e-9):
        raise ValueError(
            f"joint {joint.name!r} has J(0) != I; MJCF export assumes zero rest config"
        )


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def export_mjcf(
    spec: SkeletonSpec,
    group_scales: Optional[np.ndarray] = None,
    coupled_knee: str = "coupled",
    model_name: Optional[str] = None,
    marker_sites: bool = True,
    visual_geoms: bool = False,
    subject_mass: Optional[float] = None,
    bone_meshes: Optional[dict[str, list[BoneMesh]]] = None,
    meshdir: str = MESHDIR_REL,
) -> MjcfExportResult:
    """Build an MJCF string for the Newton MuJoCo solver from a fitted skeleton.

    ``coupled_knee``:
      - ``"coupled"`` (default): full fidelity — coupled SimmSpline DOFs emitted as
        extra joints tied by ``<equality><joint>`` quartic ``polycoef`` fits.
      - ``"hinge"``: coupled DOFs dropped (pure flexion hinge), for a lean model.

    ``visual_geoms``: if True, add non-colliding capsule/sphere geoms along each segment
    (parent origin -> each child attachment, sphere at leaves) so the skeleton is visible
    in a renderer. Geoms carry ``density=0`` + ``contype/conaffinity=0`` and the bodies
    keep their explicit ``<inertial>``, so FK/dynamics are unchanged.

    ``bone_meshes``: optional ``body name -> [BoneMesh, ...]`` (e.g. from
    :func:`biomech.export.bone_geometry.default_bone_geometry`). When given, each body is
    drawn with its actual bone mesh(es) instead of the capsule placeholder: a ``<asset>``
    block of ``<mesh>`` entries (scaled by the subject's per-body group scale) plus
    non-colliding ``type="mesh"`` geoms. ``meshdir`` is the MJCF-relative mesh directory
    written to ``<compiler meshdir=...>``. Bodies without a bone mesh fall back to the
    capsule/sphere placeholder when ``visual_geoms`` is set. Meshes are visual-only
    (``density=0`` + ``contype/conaffinity=0``), so FK/dynamics are unchanged.

    ``subject_mass``: if given (kg), rescale the model's segment masses so the whole-body
    mass equals the subject's measured mass (e.g. from the Plug-in-Gait ``.mp``), preserving
    the model's mass *distribution*. Each body's inertia is scaled consistently by the mass
    ratio and the (anisotropic) group geometric scales (products exactly, diagonal terms via
    the ``I_xx = int(y^2+z^2)dm`` split), so the exported robot is anthropometrically the
    subject: subject limb lengths (``group_scales``) *and* subject mass/inertia.
    """
    from biomech.skeleton import spatial as S

    if coupled_knee not in ("coupled", "hinge"):
        raise ValueError("coupled_knee must be 'coupled' or 'hinge'")
    sm = _ScaleMap(spec, group_scales)

    mass_ratio = 1.0
    if subject_mass is not None:
        model_mass = float(sum(max(b.mass, 0.0) for b in spec.bodies))
        if model_mass > 0.0 and subject_mass > 0.0:
            mass_ratio = float(subject_mass) / model_mass

    # joint whose child is body i
    joint_of_body = {j.child_body: j for j in spec.joints}
    children: dict[Optional[str], list[str]] = {}
    for j in spec.joints:
        children.setdefault(j.parent_body, []).append(j.child_body)

    for j in spec.joints:
        _assert_zero_config(j)

    # emit per-body joints & collect equality/report info
    body_dofs: dict[str, list[_MjJoint]] = {}
    coord_indep_name: dict[tuple[str, str], str] = {}  # (joint,coord) -> indep mj name
    equalities: list[tuple[str, str, np.ndarray]] = []
    report: dict = {}
    joint_names: list[str] = []

    for b in spec.bodies:
        j = joint_of_body[b.name]
        if j.parent_body is None:
            body_dofs[b.name] = []  # free joint emitted separately
            continue
        _, Tc = _scaled_frames(j, sm)
        Rc, tc = Tc[:3, :3], Tc[:3, 3]
        dofs = _joint_dofs(j, Rc, tc, coupled_knee)
        body_dofs[b.name] = dofs
        for d in dofs:
            if d.independent:
                coord_indep_name[(j.name, d.coord)] = d.name

    # per-child offset from its (real) parent body at zero config -> bone endpoints
    child_offset: dict[str, np.ndarray] = {}
    if visual_geoms:
        for b in spec.bodies:
            j = joint_of_body[b.name]
            if j.parent_body is None:
                continue
            Tp, Tc = _scaled_frames(j, sm)
            M = Tp @ _joint_transform(j, np.zeros(j.num_dofs)) @ S.se3_inverse_np(Tc)
            child_offset[b.name] = M[:3, 3]

    # second pass: equality constraints + diagnostics for dependent DOFs
    for j in spec.joints:
        dep = [
            d for d in body_dofs.get(j.child_body, []) if not d.independent
        ]
        if not dep and coupled_knee == "hinge":
            # in hinge mode, report what was dropped for coupled joints
            dropped = _dropped_report(j)
            if dropped:
                report[j.name] = dropped
            continue
        for d in dep:
            indep = coord_indep_name.get((j.name, d.coord))
            lo, hi = _coord_limits(j, d.coord)
            coef, resid = _fit_polycoef(d.func, lo, hi)
            equalities.append((d.name, indep, coef))
            report.setdefault(j.name, {})[d.name] = {
                "type": d.kind,
                "coord": d.coord,
                "coord_range": [lo, hi],
                "value_range": [float(d.func.value(lo)), float(d.func.value(hi))],
                "polycoef": coef.tolist(),
                "max_abs_fit_residual": float(resid),
            }

    # ---- build XML ----
    name = model_name or spec.name or "biomech"

    # Restrict bone meshes to bodies actually present in this skeleton, and pre-scale
    # each mesh by its body's per-body group scale (generic display scale is folded in).
    body_meshes: dict[str, list[BoneMesh]] = {}
    mesh_assets: list[tuple[str, str, np.ndarray]] = []  # (name, file, scale3)
    if bone_meshes:
        seen: set[str] = set()
        for b in spec.bodies:
            ms = bone_meshes.get(b.name)
            if not ms:
                continue
            body_meshes[b.name] = ms
            cs = sm.of(b.name)
            for m in ms:
                if m.stem in seen:
                    continue
                seen.add(m.stem)
                mesh_assets.append((m.stem, f"{m.stem}.stl", np.asarray(m.scale) * cs))

    lines: list[str] = []
    lines.append(f'<mujoco model="{name}">')
    compiler = '  <compiler angle="radian" autolimits="true" balanceinertia="true"'
    if mesh_assets:
        compiler += f' meshdir="{meshdir}"'
    compiler += "/>"
    lines.append(compiler)
    if mesh_assets:
        lines.append("  <asset>")
        for mname, mfile, mscale in mesh_assets:
            lines.append(
                f'    <mesh name="{mname}" file="{mfile}" scale="{_fmt(mscale)}"/>'
            )
        lines.append("  </asset>")
    lines.append("  <worldbody>")

    body_order: list[str] = []
    real_body_row: dict = {}

    def _emit_geoms(body_name: str, indent: str):
        """Visual-only capsules (parent->child bones) + a sphere at leaf bodies."""
        if body_name in body_meshes:
            return  # bone mesh replaces the capsule placeholder for this body
        if not visual_geoms:
            return
        style = (
            'group="0" contype="0" conaffinity="0" density="0" '
            'rgba="0.78 0.80 0.86 1"'
        )
        drew = False
        for child in children.get(body_name, []):
            off = child_offset.get(child)
            if off is None:
                continue
            length = float(np.linalg.norm(off))
            if length < 5e-3:
                continue
            r = float(min(0.045, max(0.018, 0.12 * length)))
            lines.append(
                f'{indent}<geom type="capsule" fromto="0 0 0 {_fmt(off)}" '
                f'size="{repr(r)}" {style}/>'
            )
            drew = True
        if not drew:
            lines.append(f'{indent}<geom type="sphere" size="0.035" pos="0 0 0" {style}/>')

    def _emit_mesh_geoms(body_name: str, indent: str):
        """Visual-only bone mesh geom(s) for a body (identity placement for Rajagopal)."""
        from biomech.skeleton import spatial as S

        style = (
            'group="1" contype="0" conaffinity="0" density="0" '
            'rgba="0.90 0.88 0.80 1"'
        )
        for m in body_meshes.get(body_name, []):
            attrs = f'type="mesh" mesh="{m.stem}"'
            if not m.is_identity_placement:
                rx, ry, rz = (float(v) for v in m.transform[:3])
                tx, ty, tz = (float(v) for v in m.transform[3:6])
                R = (
                    S.rodrigues_np(_UNIT_X, rx)
                    @ S.rodrigues_np(_UNIT_Y, ry)
                    @ S.rodrigues_np(_UNIT_Z, rz)
                )
                attrs += (
                    f' pos="{_fmt([tx, ty, tz])}" '
                    f'quat="{_fmt(_rotmat_to_quat_wxyz(R))}"'
                )
            lines.append(f"{indent}<geom {attrs} {style}/>")

    def _emit_joint(d: _MjJoint, indent: str):
        joint_names.append(d.name)
        attrs = (
            f'name="{d.name}" type="{d.kind}" '
            f'axis="{_fmt(d.axis)}" pos="{_fmt(d.anchor)}"'
        )
        if d.range is not None:
            attrs += f' range="{_fmt(d.range)}"'
        else:
            attrs += ' limited="false"'
        lines.append(f"{indent}<joint {attrs}/>")

    def _emit_payload(body_name: str, indent: str):
        """inertial + marker sites on the real body."""
        bspec = spec.body(body_name)
        cs = sm.of(body_name)
        com = bspec.com * cs
        mass = max(bspec.mass * mass_ratio, 1e-9)
        # Scale inertia to the subject: mass ratio x anisotropic geometry. Products
        # (Ixy,Ixz,Iyz) scale exactly by the pairwise axis factors; diagonal terms use
        # the I_xx = int(y^2+z^2)dm split (mean of the two orthogonal axis factors^2).
        sx, sy, sz = float(cs[0]), float(cs[1]), float(cs[2])
        ixx, iyy, izz, ixy, ixz, iyz = (float(v) for v in bspec.inertia)
        inertia = np.array([
            ixx * mass_ratio * 0.5 * (sy * sy + sz * sz),
            iyy * mass_ratio * 0.5 * (sx * sx + sz * sz),
            izz * mass_ratio * 0.5 * (sx * sx + sy * sy),
            ixy * mass_ratio * sx * sy,
            ixz * mass_ratio * sx * sz,
            iyz * mass_ratio * sy * sz,
        ], dtype=np.float64)
        lines.append(
            f'{indent}<inertial pos="{_fmt(com)}" mass="{repr(float(mass))}" '
            f'fullinertia="{_fmt(inertia)}"/>'
        )
        if marker_sites:
            for m in spec.markers:
                if m.body == body_name:
                    off = m.offset * cs
                    lines.append(
                        f'{indent}<site name="mk_{m.name}" pos="{_fmt(off)}" '
                        f'size="0.01" group="4"/>'
                    )

    def write_body(body_name: str, indent: str):
        """Emit the real body, splitting multi-DOF joints into massless dummy chains.

        A dummy body per extra DOF prevents Newton from merging joints into a single
        compound joint (which changes rotation semantics and drops joint names needed
        by equality constraints). The chain's FK equals the single-body-multi-joint
        MJCF, which is bit-exact vs the Warp/DART skeleton under MuJoCo.
        """
        from biomech.skeleton import spatial as S

        j = joint_of_body[body_name]
        if j.parent_body is None:
            lines.append(f'{indent}<body name="{body_name}" pos="0 0 0" quat="1 0 0 0">')
            lines.append(f'{indent}  <freejoint name="{j.name}"/>')
            real_body_row[body_name] = len(body_order)
            body_order.append(body_name)
            _emit_payload(body_name, indent + "  ")
            _emit_geoms(body_name, indent + "  ")
            _emit_mesh_geoms(body_name, indent + "  ")
            for child in children.get(body_name, []):
                write_body(child, indent + "  ")
            lines.append(f"{indent}</body>")
            return

        Tp, Tc = _scaled_frames(j, sm)
        M = Tp @ _joint_transform(j, np.zeros(j.num_dofs)) @ S.se3_inverse_np(Tc)
        Mpos, Mquat = M[:3, 3], _rotmat_to_quat_wxyz(M[:3, :3])
        dofs = body_dofs[body_name]

        # chain node frames: first carries M, rest identity; last node is the real body
        n_dummy = max(len(dofs) - 1, 0)
        cur = indent
        for i in range(n_dummy):
            dn = f"{body_name}__q{i}"
            pos = Mpos if i == 0 else np.zeros(3)
            quat = Mquat if i == 0 else np.array([1.0, 0.0, 0.0, 0.0])
            lines.append(f'{cur}<body name="{dn}" pos="{_fmt(pos)}" quat="{_fmt(quat)}">')
            _emit_joint(dofs[i], cur + "  ")
            lines.append(
                f'{cur}  <inertial pos="0 0 0" mass="1e-06" '
                f'diaginertia="1e-09 1e-09 1e-09"/>'
            )
            body_order.append(dn)
            cur = cur + "  "

        # real body
        if n_dummy == 0:
            pos, quat = Mpos, Mquat
        else:
            pos, quat = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
        lines.append(f'{cur}<body name="{body_name}" pos="{_fmt(pos)}" quat="{_fmt(quat)}">')
        if dofs:
            _emit_joint(dofs[-1], cur + "  ")
        real_body_row[body_name] = len(body_order)
        body_order.append(body_name)
        _emit_payload(body_name, cur + "  ")
        _emit_geoms(body_name, cur + "  ")
        _emit_mesh_geoms(body_name, cur + "  ")
        for child in children.get(body_name, []):
            write_body(child, cur + "  ")
        lines.append(f"{cur}</body>")
        for _ in range(n_dummy):
            cur = cur[:-2]
            lines.append(f"{cur}</body>")

    roots = children.get(None, [])
    for r in roots:
        write_body(r, "    ")

    lines.append("  </worldbody>")

    if equalities:
        lines.append("  <equality>")
        for dep_name, indep_name, coef in equalities:
            if indep_name is None:
                continue
            lines.append(
                f'    <joint joint1="{dep_name}" joint2="{indep_name}" '
                f'polycoef="{_fmt(coef)}"/>'
            )
        lines.append("  </equality>")

    lines.append("</mujoco>")
    xml = "\n".join(lines) + "\n"

    # free root contributes 7 qpos coords; every other mj joint contributes 1
    qpos_dim = 7 * sum(1 for j in spec.joints if j.parent_body is None) + len(joint_names)

    return MjcfExportResult(
        xml=xml,
        body_names=body_order,
        real_body_names=[b.name for b in spec.bodies],
        real_body_row=real_body_row,
        joint_names=joint_names,
        qpos_dim=qpos_dim,
        coupled_report=report,
    )


def _coord_limits(joint: JointSpec, coord: str) -> tuple[float, float]:
    for c in joint.coordinates:
        if c.name == coord:
            return c.limit_lo, c.limit_hi
    raise KeyError(coord)


def _fit_polycoef(func, lo: float, hi: float, n: int = 200) -> tuple[np.ndarray, float]:
    """Quartic (MuJoCo polycoef) fit of a coupling function over [lo, hi].

    MuJoCo joint equality: q_dep = c0 + c1 x + c2 x² + c3 x³ + c4 x⁴, x = q_indep
    (references are 0). Returns (coef[5], max_abs_residual).
    """
    xs = np.linspace(lo, hi, n)
    ys = np.array([func.value(float(x)) for x in xs], dtype=np.float64)
    deg = min(4, n - 1)
    p = np.polyfit(xs, ys, deg)  # highest power first
    coef = np.zeros(5, dtype=np.float64)
    coef[: deg + 1] = p[::-1]  # -> ascending powers c0..c4
    resid = float(np.max(np.abs(np.polyval(p, xs) - ys)))
    return coef, resid


def _dropped_report(joint: JointSpec) -> dict:
    """For hinge mode: magnitude of coupled rotation/translation dropped."""
    if joint.joint_class not in ("CustomJoint",):
        return {}
    out: dict = {}
    for a in joint.transform_axes:
        if a.coordinate is None or _is_identity_linear(a.function):
            continue
        if isinstance(a.function, ConstantFunctionSpec):
            continue
        lo, hi = _coord_limits(joint, a.coordinate)
        xs = np.linspace(lo, hi, 200)
        ys = np.array([a.function.value(float(x)) for x in xs])
        out[a.name] = {
            "kind": a.kind,
            "coord": a.coordinate,
            "max_abs_value": float(np.max(np.abs(ys))),
        }
    return out


def write_mjcf(spec: SkeletonSpec, path, **kwargs) -> MjcfExportResult:
    res = export_mjcf(spec, **kwargs)
    with open(path, "w") as f:
        f.write(res.xml)
    return res


# ---------------------------------------------------------------------------
# DART q -> Newton/MuJoCo qpos (for validation & driving the exported model)
# ---------------------------------------------------------------------------


def dart_q_to_mjcf_qpos(
    spec: SkeletonSpec,
    q: np.ndarray,
    group_scales: Optional[np.ndarray] = None,
    coupled_knee: str = "coupled",
) -> np.ndarray:
    """Map a DART pose ``q`` to the exported model's Newton ``joint_q`` array.

    Coord order matches ``export_mjcf``: bodies in tree order; root -> free joint
    ``[px,py,pz, qx,qy,qz,qw]`` (Newton uses xyzw); every other joint -> one coord in
    emit order (slides before hinges). Coupled DOFs get their spline/coupling value.
    """
    from biomech.skeleton.skeleton import fk_numpy

    sm = _ScaleMap(spec, group_scales)
    q = np.asarray(q, dtype=np.float64).ravel()
    joint_of_body = {j.child_body: j for j in spec.joints}

    # dof offset per joint (joints parallel to bodies in tree order)
    dof_start: dict[str, int] = {}
    d = 0
    for j in spec.joints:
        dof_start[j.child_body] = d
        d += j.num_dofs

    world, _ = fk_numpy(spec, q, group_scales)

    out: list[float] = []
    for b in spec.bodies:
        j = joint_of_body[b.name]
        d0 = dof_start[b.name]
        q_local = q[d0 : d0 + j.num_dofs]
        cvals = {c.name: float(q_local[k]) for k, c in enumerate(j.coordinates)}
        if j.parent_body is None:
            T = world[b.name]
            quat = _rotmat_to_quat_wxyz(T[:3, :3])  # wxyz
            out.extend([T[0, 3], T[1, 3], T[2, 3]])
            out.extend([quat[1], quat[2], quat[3], quat[0]])  # -> xyzw for Newton
            continue
        _, Tc = _scaled_frames(j, sm)
        Rc, tc = Tc[:3, :3], Tc[:3, 3]
        for dof in _joint_dofs(j, Rc, tc, coupled_knee):
            x = cvals[dof.coord]
            if dof.independent:
                out.append(x)
            else:
                out.append(float(dof.func.value(x)))
    return np.array(out, dtype=np.float64)
