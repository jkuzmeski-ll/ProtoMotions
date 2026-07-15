# SPDX-License-Identifier: MIT
#
# Windows-native, Warp-accelerated forward kinematics for an OpenSim skeleton,
# porting the value path of Nimble/DART's tree FK:
#   world(body) = world(parent) * ( T_parent * T_joint(q) * T_child^-1 )
# with anisotropic per-segment (group) scaling applied to the joint offset
# translations (Joint::setParentScale/setChildScale) and to marker local offsets
# (Skeleton::getMarkerWorldPositions), and joint transforms per DART type:
#   * CustomJoint / EulerJoint / EulerFreeJoint : Euler(order, flip)+translation
#     driven through coupling functions (SimmSpline for the gold-standard knee),
#   * RevoluteJoint (from OpenSim PinJoint) : Rodrigues about the joint Z axis,
#   * UniversalJoint : Rodrigues about X then Y,
#   * WeldJoint : identity.
#
# Reference C++: dart/dynamics/{Skeleton,BodyNode,Joint,CustomJoint,EulerJoint,
# EulerFreeJoint,RevoluteJoint,UniversalJoint}.cpp and dart/math/Geometry.cpp.
# The FK runs as a single Warp kernel batched over frames (float64 for fit parity),
# keeping it differentiable w.r.t. q (and group scales) via Warp autodiff. A plain
# NumPy reference (``fk_numpy``) mirrors it for validation.

"""Batched Warp FK for the OpenSim ``SkeletonSpec`` (M2b)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from biomech.osim.spec import SkeletonSpec
from biomech.skeleton import functions as F
from biomech.skeleton import spatial as S

# Joint kinds (must match the kernel switch).
JK_CUSTOM = 0
JK_REVOLUTE = 1
JK_UNIVERSAL = 2
JK_WELD = 3

_UNIT_Z = np.array([0.0, 0.0, 1.0])
_UNIT_X = np.array([1.0, 0.0, 0.0])
_UNIT_Y = np.array([0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# Host-side flattened topology
# ---------------------------------------------------------------------------


@dataclass
class _Topology:
    num_joints: int
    num_bodies: int
    num_markers: int
    num_dofs: int
    num_groups: int

    j_kind: np.ndarray  # (J,) int32
    j_parent_body: np.ndarray  # (J,) int32 (-1 root)
    j_dof_start: np.ndarray  # (J,) int32
    j_ndof: np.ndarray  # (J,) int32
    j_parent_group: np.ndarray  # (J,) int32 (-1 -> unit scale)
    j_child_group: np.ndarray  # (J,) int32
    T_parent: np.ndarray  # (J,4,4) float64 (unit-scale)
    T_child: np.ndarray  # (J,4,4) float64 (unit-scale)
    axis_order: np.ndarray  # (J,) int32
    flip: np.ndarray  # (J,3) float64
    slot_fid: np.ndarray  # (J,6) int32
    slot_driver: np.ndarray  # (J,6) int32 (-1 -> x=0)
    rev_axis: np.ndarray  # (J,3) float64
    uni_axis1: np.ndarray  # (J,3) float64
    uni_axis2: np.ndarray  # (J,3) float64

    m_body: np.ndarray  # (M,) int32
    m_offset: np.ndarray  # (M,3) float64
    m_group: np.ndarray  # (M,) int32

    table: F.FunctionTable


def _build_topology(spec: SkeletonSpec) -> _Topology:
    body_index = {b.name: i for i, b in enumerate(spec.bodies)}

    # body -> group index (scale_groups is a list of body-name lists)
    body_group = np.zeros(spec.num_bodies, dtype=np.int32)
    for gi, group in enumerate(spec.scale_groups):
        for name in group:
            body_group[body_index[name]] = gi

    J = spec.num_joints
    builder = F.FunctionTableBuilder()

    j_kind = np.zeros(J, dtype=np.int32)
    j_parent_body = np.full(J, -1, dtype=np.int32)
    j_dof_start = np.zeros(J, dtype=np.int32)
    j_ndof = np.zeros(J, dtype=np.int32)
    j_parent_group = np.full(J, -1, dtype=np.int32)
    j_child_group = np.zeros(J, dtype=np.int32)
    T_parent = np.tile(np.eye(4), (J, 1, 1))
    T_child = np.tile(np.eye(4), (J, 1, 1))
    axis_order = np.full(J, -1, dtype=np.int32)
    flip = np.ones((J, 3), dtype=np.float64)
    slot_fid = np.zeros((J, 6), dtype=np.int32)
    slot_driver = np.full((J, 6), -1, dtype=np.int32)
    rev_axis = np.tile(_UNIT_Z, (J, 1))
    uni_axis1 = np.tile(_UNIT_X, (J, 1))
    uni_axis2 = np.tile(_UNIT_Y, (J, 1))

    dof = 0
    # A constant-zero function id, reused for empty slots.
    zero_fid = builder.add(_const0())

    for j, joint in enumerate(spec.joints):
        child_idx = body_index[joint.child_body]
        assert child_idx == j, "joints must be parallel to bodies (tree order)"
        j_dof_start[j] = dof
        j_ndof[j] = joint.num_dofs
        dof += joint.num_dofs

        j_child_group[j] = body_group[child_idx]
        if joint.parent_body is not None:
            pidx = body_index[joint.parent_body]
            j_parent_body[j] = pidx
            j_parent_group[j] = body_group[pidx]

        T_parent[j] = joint.T_parent
        T_child[j] = joint.T_child

        # default: fill all 6 slots with the shared constant-zero function
        for s in range(6):
            slot_fid[j, s] = zero_fid
            slot_driver[j, s] = -1

        if joint.joint_class == "PinJoint":
            j_kind[j] = JK_REVOLUTE
            rev_axis[j] = _UNIT_Z
        elif joint.joint_class == "UniversalJoint":
            j_kind[j] = JK_UNIVERSAL
            uni_axis1[j] = _UNIT_X
            uni_axis2[j] = _UNIT_Y
        elif joint.joint_class == "WeldJoint":
            j_kind[j] = JK_WELD
        elif joint.joint_class == "CustomJoint":
            j_kind[j] = JK_CUSTOM
            axis_order[j] = S.axis_order_code(joint.axis_order)
            flip[j] = joint.flip_axis_map
            local = {c.name: k for k, c in enumerate(joint.coordinates)}

            rotations = [a for a in joint.transform_axes if a.kind == "rotation"]
            translations = [
                a for a in joint.transform_axes if a.kind == "translation"
            ]
            assert len(rotations) == 3, joint.name
            # rotation slots 0,1,2 in document order (CustomJoint::getEulerPositions)
            for s in range(3):
                a = rotations[s]
                slot_fid[j, s] = builder.add(a.function)
                slot_driver[j, s] = local.get(a.coordinate, -1)
            # translation slots 3,4,5 mapped by physical axis (createCustomJoint)
            for a in translations:
                comp = int(np.argmax(np.abs(a.axis)))  # X->0 Y->1 Z->2
                s = 3 + comp
                slot_fid[j, s] = builder.add(a.function)
                slot_driver[j, s] = local.get(a.coordinate, -1)
        else:
            raise ValueError(f"unsupported joint class {joint.joint_class}")

    m_body = np.array([body_index[m.body] for m in spec.markers], dtype=np.int32)
    m_offset = np.array([m.offset for m in spec.markers], dtype=np.float64)
    m_group = np.array(
        [body_group[body_index[m.body]] for m in spec.markers], dtype=np.int32
    )

    return _Topology(
        num_joints=J,
        num_bodies=spec.num_bodies,
        num_markers=len(spec.markers),
        num_dofs=spec.num_dofs,
        num_groups=len(spec.scale_groups),
        j_kind=j_kind,
        j_parent_body=j_parent_body,
        j_dof_start=j_dof_start,
        j_ndof=j_ndof,
        j_parent_group=j_parent_group,
        j_child_group=j_child_group,
        T_parent=T_parent,
        T_child=T_child,
        axis_order=axis_order,
        flip=flip,
        slot_fid=slot_fid,
        slot_driver=slot_driver,
        rev_axis=rev_axis,
        uni_axis1=uni_axis1,
        uni_axis2=uni_axis2,
        m_body=m_body,
        m_offset=m_offset,
        m_group=m_group,
        table=builder.build(),
    )


def _const0():
    from biomech.osim.spec import ConstantFunctionSpec

    return ConstantFunctionSpec(0.0)


# ---------------------------------------------------------------------------
# NumPy reference FK (single pose)
# ---------------------------------------------------------------------------


def fk_numpy(spec: SkeletonSpec, q: np.ndarray, group_scales: np.ndarray | None = None):
    """Reference forward kinematics for one pose.

    Returns ``(body_transforms, marker_positions)`` where ``body_transforms`` is a
    dict ``{body_name: (4,4)}`` and ``marker_positions`` a dict ``{name: (3,)}``,
    all in the model's native OpenSim (Y-up, meters) frame.
    """
    q = np.asarray(q, dtype=np.float64).ravel()
    G = len(spec.scale_groups)
    if group_scales is None:
        group_scales = np.ones(3 * G, dtype=np.float64)
    scales = np.asarray(group_scales, dtype=np.float64).reshape(G, 3)

    body_index = {b.name: i for i, b in enumerate(spec.bodies)}
    body_group = {}
    for gi, group in enumerate(spec.scale_groups):
        for name in group:
            body_group[name] = gi

    def body_scale(name):
        return scales[body_group[name]]

    world: dict[str, np.ndarray] = {}
    dof = 0
    for joint in spec.joints:
        ndof = joint.num_dofs
        q_local = q[dof : dof + ndof]
        dof += ndof

        Tp = joint.T_parent.copy()
        if joint.parent_body is not None:
            Tp[:3, 3] *= body_scale(joint.parent_body)
        Tc = joint.T_child.copy()
        Tc[:3, 3] *= body_scale(joint.child_body)

        Tj = _joint_transform_np(joint, q_local)
        Trel = Tp @ Tj @ S.se3_inverse_np(Tc)

        if joint.parent_body is None:
            world[joint.child_body] = Trel
        else:
            world[joint.child_body] = world[joint.parent_body] @ Trel

    markers = {}
    for m in spec.markers:
        T = world[m.body]
        local = body_scale(m.body) * m.offset
        markers[m.name] = T[:3, :3] @ local + T[:3, 3]

    return world, markers


def _joint_transform_np(joint, q_local: np.ndarray) -> np.ndarray:
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
    # custom family
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


# ---------------------------------------------------------------------------
# Warp kernel
# ---------------------------------------------------------------------------


@wp.func
def _group_scale(group_scales: wp.array(dtype=wp.float64), g: wp.int32) -> wp.vec3d:
    if g < 0:
        return wp.vec3d(wp.float64(1.0), wp.float64(1.0), wp.float64(1.0))
    return wp.vec3d(
        group_scales[3 * g], group_scales[3 * g + 1], group_scales[3 * g + 2]
    )


@wp.func
def _joint_trel(
    f: wp.int32,
    j: wp.int32,
    pdof: wp.int32,          # global DOF index to perturb (-1 = none)
    peps: wp.float64,        # perturbation amount added to q[.,pdof]
    q: wp.array2d(dtype=wp.float64),
    group_scales: wp.array(dtype=wp.float64),
    j_kind: wp.array(dtype=wp.int32),
    j_dof_start: wp.array(dtype=wp.int32),
    j_parent_group: wp.array(dtype=wp.int32),
    j_child_group: wp.array(dtype=wp.int32),
    T_parent: wp.array(dtype=wp.mat44d),
    T_child: wp.array(dtype=wp.mat44d),
    axis_order: wp.array(dtype=wp.int32),
    flip: wp.array(dtype=wp.vec3d),
    slot_fid: wp.array2d(dtype=wp.int32),
    slot_driver: wp.array2d(dtype=wp.int32),
    rev_axis: wp.array(dtype=wp.vec3d),
    uni_axis1: wp.array(dtype=wp.vec3d),
    uni_axis2: wp.array(dtype=wp.vec3d),
    fn_type: wp.array(dtype=wp.int32),
    fn_p0: wp.array(dtype=wp.float64),
    fn_p1: wp.array(dtype=wp.float64),
    fn_kstart: wp.array(dtype=wp.int32),
    fn_kcount: wp.array(dtype=wp.int32),
    fn_x: wp.array(dtype=wp.float64),
    fn_y: wp.array(dtype=wp.float64),
    fn_b: wp.array(dtype=wp.float64),
    fn_c: wp.array(dtype=wp.float64),
    fn_d: wp.array(dtype=wp.float64),
) -> wp.mat44d:
    """Local joint transform ``Trel = Tp * Tj(q) * Tc^-1`` for joint ``j`` at frame ``f``.

    If ``pdof >= 0`` the single generalized coordinate with that global index is
    perturbed by ``peps`` (used by the finite-difference Jacobian kernels). This is the
    exact per-joint value path shared by the base FK kernel and the FD Jacobian kernel so
    the two can never drift apart.
    """
    dof0 = j_dof_start[j]
    kind = j_kind[j]

    R = wp.identity(n=3, dtype=wp.float64)
    t = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))

    if kind == 0:  # custom family
        fl = flip[j]
        e = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
        for s in range(3):
            drv = slot_driver[j, s]
            x = wp.float64(0.0)
            if drv >= 0:
                x = q[f, dof0 + drv]
                if (dof0 + drv) == pdof:
                    x = x + peps
            val = F.eval_function(
                slot_fid[j, s], x, fn_type, fn_p0, fn_p1, fn_kstart,
                fn_kcount, fn_x, fn_y, fn_b, fn_c, fn_d,
            )
            e[s] = fl[s] * val
        R = S.euler_to_matrix(axis_order[j], e)
        for s in range(3):
            drv = slot_driver[j, 3 + s]
            x = wp.float64(0.0)
            if drv >= 0:
                x = q[f, dof0 + drv]
                if (dof0 + drv) == pdof:
                    x = x + peps
            t[s] = F.eval_function(
                slot_fid[j, 3 + s], x, fn_type, fn_p0, fn_p1, fn_kstart,
                fn_kcount, fn_x, fn_y, fn_b, fn_c, fn_d,
            )
    elif kind == 1:  # revolute
        x = q[f, dof0]
        if dof0 == pdof:
            x = x + peps
        R = S.rodrigues(rev_axis[j], x)
    elif kind == 2:  # universal
        x0 = q[f, dof0]
        if dof0 == pdof:
            x0 = x0 + peps
        x1 = q[f, dof0 + 1]
        if (dof0 + 1) == pdof:
            x1 = x1 + peps
        R = S.rodrigues(uni_axis1[j], x0) * S.rodrigues(uni_axis2[j], x1)
    # kind == 3 weld -> identity

    Tj = S.make_transform(R, t)
    Tp = S.scale_translation(T_parent[j], _group_scale(group_scales, j_parent_group[j]))
    Tc = S.scale_translation(T_child[j], _group_scale(group_scales, j_child_group[j]))
    return Tp * Tj * S.se3_inverse(Tc)


@wp.kernel
def _fk_kernel(
    q: wp.array2d(dtype=wp.float64),  # (F, ndof)
    group_scales: wp.array(dtype=wp.float64),  # (3G,)
    j_kind: wp.array(dtype=wp.int32),
    j_parent_body: wp.array(dtype=wp.int32),
    j_dof_start: wp.array(dtype=wp.int32),
    j_parent_group: wp.array(dtype=wp.int32),
    j_child_group: wp.array(dtype=wp.int32),
    T_parent: wp.array(dtype=wp.mat44d),
    T_child: wp.array(dtype=wp.mat44d),
    axis_order: wp.array(dtype=wp.int32),
    flip: wp.array(dtype=wp.vec3d),
    slot_fid: wp.array2d(dtype=wp.int32),
    slot_driver: wp.array2d(dtype=wp.int32),
    rev_axis: wp.array(dtype=wp.vec3d),
    uni_axis1: wp.array(dtype=wp.vec3d),
    uni_axis2: wp.array(dtype=wp.vec3d),
    # function table
    fn_type: wp.array(dtype=wp.int32),
    fn_p0: wp.array(dtype=wp.float64),
    fn_p1: wp.array(dtype=wp.float64),
    fn_kstart: wp.array(dtype=wp.int32),
    fn_kcount: wp.array(dtype=wp.int32),
    fn_x: wp.array(dtype=wp.float64),
    fn_y: wp.array(dtype=wp.float64),
    fn_b: wp.array(dtype=wp.float64),
    fn_c: wp.array(dtype=wp.float64),
    fn_d: wp.array(dtype=wp.float64),
    num_bodies: wp.int32,
    # outputs
    world: wp.array2d(dtype=wp.mat44d),  # (F, B)
):
    f = wp.tid()
    for b in range(num_bodies):
        Trel = _joint_trel(
            f, b, -1, wp.float64(0.0), q, group_scales, j_kind, j_dof_start,
            j_parent_group, j_child_group, T_parent, T_child, axis_order, flip,
            slot_fid, slot_driver, rev_axis, uni_axis1, uni_axis2, fn_type, fn_p0,
            fn_p1, fn_kstart, fn_kcount, fn_x, fn_y, fn_b, fn_c, fn_d,
        )
        pb = j_parent_body[b]
        if pb < 0:
            world[f, b] = Trel
        else:
            world[f, b] = world[f, pb] * Trel


@wp.kernel
def _fk_pert_kernel(
    q: wp.array2d(dtype=wp.float64),  # (F, ndof)
    group_scales: wp.array(dtype=wp.float64),
    j_kind: wp.array(dtype=wp.int32),
    j_parent_body: wp.array(dtype=wp.int32),
    j_dof_start: wp.array(dtype=wp.int32),
    j_parent_group: wp.array(dtype=wp.int32),
    j_child_group: wp.array(dtype=wp.int32),
    T_parent: wp.array(dtype=wp.mat44d),
    T_child: wp.array(dtype=wp.mat44d),
    axis_order: wp.array(dtype=wp.int32),
    flip: wp.array(dtype=wp.vec3d),
    slot_fid: wp.array2d(dtype=wp.int32),
    slot_driver: wp.array2d(dtype=wp.int32),
    rev_axis: wp.array(dtype=wp.vec3d),
    uni_axis1: wp.array(dtype=wp.vec3d),
    uni_axis2: wp.array(dtype=wp.vec3d),
    fn_type: wp.array(dtype=wp.int32),
    fn_p0: wp.array(dtype=wp.float64),
    fn_p1: wp.array(dtype=wp.float64),
    fn_kstart: wp.array(dtype=wp.int32),
    fn_kcount: wp.array(dtype=wp.int32),
    fn_x: wp.array(dtype=wp.float64),
    fn_y: wp.array(dtype=wp.float64),
    fn_b: wp.array(dtype=wp.float64),
    fn_c: wp.array(dtype=wp.float64),
    fn_d: wp.array(dtype=wp.float64),
    eps: wp.float64,
    num_bodies: wp.int32,
    # output: perturbed world transforms (F, ndof, 2, B); s=0 -> +eps, s=1 -> -eps
    world_pert: wp.array4d(dtype=wp.mat44d),
):
    f, i, s = wp.tid()
    peps = eps
    if s == 1:
        peps = -eps
    for b in range(num_bodies):
        Trel = _joint_trel(
            f, b, i, peps, q, group_scales, j_kind, j_dof_start,
            j_parent_group, j_child_group, T_parent, T_child, axis_order, flip,
            slot_fid, slot_driver, rev_axis, uni_axis1, uni_axis2, fn_type, fn_p0,
            fn_p1, fn_kstart, fn_kcount, fn_x, fn_y, fn_b, fn_c, fn_d,
        )
        pb = j_parent_body[b]
        if pb < 0:
            world_pert[f, i, s, b] = Trel
        else:
            world_pert[f, i, s, b] = world_pert[f, i, s, pb] * Trel


@wp.kernel
def _jac_central_diff_kernel(
    world_pert: wp.array4d(dtype=wp.mat44d),  # (F, ndof, 2, B)
    group_scales: wp.array(dtype=wp.float64),
    m_body: wp.array(dtype=wp.int32),
    m_offset: wp.array(dtype=wp.vec3d),
    m_group: wp.array(dtype=wp.int32),
    eps: wp.float64,
    jac: wp.array4d(dtype=wp.float64),  # (F, M, 3, ndof)
):
    f, m, i = wp.tid()
    s = _group_scale(group_scales, m_group[m])
    off = m_offset[m]
    local = wp.vec3d(off[0] * s[0], off[1] * s[1], off[2] * s[2])
    b = m_body[m]
    pp = S.transform_point(world_pert[f, i, 0, b], local)
    pm = S.transform_point(world_pert[f, i, 1, b], local)
    inv2e = wp.float64(1.0) / (wp.float64(2.0) * eps)
    jac[f, m, 0, i] = (pp[0] - pm[0]) * inv2e
    jac[f, m, 1, i] = (pp[1] - pm[1]) * inv2e
    jac[f, m, 2, i] = (pp[2] - pm[2]) * inv2e


@wp.kernel
def _marker_kernel(
    world: wp.array2d(dtype=wp.mat44d),  # (F, B)
    group_scales: wp.array(dtype=wp.float64),
    m_body: wp.array(dtype=wp.int32),
    m_offset: wp.array(dtype=wp.vec3d),
    m_group: wp.array(dtype=wp.int32),
    markers: wp.array2d(dtype=wp.vec3d),  # (F, M)
):
    f, m = wp.tid()
    s = _group_scale(group_scales, m_group[m])
    off = m_offset[m]
    local = wp.vec3d(off[0] * s[0], off[1] * s[1], off[2] * s[2])
    markers[f, m] = S.transform_point(world[f, m_body[m]], local)


@wp.kernel
def _marker_sqerr_kernel(
    markers: wp.array2d(dtype=wp.vec3d),  # (F, M)
    obs: wp.array2d(dtype=wp.vec3d),  # (F, M)
    weights: wp.array2d(dtype=wp.float64),  # (F, M) = per-marker weight * mask
    loss: wp.array(dtype=wp.float64),  # (1,) accumulator
):
    f, m = wp.tid()
    d = markers[f, m] - obs[f, m]
    w = weights[f, m]
    e = w * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    wp.atomic_add(loss, 0, e)


# ---------------------------------------------------------------------------
# WarpSkeleton wrapper
# ---------------------------------------------------------------------------


class WarpSkeleton:
    """Batched, differentiable Warp FK for a parsed :class:`SkeletonSpec`."""

    def __init__(self, spec: SkeletonSpec, device: str | None = None):
        self.spec = spec
        self.device = device or ("cuda" if wp.get_cuda_device_count() > 0 else "cpu")
        self.topo = _build_topology(spec)
        self._upload()

    def _upload(self):
        t = self.topo
        d = self.device
        f64 = wp.float64
        self.d_j_kind = wp.array(t.j_kind, dtype=wp.int32, device=d)
        self.d_j_parent_body = wp.array(t.j_parent_body, dtype=wp.int32, device=d)
        self.d_j_dof_start = wp.array(t.j_dof_start, dtype=wp.int32, device=d)
        self.d_j_parent_group = wp.array(t.j_parent_group, dtype=wp.int32, device=d)
        self.d_j_child_group = wp.array(t.j_child_group, dtype=wp.int32, device=d)
        self.d_T_parent = wp.array(t.T_parent, dtype=wp.mat44d, device=d)
        self.d_T_child = wp.array(t.T_child, dtype=wp.mat44d, device=d)
        self.d_axis_order = wp.array(t.axis_order, dtype=wp.int32, device=d)
        self.d_flip = wp.array(t.flip, dtype=wp.vec3d, device=d)
        self.d_slot_fid = wp.array(t.slot_fid, dtype=wp.int32, device=d)
        self.d_slot_driver = wp.array(t.slot_driver, dtype=wp.int32, device=d)
        self.d_rev_axis = wp.array(t.rev_axis, dtype=wp.vec3d, device=d)
        self.d_uni_axis1 = wp.array(t.uni_axis1, dtype=wp.vec3d, device=d)
        self.d_uni_axis2 = wp.array(t.uni_axis2, dtype=wp.vec3d, device=d)
        self.d_m_body = wp.array(t.m_body, dtype=wp.int32, device=d)
        self.d_m_offset = wp.array(t.m_offset, dtype=wp.vec3d, device=d)
        self.d_m_group = wp.array(t.m_group, dtype=wp.int32, device=d)
        tb = t.table
        self.d_fn_type = wp.array(tb.fn_type, dtype=wp.int32, device=d)
        self.d_fn_p0 = wp.array(tb.fn_p0, dtype=f64, device=d)
        self.d_fn_p1 = wp.array(tb.fn_p1, dtype=f64, device=d)
        self.d_fn_kstart = wp.array(tb.fn_kstart, dtype=wp.int32, device=d)
        self.d_fn_kcount = wp.array(tb.fn_kcount, dtype=wp.int32, device=d)
        self.d_fn_x = wp.array(tb.fn_x, dtype=f64, device=d)
        self.d_fn_y = wp.array(tb.fn_y, dtype=f64, device=d)
        self.d_fn_b = wp.array(tb.fn_b, dtype=f64, device=d)
        self.d_fn_c = wp.array(tb.fn_c, dtype=f64, device=d)
        self.d_fn_d = wp.array(tb.fn_d, dtype=f64, device=d)

    def forward(self, q: np.ndarray, group_scales: np.ndarray | None = None):
        """Batched FK. ``q`` is ``(F, ndof)`` (or ``(ndof,)``), one scale set.

        Returns ``(world, markers)`` numpy arrays of shape ``(F, B, 4, 4)`` and
        ``(F, M, 3)`` in the model's native OpenSim frame.
        """
        _, _, d_world, d_markers = self._run(q, group_scales, requires_grad=False)
        return d_world.numpy(), d_markers.numpy()

    def _run_wp(self, d_q, d_scales):
        """Device-resident FK from existing Warp arrays (no host round trip).

        Used by the Torch/Warp IK path: Torch owns ``q`` and scales on CUDA, Warp wraps
        those tensors and launches FK/marker kernels, and Torch consumes the resulting
        device arrays via DLPack.
        """
        Fr = int(d_q.shape[0])
        t = self.topo
        d = self.device
        d_world = wp.zeros((Fr, t.num_bodies), dtype=wp.mat44d, device=d)
        d_markers = wp.zeros((Fr, t.num_markers), dtype=wp.vec3d, device=d)
        wp.launch(
            _fk_kernel,
            dim=Fr,
            inputs=[
                d_q, d_scales, self.d_j_kind, self.d_j_parent_body,
                self.d_j_dof_start, self.d_j_parent_group, self.d_j_child_group,
                self.d_T_parent, self.d_T_child, self.d_axis_order, self.d_flip,
                self.d_slot_fid, self.d_slot_driver, self.d_rev_axis,
                self.d_uni_axis1, self.d_uni_axis2, self.d_fn_type, self.d_fn_p0,
                self.d_fn_p1, self.d_fn_kstart, self.d_fn_kcount, self.d_fn_x,
                self.d_fn_y, self.d_fn_b, self.d_fn_c, self.d_fn_d, t.num_bodies,
            ],
            outputs=[d_world],
            device=d,
        )
        wp.launch(
            _marker_kernel,
            dim=(Fr, t.num_markers),
            inputs=[d_world, d_scales, self.d_m_body, self.d_m_offset, self.d_m_group],
            outputs=[d_markers],
            device=d,
        )
        return d_world, d_markers

    def _run(self, q, group_scales, requires_grad, tape=None):
        """Allocate device arrays and launch the FK + marker kernels.

        Returns ``(d_q, d_scales, d_world, d_markers)``. If ``tape`` is given the
        launches are recorded on it (for Warp autodiff).
        """
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        Fr = q.shape[0]
        t = self.topo
        G = t.num_groups
        if group_scales is None:
            group_scales = np.ones(3 * G, dtype=np.float64)
        group_scales = np.asarray(group_scales, dtype=np.float64).ravel()

        d = self.device
        d_q = wp.array(q, dtype=wp.float64, device=d, requires_grad=requires_grad)
        d_scales = wp.array(
            group_scales, dtype=wp.float64, device=d, requires_grad=requires_grad
        )
        d_world = wp.zeros(
            (Fr, t.num_bodies), dtype=wp.mat44d, device=d, requires_grad=requires_grad
        )
        d_markers = wp.zeros(
            (Fr, t.num_markers), dtype=wp.vec3d, device=d, requires_grad=requires_grad
        )

        def _launch():
            wp.launch(
                _fk_kernel,
                dim=Fr,
                inputs=[
                    d_q, d_scales, self.d_j_kind, self.d_j_parent_body,
                    self.d_j_dof_start, self.d_j_parent_group, self.d_j_child_group,
                    self.d_T_parent, self.d_T_child, self.d_axis_order, self.d_flip,
                    self.d_slot_fid, self.d_slot_driver, self.d_rev_axis,
                    self.d_uni_axis1, self.d_uni_axis2, self.d_fn_type, self.d_fn_p0,
                    self.d_fn_p1, self.d_fn_kstart, self.d_fn_kcount, self.d_fn_x,
                    self.d_fn_y, self.d_fn_b, self.d_fn_c, self.d_fn_d, t.num_bodies,
                ],
                outputs=[d_world],
                device=d,
            )
            wp.launch(
                _marker_kernel,
                dim=(Fr, t.num_markers),
                inputs=[
                    d_world, d_scales, self.d_m_body, self.d_m_offset, self.d_m_group
                ],
                outputs=[d_markers],
                device=d,
            )

        if tape is not None:
            with tape:
                _launch()
        else:
            _launch()
        return d_q, d_scales, d_world, d_markers

    def marker_jacobian_wrt_q(
        self, q: np.ndarray, group_scales: np.ndarray | None = None
    ) -> np.ndarray:
        """Marker Jacobian ``d(marker world pos)/dq`` (Warp autodiff).

        ``q`` may be a single pose ``(ndof,)`` or a batch ``(F, ndof)``. Returns
        ``(M, 3, ndof)`` for a single pose, or ``(F, M, 3, ndof)`` for a batch. The
        reverse-mode sweep costs ``3*M`` backward passes **regardless of F** (each
        seeded output component back-props for every frame at once), so it is cheap
        to Jacobian a whole trial. Flows through the SimmSpline coupling exactly.
        """
        single = np.asarray(q).ndim == 1
        q2d = np.atleast_2d(np.asarray(q, dtype=np.float64))
        Fr = q2d.shape[0]
        t = self.topo
        M = t.num_markers
        tape = wp.Tape()
        d_q, _, _, d_markers = self._run(
            q2d, group_scales, requires_grad=True, tape=tape
        )
        jac = np.zeros((Fr, M, 3, t.num_dofs), dtype=np.float64)
        seed = np.zeros((Fr, M, 3), dtype=np.float64)
        for mi in range(M):
            for c in range(3):
                seed[:] = 0.0
                seed[:, mi, c] = 1.0
                d_markers.grad = wp.array(seed, dtype=wp.vec3d, device=self.device)
                tape.backward()
                jac[:, mi, c, :] = d_q.grad.numpy()
                tape.zero()
        return jac[0] if single else jac

    def _jac_buffers(self, Fr: int):
        """Cached device scratch for the FD Jacobian, keyed by frame count."""
        cache = getattr(self, "_jacbuf", None)
        if cache is None:
            cache = {}
            self._jacbuf = cache
        if Fr not in cache:
            t = self.topo
            d = self.device
            world_pert = wp.zeros(
                (Fr, t.num_dofs, 2, t.num_bodies), dtype=wp.mat44d, device=d
            )
            jac = wp.zeros((Fr, t.num_markers, 3, t.num_dofs), dtype=wp.float64, device=d)
            cache[Fr] = (world_pert, jac)
        return cache[Fr]

    def marker_jacobian_wrt_q_fd_wp(self, d_q, d_scales, eps: float = 1e-6):
        """Device-resident fast marker Jacobian by central differences.

        ``d_q`` is a Warp view of a CUDA tensor with shape ``(F, ndof)`` and
        ``d_scales`` has shape ``(3G,)``. Returns a Warp array ``(F,M,3,ndof)``;
        nothing is copied to the host.
        """
        Fr = int(d_q.shape[0])
        t = self.topo
        d = self.device
        world_pert, jac = self._jac_buffers(Fr)
        epsd = wp.float64(eps)
        wp.launch(
            _fk_pert_kernel,
            dim=(Fr, t.num_dofs, 2),
            inputs=[
                d_q, d_scales, self.d_j_kind, self.d_j_parent_body,
                self.d_j_dof_start, self.d_j_parent_group, self.d_j_child_group,
                self.d_T_parent, self.d_T_child, self.d_axis_order, self.d_flip,
                self.d_slot_fid, self.d_slot_driver, self.d_rev_axis,
                self.d_uni_axis1, self.d_uni_axis2, self.d_fn_type, self.d_fn_p0,
                self.d_fn_p1, self.d_fn_kstart, self.d_fn_kcount, self.d_fn_x,
                self.d_fn_y, self.d_fn_b, self.d_fn_c, self.d_fn_d, epsd, t.num_bodies,
            ],
            outputs=[world_pert],
            device=d,
        )
        wp.launch(
            _jac_central_diff_kernel,
            dim=(Fr, t.num_markers, t.num_dofs),
            inputs=[world_pert, d_scales, self.d_m_body, self.d_m_offset,
                    self.d_m_group, epsd],
            outputs=[jac],
            device=d,
        )
        return jac

    def marker_jacobian_wrt_q_fd(
        self,
        q: np.ndarray,
        group_scales: np.ndarray | None = None,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Fast marker Jacobian ``d(marker world pos)/dq`` by GPU central differences.

        Same result as :meth:`marker_jacobian_wrt_q` (autodiff) but computed in **two
        kernel launches** instead of ``3*M`` sequential reverse-mode passes: one launch
        perturbs every DOF by ``+/-eps`` in parallel over ``(frame, dof, sign)`` and
        writes the perturbed body transforms, the second differences the marker
        positions. Nothing round-trips to the host between launches. This is the hot
        path for the batched IK; it is ~1-2 orders of magnitude faster than the autodiff
        Jacobian and agrees with it to ``~1e-8`` (central difference, float64).

        ``q`` may be ``(ndof,)`` or ``(F, ndof)``; returns ``(M, 3, ndof)`` or
        ``(F, M, 3, ndof)`` respectively.
        """
        single = np.asarray(q).ndim == 1
        q2d = np.atleast_2d(np.asarray(q, dtype=np.float64))
        Fr = q2d.shape[0]
        t = self.topo
        G = t.num_groups
        d = self.device
        if group_scales is None:
            group_scales = np.ones(3 * G, dtype=np.float64)
        gs = np.asarray(group_scales, dtype=np.float64).ravel()
        d_q = wp.array(q2d, dtype=wp.float64, device=d)
        d_scales = wp.array(gs, dtype=wp.float64, device=d)
        world_pert, jac = self._jac_buffers(Fr)
        epsd = wp.float64(eps)
        wp.launch(
            _fk_pert_kernel,
            dim=(Fr, t.num_dofs, 2),
            inputs=[
                d_q, d_scales, self.d_j_kind, self.d_j_parent_body,
                self.d_j_dof_start, self.d_j_parent_group, self.d_j_child_group,
                self.d_T_parent, self.d_T_child, self.d_axis_order, self.d_flip,
                self.d_slot_fid, self.d_slot_driver, self.d_rev_axis,
                self.d_uni_axis1, self.d_uni_axis2, self.d_fn_type, self.d_fn_p0,
                self.d_fn_p1, self.d_fn_kstart, self.d_fn_kcount, self.d_fn_x,
                self.d_fn_y, self.d_fn_b, self.d_fn_c, self.d_fn_d, epsd, t.num_bodies,
            ],
            outputs=[world_pert],
            device=d,
        )
        wp.launch(
            _jac_central_diff_kernel,
            dim=(Fr, t.num_markers, t.num_dofs),
            inputs=[world_pert, d_scales, self.d_m_body, self.d_m_offset,
                    self.d_m_group, epsd],
            outputs=[jac],
            device=d,
        )
        out = jac.numpy()
        return out[0] if single else out

    def marker_loss_grads(
        self,
        q: np.ndarray,
        group_scales: np.ndarray,
        offsets: np.ndarray,
        observed: np.ndarray,
        weights: np.ndarray,
    ):
        """Weighted marker loss and its gradients w.r.t. group scales and offsets.

        This is the outer-loop objective of the bilevel marker fit (M2d): with the
        poses ``q`` held **fixed** (valid at the inner-IK optimum by the envelope
        theorem), it evaluates

            L = sum_{f,m} weights[f,m] * || marker(q_f, scales, offset_m) - obs ||^2

        and returns ``(L, dL/d(group_scales) (3G,), dL/d(offsets) (M,3))`` via a single
        Warp autodiff backward pass. The scale gradient flows through BOTH the joint
        offset scaling in FK and the marker offset scaling; the offset gradient is the
        per-marker block. ``weights`` is ``(F, M)`` (per-marker weight times visibility
        mask); ``observed`` is ``(F, M, 3)`` with NaNs allowed where masked to zero.
        """
        q2d = np.atleast_2d(np.asarray(q, dtype=np.float64))
        Fr = q2d.shape[0]
        t = self.topo
        M = t.num_markers
        G = t.num_groups
        d = self.device
        gs = np.asarray(group_scales, dtype=np.float64).ravel()
        if gs.size == 0:
            gs = np.ones(3 * G, dtype=np.float64)
        off = np.asarray(offsets, dtype=np.float64).reshape(M, 3)
        obs = np.asarray(observed, dtype=np.float64).reshape(Fr, M, 3)
        obs = np.where(np.isfinite(obs), obs, 0.0)
        w = np.asarray(weights, dtype=np.float64).reshape(Fr, M)

        d_q = wp.array(q2d, dtype=wp.float64, device=d, requires_grad=False)
        d_scales = wp.array(gs, dtype=wp.float64, device=d, requires_grad=True)
        d_offsets = wp.array(off, dtype=wp.vec3d, device=d, requires_grad=True)
        d_world = wp.zeros(
            (Fr, t.num_bodies), dtype=wp.mat44d, device=d, requires_grad=True
        )
        d_markers = wp.zeros(
            (Fr, M), dtype=wp.vec3d, device=d, requires_grad=True
        )
        d_obs = wp.array(obs, dtype=wp.vec3d, device=d, requires_grad=False)
        d_w = wp.array(w, dtype=wp.float64, device=d, requires_grad=False)
        d_loss = wp.zeros(1, dtype=wp.float64, device=d, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(
                _fk_kernel,
                dim=Fr,
                inputs=[
                    d_q, d_scales, self.d_j_kind, self.d_j_parent_body,
                    self.d_j_dof_start, self.d_j_parent_group, self.d_j_child_group,
                    self.d_T_parent, self.d_T_child, self.d_axis_order, self.d_flip,
                    self.d_slot_fid, self.d_slot_driver, self.d_rev_axis,
                    self.d_uni_axis1, self.d_uni_axis2, self.d_fn_type, self.d_fn_p0,
                    self.d_fn_p1, self.d_fn_kstart, self.d_fn_kcount, self.d_fn_x,
                    self.d_fn_y, self.d_fn_b, self.d_fn_c, self.d_fn_d, t.num_bodies,
                ],
                outputs=[d_world],
                device=d,
            )
            wp.launch(
                _marker_kernel,
                dim=(Fr, M),
                inputs=[d_world, d_scales, self.d_m_body, d_offsets, self.d_m_group],
                outputs=[d_markers],
                device=d,
            )
            wp.launch(
                _marker_sqerr_kernel,
                dim=(Fr, M),
                inputs=[d_markers, d_obs, d_w],
                outputs=[d_loss],
                device=d,
            )
        loss = float(d_loss.numpy()[0])
        d_loss.grad = wp.array(np.ones(1), dtype=wp.float64, device=d)
        tape.backward()
        return loss, d_scales.grad.numpy(), d_offsets.grad.numpy()

    def set_marker_offsets(self, offsets: np.ndarray) -> None:
        """Replace the marker local offsets used by FK / IK (for the M2d offset fit).

        Updates both the device array consumed by the Warp kernels and the host copy
        used by ``fk_numpy``, so subsequent ``forward`` / ``marker_jacobian_wrt_q`` /
        IK calls reflect the new offsets. ``offsets`` is ``(M, 3)``.
        """
        off = np.asarray(offsets, dtype=np.float64).reshape(self.topo.num_markers, 3)
        self.topo.m_offset = off
        self.d_m_offset = wp.array(off, dtype=wp.vec3d, device=self.device)

    def marker_offsets(self) -> np.ndarray:
        """Current marker local offsets ``(M, 3)``."""
        return np.asarray(self.topo.m_offset, dtype=np.float64).reshape(-1, 3)

    def body_names(self) -> list[str]:
        return [b.name for b in self.spec.bodies]

    def marker_names(self) -> list[str]:
        return [m.name for m in self.spec.markers]
