# SPDX-License-Identifier: MIT
#
# Windows-native port of the SO3/SE3 pieces of Nimble's ``dart/math/Geometry.cpp``
# that the OpenSim skeleton FK needs: the four Euler rotation orders used by
# ``EulerJoint``/``EulerFreeJoint``/``CustomJoint`` and the ``expAngular`` (Rodrigues)
# map used by ``RevoluteJoint``/``UniversalJoint``. Both a plain-NumPy reference
# (for parity debugging) and Warp device functions (float64, for the batched FK
# kernel) are provided and kept bit-identical.
#
# The matrix element assignments are copied verbatim from Nimble/OpenSim:
#   eulerXYZToMatrix  Geometry.cpp:1767
#   eulerXZYToMatrix  Geometry.cpp:2122
#   eulerZXYToMatrix  Geometry.cpp:2565
#   eulerZYXToMatrix  Geometry.cpp:2880
# and ``EulerJoint::convertToRotation`` applies them to ``positions ⊙ flipAxisMap``
# (EulerJoint.cpp:242).

"""SO3/SE3 device + host math for the biomech skeleton FK (M2b)."""

from __future__ import annotations

import numpy as np
import warp as wp

# Euler axis-order codes (must match ``skeleton.skeleton`` builder + Warp kernel).
AXIS_ORDER_XYZ = 0
AXIS_ORDER_ZYX = 1
AXIS_ORDER_ZXY = 2
AXIS_ORDER_XZY = 3

_AXIS_ORDER_FROM_STR = {
    "XYZ": AXIS_ORDER_XYZ,
    "ZYX": AXIS_ORDER_ZYX,
    "ZXY": AXIS_ORDER_ZXY,
    "XZY": AXIS_ORDER_XZY,
}


def axis_order_code(name: str) -> int:
    return _AXIS_ORDER_FROM_STR[name]


# ---------------------------------------------------------------------------
# NumPy reference (exact ports)
# ---------------------------------------------------------------------------


def euler_xyz_to_matrix_np(a: np.ndarray) -> np.ndarray:
    cx, sx = np.cos(a[0]), np.sin(a[0])
    cy, sy = np.cos(a[1]), np.sin(a[1])
    cz, sz = np.cos(a[2]), np.sin(a[2])
    r = np.empty((3, 3), dtype=np.float64)
    r[0, 0] = cy * cz
    r[1, 0] = cx * sz + cz * sx * sy
    r[2, 0] = sx * sz - cx * cz * sy
    r[0, 1] = -cy * sz
    r[1, 1] = cx * cz - sx * sy * sz
    r[2, 1] = cz * sx + cx * sy * sz
    r[0, 2] = sy
    r[1, 2] = -cy * sx
    r[2, 2] = cx * cy
    return r


def euler_xzy_to_matrix_np(a: np.ndarray) -> np.ndarray:
    cx, sx = np.cos(a[0]), np.sin(a[0])
    cz, sz = np.cos(a[1]), np.sin(a[1])
    cy, sy = np.cos(a[2]), np.sin(a[2])
    r = np.empty((3, 3), dtype=np.float64)
    r[0, 0] = cy * cz
    r[1, 0] = sx * sy + cx * cy * sz
    r[2, 0] = -cx * sy + cy * sx * sz
    r[0, 1] = -sz
    r[1, 1] = cx * cz
    r[2, 1] = cz * sx
    r[0, 2] = cz * sy
    r[1, 2] = -cy * sx + cx * sy * sz
    r[2, 2] = cx * cy + sx * sy * sz
    return r


def euler_zxy_to_matrix_np(a: np.ndarray) -> np.ndarray:
    cz, sz = np.cos(a[0]), np.sin(a[0])
    cx, sx = np.cos(a[1]), np.sin(a[1])
    cy, sy = np.cos(a[2]), np.sin(a[2])
    r = np.empty((3, 3), dtype=np.float64)
    r[0, 0] = cy * cz - sx * sy * sz
    r[1, 0] = cz * sx * sy + cy * sz
    r[2, 0] = -cx * sy
    r[0, 1] = -cx * sz
    r[1, 1] = cx * cz
    r[2, 1] = sx
    r[0, 2] = cz * sy + cy * sx * sz
    r[1, 2] = -cy * cz * sx + sy * sz
    r[2, 2] = cx * cy
    return r


def euler_zyx_to_matrix_np(a: np.ndarray) -> np.ndarray:
    cz, sz = np.cos(a[0]), np.sin(a[0])
    cy, sy = np.cos(a[1]), np.sin(a[1])
    cx, sx = np.cos(a[2]), np.sin(a[2])
    r = np.empty((3, 3), dtype=np.float64)
    r[0, 0] = cz * cy
    r[1, 0] = sz * cy
    r[2, 0] = -sy
    r[0, 1] = cz * sy * sx - sz * cx
    r[1, 1] = sz * sy * sx + cz * cx
    r[2, 1] = cy * sx
    r[0, 2] = cz * sy * cx + sz * sx
    r[1, 2] = sz * sy * cx - cz * sx
    r[2, 2] = cy * cx
    return r


def euler_to_matrix_np(order: int, e: np.ndarray) -> np.ndarray:
    """``e`` is the (already flip-multiplied) Euler vector, in transform-axis order."""
    if order == AXIS_ORDER_XYZ:
        return euler_xyz_to_matrix_np(e)
    if order == AXIS_ORDER_XZY:
        return euler_xzy_to_matrix_np(e)
    if order == AXIS_ORDER_ZXY:
        return euler_zxy_to_matrix_np(e)
    if order == AXIS_ORDER_ZYX:
        return euler_zyx_to_matrix_np(e)
    raise ValueError(f"bad axis order code {order}")


def rodrigues_np(axis: np.ndarray, angle: float) -> np.ndarray:
    """Port of ``math::expAngular(axis * angle)`` rotation part (Rodrigues)."""
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n == 0.0:
        return np.eye(3, dtype=np.float64)
    k = axis / n
    theta = angle * n
    kx, ky, kz = k
    K = np.array(
        [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float64
    )
    s, c = np.sin(theta), np.cos(theta)
    return np.eye(3, dtype=np.float64) + s * K + (1.0 - c) * (K @ K)


def se3_inverse_np(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


# ---------------------------------------------------------------------------
# Warp device functions (float64)
# ---------------------------------------------------------------------------


@wp.func
def euler_to_matrix(order: wp.int32, e: wp.vec3d) -> wp.mat33d:
    a0 = e[0]
    a1 = e[1]
    a2 = e[2]
    r = wp.mat33d(
        wp.float64(0.0), wp.float64(0.0), wp.float64(0.0),
        wp.float64(0.0), wp.float64(0.0), wp.float64(0.0),
        wp.float64(0.0), wp.float64(0.0), wp.float64(0.0),
    )
    if order == 0:  # XYZ : a0=x a1=y a2=z
        cx = wp.cos(a0); sx = wp.sin(a0)
        cy = wp.cos(a1); sy = wp.sin(a1)
        cz = wp.cos(a2); sz = wp.sin(a2)
        r[0, 0] = cy * cz
        r[1, 0] = cx * sz + cz * sx * sy
        r[2, 0] = sx * sz - cx * cz * sy
        r[0, 1] = -cy * sz
        r[1, 1] = cx * cz - sx * sy * sz
        r[2, 1] = cz * sx + cx * sy * sz
        r[0, 2] = sy
        r[1, 2] = -cy * sx
        r[2, 2] = cx * cy
    elif order == 3:  # XZY : a0=x a1=z a2=y
        cx = wp.cos(a0); sx = wp.sin(a0)
        cz = wp.cos(a1); sz = wp.sin(a1)
        cy = wp.cos(a2); sy = wp.sin(a2)
        r[0, 0] = cy * cz
        r[1, 0] = sx * sy + cx * cy * sz
        r[2, 0] = -cx * sy + cy * sx * sz
        r[0, 1] = -sz
        r[1, 1] = cx * cz
        r[2, 1] = cz * sx
        r[0, 2] = cz * sy
        r[1, 2] = -cy * sx + cx * sy * sz
        r[2, 2] = cx * cy + sx * sy * sz
    elif order == 2:  # ZXY : a0=z a1=x a2=y
        cz = wp.cos(a0); sz = wp.sin(a0)
        cx = wp.cos(a1); sx = wp.sin(a1)
        cy = wp.cos(a2); sy = wp.sin(a2)
        r[0, 0] = cy * cz - sx * sy * sz
        r[1, 0] = cz * sx * sy + cy * sz
        r[2, 0] = -cx * sy
        r[0, 1] = -cx * sz
        r[1, 1] = cx * cz
        r[2, 1] = sx
        r[0, 2] = cz * sy + cy * sx * sz
        r[1, 2] = -cy * cz * sx + sy * sz
        r[2, 2] = cx * cy
    else:  # ZYX (order == 1) : a0=z a1=y a2=x
        cz = wp.cos(a0); sz = wp.sin(a0)
        cy = wp.cos(a1); sy = wp.sin(a1)
        cx = wp.cos(a2); sx = wp.sin(a2)
        r[0, 0] = cz * cy
        r[1, 0] = sz * cy
        r[2, 0] = -sy
        r[0, 1] = cz * sy * sx - sz * cx
        r[1, 1] = sz * sy * sx + cz * cx
        r[2, 1] = cy * sx
        r[0, 2] = cz * sy * cx + sz * sx
        r[1, 2] = sz * sy * cx - cz * sx
        r[2, 2] = cy * cx
    return r


@wp.func
def rodrigues(axis: wp.vec3d, angle: wp.float64) -> wp.mat33d:
    # axes in the target model are exact unit vectors, but normalize defensively.
    n = wp.length(axis)
    ident = wp.identity(n=3, dtype=wp.float64)
    if n == wp.float64(0.0):
        return ident
    k = axis / n
    theta = angle * n
    kx = k[0]
    ky = k[1]
    kz = k[2]
    K = wp.mat33d(
        wp.float64(0.0), -kz, ky,
        kz, wp.float64(0.0), -kx,
        -ky, kx, wp.float64(0.0),
    )
    s = wp.sin(theta)
    c = wp.cos(theta)
    return ident + s * K + (wp.float64(1.0) - c) * (K * K)


@wp.func
def make_transform(R: wp.mat33d, t: wp.vec3d) -> wp.mat44d:
    T = wp.identity(n=4, dtype=wp.float64)
    for i in range(3):
        for j in range(3):
            T[i, j] = R[i, j]
        T[i, 3] = t[i]
    return T


@wp.func
def se3_inverse(T: wp.mat44d) -> wp.mat44d:
    out = wp.identity(n=4, dtype=wp.float64)
    # R^T
    for i in range(3):
        for j in range(3):
            out[i, j] = T[j, i]
    # -R^T t
    for i in range(3):
        acc = wp.float64(0.0)
        for j in range(3):
            acc += T[j, i] * T[j, 3]
        out[i, 3] = -acc
    return out


@wp.func
def transform_point(T: wp.mat44d, p: wp.vec3d) -> wp.vec3d:
    return wp.vec3d(
        T[0, 0] * p[0] + T[0, 1] * p[1] + T[0, 2] * p[2] + T[0, 3],
        T[1, 0] * p[0] + T[1, 1] * p[1] + T[1, 2] * p[2] + T[1, 3],
        T[2, 0] * p[0] + T[2, 1] * p[1] + T[2, 2] * p[2] + T[2, 3],
    )


@wp.func
def scale_translation(T: wp.mat44d, s: wp.vec3d) -> wp.mat44d:
    out = T
    out[0, 3] = T[0, 3] * s[0]
    out[1, 3] = T[1, 3] * s[1]
    out[2, 3] = T[2, 3] * s[2]
    return out
