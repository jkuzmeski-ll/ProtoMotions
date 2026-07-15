# SPDX-License-Identifier: MIT
#
# Backend-agnostic OpenSim skeleton specification produced by ``biomech.osim.parser``.
#
# These dataclasses are a faithful, Windows-native re-expression of the pieces of
# a DART ``Skeleton`` that Nimble's ``OpenSimParser`` builds from an ``.osim`` file:
# bodies (mass/COM/inertia), joints (relative frames + coordinate-driven transform
# axes), markers (body + local offset), and the per-segment group-scale grouping.
# The ``biomech.skeleton`` Warp kinematics (M2b) consume this spec; nothing here
# depends on Newton/MuJoCo/Warp so it stays a pure, testable data layer.

"""OpenSim ``SkeletonSpec`` dataclasses (see ``docs/20_nimble_port_plan.md`` M2a)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from biomech.skeleton.simmspline import SimmSpline

# ---------------------------------------------------------------------------
# Coordinate-coupling functions (the ``<function>`` inside each TransformAxis)
# ---------------------------------------------------------------------------


@dataclass
class LinearFunctionSpec:
    """OpenSim ``LinearFunction``: ``y = slope * x + intercept``.

    OpenSim serializes ``<coefficients> slope intercept</coefficients>``.
    """

    slope: float
    intercept: float

    def value(self, x: float) -> float:
        return self.slope * float(x) + self.intercept

    def derivative(self, order: int, x: float) -> float:
        if order == 1:
            return self.slope
        return 0.0


@dataclass
class ConstantFunctionSpec:
    """OpenSim ``Constant``: ``y = value`` (an un-driven transform axis)."""

    constant: float

    def value(self, x: float) -> float:
        return self.constant

    def derivative(self, order: int, x: float) -> float:
        return 0.0


@dataclass
class SimmSplineSpec:
    """OpenSim ``SimmSpline`` (natural cubic with SIMM end conditions)."""

    x: list[float]
    y: list[float]
    _spline: Optional[SimmSpline] = field(default=None, repr=False, compare=False)

    def spline(self) -> SimmSpline:
        if self._spline is None:
            self._spline = SimmSpline(self.x, self.y)
        return self._spline

    def value(self, x: float) -> float:
        return self.spline().calc_value(float(x))

    def derivative(self, order: int, x: float) -> float:
        return self.spline().calc_derivative(order, float(x))


@dataclass
class PolynomialFunctionSpec:
    """OpenSim ``PolynomialFunction`` (coefficients highest-order first)."""

    coefficients: list[float]

    def value(self, x: float) -> float:
        return float(np.polyval(self.coefficients, float(x)))

    def derivative(self, order: int, x: float) -> float:
        c = np.polynomial.polynomial.Polynomial(list(reversed(self.coefficients)))
        return float(c.deriv(order)(float(x)))


@dataclass
class PiecewiseLinearFunctionSpec:
    """OpenSim ``PiecewiseLinearFunction`` (linear interpolation of knots)."""

    x: list[float]
    y: list[float]

    def value(self, x: float) -> float:
        return float(np.interp(float(x), self.x, self.y))

    def derivative(self, order: int, x: float) -> float:
        if order != 1:
            return 0.0
        xs = float(x)
        xk = self.x
        for i in range(len(xk) - 1):
            if xk[i] <= xs <= xk[i + 1]:
                return (self.y[i + 1] - self.y[i]) / (xk[i + 1] - xk[i])
        return 0.0


@dataclass
class MultiplierFunctionSpec:
    """OpenSim ``MultiplierFunction``: ``scale * inner(x)``."""

    scale: float
    inner: "CouplingFunction"

    def value(self, x: float) -> float:
        return self.scale * self.inner.value(x)

    def derivative(self, order: int, x: float) -> float:
        return self.scale * self.inner.derivative(order, x)


CouplingFunction = Union[
    LinearFunctionSpec,
    ConstantFunctionSpec,
    SimmSplineSpec,
    PolynomialFunctionSpec,
    PiecewiseLinearFunctionSpec,
    MultiplierFunctionSpec,
]


# ---------------------------------------------------------------------------
# Structural specs
# ---------------------------------------------------------------------------


@dataclass
class TransformAxisSpec:
    """One of the 6 ``TransformAxis`` entries of an OpenSim ``SpatialTransform``.

    ``kind`` is ``"rotation"`` for the first 3 and ``"translation"`` for the last
    3. ``coordinate`` is the driving DOF name (or ``None`` for a constant axis).
    """

    name: str
    kind: str
    axis: np.ndarray  # shape (3,)
    coordinate: Optional[str]
    function: CouplingFunction


@dataclass
class CoordinateSpec:
    """A generalized coordinate (``<Coordinate>``)."""

    name: str
    motion_type: str
    default_value: float
    default_speed_value: float
    range_lo: float
    range_hi: float
    clamped: bool
    locked: bool

    @property
    def limit_lo(self) -> float:
        """Effective DART position lower limit (locked DOFs collapse to default)."""
        return self.default_value if self.locked else self.range_lo

    @property
    def limit_hi(self) -> float:
        return self.default_value if self.locked else self.range_hi


@dataclass
class JointSpec:
    """A joint connecting ``parent_body`` -> ``child_body``.

    ``joint_class`` is the raw OpenSim tag (``CustomJoint`` / ``PinJoint`` /
    ``UniversalJoint`` / ``WeldJoint``). ``nimble_type`` is the DART joint type
    Nimble would pick (``EulerFreeJoint`` / ``EulerJoint`` / ``CustomJointN`` /
    ``RevoluteJoint`` / ``UniversalJoint`` / ``WeldJoint``); it is informational and
    used only for parity checks — the ``biomech.skeleton`` FK treats every joint
    through the generic CustomJoint canonical form.

    ``T_parent`` / ``T_child`` are 4x4 homogeneous transforms
    (``eulerXYZ(orientation) `` + translation), exactly as Nimble builds them.
    ``axis_order`` / ``flip_axis_map`` describe the Euler rotation convention for
    Euler-family joints (``None`` for revolute/universal/weld).
    """

    name: str
    joint_class: str
    nimble_type: str
    parent_body: Optional[str]
    child_body: str
    T_parent: np.ndarray  # (4, 4)
    T_child: np.ndarray  # (4, 4)
    coordinates: list[CoordinateSpec]
    transform_axes: list[TransformAxisSpec]
    axis_order: Optional[str]
    flip_axis_map: Optional[np.ndarray]  # (3,)
    reverse: bool = False

    @property
    def num_dofs(self) -> int:
        return len(self.coordinates)

    @property
    def dof_names(self) -> list[str]:
        return [c.name for c in self.coordinates]


@dataclass
class BodySpec:
    """A rigid body (``<Body>``)."""

    index: int
    name: str
    parent_body: Optional[str]
    parent_joint: str
    mass: float
    com: np.ndarray  # (3,) mass_center
    inertia: np.ndarray  # (6,) [xx, yy, zz, xy, xz, yz]


@dataclass
class MarkerSpec:
    """A marker (``<Marker>``): a body plus a local offset."""

    name: str
    body: str
    offset: np.ndarray  # (3,)
    fixed: bool

    @property
    def anatomical(self) -> bool:
        # Nimble treats markers with fixed="true" as anatomical landmarks.
        return self.fixed


@dataclass
class SkeletonSpec:
    """A parsed OpenSim model, backend-agnostic.

    Bodies and joints are stored in DART tree order (which, for the target
    models, equals BodySet document order after skipping ``ground`` and the
    constraint-driven patellae). ``joints[i]`` is the joint whose child is
    ``bodies[i]``. DOFs are numbered by iterating joints in this order and their
    coordinates in ``CoordinateSet`` order.
    """

    name: str
    frame: str
    bodies: list[BodySpec]
    joints: list[JointSpec]
    markers: list[MarkerSpec]
    scale_groups: list[list[str]]
    length_units: str = "meters"
    force_units: str = "N"

    # -- derived ------------------------------------------------------------
    @property
    def num_bodies(self) -> int:
        return len(self.bodies)

    @property
    def num_joints(self) -> int:
        return len(self.joints)

    @property
    def num_dofs(self) -> int:
        return sum(j.num_dofs for j in self.joints)

    @property
    def group_scales_dim(self) -> int:
        return 3 * len(self.scale_groups)

    @property
    def dof_names(self) -> list[str]:
        names: list[str] = []
        for j in self.joints:
            names.extend(j.dof_names)
        return names

    def dof_index_map(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.dof_names)}

    def body(self, name: str) -> BodySpec:
        for b in self.bodies:
            if b.name == name:
                return b
        raise KeyError(name)

    def joint(self, name: str) -> JointSpec:
        for j in self.joints:
            if j.name == name:
                return j
        raise KeyError(name)

    def marker(self, name: str) -> MarkerSpec:
        for m in self.markers:
            if m.name == name:
                return m
        raise KeyError(name)
