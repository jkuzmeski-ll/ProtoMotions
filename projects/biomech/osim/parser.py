# SPDX-License-Identifier: MIT
#
# Windows-native port of the parsing half of Nimble's
# ``dart/biomechanics/OpenSimParser::readOsim30`` (the OpenSim v3 ".osim" format,
# schema version 30000, which the target Rajagopal2015 model uses).
#
# We reproduce exactly the parts that define gold-standard kinematics:
#   * body mass / mass_center / inertia,
#   * joint relative frames  T_parent = eulerXYZ(orientation_in_parent)+location_in_parent
#                            T_child  = eulerXYZ(orientation)+location
#     (verified bit-for-bit against Nimble; see docs/refs/rajagopal2015_structure.json),
#   * the 6-axis SpatialTransform (axis + driving coordinate + coupling function),
#   * Euler AxisOrder / FlipAxisMap  (ports of getAxisOrder / getAxisFlips),
#   * markers (body + local offset) and per-segment scale groups.
#
# Bodies ``ground`` and the constraint-driven ``patella_{r,l}`` are skipped, exactly
# as Nimble does; after that the BodySet document order equals DART tree order.

"""OpenSim ``.osim`` -> :class:`~biomech.osim.spec.SkeletonSpec` parser (M2a)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np

from biomech.osim.spec import (
    BodySpec,
    ConstantFunctionSpec,
    CoordinateSpec,
    CouplingFunction,
    JointSpec,
    LinearFunctionSpec,
    MarkerSpec,
    PiecewiseLinearFunctionSpec,
    PolynomialFunctionSpec,
    SimmSplineSpec,
    SkeletonSpec,
    TransformAxisSpec,
)

# Bodies Nimble drops from the skeleton (ground root + constraint-driven patellae).
_SKIP_BODIES = {"ground", "patella_r", "patella_l"}

_UNIT_AXES = {
    (1, 0, 0): "X",
    (0, 1, 0): "Y",
    (0, 0, 1): "Z",
}


# ---------------------------------------------------------------------------
# small XML / math helpers
# ---------------------------------------------------------------------------


def _text(elem: Optional[ET.Element]) -> str:
    if elem is None or elem.text is None:
        return ""
    return elem.text.strip()


def _read_vec(elem: Optional[ET.Element]) -> np.ndarray:
    s = _text(elem)
    if not s:
        return np.zeros(0, dtype=np.float64)
    return np.array([float(v) for v in s.split()], dtype=np.float64)


def _read_bool(elem: Optional[ET.Element]) -> bool:
    return _text(elem).lower() == "true"


def euler_xyz_to_matrix(angle: np.ndarray) -> np.ndarray:
    """Port of Nimble ``math::eulerXYZToMatrix`` (R = Rx(a).Ry(b).Rz(c))."""
    cx, sx = np.cos(angle[0]), np.sin(angle[0])
    cy, sy = np.cos(angle[1]), np.sin(angle[1])
    cz, sz = np.cos(angle[2]), np.sin(angle[2])
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


def _make_transform(orientation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = euler_xyz_to_matrix(orientation)
    t[:3, 3] = translation
    return t


def _axis_order_and_flip(
    rot_axes: list[np.ndarray],
) -> tuple[Optional[str], Optional[np.ndarray]]:
    """Ports of Nimble ``getAxisOrder`` / ``getAxisFlips`` for 3 rotational axes.

    Returns e.g. ``("ZXY", array([1., 1., 1.]))``. The axis order matches the
    absolute axes to one of XYZ/ZYX/ZXY/XZY; the flip map is -1 where the axis is
    a negative unit vector, else +1.
    """
    abs_letters = []
    for a in rot_axes:
        av = np.abs(a)
        r = (int(round(av[0])), int(round(av[1])), int(round(av[2])))
        abs_letters.append(_UNIT_AXES.get(r))
    order = "".join(letter for letter in abs_letters if letter is not None)
    valid = {"XYZ", "ZYX", "ZXY", "XZY"}
    if order not in valid:
        raise ValueError(f"Unsupported Euler axis order from axes {rot_axes!r}")

    flips = np.ones(3, dtype=np.float64)
    for i, a in enumerate(rot_axes):
        av = np.rint(a)
        r = (int(av[0]), int(av[1]), int(av[2]))
        if r in {(-1, 0, 0), (0, -1, 0), (0, 0, -1)}:
            flips[i] = -1.0
    return order, flips


# ---------------------------------------------------------------------------
# function parsing
# ---------------------------------------------------------------------------


def _parse_axis_function(
    func_elem: Optional[ET.Element], axis: np.ndarray
) -> tuple[CouplingFunction, np.ndarray]:
    """Parse a ``<function>`` and return ``(function, axis)``.

    Reproduces Nimble's per-axis handling (OpenSimParser.cpp:5036-5199):
    a ``MultiplierFunction`` folds its ``scale`` into the leaf function, and a
    ``LinearFunction`` with slope ``-1`` bakes the sign into the axis (slope set
    to ``+1``). Both affect the axis passed to ``getAxisOrder``/``getAxisFlips``.
    """
    axis = axis.astype(np.float64).copy()
    if func_elem is None:
        return ConstantFunctionSpec(0.0), axis

    child = None
    for c in func_elem:
        child = c
        break
    if child is None:
        return ConstantFunctionSpec(0.0), axis

    scale = 1.0
    if child.tag == "MultiplierFunction":
        scale = float(_text(child.find("scale")) or 1.0)
        inner = child.find("function")
        child = None
        if inner is not None:
            for c in inner:
                child = c
                break
        if child is None:
            return ConstantFunctionSpec(0.0), axis

    tag = child.tag
    if tag == "LinearFunction":
        coeffs = _read_vec(child.find("coefficients"))
        slope = float(coeffs[0]) if coeffs.size > 0 else 1.0
        intercept = float(coeffs[1]) if coeffs.size > 1 else 0.0
        if slope == -1.0:
            axis = -axis
            slope = 1.0
        return LinearFunctionSpec(slope=slope * scale, intercept=intercept * scale), axis
    if tag == "Constant":
        value = float(_text(child.find("value")) or 0.0) * scale
        return ConstantFunctionSpec(value), axis
    if tag in ("SimmSpline", "NaturalCubicSpline"):
        x = _read_vec(child.find("x")).tolist()
        y = (_read_vec(child.find("y")) * scale).tolist()
        return SimmSplineSpec(x=x, y=y), axis
    if tag == "PolynomialFunction":
        coeffs = (_read_vec(child.find("coefficients")) * scale).tolist()
        return PolynomialFunctionSpec(coeffs), axis
    if tag == "PiecewiseLinearFunction":
        x = _read_vec(child.find("x")).tolist()
        y = (_read_vec(child.find("y")) * scale).tolist()
        return PiecewiseLinearFunctionSpec(x=x, y=y), axis
    raise ValueError(f"Unsupported OpenSim function type: {tag}")


def _parse_coordinate(coord_elem: ET.Element) -> CoordinateSpec:
    rng = _read_vec(coord_elem.find("range"))
    lo = float(rng[0]) if rng.size > 0 else -np.inf
    hi = float(rng[1]) if rng.size > 1 else np.inf
    return CoordinateSpec(
        name=coord_elem.attrib["name"].strip(),
        motion_type=_text(coord_elem.find("motion_type")) or "rotational",
        default_value=float(_text(coord_elem.find("default_value")) or 0.0),
        default_speed_value=float(_text(coord_elem.find("default_speed_value")) or 0.0),
        range_lo=lo,
        range_hi=hi,
        clamped=_read_bool(coord_elem.find("clamped")),
        locked=_read_bool(coord_elem.find("locked")),
    )


def _parse_coordinates(joint_detail: ET.Element) -> list[CoordinateSpec]:
    coords: list[CoordinateSpec] = []
    cset = joint_detail.find("CoordinateSet")
    if cset is None:
        return coords
    objs = cset.find("objects")
    if objs is None:
        return coords
    for c in objs.findall("Coordinate"):
        coords.append(_parse_coordinate(c))
    return coords


def _parse_spatial_transform(
    joint_detail: ET.Element,
) -> list[TransformAxisSpec]:
    axes: list[TransformAxisSpec] = []
    st = joint_detail.find("SpatialTransform")
    if st is None:
        return axes
    for ax in st.findall("TransformAxis"):
        name = ax.attrib.get("name", "")
        kind = "translation" if name.startswith("translation") else "rotation"
        coord = _text(ax.find("coordinates")) or None
        # v3 files wrap the function in <function>; fall back to the axis itself.
        func_elem = ax.find("function")
        if func_elem is None:
            func_elem = ax
        function, baked_axis = _parse_axis_function(func_elem, _read_vec(ax.find("axis")))
        axes.append(
            TransformAxisSpec(
                name=name,
                kind=kind,
                axis=baked_axis,
                coordinate=coord,
                function=function,
            )
        )
    return axes


# ---------------------------------------------------------------------------
# joint parsing / classification
# ---------------------------------------------------------------------------

_JOINT_TAGS = ("CustomJoint", "PinJoint", "UniversalJoint", "WeldJoint")


def _find_joint_detail(joint_elem: ET.Element) -> Optional[ET.Element]:
    for tag in _JOINT_TAGS:
        detail = joint_elem.find(tag)
        if detail is not None:
            return detail
    return None


def _classify(
    joint_class: str,
    coordinates: list[CoordinateSpec],
    axes: list[TransformAxisSpec],
) -> tuple[str, Optional[str], Optional[np.ndarray]]:
    """Return (nimble_type, axis_order, flip_axis_map) matching Nimble's parse.

    Euler-family joints (CustomJoint) carry an axis order + flip; revolute /
    universal / weld carry ``None`` (matching the golden structure JSON).
    """
    if joint_class == "PinJoint":
        return "RevoluteJoint", None, None
    if joint_class == "UniversalJoint":
        return "UniversalJoint", None, None
    if joint_class == "WeldJoint":
        return "WeldJoint", None, None
    if joint_class != "CustomJoint":
        raise ValueError(f"Unsupported joint class: {joint_class}")

    rotations = [a for a in axes if a.kind == "rotation"]
    translations = [a for a in axes if a.kind == "translation"]
    rot_axes = [a.axis for a in rotations[:3]]
    axis_order, flips = _axis_order_and_flip(rot_axes)

    ndof = len(coordinates)
    if ndof == 6:
        nimble_type = "EulerFreeJoint"
    elif (
        ndof == 3
        and all(isinstance(a.function, LinearFunctionSpec) for a in rotations)
        and all(isinstance(a.function, ConstantFunctionSpec) for a in translations)
    ):
        nimble_type = "EulerJoint"
    else:
        nimble_type = f"CustomJoint{ndof}"
    return nimble_type, axis_order, flips


def _parse_joint(joint_detail: ET.Element, child_body: str) -> JointSpec:
    joint_class = joint_detail.tag
    name = joint_detail.attrib.get("name", "")

    parent_name = _text(joint_detail.find("parent_body")) or None
    if parent_name in _SKIP_BODIES:
        # e.g. the pelvis' parent is "ground"; DART's root has no parent body.
        parent_name = None

    loc_in_parent = _read_vec(joint_detail.find("location_in_parent"))
    ori_in_parent = _read_vec(joint_detail.find("orientation_in_parent"))
    loc_in_child = _read_vec(joint_detail.find("location"))
    ori_in_child = _read_vec(joint_detail.find("orientation"))

    t_parent = _make_transform(ori_in_parent, loc_in_parent)
    t_child = _make_transform(ori_in_child, loc_in_child)

    reverse = _read_bool(joint_detail.find("reverse"))
    if reverse:
        raise NotImplementedError(
            f"Joint '{name}' has reverse=true, which is not yet supported."
        )

    coordinates = _parse_coordinates(joint_detail)
    axes = _parse_spatial_transform(joint_detail)
    nimble_type, axis_order, flips = _classify(joint_class, coordinates, axes)

    return JointSpec(
        name=name,
        joint_class=joint_class,
        nimble_type=nimble_type,
        parent_body=parent_name,
        child_body=child_body,
        T_parent=t_parent,
        T_child=t_child,
        coordinates=coordinates,
        transform_axes=axes,
        axis_order=axis_order,
        flip_axis_map=flips,
        reverse=reverse,
    )


# ---------------------------------------------------------------------------
# top-level parse
# ---------------------------------------------------------------------------


def _parse_body(index: int, body_elem: ET.Element, parent_joint: JointSpec) -> BodySpec:
    name = body_elem.attrib["name"].strip()
    inertia = np.array(
        [
            float(_text(body_elem.find(k)) or 0.0)
            for k in (
                "inertia_xx",
                "inertia_yy",
                "inertia_zz",
                "inertia_xy",
                "inertia_xz",
                "inertia_yz",
            )
        ],
        dtype=np.float64,
    )
    return BodySpec(
        index=index,
        name=name,
        parent_body=parent_joint.parent_body,
        parent_joint=parent_joint.name,
        mass=float(_text(body_elem.find("mass")) or 0.0),
        com=_read_vec(body_elem.find("mass_center")),
        inertia=inertia,
    )


def _parse_markers(model_elem: ET.Element) -> list[MarkerSpec]:
    markers: list[MarkerSpec] = []
    mset = model_elem.find("MarkerSet")
    if mset is None:
        return markers
    objs = mset.find("objects")
    if objs is None:
        return markers
    for m in objs.findall("Marker"):
        markers.append(
            MarkerSpec(
                name=m.attrib["name"].strip(),
                body=_text(m.find("body")),
                offset=_read_vec(m.find("location")),
                fixed=_read_bool(m.find("fixed")),
            )
        )
    return markers


def parse_osim(path: str) -> SkeletonSpec:
    """Parse an OpenSim ``.osim`` (schema 30000) into a :class:`SkeletonSpec`."""
    tree = ET.parse(path)
    root = tree.getroot()
    model = root.find("Model")
    if model is None:
        raise ValueError(f"{path}: no <Model> element found")

    model_name = model.attrib.get("name", "").strip() or "model"
    length_units = _text(model.find("length_units")) or "meters"
    force_units = _text(model.find("force_units")) or "N"

    bodies: list[BodySpec] = []
    joints: list[JointSpec] = []

    body_set = model.find("BodySet")
    if body_set is None:
        raise ValueError(f"{path}: no <BodySet> element found")
    objs = body_set.find("objects")
    if objs is None:
        raise ValueError(f"{path}: no <BodySet>/<objects> element found")

    index = 0
    for body_elem in objs.findall("Body"):
        name = body_elem.attrib["name"].strip()
        if name in _SKIP_BODIES:
            continue
        joint_elem = body_elem.find("Joint")
        joint_detail = _find_joint_detail(joint_elem) if joint_elem is not None else None
        if joint_detail is None:
            # Body with no real joint (only the skipped ground has this).
            continue
        joint = _parse_joint(joint_detail, child_body=name)
        body = _parse_body(index, body_elem, joint)
        joints.append(joint)
        bodies.append(body)
        index += 1

    markers = _parse_markers(model)

    # Rajagopal-style models scale each body as its own group; there is no
    # explicit ScaleSet, so the group set is one singleton group per body, in
    # body order. (Matches docs/refs/rajagopal2015_structure.json.)
    scale_groups = [[b.name] for b in bodies]

    return SkeletonSpec(
        name=model_name,
        frame="opensim_y_up_meters",
        bodies=bodies,
        joints=joints,
        markers=markers,
        scale_groups=scale_groups,
        length_units=length_units,
        force_units=force_units,
    )
