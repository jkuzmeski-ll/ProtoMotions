# SPDX-License-Identifier: MIT
#
# Parse per-body display geometry (bone meshes) from an OpenSim ``.osim`` model so the
# MJCF exporter can render each body as its actual bone(s) instead of capsule/sphere
# placeholders. For the Rajagopal model every ``<DisplayGeometry>`` sits at an identity
# in-body transform with unit generic scale, so placement reduces to "mesh in body
# frame"; the exporter still applies the subject's per-body group scale on top.

"""Per-body bone-mesh geometry parsed from an OpenSim model's ``<DisplayGeometry>``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

# The bundled generic Rajagopal model whose body frames the fitted SkeletonSpec inherits.
DEFAULT_OSIM = Path(__file__).resolve().parents[1] / "models" / "rajagopal_data" / "Rajagopal2015.osim"

# Where converted STL meshes live in the ProtoMotions asset tree, and the MJCF-relative
# ``meshdir`` (the MJCF lives in ``data/assets/mjcf/``; meshes in ``data/assets/mesh/``).
MESH_ASSET_SUBDIR = "mesh/biomech_rajagopal"
MESHDIR_REL = "../mesh/biomech_rajagopal/"

# Raw base URL for the O2MConverter-hosted OpenSim Rajagopal Geometry (.vtp) files.
O2M_GEOMETRY_URL = (
    "https://raw.githubusercontent.com/aikkala/O2MConverter/master/"
    "models/opensim/rajagopal_walking/Geometry"
)


@dataclass
class BoneMesh:
    """One display mesh attached to a body, expressed in that body's local frame."""

    stem: str  # mesh basename without extension, e.g. "r_femur" (used as MJCF mesh name)
    vtp_file: str  # source geometry file name, e.g. "r_femur.vtp"
    transform: np.ndarray = field(  # (6,) OpenSim in-body transform: rx ry rz tx ty tz
        default_factory=lambda: np.zeros(6, dtype=np.float64)
    )
    scale: np.ndarray = field(  # (3,) generic display scale_factors
        default_factory=lambda: np.ones(3, dtype=np.float64)
    )

    @property
    def is_identity_placement(self) -> bool:
        return bool(np.allclose(self.transform, 0.0) and np.allclose(self.scale, 1.0))


def parse_display_geometry(osim_path: str | Path = DEFAULT_OSIM) -> dict[str, list[BoneMesh]]:
    """Map ``body name -> [BoneMesh, ...]`` from an OpenSim model's display geometry.

    Reads the legacy ``<Body>/<VisibleObject>/<GeometrySet>/objects/<DisplayGeometry>``
    layout used by the Rajagopal2015 model.
    """
    root = ET.parse(str(osim_path)).getroot()
    out: dict[str, list[BoneMesh]] = {}
    for body in root.iter("Body"):
        name = body.get("name")
        if not name:
            continue
        meshes: list[BoneMesh] = []
        for dg in body.findall(".//VisibleObject/GeometrySet/objects/DisplayGeometry"):
            gf = (dg.findtext("geometry_file") or "").strip()
            if not gf:
                continue
            tf = np.fromstring(dg.findtext("transform") or "0 0 0 0 0 0", sep=" ")
            sc = np.fromstring(dg.findtext("scale_factors") or "1 1 1", sep=" ")
            if tf.size != 6:
                tf = np.zeros(6, dtype=np.float64)
            if sc.size != 3:
                sc = np.ones(3, dtype=np.float64)
            meshes.append(BoneMesh(stem=Path(gf).stem, vtp_file=gf, transform=tf, scale=sc))
        if meshes:
            out[name] = meshes
    return out


def default_bone_geometry() -> dict[str, list[BoneMesh]]:
    """Bone-mesh map parsed from the bundled generic Rajagopal model."""
    return parse_display_geometry(DEFAULT_OSIM)


def bone_mesh_dir(asset_root: str | Path) -> Path:
    """Absolute directory the converted STL bone meshes live in."""
    return Path(asset_root) / MESH_ASSET_SUBDIR


def bone_meshes_available(asset_root: str | Path) -> bool:
    """True if the converted STL bone meshes are present under ``asset_root``.

    Checks a small core set of stems (pelvis/femur/foot/toes) rather than every file,
    which is enough to tell a converted tree from an un-converted one. Run
    ``tools/convert_bone_meshes.py`` to populate the directory.
    """
    d = bone_mesh_dir(asset_root)
    core = ("r_pelvis", "r_femur", "r_foot", "r_bofoot")
    return d.is_dir() and all((d / f"{s}.stl").exists() for s in core)
