# SPDX-License-Identifier: MIT
#
# Minimal reader for the ASCII VTK PolyData (``.vtp``) bone meshes shipped with the
# OpenSim Rajagopal model (as re-hosted by the O2MConverter project). Only the
# ``format="ascii"`` PolyData variant is supported -- that is what the Rajagopal
# Geometry folder uses -- so no VTK/meshio dependency is required (stdlib XML only).
#
# The reader returns triangulated (vertices, faces) suitable for writing an STL/OBJ
# that MuJoCo / Newton can load (their mesh loaders accept STL/OBJ/MSH, not ``.vtp``).

"""Dependency-light reader for ASCII ``.vtp`` (VTK PolyData) bone meshes."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _floats(text: str) -> np.ndarray:
    return np.fromstring(text.replace("\n", " "), sep=" ", dtype=np.float64)


def _ints(text: str) -> np.ndarray:
    return np.fromstring(text.replace("\n", " "), sep=" ", dtype=np.int64)


def read_vtp(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an ASCII ``.vtp`` PolyData file into (vertices, triangle_faces).

    Returns
    -------
    vertices : (N, 3) float64
    faces : (M, 3) int64  -- polygons are fan-triangulated so faces are triangles.

    Raises
    ------
    ValueError
        If the file is not ASCII PolyData (e.g. binary/appended data arrays), which
        the Rajagopal Geometry meshes never are.
    """
    root = ET.parse(str(path)).getroot()
    if root.get("type") != "PolyData":
        raise ValueError(f"{path}: not a VTK PolyData file (type={root.get('type')!r})")

    piece = root.find("./PolyData/Piece")
    if piece is None:
        raise ValueError(f"{path}: missing PolyData/Piece")

    # --- points ---
    pts_da = piece.find("./Points/DataArray")
    if pts_da is None or (pts_da.get("format") or "ascii") != "ascii":
        raise ValueError(f"{path}: only ASCII Points DataArray is supported")
    verts = _floats(pts_da.text or "").reshape(-1, 3)

    # --- polygon connectivity + offsets ---
    conn_da = piece.find("./Polys/DataArray[@Name='connectivity']")
    off_da = piece.find("./Polys/DataArray[@Name='offsets']")
    if conn_da is None or off_da is None:
        raise ValueError(f"{path}: missing Polys connectivity/offsets")
    conn = _ints(conn_da.text or "")
    offsets = _ints(off_da.text or "")

    tris: list[tuple[int, int, int]] = []
    start = 0
    for end in offsets:
        poly = conn[start:end]
        start = int(end)
        # fan-triangulate any polygon with 3+ vertices
        for k in range(1, len(poly) - 1):
            tris.append((int(poly[0]), int(poly[k]), int(poly[k + 1])))

    faces = np.asarray(tris, dtype=np.int64) if tris else np.zeros((0, 3), np.int64)
    return verts, faces


def vtp_to_stl(src: str | Path, dst: str | Path) -> Path:
    """Convert an ASCII ``.vtp`` mesh to a binary ``.stl`` at ``dst``."""
    import trimesh

    verts, faces = read_vtp(src)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(dst), file_type="stl")
    return dst
