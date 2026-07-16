# SPDX-License-Identifier: MIT
#
# Convert the OpenSim Rajagopal bone meshes (``.vtp``) into the STL meshes the biomech
# MJCF references, writing them into the ProtoMotions asset tree at
# ``protomotions/data/assets/mesh/biomech_rajagopal/``. Only the meshes attached to
# bodies that actually appear in the exported skeleton are converted.
#
# Meshes are fetched from the O2MConverter project's re-hosted OpenSim geometry by
# default, or read from a local ``--geometry-dir`` (e.g. a local OpenSim install's
# ``Geometry`` folder). MuJoCo/Newton load STL/OBJ/MSH but not ``.vtp``, hence the
# conversion. The bone geometry originates from the OpenSim Rajagopal2015 model.
#
# Run from the repo root::
#
#     python projects/biomech/tools/convert_bone_meshes.py            # download + convert
#     python projects/biomech/tools/convert_bone_meshes.py --geometry-dir path/to/Geometry

"""Convert OpenSim Rajagopal ``.vtp`` bone meshes to STL in the ProtoMotions asset tree."""

from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # projects/

from biomech.export.bone_geometry import (  # noqa: E402
    DEFAULT_OSIM,
    MESH_ASSET_SUBDIR,
    O2M_GEOMETRY_URL,
    parse_display_geometry,
)
from biomech.export.vtp import vtp_to_stl  # noqa: E402
from biomech.osim import parse_osim  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_PM_ASSETS = _REPO / "protomotions" / "data" / "assets"

_NOTICE = """\
Bone meshes for the `biomech` robot.

Source: OpenSim Rajagopal2015 musculoskeletal model geometry, as re-hosted by the
O2MConverter project (https://github.com/aikkala/O2MConverter, Apache-2.0). Converted
from the original ASCII VTK PolyData (.vtp) to STL by
projects/biomech/tools/convert_bone_meshes.py. Regenerate with that tool; do not edit
these files by hand.

These meshes are visual-only in the MJCF (density=0, contype/conaffinity=0).
"""


def _required_vtp(osim_path: Path) -> list[str]:
    """The .vtp files attached to bodies that appear in the exported skeleton."""
    disp = parse_display_geometry(osim_path)
    spec = parse_osim(str(osim_path))
    body_names = {b.name for b in spec.bodies}
    files: list[str] = []
    seen: set[str] = set()
    for name in body_names:
        for m in disp.get(name, []):
            if m.vtp_file not in seen:
                seen.add(m.vtp_file)
                files.append(m.vtp_file)
    return sorted(files)


def _fetch(url: str, dst: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 (trusted raw URL)
        dst.write_bytes(r.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--geometry-dir",
        type=Path,
        default=None,
        help="Local folder of .vtp meshes (default: download from O2MConverter).",
    )
    ap.add_argument("--osim", type=Path, default=DEFAULT_OSIM, help="Source .osim model.")
    ap.add_argument(
        "--out",
        type=Path,
        default=_PM_ASSETS / MESH_ASSET_SUBDIR,
        help="Output STL directory (in the ProtoMotions asset tree).",
    )
    ap.add_argument("--force", action="store_true", help="Reconvert even if STL exists.")
    args = ap.parse_args()

    vtp_files = _required_vtp(args.osim)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "NOTICE.txt").write_text(_NOTICE)

    print(f"{len(vtp_files)} bone meshes -> {out_dir}")
    tmp_ctx = tempfile.TemporaryDirectory() if args.geometry_dir is None else None
    tmp_dir = Path(tmp_ctx.name) if tmp_ctx is not None else None

    n_done = n_skip = 0
    for vf in vtp_files:
        stl = out_dir / f"{Path(vf).stem}.stl"
        if stl.exists() and not args.force:
            n_skip += 1
            continue
        if args.geometry_dir is not None:
            src = args.geometry_dir / vf
            if not src.exists():
                print(f"  MISSING {src}")
                return 2
        else:
            src = tmp_dir / vf  # type: ignore[operator]
            _fetch(f"{O2M_GEOMETRY_URL}/{vf}", src)
        vtp_to_stl(src, stl)
        n_done += 1
        print(f"  {vf} -> {stl.name}")

    if tmp_ctx is not None:
        tmp_ctx.cleanup()
    print(f"done: {n_done} converted, {n_skip} up-to-date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
