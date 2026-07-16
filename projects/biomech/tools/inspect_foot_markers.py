# SPDX-License-Identifier: MIT
"""Quick audit of foot/ankle marker availability + geometry in the S001 C3D trials.

Confirms which foot markers are actually populated (non-NaN) in the static (Cal) and
dynamic (Trial) captures and prints their mean positions + key inter-marker distances,
so the enriched Rajagopal marker map / new model-marker offsets can be grounded in the
real data rather than guessed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "projects")
from biomech.session import load_session  # noqa: E402

DATA = Path("projects/data/S001")

FOOT_LABELS = [
    "RHEE", "RHEE2", "RHEE3", "RMTH1", "RMTH5", "RTOE", "RHLX", "RANK", "RMANK",
    "LHEE", "LHEE2", "LHEE3", "LMTH1", "LMTH5", "LTOE", "LHLX", "LANK", "LMANK",
]


def audit(session, name):
    labels = list(session.marker_labels)
    pts = np.asarray(session.markers, dtype=np.float64)  # (F, M, 3)
    print(f"\n=== {name}: {pts.shape[0]} frames, {len(labels)} markers ===")
    idx = {lbl: i for i, lbl in enumerate(labels)}
    means = {}
    for lbl in FOOT_LABELS:
        if lbl not in idx:
            print(f"  {lbl:6s}  MISSING from labels")
            continue
        col = pts[:, idx[lbl], :]
        finite = np.isfinite(col).all(axis=1)
        frac = finite.mean()
        mean = np.nanmean(col, axis=0)
        means[lbl] = mean
        print(f"  {lbl:6s}  present {frac*100:5.1f}%   mean(lab m)= "
              f"[{mean[0]:+.3f} {mean[1]:+.3f} {mean[2]:+.3f}]")
    # Key right-foot distances (lab frame, so just Euclidean).
    def d(a, b):
        if a in means and b in means:
            return float(np.linalg.norm(means[a] - means[b]))
        return float("nan")
    print("  -- right foot distances (m) --")
    print(f"     HEE-TOE  {d('RHEE','RTOE'):.3f}   HEE-MTH5 {d('RHEE','RMTH5'):.3f}   "
          f"HEE-MTH1 {d('RHEE','RMTH1'):.3f}")
    print(f"     MTH1-MTH5 {d('RMTH1','RMTH5'):.3f}  TOE-HLX  {d('RTOE','RHLX'):.3f}   "
          f"ANK-MANK {d('RANK','RMANK'):.3f}")
    print(f"     HEE-HEE2 {d('RHEE','RHEE2'):.3f}   HEE-HEE3 {d('RHEE','RHEE3'):.3f}   "
          f"HEE2-HEE3 {d('RHEE2','RHEE3'):.3f}")
    return means


def main():
    static = load_session(
        c3d_path=str(DATA / "Cal 101.v3d.c3d"),
        subject_mp_path=str(DATA / "S001.mp"),
        filter_cutoff_hz=None,
    )
    audit(static, "static (Cal 101)")

    trial = load_session(
        c3d_path=str(DATA / "Trial 101.v3d.c3d"),
        subject_mp_path=str(DATA / "S001.mp"),
        filter_cutoff_hz=None,
    )
    audit(trial, "dynamic (Trial 101)")


if __name__ == "__main__":
    main()
