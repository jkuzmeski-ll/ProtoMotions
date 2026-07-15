# biomech — gold-standard capture ingestion (Milestone 1)

Local, dependency-light ingestion of mocap + instrumented-treadmill captures into
one explicitly-framed, unit-checked, time-aligned `CaptureSession`. This is the
first stage of the pipeline that turns real captures into biomechanically
gold-standard skeleton motions for Newton contact-model research.

Everything runs locally. No `ezc3d`/`c3d`/web service is required — only `numpy`.

## What Milestone 1 delivers

- A full C3D reader (`io/c3d.py`) that returns **dense, frame-indexed** marker
  arrays in meters (NaN in gaps) plus all analog channels at the analog rate,
  with C3D scaling applied. Supports DEC/Intel float and scaled-integer storage.
- Force-plate extraction (`io/force_plate.py`): per-belt GRF, COP (world frame),
  and free vertical moment from the C3D `FORCE_PLATFORM` group.
- Treadmill belt-speed reader/resampler (`io/treadmill.py`) for the Visual3D
  `SPEEDCHANGE` exports.
- A unified `CaptureSession` (`session.py`) with two explicit time bases
  (`t_point` at mocap rate, `t_analog` at analog rate), belt speed resampled onto
  both, parsed gait events, and optional `.mp` subject metadata.
- Explicit frame/unit conventions (`frames.py`) attached to every session.
- Self-contained tests (`tests/`) + a runner that needs no pytest.

## Conventions (enforced at ingest)

- **Units:** meters, Newtons, Newton-metres, seconds. Marker mm→m and moment
  N·mm→N·m conversions happen at read time.
- **Frames:** `LAB`, `FORCE_PLATE`, `TREADMILL`, `WORLD`. For the S001 capture the
  lab frame is **Z-up, right-handed, meters** and `WORLD == LAB` (identity).
  OpenSim (Y-up) conversion is intentionally *not* applied here; it is a
  downstream skeleton concern.
- **GRF sign:** raw channels record force *onto the plate* (vertical negative in
  stance). `ForcePlate.grf` is the force *onto the subject* (+Z up in stance).
- **COP** is returned in the world frame and is `NaN` below the contact
  threshold (`fz_threshold`, default 20 N), since `COP = M/Fz` is unreliable at
  low load.
- **Filtering:** a zero-phase (`filtfilt`) 4th-order Butterworth low-pass is applied
  at ingest to **both** kinematics (markers, at the point rate) and kinetics (force/
  moment, at the analog rate) at a matched **20 Hz** cutoff (`load_session(...,
  filter_cutoff_hz=20.0, filter_order=4)`; pass `filter_cutoff_hz=None` to disable).
  Matching the two bandwidths avoids inverse-dynamics artifacts (Kristianslund et al.
  2012). Marker NaN gaps are preserved (each contiguous run is filtered separately);
  COP/free-moment are derived from the filtered force/moment. The applied filter is
  recorded in `session.filter_info` and `session.report()["preprocessing"]`.

## Usage

From the repository root:

```bash
# Print + save a Milestone-1 report for S001 (defaults point at projects/data/S001).
python projects/biomech/load_capture.py \
    --report "projects/data/S001/Trial101_session_report.json"

# Run the test suite (no pytest needed).
python projects/biomech/run_tests.py
```

Programmatically:

```python
import sys; sys.path.insert(0, "projects")
from biomech.session import load_session

session = load_session(
    c3d_path="projects/data/S001/Trial 101.v3d.c3d",
    left_belt_path="projects/data/S001/LeftBelt101.txt",
    right_belt_path="projects/data/S001/RightBelt101.txt",
    subject_mp_path="projects/data/S001/S001.mp",
    # belt_rate_hz=300.0,  # set once the true belt log rate is confirmed
)

markers = session.markers                 # [n_frames, n_markers, 3] m, NaN gaps
grf = session.force_plates[0].grf         # [n_analog, 3] N, +Z up in stance
cop = session.force_plates[0].cop_world   # [n_analog, 3] m, NaN in swing
belt = session.belt_speed_analog["left"]  # [n_analog] m/s on the analog timeline
```

## S001 capture facts (Motek split-belt treadmill)

- Markers: 72 @ 100 Hz. Analog: 249 channels @ 2000 Hz (20 samples/frame).
- Two AMTI-type (C3D TYPE 2) belts; channels 1–6 = belt at +x, 7–12 = belt at −x.
  Belts share the inner edge at x = 0, so per-plate COP tails legitimately cross
  the boundary during weight transfer.
- Belt-speed logs carry **no sample rate** and use a separate clock. The loader
  infers ~298 Hz from the C3D duration and records a warning; pass
  `belt_rate_hz` once the true rate is confirmed.
- `Trial 101` has no stored C3D gait events (`EVENT:USED = 0`).

## Known open items (for later milestones)

- Confirm the true belt-log sample rate and mocap↔belt time offset.
- Assign each belt to left/right foot (needs subject facing direction).
- COP/free-moment are validated for axis-aligned plates only; rotated plates
  would need the plate→world rotation applied (flagged in `ForcePlate.warnings`).
```
