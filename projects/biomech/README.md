# biomech - C3D to ProtoMotions

The unified pipeline turns marker and instrumented-treadmill C3D data into a
subject-specific Rajagopal MJCF and a simulator-body-aligned ProtoMotions motion.
It connects capture ingestion, marker cleanup, symmetric subject scaling, static foot
calibration, robust marker fitting, treadmill-to-overground conversion, measured
ground/contact data, subject mass, collision geometry, and ProtoMotions validation in
one command.

## Unified pipeline

From the repository root, the S001 fidelity path is:

```powershell
.venv\Scripts\python.exe projects/biomech/c3d_to_protomotions.py `
    --trial "projects/data/S001/Trial 101.v3d.c3d" `
    --static "projects/data/S001/Cal 101.v3d.c3d" `
    --left-belt "projects/data/S001/LeftBelt101.txt" `
    --right-belt "projects/data/S001/RightBelt101.txt" `
    --speedchange "projects/data/S001/Speedchange101.txt" `
    --subject-mp "projects/data/S001/S001.mp" `
    --subject-id S001 --phase walk --device cuda:0
```

The command creates a content-addressed directory under `outputs/biomech/` with:

- a subject-mass-matched base MJCF;
- subject-sized box and sphere foot-collision MJCF variants;
- a ProtoMotions `.motion` with exact simulator body/DOF order and measured contacts;
- `fit/reconstruction.npz` containing raw marker-optimal and delivered poses, scales,
  marker offsets, weights, GRF/COP, and correction provenance;
- `manifest.json` containing input hashes, all settings, fit/contact metrics, artifact
  hashes, quality gates, and the exact training environment/command.

Use `--frames 150` for a fast integration/smoke bundle. Omitting it, as above, reconstructs
the complete selected protocol phase.

Running the same command again verifies and reuses the completed bundle. Content-addressed
bundles are immutable: `--force` recomputes and verifies equivalent arrays and deterministic
XML rather than replacing a live bundle. By default the pipeline fails rather than silently dropping
requested belt, force, mass, contact, or validation stages.

The default fitting path is the locally validated S001 Plug-in-Gait mapping. A new marker
protocol needs an explicit `MarkerMap`; it is not guessed from label names.

`--anthropometric-prior` is available for experiments but is off by default: the current
`.mp` length prior degraded S001 marker/gait accuracy in local benchmarks. Subject mass is
still always applied when supplied.

The manifest reports marker-optimal and final foot-corrected errors separately. Both must
pass. The final correction may improve the anatomical sole/contact frame while worsening
shoe-marker reprojection, so a good raw fit never hides a poor delivered motion.
Measured contact labels use the configured fixed vertical-force threshold (50 N by
default), plus each heel/toe collider's height, rather than a clip-relative force peak.

### Training the generated subject

Read the three values from `manifest.json` under `training`. In PowerShell:

```powershell
$env:BIOMECH_ASSET_ROOT="<bundle>/assets"
$env:BIOMECH_ASSET_STEM="<manifest export.asset_stem>"
$env:BIOMECH_FOOT_COLLISION="boxes"
python protomotions/train_agent.py --robot-name biomech --simulator newton `
    --experiment-path projects/biomech/experiments/mimic_newton.py `
    --experiment-name biomech_subject_mimic `
    --motion-file "<bundle>/motions/<asset-stem>.motion" `
    --num-envs 1024 --batch-size 16384 --ngpu 1
```

`--dynamics-diagnostics` adds a corrected-frame inverse-dynamics shadow report. It does
not alter the delivered motion or inertial parameters: measured free-moment sign,
wrench-preserving analog downsampling, and real-data RRA/inertial export are not yet
validated strongly enough to be production transformations.

The coupled Rajagopal knee retains all dependent coordinates in observations and motion
references, but they are passive in Newton. Only the 25 independent coordinates receive
PD actuators; equality constraints generate the eight dependent spline coordinates.

### Validated S001 Result

The full documented walk command was run over frames 1469-7469 (6,000 frames, 59.99 s).
The resulting local bundle is `outputs/biomech/biomech_s001_9778c180e2f4`.

- Raw Euclidean marker RMS: 23.52 mm; median anatomical-marker RMS: 9.58 mm.
- Delivered foot-corrected RMS: 31.22 mm; anatomical median: 19.88 mm.
- Exported mass: 81.650018 kg; measured loaded GRF: 1.003 times bodyweight.
- Loaded-stance plantar-patch p95 slip: 0.102 m/s right, 0.137 m/s left.
- Contact-label precision/recall: 1.0/1.0 for both feet.
- ProtoMotions MotionLib validation passed at 38 bodies and 33 state DOFs.

A bounded Newton PPO epoch on this full bundle (32 environments, 1,024 collected frames)
completed and saved a checkpoint. It had finite reward/losses, contact-match reward 0.973,
and zero actor/critic bad-gradient counts. This is a runtime smoke test, not a converged
policy evaluation.

## Capture ingestion

Everything runs locally. The C3D parser itself needs no `ezc3d`, `c3d`, or web service;
the full fitting/export command also uses the repository's NumPy/SciPy, Warp, MuJoCo,
PyTorch, and ProtoMotions environment.

### What ingestion delivers

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
