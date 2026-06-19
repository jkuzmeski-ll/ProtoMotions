# Biomechanics Retargeting

Tools for turning lab motion capture (Visual3D / Vicon C3D) into subject-specific
MS-Human lower-body digital twins and ProtoMotions `.motion` files.

All commands are run from the repository root with the project virtualenv
(`.venv/bin/python`). The MS-Human marker-IK retargeter requires the Newton
simulator (`pip install "newton[examples]"`).

## Folder Layout

```
data/biomechanics_retargeting/
  scripts/      # CLI tools (this pipeline)
  utils/        # shared helpers: c3d_io, treadmill_overground
  configs/      # marker scaling / site configs (JSON)
  retargeted/   # outputs: proto/ (.motion), plots/, *.rms.json, *_scaling_report.json
```

Robot configs (`ms_human_lower*`) and MJCF assets stay under `protomotions/`.

## Pipeline Overview

```
C3D mocap ─▶ (0) convert base MJCF ─▶ (1) scale to subject ─▶ (2) add marker sites
          ─▶ (3) edit/verify sites ─▶ (4) marker-IK retarget ─▶ (5) post-process ─▶ (6) play back
```

---

## 0. Convert MS-Human-700 to a ProtoMotions base asset (one-time)

Only needed once, from a local clone of the MS-Human-700 repo. Produces
`MS-Human-700-Locomotion-Simple.xml` (the base model everything scales from).

```bash
.venv/bin/python data/biomechanics_retargeting/scripts/convert_ms_human_lowerbody_to_proto.py \
    /path/to/MS-Human-700 \
    --output-dir protomotions/data/assets/mjcf/ms_human_700
```

## 1. Scale the base model to a subject

Uses a static-pose window in the subject's C3D to size pelvis/thigh/shank/foot.
Writes a scaled MJCF (into `protomotions/...`) and a scaling report (into
`retargeted/`).

```bash
.venv/bin/python data/biomechanics_retargeting/scripts/scale_ms_human_to_subject.py \
    c3dproto/S003/S003.c3d \
    --marker-config data/biomechanics_retargeting/configs/ms_human_700_cal101_marker_scaling_config.json \
    --output-xml protomotions/data/assets/mjcf/ms_human_700/MS-Human-700-Locomotion-S003.xml \
    --report data/biomechanics_retargeting/retargeted/S003_scaling_report.json \
    --static-start 1 --static-end 100
```

Optionally strip the upper body to a pelvis-and-legs-only asset:

```bash
.venv/bin/python data/biomechanics_retargeting/scripts/create_ms_human_lower_only_asset.py \
    --input-xml protomotions/data/assets/mjcf/ms_human_700/MS-Human-700-Locomotion-S003.xml \
    --output-xml protomotions/data/assets/mjcf/ms_human_700/MS-Human-700-Locomotion-S003-LowerOnly.xml
```

## 2. Attach calibration marker sites to the scaled MJCF

Adds `mocap_*` `<site>` elements from a static C3D window so the IK retargeter
has anatomical marker offsets. Prefer doing this during scaling with
`scale_ms_human_to_subject.py --add-marker-sites` so the subject MJCF is born
with the fixed sites:

```bash
.venv/bin/python data/biomechanics_retargeting/scripts/scale_ms_human_to_subject.py \
    c3dproto/S003/S003.c3d \
    --marker-config data/biomechanics_retargeting/configs/ms_human_700_cal101_marker_scaling_config.json \
    --output-xml protomotions/data/assets/mjcf/ms_human_700/MS-Human-700-Locomotion-S003.xml \
    --report data/biomechanics_retargeting/retargeted/S003_scaling_report.json \
    --static-start 1 --static-end 100 \
    --add-marker-sites --strict-marker-sites
```

If the scaled/lower-only MJCF already exists, this helper updates that asset
in-place without creating a parallel MJCF:

```bash
.venv/bin/python data/biomechanics_retargeting/scripts/add_mjcf_marker_sites_from_c3d.py \
    c3dproto/S003/S003.c3d \
    --mjcf protomotions/data/assets/mjcf/ms_human_700/MS-Human-700-Locomotion-S003-LowerOnly.xml \
    --marker-config data/biomechanics_retargeting/configs/ms_human_700_cal101_marker_scaling_config.json
```

## 3. Inspect / edit marker sites (optional)

Browser-based editor (Viser, drag markers in 3D):

```bash
.venv/bin/python data/biomechanics_retargeting/scripts/edit_mjcf_marker_sites_viser.py \
    --mjcf protomotions/data/assets/mjcf/ms_human_700/MS-Human-700-Locomotion-S003-LowerOnly.xml \
    --site-prefix mocap_
```

MuJoCo passive-viewer editor (keyboard hotkeys):

```bash
.venv/bin/python data/biomechanics_retargeting/scripts/edit_mjcf_marker_sites.py \
    --mjcf protomotions/data/assets/mjcf/ms_human_700/MS-Human-700-Locomotion-S003-LowerOnly.xml
```

Overlay scaled model + static markers to verify the fit:

```bash
.venv/bin/python data/biomechanics_retargeting/scripts/static_calibration_marker_viewer.py \
    c3dproto/S003/S003.c3d \
    --robot-name ms_human_lower_s003 --simulator newton
```

## 4. Marker-IK retarget a trial (main step)

Solves per-frame poses that match the C3D markers (Newton Warp IK backend).
Writes a ProtoMotions `.motion` into `retargeted/proto/`.

```bash
.venv/bin/python data/biomechanics_retargeting/scripts/ik_retarget_c3d_to_ms_human_lower.py \
    c3dproto/S003/S003.c3d \
    --robot-name ms_human_lower_s003 \
    --output data/biomechanics_retargeting/retargeted/proto/S003_marker_newton.motion \
    --report data/biomechanics_retargeting/retargeted/S003_marker_newton.rms.json \
    --backend newton --newton-iterations 60
```

### Fixed-site marker workflow

For marker overlays that should line up with the mesh, use fixed MJCF `mocap_*`
sites: first bake the sites from a **static/neutral** calibration window, then
retarget the dynamic trial with `--marker-offset-source site`. Do **not** bake
sites from a full dynamic gait trial: the resulting trial-average sites encode
soft-tissue motion and can bias the model toward bent knees.

For S081, bake the s081-clusters sites from the subject's dedicated Cal 101
static standing trial directly into the subject's default MJCF. Cal 101
contains the same cluster marker set as Trial 101, but without gait-phase
soft-tissue motion. Prefer doing this during scaling:

```bash
.venv/bin/python data/biomechanics_retargeting/scripts/scale_ms_human_to_subject.py \
    "c3dproto/Cal 101.v3d.c3d" \
    --marker-config data/biomechanics_retargeting/configs/ms_human_700_cal101_marker_scaling_config.json \
    --output-xml protomotions/data/assets/mjcf/ms_human_700/MS-Human-700-Locomotion-S081.xml \
    --report data/biomechanics_retargeting/retargeted/S081_scaling_report.json \
    --add-marker-sites --strict-marker-sites
```

Then create/update the lower-only asset from that scaled MJCF. The lower-body
extraction preserves the pelvis/leg `mocap_*` sites. If the lower-only asset
already exists and you only need to refresh sites, update it in-place:

```bash
.venv/bin/python data/biomechanics_retargeting/scripts/add_mjcf_marker_sites_from_c3d.py \
    "c3dproto/Cal 101.v3d.c3d" \
    --mjcf protomotions/data/assets/mjcf/ms_human_700/MS-Human-700-Locomotion-S081-LowerOnly.xml \
    --marker-config data/biomechanics_retargeting/configs/ms_human_700_cal101_marker_scaling_config.json \
    --strict
```

Then retarget the full dynamic trial against those fixed static sites. No
`--asset-mjcf` override is needed because the robot config already points at the
updated `MS-Human-700-Locomotion-S081-LowerOnly.xml`:

```bash
.venv/bin/python data/biomechanics_retargeting/scripts/ik_retarget_c3d_to_ms_human_lower.py \
    "c3dproto/Trial 101.v3d.c3d" \
    --robot-name ms_human_lower_s081 \
    --marker-set s081-clusters \
    --marker-offset-source site \
    --max-frames 0 \
    --newton-iterations 160 \
    --output data/biomechanics_retargeting/retargeted/proto/S081_trial101_marker_newton_site.motion \
    --report data/biomechanics_retargeting/retargeted/S081_trial101_marker_newton_site.rms.json
```

The first command updates the default robot MJCF in-place; it does not create a
new parallel asset. The final dynamic motion to inspect is
`S081_trial101_marker_newton_site.motion`.

During `env_kinematic_playback.py`, `--robot-name ms_human_lower_s081` still logs
the robot config's default MJCF path. That path should now be the updated asset
with the Cal 101 `mocap_*` sites, so the solve and playback use the same robot.

`ik_retarget_c3d_to_ms_human_lower.py` refuses to export marker sites from
windows longer than 500 loaded frames by default. Use `--max-frames`,
`--start-frame`, or `--end-frame` to select a static calibration window; pass
`--allow-dynamic-marker-site-export` only when dynamic-window site baking is
intentional.

Quick Visual3D joint-angle baseline (no IK, useful warm start / sanity check):

```bash
.venv/bin/python data/biomechanics_retargeting/scripts/retarget_c3d_to_ms_human_lower.py \
    c3dproto/S003/S003.c3d \
    --robot-name ms_human_lower_s003 \
    --output data/biomechanics_retargeting/retargeted/proto/S003_v3d_angles.motion
```

## 5. Post-process: treadmill to overground (optional)

Converts an in-place treadmill `.motion` into forward overground travel.

```bash
.venv/bin/python data/biomechanics_retargeting/scripts/map_treadmill_motion_to_overground.py \
    data/biomechanics_retargeting/retargeted/proto/S003_marker_newton.motion \
    --output data/biomechanics_retargeting/retargeted/proto/S003_marker_newton_overground.motion \
    --robot-name ms_human_lower_s003 \
    --speed-mps 1.2
```

## 6. Play back / visualize a retargeted motion

```bash
.venv/bin/python examples/env_kinematic_playback.py \
    --experiment-path=examples/experiments/mimic/mlp.py \
    --motion-file data/biomechanics_retargeting/retargeted/proto/S003_marker_newton.motion \
    --robot-name ms_human_lower_s003 \
    --simulator newton --num-envs 1
```

---

## Notes

- Replace `s003` / `S003` and the C3D path with your subject (e.g. `ms_human_lower_s081`).
- Each script supports `--help` for the full option list.
- Robot config names: `ms_human_lower`, `ms_human_lower_s003`, `ms_human_lower_s081`.
- `retarget_g1_motion_to_ms_human_lower.py` is a separate G1-to-MS-Human demo, not part of the C3D pipeline.
