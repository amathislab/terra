# 1. Motion files and transformation

TERRA first turns source recordings into a body motion with known units, axes, frame
rate, and identity. AMASS already provides compatible SMPL-H archives. Marker recordings
need fitting before retargeting; the dataset converters also write a manifest of
accepted and failed fits so you can inspect cohort membership.

## Supported inputs

| Format | Typical source | Entry point | Extra input |
|---|---|---|---|
| `.npz` SMPL-H | AMASS or a TERRA converter | `terra retarget` directly | Neutral SMPL-H model |
| `.c3d` markers | Marker capture | `terra retarget` or dataset converter | Marker-fitting model root; `--extra c3d` |
| `.trc` markers | Marker capture, e.g. Gait120 | `terra retarget` or dataset converter | Marker-fitting model root; correct `--trc-up-axis` |
| `.mat` markers | Study-specific MATLAB container | `terra retarget` or dataset converter | Marker-fitting model root and `--mat-schema` JSON |

A marker file does not by itself describe its subject shape or the target MyoFullBody
trajectory. The marker fitter estimates a SMPL-H motion, then TERRA retargets it. For a
first run, use an existing AMASS `.npz` and skip this fitting stage.

## Check one SMPL-H archive

`terra.retarget` accepts the following fields in an `.npz`: `trans` with shape
`(frames, 3)` in metres, a shape vector `betas` with at least 10 values, scalar `gender`
(`neutral`, `female`, or `male`), `poses` with at least 66 columns or canonical
`pose_aa` with 72 columns, and a positive scalar `fps`, `mocap_framerate`, or
`mocap_frame_rate`. At least three frames and finite values are required. For AMASS
`poses`, TERRA tracks the 22-body subset and fills the two hand-root slots with zeros.

After [installation](installation.md) and [data setup](data.md), inspect the KIT
tutorial motion with the same loader used by retargeting:

```bash
export TERRA_DATA_ROOT="$HOME/terra-data"
export MOTION_FILE="$TERRA_DATA_ROOT/AMASS/KIT/3/upstairs04_poses.npz"
python - <<'PY'
import os
from terra.smplh import load_smplh_motion
motion = load_smplh_motion(os.environ["MOTION_FILE"])
print("frames:", len(motion["trans"]))
print("fps:", motion["fps"])
print("pose shape:", motion["pose_aa"].shape)
print("translation shape:", motion["trans"].shape)
print("gender:", motion["gender"])
PY
```

The loader uses NumPy with pickle disabled and reports missing fields, shape mismatches,
invalid frame rates, and non-finite values. A `model.npz` is a body model, not a motion
archive.

## Retarget marker files directly

A direct marker run needs a surface model for fitting and the neutral SMPL-H model
for robot retargeting. The default surface fit uses SMPL-X, which requires its
[separately downloaded model](data.md#marker-fitting-model-choice). The examples
below explicitly fit a SMPL-H surface so the already downloaded neutral SMPL-H
model can serve both roles. Export `TERRA_MODEL_ROOT` and `TERRA_ARTIFACT_ROOT` as
shown in [Data setup](data.md#the-first-motion-layout); for C3D, also install the
[`c3d` extra](installation.md#choose-an-environment). These are single-motion examples; use dataset converters for larger selections.

For a C3D with enough recognizable body markers:

```bash
terra retarget /absolute/path/to/trial.c3d \
  --output-root "$TERRA_ARTIFACT_ROOT/direct-markers" \
  --name Study/Subject/Trial \
  --c3d-model-path "$TERRA_MODEL_ROOT" \
  --smpl-model-path "$TERRA_MODEL_ROOT" \
  --c3d-options '{"surface_model_type":"smplh"}'
```

For a TRC, choose the vertical-axis conversion that matches the recording.
`y` applies the Gait120 `[X, -Z, Y]` transform; `z` preserves recorded XYZ.
TERRA normalizes marker coordinates to metres and Z-up before fitting. Missing
or all-zero TRC marker triples become NaNs; the fitter still needs enough
recognized, observed markers to estimate the body:

```bash
terra retarget /absolute/path/to/trial.trc \
  --output-root "$TERRA_ARTIFACT_ROOT/direct-markers" \
  --name Study/Subject/TRCTrial \
  --c3d-model-path "$TERRA_MODEL_ROOT" \
  --smpl-model-path "$TERRA_MODEL_ROOT" \
  --c3d-options '{"surface_model_type":"smplh"}' \
  --trc-up-axis y
```

MAT containers have no standard marker layout. TERRA reads SciPy-compatible MAT
files; export a MATLAB v7.3/HDF5 file as v7.2 before using this path. Suppose
your file contains a `markers` tensor shaped `(time, markers, coordinates)`,
`marker_labels`, and `marker_rate`. A schema for millimetres recorded Y-up is:

```json
{
  "version": 1,
  "positions_path": "markers",
  "axis_order": "tmc",
  "labels_path": "marker_labels",
  "fps_path": "marker_rate",
  "units": "mm",
  "axes": ["x", "-z", "y"]
}
```

Save this as `trial-schema.json`, changing the field names, units, and axes to
match your recording. Check extraction before fitting:

```bash
python - <<'PY'
from terra.mat import load_mat_markers
motion = load_mat_markers("/absolute/path/to/trial.mat", "trial-schema.json")
print(len(motion.positions), "frames;", len(motion.labels), "markers;", motion.fps, "Hz")
PY
terra retarget /absolute/path/to/trial.mat \
  --output-root "$TERRA_ARTIFACT_ROOT/direct-markers" \
  --name Study/Subject/MATTrial \
  --c3d-model-path "$TERRA_MODEL_ROOT" \
  --smpl-model-path "$TERRA_MODEL_ROOT" \
  --c3d-options '{"surface_model_type":"smplh"}' \
  --mat-schema trial-schema.json
```

Nested MATLAB records may need `--mat-selector NAME=INDEX`; see
[`terra.mat`](../src/terra/mat.py) and `terra retarget --help`. Marker fitting
requires a recognizable marker set; inspect the reported fit and analysis before
using the trajectory. For dataset conversion, examine failed fits in the generated
manifest rather than silently dropping them.

## Convert a dataset

For a complete example, see [Darmstadt download to pre-trained policy playback](dataset-workflows.md#non-amass-dataset-to-a-pre-trained-policy).

Run a converter when you need many recordings, reproducible selection, and conversion
manifests:

```bash
terra convert --help
terra convert gait120 --help
terra convert darmstadt --help
terra convert vielemeyer --help
terra convert prism --help
```

Each converter's `--help` gives its input and output roots. Converters write SMPL-H
archives and a manifest identifying accepted fits and calibration clips. Marker
coordinates are converted to metres in a Z-up frame; physiological streams retain
explicit rates and alignment. AMASS needs no conversion.

Next: [Terrain reconstruction](terrain-reconstruction.md) or the
[one-motion retarget walkthrough](motion-retargeting.md).
