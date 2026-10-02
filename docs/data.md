# Data and model setup

TERRA ships code and configuration examples. Source recordings and body models
have their own access terms. Start with **one** AMASS-compatible SMPL-H motion and
the neutral SMPL-H model; the other datasets are needed for their respective conversions.

## The first-motion layout

Choose any writable directories and export:

```bash
export TERRA_DATA_ROOT="$HOME/terra-data"
export TERRA_MODEL_ROOT="$HOME/terra-models/smplh"
export TERRA_ARTIFACT_ROOT="$HOME/terra-results"
mkdir -p "$TERRA_DATA_ROOT/AMASS" "$TERRA_ARTIFACT_ROOT"
```

1. Register or sign in at [AMASS Downloads](https://amass.is.tue.mpg.de/download.php),
   accept its terms, and download a collection in SMPL-H format. Extract it under
   `$TERRA_DATA_ROOT/AMASS`, retaining the collection/subject/motion directories.
2. Register or sign in at the [MANO/SMPL-H downloads](https://mano.is.tue.mpg.de/download.php)
   and accept the provider terms. Download **both** the SMPL-H models (the archive
   containing `smplh/neutral/model.npz`) and MANO v1.2 (containing
   `mano_v1_2/models/MANO_LEFT.pkl` and `MANO_RIGHT.pkl`). Extract both under
   `$TERRA_MODEL_ROOT`, retaining those directories, then run:

   ```bash
   python scripts/models/make_smplh_neutral.py "$TERRA_MODEL_ROOT"
   ```

   This combines the neutral body arrays and hand PCA data into
   `$TERRA_MODEL_ROOT/SMPLH_NEUTRAL.pkl`. It needs NumPy from the base environment;
   chumpy is not needed. The male/female `.pkl` files in the MANO archive are not
   the neutral model. Keep the generated model local under its provider's terms.
3. Find an extracted motion file and check both paths using the commands below. The
   model resolver also accepts a `smplh/` child directory, but the quickstart assumes
   the direct layout.

The provider download pages require sign-in, so TERRA does not provide a script that
fetches these licensed files for you.

```text
$TERRA_DATA_ROOT/
└── AMASS/
    └── <collection>/<subject>/<motion>_poses.npz
$TERRA_MODEL_ROOT/
└── SMPLH_NEUTRAL.pkl
$TERRA_ARTIFACT_ROOT/
└── quickstart/                  # created by the quickstart
```

Choose an actual archive; the names in the tree are placeholders:

```bash
find "$TERRA_DATA_ROOT/AMASS" -type f -name '*_poses.npz' | head
test -f "$TERRA_MODEL_ROOT/SMPLH_NEUTRAL.pkl"
```

Set `MOTION_FILE` to a line printed by `find`, then use the
[README quickstart](../README.md#first-result-retarget-one-smpl-h-motion). The
[motion-file guide](motion-files.md) explains how to inspect archive fields safely.
SMPL-H model files must contain the required hand PCA data; an arbitrary AMASS
`model.npz` is not a replacement.

## Marker-fitting model choice

Direct `terra retarget` calls on C3D, TRC, or MAT markers default to a **SMPL-X**
surface fit. For that default, register and accept the terms at the
[SMPL-X download](https://smpl-x.is.tue.mpg.de/) and place a neutral
`SMPLX_NEUTRAL.pkl` or `SMPLX_NEUTRAL.npz` in a separate model directory. Pass
that directory as `--c3d-model-path`; keep `--smpl-model-path` pointed at the
neutral SMPL-H model used for robot retargeting.

You can fit markers with the SMPL-H model already downloaded for the first-motion
walkthrough by passing `--c3d-model-path "$TERRA_MODEL_ROOT"` and
`--c3d-options '{"surface_model_type":"smplh"}'`. The
[direct marker examples](motion-files.md#retarget-marker-files-directly) use this
choice. The dataset converters use SMPL-H marker fitting and take the model
root from `--smpl-model-path`.

## Full-source datasets

For dataset converters, arrange the extracted inputs under
`TERRA_DATA_ROOT` as follows. Download and accept each source's terms before running its
converter.

| Source | Obtain from | Expected directory |
|---|---|---|
| AMASS SMPL-H motions | [AMASS](https://amass.is.tue.mpg.de/) | `AMASS/` |
| Gait120 markers | [Motion capture](https://doi.org/10.6084/m9.figshare.27677016.v1) | `Gait120-original/extracted/` |
| Darmstadt stairs | [TU Darmstadt](https://doi.org/10.48328/tudatalib-1182) | `Darmstadt-Stair-Ambulation/` |
| Vielemeyer ramps | [Figshare](https://doi.org/10.6084/m9.figshare.29300888.v1) | `Vielemeyer-Ramp-Walking/` |
| PRISM | [PRISM dataset repository](https://github.com/RyosukeHori/PRISM) | `PRISM/` |

Before starting a full conversion, spot-check the paths the loaders actually read:

```text
Gait120-original/extracted/S001/MotionCapture/LevelWalking/TRC/Trial01/Step01.trc
Darmstadt-Stair-Ambulation/Marker1.mat
Darmstadt-Stair-Ambulation/touchdowns/Processed/Touchdowns/Touchdowns1.mat
Vielemeyer-Ramp-Walking/raw/Ref_*/<condition>/*.c3d
```

Darmstadt conversion needs the marker files and trial touchdown records in
`Processed.zip`. EMG, force files, and group averages are not required for motion
conversion. Gait120 conversion needs only the TRC marker recordings.
The Vielemeyer loader also accepts `raw-incomplete/Ref_*/<condition>/*.c3d`.
These are path patterns for checking extraction, not a requirement to create a sample
file when that subject or trial is absent.

For PRISM, submit the access form linked from the dataset repository, then extract the
provider archive so a take is at
`$TERRA_DATA_ROOT/PRISM/data/PRISM/subj001/take002.pkl`. The PRISM converter
accepts that extraction root or the inner `data/PRISM` directory. Check with
`find "$TERRA_DATA_ROOT/PRISM" -path "*/subj*/take*.pkl" | head` before converting.

Raw marker datasets need conversion to TERRA's SMPL-H motion archives; see
[Motion files](motion-files.md#convert-a-dataset).

The marker converters default to an L2 pose prior, 100 Stage-I iterations, 12 calibration
frames, and 80 Stage-II iterations with residual-gate retries at 50 Hz. You can configure
a MoSh++ GMM prior to use a different fitting objective. SMPL-H fitting does not need
the SMPL-X head-marker correction asset.

This repository does not include licensed raw data, body models, converted trajectories,
or policy weights. Obtain inputs from their providers, run a converter where needed,
and review its output manifest and failed fits before reconstruction or retargeting.
