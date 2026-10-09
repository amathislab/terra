# Process and evaluate a motion collection

`terra run` applies one dataset config to an explicit motion selection. `terra
evaluate` scores the resulting trajectories on CPU; PPO policy evaluation is a
separate GPU workflow described in [Policy training](policy-training.md).

## Non-AMASS dataset to a pre-trained policy

This example uses a Darmstadt stair-descent recording. Run commands from the
repository root with the TERRA environment active. The path is:

```text
download markers → convert to SMPL-H → fit terrain and retarget
                 → prepare a policy dataset → play the downloaded checkpoint
```

This workflow requires the neutral SMPL-H model from [Data and model setup](data.md#the-first-motion-layout).
The base environment supports this MATLAB dataset and CPU playback; no `c3d`
or `cuda` extra is needed. Marker conversion can be slow on CPU.

### 1. Download the recordings

Keep your existing `TERRA_DATA_ROOT`, `TERRA_MODEL_ROOT`, and
`TERRA_ARTIFACT_ROOT` settings. If you have not set them, use the directories in
[Data and model setup](data.md#the-first-motion-layout). `TERRA_MODEL_ROOT`
must contain `SMPLH_NEUTRAL.pkl`.

Download `Preprocessed.zip` and `Processed.zip` from the
[Darmstadt dataset page](https://doi.org/10.48328/tudatalib-1182), under its
access terms. Extract the `Marker*.mat` files from `Preprocessed/Marker/` in
`Preprocessed.zip` directly into `$TERRA_DATA_ROOT/Darmstadt-Stair-Ambulation/`.
Extract `Processed/Touchdowns/` from `Processed.zip` into that directory's
`touchdowns/` subdirectory, retaining the `Processed/Touchdowns/` path. The converter needs the trial
records in `Processed/Touchdowns/`, not EMG, forces, or group averages.

For a smaller first run, keep only subject 5's `Marker5.mat` in the input root.
The converter processes every subject whose `Marker*.mat` file is present, and
fits all available trials for those subjects. It has no single-motion filter.
Check this layout before conversion:

```bash
test -f "$TERRA_MODEL_ROOT/SMPLH_NEUTRAL.pkl"
test -f "$TERRA_DATA_ROOT/Darmstadt-Stair-Ambulation/Marker5.mat"
test -f "$TERRA_DATA_ROOT/Darmstadt-Stair-Ambulation/touchdowns/Processed/Touchdowns/Touchdowns5.mat"
```

### 2. Convert markers to body motion

```bash
terra convert darmstadt \
  --input-root "$TERRA_DATA_ROOT/Darmstadt-Stair-Ambulation" \
  --output-root "$TERRA_ARTIFACT_ROOT/darmstadt/smplh" \
  --smpl-model-path "$TERRA_MODEL_ROOT" \
  --device cpu
```

This fits SMPL-H to the markers, crops stair traversals using touchdown records,
and writes `.npz` body motions plus `manifest.csv`. On a machine with a compatible
PyTorch CUDA device, use `--device cuda` to accelerate marker fitting. This is
separate from the JAX `cuda` extra used for PPO training.

Check the conversion summary and `manifest.csv`. A rejected fit is recorded with
`fit_passed=False`; do not use it for the policy. The converter exits with code 2
if any fit fails, but retains successful fits. You can use a successful motion
without rerunning the rejected ones. Reruns reuse existing conversion outputs.

### 3. Fit terrain and retarget one motion

Use the motion identifier from the conversion manifest, without `.npz`. This
example selects the stair descent used in the supplementary montage:

```bash
export MOTION_ID="Darmstadt/D05/config05/trial05/descent_stageii"
terra run darmstadt --motion "$MOTION_ID" --dry-run
```

Check that the preview selects one motion with `fit_passed: true` and no missing
source files. Then run:

```bash
terra run darmstadt --motion "$MOTION_ID"
```

The bundled Darmstadt config reads the converter outputs from
`$TERRA_ARTIFACT_ROOT/darmstadt/smplh`. It fits terrain and retargets the motion
together, using the dataset's retargeting settings. There is no separate terrain
command or hand-written selection CSV. Results go to `darmstadt/cache`, and the
run record goes to `darmstadt/retarget`, both under `TERRA_ARTIFACT_ROOT`.
Check that the run reports `ok` or `cached` for the selected motion.

### 4. Prepare the motion for policy playback

The following commands create the selection CSV automatically from the completed
run, then create the cache and JSON record that playback reads:

```bash
mkdir -p "$TERRA_ARTIFACT_ROOT/darmstadt/policy"
terra train select \
  --run "$TERRA_ARTIFACT_ROOT/darmstadt/retarget" \
  --terrain-mode mixed \
  --out "$TERRA_ARTIFACT_ROOT/darmstadt/policy/selection.csv"
terra train materialize \
  --selection-manifest "$TERRA_ARTIFACT_ROOT/darmstadt/policy/selection.csv" \
  --destination-cache "$TERRA_ARTIFACT_ROOT/darmstadt/policy/cache" \
  --terrain-mode mixed \
  --record "$TERRA_ARTIFACT_ROOT/darmstadt/policy/materialization.json"
export POLICY_DATASET="$TERRA_ARTIFACT_ROOT/darmstadt/policy/materialization.json"
```

These `train` commands prepare data for the policy. They do not train
a policy. This short stair traversal needs no segmentation. For a collection
with long recordings, use [the segmentation step](policy-training.md#prepare-a-small-local-selection)
before materialization. The selection includes the passing motions from the latest
run record.

### 5. Download and play TERRA-4B

Skip the download if you already have this checkpoint; keep `POLICY_CHECKPOINT`
pointed at its local directory. Otherwise:

```bash
export POLICY_CHECKPOINT="$TERRA_ARTIFACT_ROOT/checkpoints/TERRA-4B"
uvx --from huggingface_hub hf download merc-s/TERRA-4B \
  --revision b89a604d0687549f2678bb1b6a436dea058cd8b9 \
  --local-dir "$POLICY_CHECKPOINT"
```

The checkpoint download contains no motion data.

Open the native MuJoCo GUI on a machine with a desktop display:

```bash
python scripts/terra/play_policy.py \
  --checkpoint "$POLICY_CHECKPOINT" \
  --materialization-record "$POLICY_DATASET" \
  --motion "$MOTION_ID" \
  --stochastic
```

To save a video instead, run:

```bash
python scripts/terra/play_policy.py \
  --checkpoint "$POLICY_CHECKPOINT" \
  --materialization-record "$POLICY_DATASET" \
  --motion "$MOTION_ID" \
  --stochastic \
  --video-dir "$TERRA_ARTIFACT_ROOT/darmstadt/policy/videos"
```

The file is `videos/$MOTION_ID/policy.mp4` beneath
`$TERRA_ARTIFACT_ROOT/darmstadt/policy/`. Headless rendering needs the system
libraries in [Optional rendering](installation.md#optional-rendering).
For playback limits, retries, and action settings, see
[Watch a trained policy](policy-training.md#watch-a-trained-policy).

### Use another dataset

The process is the same for `gait120`, `vielemeyer`, and `prism`: obtain the source
files, run `terra convert DATASET`, then `terra run DATASET --motion ID`. Use that
converter's `--help` for its paths and options, and the generated manifest for
motion identifiers. Vielemeyer C3D inputs need the `c3d` extra. PRISM needs the
provider's access approval. See [Full-source datasets](data.md#full-source-datasets)
for download links and extraction layouts.