# 4. Policy training

TERRA trains a terrain-conditioned PPO policy to follow retargeted MyoFullBody
trajectories. The input to training is a **verified motion selection**, not a raw AMASS
motion or a reconstruction report.

```text
source motions → retargeted artifacts → selection CSV → materialized cache
               → CUDA preflight → one-update startup check → PPO experiment
```

The one-motion path below checks that your installation can reach a PPO update. Use
a recording with several seconds of continuous motion so the tracking goal has useful
lookahead frames. This startup exercise does not establish a useful general policy.

To use a pre-trained policy, prepare and materialize a motion selection below,
then go to [Download a pre-trained checkpoint](#download-a-pre-trained-checkpoint).
You can skip the CUDA preflight, startup check, and training run for playback.

## Prepare a small local selection

First complete the
[one-motion retarget walkthrough](motion-retargeting.md#run-one-motion) and its
validation. The tutorial uses `KIT/3/upstairs04_poses.npz`; the downloaded TERRA-4B
checkpoint completes this motion in native MuJoCo. Use sampled actions
(`--stochastic`) for this checkpoint, as in the manuscript evaluation. Select the
published motion by its cache identifier. If you omitted
`--name` for `upstairs04_poses.npz`, use `--motion upstairs04_poses` below.
The `--dataset` value is a label for this selection. You do not need to rename
the cached motion. The selector validates the trajectory and sidecars before
writing the manifest:

```bash
export TERRA_ARTIFACT_ROOT="$HOME/terra-results"
mkdir -p "$TERRA_ARTIFACT_ROOT/training"
terra train select \
  --cache-root "$TERRA_ARTIFACT_ROOT/quickstart" \
  --motion FirstRun/motion \
  --dataset first-run \
  --terrain-mode mixed \
  --out "$TERRA_ARTIFACT_ROOT/training/selection.csv"
terra train segment \
  --selection-manifest "$TERRA_ARTIFACT_ROOT/training/selection.csv" \
  --out "$TERRA_ARTIFACT_ROOT/training/segmented.csv" \
  --report-out "$TERRA_ARTIFACT_ROOT/training/segments.json"
terra train materialize \
  --selection-manifest "$TERRA_ARTIFACT_ROOT/training/segmented.csv" \
  --destination-cache "$TERRA_ARTIFACT_ROOT/training/cache" \
  --terrain-mode mixed \
  --record "$TERRA_ARTIFACT_ROOT/training/materialization.json"
```

Segmentation leaves clips of 20 seconds or less intact. Longer clips become contiguous
segments of at most 10 seconds, with their source identity and split preserved. Review
`segments.json` for counts and durations. `mixed` accepts a validated flat or non-flat
trajectory. Materialization checks source artifact paths, copies or hard-links them into
a single cache, and records the selected split and source cache. Keep the manifest and
JSON record: `terra train run` reads the record rather than guessing which files to
train on.

This one-motion selection has only a training split, so the launcher also uses it for
startup validation. For a real experiment, select a larger, diverse training set and
a separate evaluation split with `terra train select --evaluation-fraction` before
segmentation. Keep source identities and their segments in the same split when building a larger
cohort. Automatic evaluation and test splits require identifiers that include the
dataset and person, such as `KIT/3/upstairs04_poses`. A default filename alone
does not identify the person. Record the selection and materialization report
alongside each training run.

## Check CUDA, then run one update

Install the `cuda` extra and use a compatible NVIDIA GPU. Training and Warp evaluation
require a working JAX CUDA device:

```bash
uv sync --locked --python 3.11 --extra cuda
source .venv/bin/activate
terra train preflight
python scripts/terra/train_smoke.py \
  --materialization-record "$TERRA_ARTIFACT_ROOT/training/materialization.json" \
  --output "$TERRA_ARTIFACT_ROOT/training/smoke"
```

The smoke script uses the production goal, observations, network, rewards, and PPO
initialization. It reduces the environment count to eight, disables validation, runs
one PPO update, and requires a `checkpoint_1` under its output. This checks startup and
one update; it does not establish task performance. If preflight fails, resolve the
device or package issue before trying PPO. `--allow-no-device` only checks
software/configuration on a CPU machine.

## Launch an experiment

Once the materialized cohort and startup check are sound, inspect a launch without
starting a long run:

```bash
terra train run \
  --materialization-record "$TERRA_ARTIFACT_ROOT/training/materialization.json" \
  --label first-run \
  --dry-run
```

For a full experiment, build a diverse, verified cohort with a separate evaluation
split created by `terra train select --evaluation-fraction`, inspect the dry-run
output, and rerun without `--dry-run`. Tune the configuration to available GPU memory and keep the materialization record, run configuration, and
checkpoints together. The one-update smoke test checks startup only; it does not
establish task performance.

## Download a pre-trained checkpoint

The [TERRA-4B checkpoint](https://huggingface.co/merc-s/TERRA-4B) is the primary
training-seed-0 policy used in the manuscript tables. It has 4,000,317,440 training
steps (PPO update 24,416). The repository is currently private; an account with
access is required. Log in on the machine where you will download it:

```bash
uvx --from huggingface_hub hf auth login
```

Download the complete checkpoint at the tested release revision:

```bash
export POLICY_REPO="merc-s/TERRA-4B"
export POLICY_REVISION="b89a604d0687549f2678bb1b6a436dea058cd8b9"
export POLICY_CHECKPOINT="${TERRA_ARTIFACT_ROOT:-$HOME/terra-results}/checkpoints/TERRA-4B"
uvx --from huggingface_hub hf download "$POLICY_REPO" \
  --revision "$POLICY_REVISION" \
  --local-dir "$POLICY_CHECKPOINT"
export POLICY_DATASET="${TERRA_ARTIFACT_ROOT:-$HOME/terra-results}/training/materialization.json"
```

The download contains the complete `checkpoint_24416` directory directly under
`$POLICY_CHECKPOINT`, including its saved configuration, state, and metadata.
Keep its files together. The playback script selects the latest completed
checkpoint in this directory. Follow [Watch a trained policy](#watch-a-trained-policy)
with these two environment variables; do not replace them with the example paths.

A checkpoint does not include the motion dataset. Create your local materialization
record with `terra train materialize`, as shown above. Use the model card's supported
robot, observations, and control rate when preparing the motions. This checkpoint
uses MyoFullBody and a 100 Hz control rate. It does not include the baseline policies
or all three training seeds. The download tool runs in a separate environment and
does not change TERRA's locked dependencies.

## Watch a trained policy

Use [play_policy.py](../scripts/terra/play_policy.py) to load a saved PPO checkpoint
and a materialized dataset. This runs the policy in native MuJoCo with the motion's
paired terrain. It uses the checkpoint's network, observations, control timestep,
and model solver settings.
Inference defaults to CPU; CUDA is not required for playback.

If you downloaded a checkpoint above, keep `POLICY_CHECKPOINT` and `POLICY_DATASET`
as set there. Otherwise, choose a saved `checkpoint_N` directory from a TERRA PPO
training run and a materialization record for the motions you want to watch:

```bash
export POLICY_CHECKPOINT="/absolute/path/to/checkpoint_500"
export POLICY_DATASET="/absolute/path/to/materialization.json"
```

You can also pass the checkpoint's immediate parent directory. The script selects
its latest completed checkpoint. The dataset can be the training selection or
another compatible selection created with `terra train materialize`. It must
point to materialized MyoFullBody trajectories and any terrain metadata.

### Open the native MuJoCo GUI

Run this on a machine with a desktop display:

```bash
python scripts/terra/play_policy.py \
  --checkpoint "$POLICY_CHECKPOINT" \
  --materialization-record "$POLICY_DATASET" \
  --stochastic
```

The script attempts every motion in the training split, in dataset order.
To choose one motion, add `--motion Dataset/Subject/motion`. Use its identifier
from the record's `motion` field, without `.npz`. Repeat `--motion` to choose
several motions. Drag the mouse to change the view and scroll to zoom.
Press Escape to close the current motion's window. The GUI retries the motion
after a fall or other episode termination, until the step limit is reached.

### Save a video

Add `--video-dir` to save a video instead of opening a window:

```bash
export POLICY_VIDEO_DIR="$HOME/terra-policy-videos"
python scripts/terra/play_policy.py \
  --checkpoint "$POLICY_CHECKPOINT" \
  --materialization-record "$POLICY_DATASET" \
  --stochastic \
  --video-dir "$POLICY_VIDEO_DIR"
```

Each motion is saved as `$POLICY_VIDEO_DIR/<motion>/policy.mp4`. The video
shows the policy alone at 1280 × 720, without debug text. Add `--show-reference`
to overlay the reference motion as a ghost body. Each video stops at the first episode termination, including
a fall, or at the step limit. Add `--repeat` to record retries until the step limit.
The scene uses a gray background, steel-blue terrain, and shadows. Muscles are blue
when inactive and blend toward red as activation rises.
Headless recording uses OSMesa; see
[Optional rendering](installation.md#optional-rendering) for system libraries.

Add `--split evaluation` to use the materialized evaluation split instead of
the training split. The materialization record must point to an existing cache.

`--steps 1000` is the default limit per motion: 10 seconds at the default 100 Hz
control rate. Increase it for longer motions. The examples use sampled actions,
which improved tracking for the TERRA-4B tutorial motion in native MuJoCo.
Omit `--stochastic` to use mean actions, the script default. For a checkpoint with
multiple training seeds, use `--train-state-seed N` to choose the seed. Playback
uses validation resets.

The global and upper-body tracking error limits default to **0.25 m**, as in the
manuscript evaluation. The saved TERRA-4B training configuration uses 0.15 m;
that stricter limit can stop an attempt while the body is still upright and moving.
Use `--tracking-threshold 0.15` to apply that limit, or another positive value for
your own checkpoint. Root-orientation and other termination rules are unchanged.
The manuscript used sampled actions; add `--stochastic` to use that setting.
Completion can differ across sampled attempts.

## Validation options

The production PPO defaults use stochastic validation with at least 100 rollouts.
You can override `experiment.validation.deterministic`, `minimum_total_rollouts`,
`rollouts_per_motion`, and `max_parallel_rollouts` for your experiment. Rollout counts
must be positive. GPU validation allocates lanes in groups of 32 and rounds the
parallel limit upward to the next group; inactive lanes are masked.
