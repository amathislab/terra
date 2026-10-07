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

## Prepare a small local selection

First complete the
[one-motion retarget walkthrough](motion-retargeting.md#run-one-motion) and its
validation. Select that published motion by its cache identifier. If you omitted
`--name` for `upstairs07_poses.npz`, use `--motion upstairs07_poses` below.
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
dataset and person, such as `KIT/3/upstairs07_poses`. A default filename alone
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

## Validation options

The production PPO defaults use stochastic validation with at least 100 rollouts.
You can override `experiment.validation.deterministic`, `minimum_total_rollouts`,
`rollouts_per_motion`, and `max_parallel_rollouts` for your experiment. Rollout counts
must be positive. GPU validation allocates lanes in groups of 32 and rounds the
parallel limit upward to the next group; inactive lanes are masked.
