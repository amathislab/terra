# Install TERRA

Follow this once, then use the
[first-motion walkthrough](../README.md#first-result-retarget-one-smpl-h-motion).
Commands assume a Linux shell at the repository root.

## Choose an environment

| Task | Python packages | Hardware |
|---|---|---|
| Inspect CLI, load a motion, retarget SMPL-H on CPU | Base lockfile | Linux x86-64, Python 3.11; CPU fitting is possible |
| Fit C3D markers | Base + `c3d` | CPU or CUDA device; marker fitting can be slow |
| Run GMR comparison | Base + `baselines` | Depends on the selected method |
| Train or evaluate a PPO policy | Base + `cuda` | Compatible NVIDIA GPU, driver, and CUDA 12 JAX runtime |
| Run tests and lint | Base + `dev` | No GPU for most checks |

Install [Git](https://git-scm.com/) and
[uv](https://docs.astral.sh/uv/getting-started/installation/). On Linux, the uv
project documents this installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new shell if the installer added uv to your `PATH`, then check `uv --version`.
You can use another installation method from the official uv guide. Clone TERRA and
run:

```bash
git clone https://github.com/amathislab/terra.git
cd terra
uv sync --locked --python 3.11
source .venv/bin/activate
terra --version
terra --help
terra retarget --help
```

`uv sync --locked` uses the checked-in lockfile, including TERRA's pinned
MuscleMimic `terra` branch, SMPL-X package, and Holosoma dependency. It needs
network access to their Git repositories and package indexes. The Linux x86-64 PyTorch wheel in this lock is a CUDA
12.6 build, even for CPU fitting. The base environment occupies about **8 GB**;
allow additional space for uv's download cache, licensed models, and generated results.

Install the extras for your path in one command. For example:

```bash
uv sync --locked --python 3.11 --extra c3d --extra baselines
uv sync --locked --python 3.11 --extra cuda --extra c3d --extra baselines
```

Each `uv sync` reconciles the same environment with the flags in that invocation. Keep
every extra you need on your last command. After activation, run `terra ...` directly.
If you use `uv run`, repeat the needed extras there too; for example,
`uv run --locked --extra cuda terra train preflight`. Otherwise,
`uv run` may resync the environment without them.

## Check inputs separately

Successful installation does not include licensed motion data or the neutral SMPL-H
model. Download these as described in [Data and models](data.md), then verify an actual
`.npz` file before running `terra retarget`. Marker inputs need a marker-fitting model
root as well as the SMPL-H retargeting root.

## Check GPU readiness

For training, sync with `--extra cuda` and run:

```bash
terra train preflight
```

On a machine without a GPU, `terra train preflight --allow-no-device`
checks software and configuration only. A successful software check does not establish
that a later GPU run will work. Use [Policy training](policy-training.md) for the
materialized cohort and one-update startup test.

## Optional rendering

The terrain image script and `terra visualize` use MuJoCo's headless OSMesa
backend by default. Install a system `libOSMesa` library if MuJoCo cannot create
a rendering context. The Python lockfile does not install this system library.
It is needed for [terrain images](terrain-reconstruction.md#view-the-terrain-fit)
and the [video review step](motion-retargeting.md#render-the-motion).

## If setup stops

| Symptom | Check |
|---|---|
| `uv` cannot fetch a Git dependency | Git and network access to the exact repositories in `pyproject.toml`; rerun `uv sync --locked` |
| `No module named musclemimic` | Activate `.venv` or use `uv run`; the compatible `terra` branch is pinned in the base installation |
| SMPL-H model root missing | Set `TERRA_MODEL_ROOT` or pass `--smpl-model-path`; verify `SMPLH_NEUTRAL.pkl` is present |
| Source file missing or archive fields rejected | Follow the [motion archive contract](motion-files.md#check-one-smpl-h-archive) |
| C3D reader missing | Sync with `--extra c3d` |
| JAX reports no CUDA device | Confirm `--extra cuda`, NVIDIA driver/device visibility, then rerun preflight |
| Output already exists | Choose a fresh `--output-root` or deliberately pass `--overwrite` to `terra retarget` |

Continue with [motion loading](motion-files.md),
[terrain reconstruction](terrain-reconstruction.md),
[retargeting](motion-retargeting.md), and [policy training](policy-training.md).

## Expected fitting warnings

On the CPU quickstart, JAX may report that the optional TPU backend could not
initialize and that it is “falling back to cpu.” These messages are expected for
CPU fitting. For GPU training, a CPU fallback needs investigation; confirm the
CUDA device with `terra train preflight`.

SMPL-H may print “SMPL+H … 16 shape coefficients” more than once while loading
models for fitting, reconstruction, and retargeting. This describes the model's
available shape coefficients and does not indicate a failed fit.

## Release checks

The default test suite needs no licensed assets. To run actual CLI-to-solver checks
for all four retargeting methods, provide a local AMASS stair motion and neutral model:

```bash
uv sync --locked --extra dev --extra baselines --extra c3d
export TERRA_TEST_MOTION="$MOTION_FILE"
export TERRA_TEST_MODEL_ROOT="$TERRA_MODEL_ROOT"
# Optional: reuse the quickstart body calibration.
export TERRA_TEST_SHAPE="$TERRA_ARTIFACT_ROOT/quickstart/MyoFullBody/shape_optimized.pkl"
.venv/bin/python -m pytest -q --runslow tests/integration
```

These tests create short motion fixtures and temporary output caches; they do not
ship model or dataset files. Distribution checks run with `pytest -q --runslow`.
