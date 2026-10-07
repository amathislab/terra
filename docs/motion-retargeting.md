# 3. Motion retargeting

Retargeting converts a human motion into a MyoFullBody trajectory. TERRA's default
path estimates terrain from the motion, fits the robot body shape if needed, solves a
terrain-aware trajectory, applies its final stability gate, and publishes an analysis
record beside the trajectory.

## Run one motion

Complete [installation](installation.md) and [data setup](data.md). Use the KIT
tutorial staircase motion below, or set `MOTION_FILE` to another AMASS-compatible
`.npz` motion:

```bash
export TERRA_DATA_ROOT="$HOME/terra-data"
export MOTION_FILE="$TERRA_DATA_ROOT/AMASS/KIT/3/upstairs04_poses.npz"
export TERRA_MODEL_ROOT="$HOME/terra-models/smplh"
export TERRA_ARTIFACT_ROOT="$HOME/terra-results"
terra retarget "$MOTION_FILE" \
  --smpl-model-path "$TERRA_MODEL_ROOT" \
  --output-root "$TERRA_ARTIFACT_ROOT/quickstart" \
  --name FirstRun/motion
```

The default method is `terra` and the default terrain mode is `auto`. To force flat
ground, add `--terrain none`. To use already known terrain, pass a compatible terrain
metadata JSON path via `--terrain`. `--name` is the portable identifier inside the
cache; it may have relative path components. If you omit `--name`, the identifier is
the input filename without its extension. For example, `upstairs04_poses.npz` becomes
`upstairs04_poses`. Spaces and other unsupported characters become underscores.
Use this identifier in the validation and visualization commands below.
Reusing an output name raises an error
unless you pass `--overwrite`.

`terra retarget --help` lists marker-specific inputs. A `.c3d`, `.trc`, or `.mat`
additionally needs `--c3d-model-path` for marker fitting. A MAT source needs
`--mat-schema`; a TRC source needs the correct up-axis choice. See
[Motion files](motion-files.md#retarget-marker-files-directly).

## Inspect and validate the published result

For the command above, the cache layout is:

```text
$TERRA_ARTIFACT_ROOT/quickstart/
└── MyoFullBody/
    ├── shape_optimized.pkl
    └── terra/
        └── FirstRun/
            ├── motion.npz
            ├── motion_analysis.npz
            └── motion_terrain.json   # present when terrain metadata is published
```

The CLI prints the actual paths as JSON. The `.npz` trajectory contains state arrays for
playback/training; `_analysis.npz` contains structured run metadata. A flat result may have no `_terrain.json`. Inspect the set with the public
validator:

```bash
python - <<'PY'
import os
from pathlib import Path
from terra import validate_retarget_artifacts
root = Path(os.environ["TERRA_ARTIFACT_ROOT"]) / "quickstart"
item = validate_retarget_artifacts(root, "FirstRun/motion")
print(item.num_frames, "frames at", item.frequency, "Hz")
print("trajectory:", item.trajectory_path)
print("analysis:", item.analysis_path)
print("terrain:", item.terrain_path)
print("non-flat:", item.nonflat_terrain)
PY
```

`validate_retarget_artifacts` checks numeric trajectory arrays and terrain metadata. Use `require_nonflat_terrain=True` only for a deliberately non-flat
selection; an arbitrary AMASS motion may be flat or fail the terrain fit.

If automatic terrain fitting rejects a motion, read the reported contact or validation
failure and try a recording with clear foot contact and enough complete frames. For an
intentionally flat-ground example, rerun the command above with `--terrain none` and
`--overwrite` while keeping `--name FirstRun/motion`; the validation command will then
check that replacement. This option changes the support geometry used by the solver, so
keep `auto` when your result needs reconstructed stairs or a ramp.

## Render the motion

A video is useful for checking body motion and foot placement after artifact
validation. Render one motion without a CSV manifest or evaluator scores:

```bash
terra visualize \
  --motion FirstRun/motion \
  --cache-root "$TERRA_ARTIFACT_ROOT/quickstart" \
  --without-scores \
  --out "$TERRA_ARTIFACT_ROOT/quickstart/videos" \
  --workers 1
```

If you omitted `--name` when retargeting `upstairs04_poses.npz`, use
`--motion upstairs04_poses` instead. The identifier is relative to the method's
cache directory. Do not include `.npz`, `MyoFullBody/terra/`, or the input file path.
Keep `--cache-root` equal to the retarget command's `--output-root`.

Repeat `--motion` to render several motions. For a larger collection, use
`--manifest /path/to/motions.csv` instead. The CSV needs a `motion` column, with one
cache identifier per row. For a motion named `upstairs04_poses`, it contains:

```csv
motion
upstairs04_poses
```

Open `$TERRA_ARTIFACT_ROOT/quickstart/videos/INDEX.md` for the MP4 link. The
default headless MuJoCo renderer uses OSMesa; see the
[installation note](installation.md#optional-rendering) if that system
library is unavailable. Add evaluator scores later for contact and failure
annotations.

## Python API

The API returns an in-memory result. Publishing is an explicit second step:

```python
from terra import retarget, save_retarget_result, validate_retarget_artifacts

result = retarget(
    "/absolute/path/to/motion_poses.npz",
    smpl_model_path="/absolute/path/to/smplh",
    cache_root="/absolute/path/to/cache",
    terrain="auto",
)
published = save_retarget_result(result, "/absolute/path/to/cache", "Study/Trial")
checked = validate_retarget_artifacts("/absolute/path/to/cache", "Study/Trial")
print(checked.trajectory_path)
```

Set `cache_root` on both calls if you want the fitted shape and artifacts together. The
[API source](../src/terra/api.py) documents dispatch for `.npz`, `.c3d`, `.trc`, and
`.mat`.

## Methods and configuration

`--method` accepts `terra`, `omniretarget`, `gmr`, and `smpl`. GMR needs the
`baselines` extra. Only TERRA and OmniRetarget can infer terrain from `auto`; for the
others use `--terrain none` or an explicit terrain JSON. `--config` takes an inline JSON object or a JSON file
of method-specific overrides. Start with the defaults, then save the exact options and
input selection used for a larger cohort.

Next: [Policy training](policy-training.md).
