# 2. Terrain reconstruction

TERRA estimates support geometry from the motion itself. It finds contact intervals at
the toes and ankles, uses the motion's stance heights as support evidence, and checks
candidate terrain against motion landmarks and free space. The fit produces a terrain
specification **and** a validation report. A flat motion can legitimately yield no
raised terrain.

The fitter groups stance heights into support levels, compares a continuous ramp with
discrete steps when raised support is present, fits the selected surfaces to contact
and free-space evidence, then validates the geometry against the full motion.

The [one-motion retarget command](motion-retargeting.md) uses `--terrain auto` and runs
reconstruction as part of retargeting. Use the standalone cohort command when you want
to inspect terrain fits before retargeting a selected set.

On the fitted SMPL-H path, ramp-versus-stair selection compares supported foot
orientation with a neutral ankle-to-toe pitch. A measured flat reference takes
precedence; otherwise TERRA uses the fitted SMPL-H model-rest pitch and records that
source in the fit report. Other landmark models need their own neutral pitch.

Dataset configs use `[terrain].posed_seat_frame = "apparatus"` by default for both
reconstruction and retargeting. This subtracts the landmark normalization's vertical
translation so posed seats share the foot surfaces' ground datum. Set `"normalized"`
only when the terrain itself is defined in normalized landmark coordinates.

## Fit one AMASS motion as a cohort of one

This example needs the
[AMASS and neutral SMPL-H layout](data.md#the-first-motion-layout), an installed TERRA
environment, and a `MOTION_FILE` inside `$TERRA_DATA_ROOT/AMASS`. Run it from the
repository root:

```bash
export MOTION_FILE="/absolute/path/to/your/motion_poses.npz"
export TERRA_ARTIFACT_ROOT="$HOME/terra-results"
python - <<'PY'
import os
from pathlib import Path
source = Path(os.environ["MOTION_FILE"]).resolve()
root = (Path(os.environ["TERRA_DATA_ROOT"]) / "AMASS").resolve()
if not source.is_file() or not source.is_relative_to(root):
    raise SystemExit("MOTION_FILE must be an existing .npz below TERRA_DATA_ROOT/AMASS")
motion_id = source.relative_to(root).with_suffix("").as_posix()
out = Path(os.environ["TERRA_ARTIFACT_ROOT"]) / "reconstruction"
out.mkdir(parents=True, exist_ok=True)
(out / "one-motion.txt").write_text(motion_id + "\n")
print("Selected:", motion_id)
PY
python - <<'PY'
from pathlib import Path
from terra.dataset_pipeline import ensure_robot_shape, load_dataset_config
config = load_dataset_config(Path("src/terra/datasets/configs/amass.toml"))
print("Shared body-shape cache:", ensure_robot_shape(config))
PY
terra reconstruct cohort \
  --method terra \
  --motions "$TERRA_ARTIFACT_ROOT/reconstruction/one-motion.txt" \
  --dataset-config src/terra/datasets/configs/amass.toml \
  --output-dir "$TERRA_ARTIFACT_ROOT/reconstruction/terra"
```

The shape step calibrates MyoFullBody to the SMPL-H model and is cached for later runs.
The AMASS config resolves `data/AMASS` to `TERRA_DATA_ROOT/AMASS` and its model/cache
paths through the exported roots. The selection file contains IDs **relative to AMASS**,
without the `.npz` suffix; it is not a list of absolute filenames. A CSV with a `motion`
column also works.

## Read the output

`$TERRA_ARTIFACT_ROOT/reconstruction/terra/` contains one
`<motion-id-with-slashes-replaced-by-__>.json` per successful fit, plus `status.csv` and
`run.json`. The per-motion JSON has `terrain`, `fit`, and `validation` sections.
`status.csv` records `ok` or `failed` for every selected motion and an
error message for failures. Rerunning the command refits selected motions and replaces
their JSON records. Start there if the command exits nonzero; failed fits remain
visible in the selected set.

```bash
cat "$TERRA_ARTIFACT_ROOT/reconstruction/terra/status.csv"
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["TERRA_ARTIFACT_ROOT"]) / "reconstruction/terra"
print(json.dumps(json.loads((root / "run.json").read_text())["counts"], indent=2))
PY
```

The cohort record is a reconstruction report. For
`terra retarget --terrain /path/to/file.json`, use a compatible terrain metadata JSON
(such as a published `_terrain.json` sidecar); the cohort report is not the CLI's
terrain input format. Use `--terrain auto` to let the one-motion retarget command
estimate terrain directly.

Next: [Motion retargeting](motion-retargeting.md).
