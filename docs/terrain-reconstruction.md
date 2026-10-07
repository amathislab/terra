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
repository root after the README quickstart:

```bash
terra reconstruct cohort \
  --motion "$MOTION_FILE" \
  --dataset-config amass \
  --smpl-model-path "$TERRA_MODEL_ROOT" \
  --cache-root "$TERRA_ARTIFACT_ROOT/quickstart" \
  --output-dir "$TERRA_ARTIFACT_ROOT/reconstruction/terra"
```

The command writes a selection automatically, prepares the fitted body shape if
needed, and fits terrain. The cache above reuses the quickstart's MyoFullBody
shape, so it is not fitted twice. `MOTION_FILE` must be below
`$TERRA_DATA_ROOT/AMASS`.

For several motions, repeat `--motion`, or pass `--motions selection.txt`. A
selection file contains IDs relative to the configured input root, without the
`.npz` suffix, for example `KIT/3/upstairs04_poses`. A CSV with a `motion` column
also works. Converted datasets use `--motions` and their bundled dataset name so
subject calibration and conversion quality remain available.

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
cat "$TERRA_ARTIFACT_ROOT/reconstruction/terra/run.json"
```

The cohort record is a reconstruction report. For
`terra retarget --terrain /path/to/file.json`, use a compatible terrain metadata JSON
(such as a published `_terrain.json` sidecar); the cohort report is not the CLI's
terrain input format. Use `--terrain auto` to let the one-motion retarget command
estimate terrain directly.

## View the terrain fit

Render each successful fit to a PNG image before retargeting:

```bash
python scripts/terra/render_terrain.py "$TERRA_ARTIFACT_ROOT/reconstruction/terra"
```

Open the PNG files in that directory. Each image shows the fitted surfaces and
floor in MuJoCo. The camera includes all terrain boxes. A flat fit shows only
the floor. Use the images to inspect the surface shape and placement, then read
the validation report to check support for the motion.

You can also pass one fit JSON file. Use `--output-dir` to save the images in
another directory. Use `--azimuth` and `--elevation` to adjust the view, in degrees.

The script uses the same headless rendering backend as the motion viewer. See
[rendering setup](installation.md#optional-rendering) if MuJoCo cannot create a
rendering context.

Next: [Motion retargeting](motion-retargeting.md).
