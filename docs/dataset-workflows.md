# Process and evaluate a motion collection

`terra run` applies one dataset config to an explicit motion selection. `terra
evaluate` scores the resulting trajectories on CPU; PPO policy evaluation is a
separate GPU workflow described in [Policy training](policy-training.md).

## Select and retarget motions

After [data setup](data.md), create a TXT file with one motion ID per line, relative
to the dataset input root and without `.npz`. For an AMASS KIT download:

```bash
mkdir -p "$TERRA_ARTIFACT_ROOT/selections"
printf '%s\n' 'KIT/3/upstairs04_poses' > "$TERRA_ARTIFACT_ROOT/selections/motions.txt"
terra run amass \
  --selection-manifest "$TERRA_ARTIFACT_ROOT/selections/motions.txt" \
  --cache-root "$TERRA_ARTIFACT_ROOT/quickstart" \
  --run-root "$TERRA_ARTIFACT_ROOT/dataset-run" \
  --dry-run
```

Replace the example ID with a motion you downloaded. Remove `--dry-run` to
retarget the selection. `--motion ID` is a repeatable alternative; `--all-motions`
explicitly selects the whole input tree. CSV selections need a `motion` column.
The bundled names are `amass`, `darmstadt`, `gait120`, `prism`, and `vielemeyer`;
you can also supply your own TOML config. Use the respective converter first for
marker datasets, then inspect its `manifest.csv` and failed fits.

A run writes `run.json`, `manifest.csv`, and per-motion terrain reports beneath
`--run-root`. Trajectory, analysis, and terrain sidecars are written beneath
`--cache-root/MyoFullBody/<method>/`. The cache above shares the quickstart body
shape. Reruns reuse complete artifact sets; `--overwrite` recomputes them.

To use standalone reconstruction records, add `--terrain-dir DIR
--terrain-method terra`. To run a comparison method, repeat the same selection
with `--method omniretarget`, `smpl`, or `gmr` after completing TERRA in the same
cache. GMR requires the `baselines` extra. Each method gets its own cache directory.

## Evaluate a dataset run

```bash
terra evaluate dataset amass \
  --manifest "$TERRA_ARTIFACT_ROOT/dataset-run/manifest.csv" \
  --cache-root "$TERRA_ARTIFACT_ROOT/quickstart" \
  --output-root "$TERRA_ARTIFACT_ROOT/evaluation/dataset"
```

This reads the explicit manifest, checks completed artifacts, and scores their motion
and terrain interactions. It keeps failed rows visible. See `terra evaluate
dataset --help` for worker count and output controls.

## Compare motion metrics

```bash
terra evaluate metrics \
  --motion-class stairs="$TERRA_ARTIFACT_ROOT/selections/motions.txt" \
  --method terra=terra \
  --cache-root "$TERRA_ARTIFACT_ROOT/quickstart" \
  --out "$TERRA_ARTIFACT_ROOT/evaluation/metrics" \
  --quality-out "$TERRA_ARTIFACT_ROOT/evaluation/quality" \
  --workers 1
```

`--motion-class LABEL=FILE` names a selection. `--method LABEL=CACHE_SUBDIR` names
a method's directory beneath `MyoFullBody`; repeat it for comparison results in
the same cache. Terrain defaults to TERRA's recorded terrain, or use
`--terrain-method self` for each method's own sidecar. Source paths recorded in
analysis are used by default; `--source-root` can point to a relocated motion
collection.

The evaluator writes per-motion metrics, method summaries, and quality details
for support, clearance, skating, collisions, and joint limits. Missing outputs
are reported as failures; `--allow-missing` permits partial comparisons. Review
these reports before [building a training selection](policy-training.md).
