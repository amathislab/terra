#!/usr/bin/env python3
"""Run one eight-environment PPO update to check a materialized cohort."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from terra.training import config_overrides

    settings = json.loads((Path(__file__).with_name("ppo_smoke_overrides.json")).read_text())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        "-u",
        "-m",
        "terra",
        "train",
        "run",
        "--algorithm",
        "ppo",
        "--materialization-record",
        str(args.materialization_record.resolve()),
        "--label",
        "release-smoke",
    ]
    overrides = config_overrides(settings)
    overrides += [
        f"hydra.run.dir={out}",
        f"experiment.checkpoint_root={out / 'checkpoints'}",
        "experiment.run_id=smoke",
    ]
    for override in overrides:
        command.extend(["--override", override])
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        TERRA_ARTIFACT_ROOT=str(out),
        WANDB_MODE="disabled",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        PYTHONUNBUFFERED="1",
    )
    (out / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    code = subprocess.call(command, cwd=ROOT, env=env)
    if code:
        raise SystemExit(code)
    print("PPO process exited successfully; inspect the update metrics and checkpoint under", out, flush=True)
    # A saved checkpoint proves the update completed, beyond configuration/import startup.
    if not list((out / "checkpoints").rglob("checkpoint_1")):
        raise SystemExit("No completed one-update checkpoint was saved.")
    print("PASS: one PPO update and checkpoint completed.", flush=True)


if __name__ == "__main__":
    main()
