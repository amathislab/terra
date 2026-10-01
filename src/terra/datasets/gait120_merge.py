"""Atomically merge validated, disjoint Gait120 conversion shards.

The full CUDA conversion can be split across isolated output roots.  This tool copies
only subject trajectories and Stage-I subject states into the canonical dataset root;
the canonical run owns `.markers`, audit tables, the manifest, and final validation.
Existing SMPL-H trajectories are refused so two workers can never silently overwrite
one another.  Subject directories containing only canonical EMG companions may be
augmented; the canonical preparation pass creates those directories before fitting.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

from terra._revision import write_git_commit
from terra.paths import StorageRoots


def _validated_subjects(source: Path) -> list[Path]:
    validation_path = source / "validation.json"
    if not validation_path.exists():
        raise ValueError(f"staging shard is not finished: {validation_path} is missing")
    validation = json.loads(validation_path.read_text())
    if not validation.get("paired_dataset_ready"):
        raise ValueError(f"staging shard did not pass paired validation: {validation}")
    subjects = sorted((source / "Gait120").glob("S[0-9][0-9][0-9]"))
    if not subjects:
        raise ValueError(f"no subject directories under {source / 'Gait120'}")
    return subjects


def merge_shard(source: Path, destination: Path, *, replace_existing: bool = False) -> list[str]:
    """Merge one completed shard, optionally replacing existing trajectories."""

    source = source.resolve()
    destination = destination.resolve()
    subjects = _validated_subjects(source)
    conflicts = [
        path.name
        for path in subjects
        if any((destination / "Gait120" / path.name).rglob("AllSteps_stageii.npz"))
        or (destination / ".stage1" / f"{path.name}.npz").exists()
    ]
    if conflicts and not replace_existing:
        raise FileExistsError(f"destination already contains: {', '.join(conflicts)}")

    missing_states = [path.name for path in subjects if not (source / ".stage1" / f"{path.name}.npz").exists()]
    if missing_states:
        raise ValueError(f"missing Stage-I states: {', '.join(missing_states)}")

    merged = []
    (destination / "Gait120").mkdir(parents=True, exist_ok=True)
    (destination / ".stage1").mkdir(parents=True, exist_ok=True)
    write_git_commit(destination)
    for subject in subjects:
        target = destination / "Gait120" / subject.name
        temporary = target.with_name(f".{subject.name}.merge-tmp")
        backup = target.with_name(f".{subject.name}.merge-backup")
        if temporary.exists():
            raise FileExistsError(f"stale merge directory exists: {temporary}")
        if backup.exists():
            raise FileExistsError(f"stale merge directory exists: {backup}")

        if target.exists():
            shutil.copytree(target, temporary)
            for source_path in subject.rglob("*"):
                relative = source_path.relative_to(subject)
                temporary_path = temporary / relative
                if source_path.is_dir():
                    temporary_path.mkdir(parents=True, exist_ok=True)
                elif source_path.name.endswith(("_stageii.npz", ".fit.json")):
                    temporary_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, temporary_path)
                elif not temporary_path.exists():
                    temporary_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, temporary_path)
        else:
            shutil.copytree(subject, temporary)

        for metadata_path in temporary.rglob("*.fit.json"):
            metadata = json.loads(metadata_path.read_text())
            metadata["source_marker_archive"] = metadata["source_marker_archive"].replace(
                str(source), str(destination), 1
            )
            metadata["stage1_state"] = metadata["stage1_state"].replace(str(source), str(destination), 1)
            marker_path = Path(metadata["source_marker_archive"])
            if not marker_path.exists():
                shutil.rmtree(temporary)
                raise ValueError(f"canonical marker archive is missing: {marker_path}")
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

        state_source = source / ".stage1" / f"{subject.name}.npz"
        state_target = destination / ".stage1" / state_source.name
        state_temporary = state_target.with_suffix(".merge-tmp")
        shutil.copy2(state_source, state_temporary)
        try:
            if target.exists():
                os.replace(target, backup)
            os.replace(temporary, target)
            os.replace(state_temporary, state_target)
        except Exception:
            if target.exists():
                shutil.rmtree(target)
            if backup.exists():
                os.replace(backup, target)
            state_temporary.unlink(missing_ok=True)
            state_target.unlink(missing_ok=True)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        merged.append(subject.name)
    return merged


def main(argv: Sequence[str] | None = None) -> int:
    roots = StorageRoots.from_environment(Path.cwd())
    parser = argparse.ArgumentParser(prog="terra convert gait120 merge", description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--destination", type=Path, default=roots.artifact_root / "gait120" / "smplh")
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Atomically replace subjects already containing SMPL-H trajectories",
    )
    args = parser.parse_args(argv)
    destination = roots.resolve_artifact(args.destination, base=Path.cwd())
    assert destination is not None

    all_merged = []
    for source in args.sources:
        source_path = roots.resolve_artifact(source, base=Path.cwd())
        assert source_path is not None
        merged = merge_shard(source_path, destination, replace_existing=args.replace_existing)
        all_merged.extend(merged)
        print(f"{source}: merged {len(merged)} subject(s): {', '.join(merged)}")
    print(f"Merged {len(all_merged)} subject(s) into {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
