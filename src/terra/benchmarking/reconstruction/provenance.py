"""Content-addressed provenance for reconstruction runs and evaluations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from terra._files import atomic_write
from terra._revision import git_commit

RUN_PROVENANCE_SCHEMA = "terra.reconstruction-provenance.v1"
METHOD_IDENTITY_SCHEMA = "terra.reconstruction-method-identity.v1"
SCIENTIFIC_IDENTITY_SCHEMA = "terra.reconstruction-scientific-identity.v1"
EVALUATION_PROVENANCE_SCHEMA = "terra.reconstruction-evaluation-provenance.v1"
RECORD_PROVENANCE_SCHEMA = "terra.reconstruction-record-provenance.v1"

_SOURCE_SUFFIXES = frozenset({".csv", ".json", ".py", ".toml"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Render a stable JSON representation suitable for scientific identities."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )


def content_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_tree_sha256(package_root: Path) -> str:
    """Hash package-owned executable code and bundled benchmark resources."""

    digest = hashlib.sha256()
    files = sorted(
        path
        for path in package_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in _SOURCE_SUFFIXES and "__pycache__" not in path.parts
    )
    for path in files:
        relative = path.relative_to(package_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def _source_state(repo_root: Path | None, package_root: Path) -> str:
    candidate = repo_root.expanduser().resolve() if repo_root is not None else package_root.parents[1]
    # A source archive can live inside another project's checkout. Only the
    # package's own repository determines whether its executable source is dirty.
    if not (candidate / ".git").exists():
        return "packaged"
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(candidate),
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                str(package_root),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "packaged"
    return "dirty" if completed.stdout.strip() else "clean"


def _input_file(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"scientific input not found: {resolved}")
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def build_run_provenance(
    method: str,
    resolved_options: Mapping[str, Any],
    selection_path: Path,
    *,
    dataset_config_path: Path | None = None,
    matrix_path: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build the complete, reviewable identity of one reconstruction cohort."""

    package_root = Path(__file__).resolve().parents[2]
    source = {
        "git_commit": git_commit(repo_root),
        "state": _source_state(repo_root, package_root),
        "tree_sha256": _source_tree_sha256(package_root),
    }
    options = json.loads(canonical_json(dict(resolved_options)))
    method_identity = {
        "schema": METHOD_IDENTITY_SCHEMA,
        "method": method,
        "resolved_options": options,
        "source": source,
    }
    method_identity_sha256 = content_sha256(method_identity)
    inputs = {
        "selection": _input_file(selection_path),
        "dataset_config": _input_file(dataset_config_path),
        "matrix": _input_file(matrix_path),
    }
    scientific_identity = {
        "schema": SCIENTIFIC_IDENTITY_SCHEMA,
        "method_identity_sha256": method_identity_sha256,
        "selection_sha256": inputs["selection"]["sha256"],
        "dataset_config_sha256": (None if inputs["dataset_config"] is None else inputs["dataset_config"]["sha256"]),
        "matrix_sha256": None if inputs["matrix"] is None else inputs["matrix"]["sha256"],
    }
    return {
        "schema": RUN_PROVENANCE_SCHEMA,
        "method_identity": method_identity,
        "method_identity_sha256": method_identity_sha256,
        "scientific_identity": scientific_identity,
        "scientific_identity_sha256": content_sha256(scientific_identity),
        "inputs": inputs,
    }


@dataclass(frozen=True)
class ValidatedRunProvenance:
    method: str
    resolved_options: dict[str, Any]
    method_identity: dict[str, Any]
    method_identity_sha256: str
    scientific_identity_sha256: str
    source: dict[str, str]


def validate_run_provenance(run: Mapping[str, Any], path: Path) -> ValidatedRunProvenance:
    """Validate internal hashes and the readable metadata duplicated in ``run.json``."""

    raw = run.get("provenance")
    if not isinstance(raw, dict) or raw.get("schema") != RUN_PROVENANCE_SCHEMA:
        raise ValueError(f"reconstruction run has no {RUN_PROVENANCE_SCHEMA} record: {path}")
    method_identity = raw.get("method_identity")
    scientific_identity = raw.get("scientific_identity")
    if not isinstance(method_identity, dict) or method_identity.get("schema") != METHOD_IDENTITY_SCHEMA:
        raise ValueError(f"reconstruction run has an invalid method identity: {path}")
    if not isinstance(scientific_identity, dict) or scientific_identity.get("schema") != SCIENTIFIC_IDENTITY_SCHEMA:
        raise ValueError(f"reconstruction run has an invalid scientific identity: {path}")
    method_hash = content_sha256(method_identity)
    scientific_hash = content_sha256(scientific_identity)
    if raw.get("method_identity_sha256") != method_hash:
        raise ValueError(f"reconstruction method identity hash mismatch: {path}")
    if scientific_identity.get("method_identity_sha256") != method_hash:
        raise ValueError(f"scientific identity names a different method identity: {path}")
    if raw.get("scientific_identity_sha256") != scientific_hash:
        raise ValueError(f"reconstruction scientific identity hash mismatch: {path}")
    method = method_identity.get("method")
    options = method_identity.get("resolved_options")
    source = method_identity.get("source")
    if not isinstance(method, str) or not isinstance(options, dict) or not isinstance(source, dict):
        raise ValueError(f"reconstruction method identity is incomplete: {path}")
    if run.get("method") != method or run.get("options") != options:
        raise ValueError(f"reconstruction run metadata disagrees with its method identity: {path}")
    required_source = {"git_commit", "state", "tree_sha256"}
    if set(source) != required_source or any(not isinstance(source[key], str) for key in required_source):
        raise ValueError(f"reconstruction source identity is incomplete: {path}")
    if source["git_commit"] != "unknown" and _GIT_COMMIT.fullmatch(source["git_commit"]) is None:
        raise ValueError(f"reconstruction source Git commit is invalid: {path}")
    if source["state"] not in {"clean", "dirty", "packaged"}:
        raise ValueError(f"reconstruction source state is invalid: {path}")
    if _SHA256.fullmatch(source["tree_sha256"]) is None:
        raise ValueError(f"reconstruction source tree hash is invalid: {path}")
    inputs = raw.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"selection", "dataset_config", "matrix"}:
        raise ValueError(f"reconstruction scientific inputs are incomplete: {path}")
    for name, identity_key in (
        ("selection", "selection_sha256"),
        ("dataset_config", "dataset_config_sha256"),
        ("matrix", "matrix_sha256"),
    ):
        record = inputs[name]
        expected = scientific_identity.get(identity_key)
        if record is None:
            if expected is not None:
                raise ValueError(f"reconstruction {name} identity is inconsistent: {path}")
            continue
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"reconstruction {name} provenance is invalid: {path}")
        if record["sha256"] != expected or _SHA256.fullmatch(str(record["sha256"])) is None:
            raise ValueError(f"reconstruction {name} hash is inconsistent: {path}")
    return ValidatedRunProvenance(
        method=method,
        resolved_options=options,
        method_identity=method_identity,
        method_identity_sha256=method_hash,
        scientific_identity_sha256=scientific_hash,
        source=source,
    )


def reconstruction_input_record(run_path: Path) -> dict[str, Any]:
    resolved = run_path.expanduser().resolve()
    try:
        run = json.loads(resolved.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid reconstruction run metadata: {resolved}") from error
    if not isinstance(run, dict):
        raise ValueError(f"reconstruction run metadata must contain an object: {resolved}")
    validated = validate_run_provenance(run, resolved)
    return {
        "run_json": str(resolved),
        "run_sha256": file_sha256(resolved),
        "method": validated.method,
        "resolved_options": validated.resolved_options,
        "method_identity": validated.method_identity,
        "method_identity_sha256": validated.method_identity_sha256,
        "scientific_identity_sha256": validated.scientific_identity_sha256,
    }


def write_evaluation_provenance(
    output_dir: Path,
    per_motion_path: Path,
    primary_run_path: Path,
    *,
    reference_run_paths: Sequence[Path] = (),
) -> Path:
    """Bind an evaluation CSV to the exact reconstruction runs it consumed."""

    output = output_dir.expanduser().resolve()

    def portable_record(run_path: Path) -> dict[str, Any]:
        record = reconstruction_input_record(run_path)
        record["run_json"] = os.path.relpath(run_path.expanduser().resolve(), output)
        return record

    payload = {
        "schema": EVALUATION_PROVENANCE_SCHEMA,
        "per_motion_sha256": file_sha256(per_motion_path.expanduser().resolve()),
        "primary_reconstruction": portable_record(primary_run_path),
        "reference_reconstructions": [portable_record(path) for path in reference_run_paths],
    }
    target = output / "evaluation.json"
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    atomic_write(target, lambda temporary: temporary.write_text(rendered))
    return target


def load_evaluation_provenance(per_motion_path: Path, expected_method: str) -> dict[str, Any]:
    """Validate one evaluation's CSV and its still-present upstream ``run.json``."""

    csv_path = per_motion_path.expanduser().resolve()
    metadata_path = csv_path.parent / "evaluation.json"
    try:
        payload = json.loads(metadata_path.read_text())
    except FileNotFoundError as error:
        raise ValueError(f"reconstruction evaluation provenance not found: {metadata_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid reconstruction evaluation provenance: {metadata_path}") from error
    if not isinstance(payload, dict) or payload.get("schema") != EVALUATION_PROVENANCE_SCHEMA:
        raise ValueError(f"invalid reconstruction evaluation provenance schema: {metadata_path}")
    if payload.get("per_motion_sha256") != file_sha256(csv_path):
        raise ValueError(f"reconstruction evaluation CSV hash mismatch: {csv_path}")
    primary = payload.get("primary_reconstruction")
    if not isinstance(primary, dict) or primary.get("method") != expected_method:
        raise ValueError(f"reconstruction evaluation method provenance mismatch: {metadata_path}")

    def validate_input(record: object) -> None:
        if not isinstance(record, dict):
            raise ValueError(f"reconstruction evaluation input is invalid: {metadata_path}")
        relative = record.get("run_json")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError(f"reconstruction evaluation run path must be relative: {metadata_path}")
        run_path = (metadata_path.parent / relative).resolve()
        upstream = reconstruction_input_record(run_path)
        if upstream["run_sha256"] != record.get("run_sha256"):
            raise ValueError(f"upstream reconstruction run hash mismatch: {run_path}")
        for key in (
            "method",
            "resolved_options",
            "method_identity",
            "method_identity_sha256",
            "scientific_identity_sha256",
        ):
            if upstream[key] != record.get(key):
                raise ValueError(f"evaluation provenance disagrees with upstream reconstruction {key}: {run_path}")

    validate_input(primary)
    references = payload.get("reference_reconstructions")
    if not isinstance(references, list):
        raise ValueError(f"reconstruction evaluation references must be a list: {metadata_path}")
    for reference in references:
        validate_input(reference)
    return payload


__all__ = [
    "EVALUATION_PROVENANCE_SCHEMA",
    "METHOD_IDENTITY_SCHEMA",
    "RECORD_PROVENANCE_SCHEMA",
    "RUN_PROVENANCE_SCHEMA",
    "SCIENTIFIC_IDENTITY_SCHEMA",
    "ValidatedRunProvenance",
    "build_run_provenance",
    "canonical_json",
    "content_sha256",
    "file_sha256",
    "load_evaluation_provenance",
    "reconstruction_input_record",
    "validate_run_provenance",
    "write_evaluation_provenance",
]
