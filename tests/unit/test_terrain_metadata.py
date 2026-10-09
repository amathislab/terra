"""Tests for the neutral terrain metadata wire format."""

from __future__ import annotations

import json

import pytest

from loco_mujoco.core.terrain import BoxSpec, TerrainSpec
from terra.terrain.metadata import TerrainMetadata


def _terrain(*, height: float, provenance: dict | None = None) -> TerrainSpec:
    return TerrainSpec(
        boxes=(
            BoxSpec(
                pos=(0.0, 0.0, height / 2.0),
                size=(0.5, 0.5, height / 2.0),
                name="terrain_box_0",
            ),
        ),
        provenance=provenance or {},
    )


def test_metadata_round_trip_is_plain_canonical_terrain_json(tmp_path):
    first = _terrain(height=0.2, provenance={"source": "test", "version": 1})
    second = _terrain(height=0.2, provenance={"version": 1, "source": "test"})
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    TerrainMetadata.from_terrain(first).save(first_path)
    TerrainMetadata.from_terrain(second).save(second_path)

    assert TerrainMetadata.load(first_path).terrain == first
    assert TerrainSpec.load(first_path) == first
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_path.read_bytes().endswith(b"\n")


def test_metadata_rejects_removed_augmentation_fields(tmp_path):
    path = tmp_path / "augmented.json"
    path.write_text(
        json.dumps(_terrain(height=0.2).to_dict() | {"source_terrain": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected terrain metadata keys"):
        TerrainMetadata.load(path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "terrain metadata must be an object"),
        ({"boxes": [{"name": "missing geometry"}]}, "invalid terrain geometry"),
    ],
)
def test_metadata_rejects_invalid_geometry(tmp_path, payload, message):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        TerrainMetadata.load(path)


def test_metadata_rejects_invalid_json(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid terrain metadata JSON"):
        TerrainMetadata.load(path)
