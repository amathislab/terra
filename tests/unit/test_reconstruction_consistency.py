from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from terra.evaluation.reconstruction_consistency import (
    ContactQuery,
    load_cohort,
    score_contacts,
    score_free_space,
    source_surface_offsets,
    summarize,
)


class _Terrain:
    def __init__(self, heights: dict[float, float], penetration: np.ndarray | None = None):
        self.heights = heights
        self._penetration = penetration
        self.boxes = ()

    @property
    def walkable(self):
        return self

    def height_at(self, x: float, _y: float) -> float:
        return self.heights[x]

    def penetration(self, points: np.ndarray) -> np.ndarray:
        if self._penetration is None:
            return np.zeros(len(points))
        assert len(points) == len(self._penetration)
        return self._penetration


def _contacts() -> tuple[ContactQuery, ...]:
    return (
        ContactQuery("L_Toe", 0, 5, np.array([0.0, 0.0, 0.01])),
        ContactQuery("L_Ankle", 0, 5, np.array([0.0, 0.1, 0.06])),
        ContactQuery("R_Toe", 0, 5, np.array([0.0, 1.0, 0.01])),
        ContactQuery("R_Ankle", 0, 5, np.array([0.0, 1.1, 0.06])),
        ContactQuery("L_Toe", 10, 15, np.array([1.0, 0.0, 0.11])),
        ContactQuery("L_Ankle", 10, 15, np.array([1.0, 0.1, 0.16])),
    )


def test_common_contact_score_uses_source_offsets_and_recovers_exact_support() -> None:
    contacts = _contacts()

    assert source_surface_offsets(contacts) == {
        "L_Ankle": pytest.approx(0.06),
        "L_Toe": pytest.approx(0.01),
        "R_Ankle": pytest.approx(0.06),
        "R_Toe": pytest.approx(0.01),
    }
    score = score_contacts(contacts, _Terrain({0.0: 0.0, 1.0: 0.10}))

    assert score["contact_height_mae_mm"] == pytest.approx(0.0)
    assert score["contact_consistent_pct"] == 100.0
    assert score["contact_penetrating_pct"] == 0.0
    assert score["contact_floating_pct"] == 0.0
    assert score["raised_precision_pct"] == 100.0
    assert score["raised_recall_pct"] == 100.0
    assert score["raised_f1_pct"] == 100.0
    assert score["support_pair_n"] == 3
    assert score["support_pair_height_mae_mm"] == pytest.approx(0.0)


def test_signed_contact_errors_separate_penetration_floating_and_false_support() -> None:
    score = score_contacts(_contacts(), _Terrain({0.0: 0.08, 1.0: 0.0}), contact_tolerance_m=0.05)

    assert score["contact_penetrating_n"] == 4
    assert score["contact_floating_n"] == 2
    assert score["raised_true_positive_n"] == 0
    assert score["raised_false_positive_n"] == 4
    assert score["raised_false_negative_n"] == 2
    assert score["raised_f1_pct"] == 0.0


def test_free_space_excludes_only_the_corresponding_contact_samples() -> None:
    joints = np.zeros((3, 2, 3))
    # Flattened penetration order is (frame, joint). The first joint is supported on
    # frames 0--1; the same depths on the second joint remain free-space violations.
    terrain = _Terrain({}, np.array([0.10, 0.10, 0.10, 0.10, 0.0, 0.0]))
    contacts = (ContactQuery("L_Toe", 0, 2, np.array([0.0, 0.0, 0.0])),)

    score = score_free_space(contacts, terrain, joints, ("L_Toe", "Pelvis"), 100.0, tolerance_m=0.03)

    assert score["free_space_query_n"] == 3
    assert score["free_space_penetrating_n"] == 2
    assert score["free_space_penetrating_pct"] == pytest.approx(200.0 / 3.0)
    assert score["free_space_penetration_max_mm"] == pytest.approx(100.0)
    assert score["free_space_violation"] is True


def test_summary_retains_failures_and_uses_common_success_denominator() -> None:
    rows = [
        {"method": "A", "motion": "one", "error": "", "contact_n": 2, "contact_height_mae_mm": 1.0},
        {"method": "A", "motion": "two", "error": "", "contact_n": 2, "contact_height_mae_mm": 3.0},
        {"method": "B", "motion": "one", "error": "", "contact_n": 2, "contact_height_mae_mm": 2.0},
        {"method": "B", "motion": "two", "error": "failed"},
    ]

    result = summarize(rows, ("A", "B"), 2)

    assert result["methods"]["A"]["successful"] == 2
    assert result["methods"]["B"]["failed"] == 1
    assert result["common_success_motions"] == 1
    assert result["common_success_methods"]["A"]["metrics"]["contact_height_mae_mm"]["mean"] == 1.0


def test_legacy_reconstruction_requires_explicit_opt_in(tmp_path: Path) -> None:
    root = tmp_path / "terrain"
    root.mkdir()
    (root / "run.json").write_text(json.dumps({"method": "old", "motions": 1, "options": {}}))
    with (root / "status.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("motion", "method", "status", "output", "error"))
        writer.writeheader()
        writer.writerow(
            {"motion": "set/take", "method": "old", "status": "ok", "output": "set__take.json", "error": ""}
        )

    with pytest.raises(ValueError, match="allow-legacy-provenance"):
        load_cohort("old", root, ("set/take",))

    cohort = load_cohort("old", root, ("set/take",), allow_legacy_provenance=True)
    assert cohort.legacy_provenance is True
