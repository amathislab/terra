from __future__ import annotations

import json

from terra.evaluation.reconstruction_figures import (
    render_voronoi_sensitivity,
    select_apparatus_representatives,
    select_prism_representatives,
)


def test_prism_representative_uses_condition_median_ground_truth_area():
    records = [
        {
            "motion": f"PRISM/subj00{index}/take001_poses",
            "condition_group": "Stairs",
            "mesh_full": {"gt_raised_area_m2": area},
        }
        for index, area in enumerate((0.2, 0.4, 1.8), start=1)
    ]

    assert select_prism_representatives(records) == {"Stairs": "PRISM/subj002/take001_poses"}


def test_apparatus_representative_uses_median_source_duration_per_condition():
    rows = [
        {
            "motion": f"Gait120/S00{index}/StairAscent/Trial01/AllSteps_stageii",
            "condition": "StairAscent",
            "expected_family": "steps",
            "frames": str(frames),
        }
        for index, frames in enumerate((100, 200, 900), start=1)
    ]

    assert select_apparatus_representatives("gait120", rows) == {
        "StairAscent": "Gait120/S002/StairAscent/Trial01/AllSteps_stageii"
    }


def test_voronoi_sensitivity_preserves_shared_cohort_counts(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "selected": 31,
                "scored": 31,
                "overall": {
                    "primary_passes": 18,
                    "observed_height_mae_mm": {"mean": 50.0},
                    "raised_terrain_coverage": {"mean": 0.69},
                    "flat_terrain_coverage": {"mean": 0.98},
                    "full_footprint_iou": {"mean": 0.57},
                    "full_support_f1_50mm": {"mean": 0.70},
                },
            }
        )
    )

    rows = render_voronoi_sensitivity({"0.2 m benchmark": summary}, tmp_path)

    assert rows[0]["selected"] == rows[0]["scored"] == 31
    assert rows[0]["primary_passes"] == 18
    assert (tmp_path / "voronoi_sensitivity.csv").is_file()
    assert (tmp_path / "voronoi_sensitivity.png").is_file()
    assert (tmp_path / "voronoi_sensitivity.pdf").is_file()
