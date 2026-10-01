from __future__ import annotations

import json

import pytest

from terra.evaluation.audit import audit_metric_rows, audit_timing_contexts


def _row(method: str = "TERRA") -> dict:
    context = {
        "cpu_model": "Test CPU",
        "cpu_request": 8,
        "worker_processes": 8,
        "threads_per_worker": 1,
        "omp_num_threads": 1,
        "mkl_num_threads": 1,
        "openblas_num_threads": 1,
        "xla_flags": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    }
    return {
        "method": method,
        "motion_class": "dataset",
        "motion": "motion",
        "error": "",
        "frames": 20,
        "fps": 10.0,
        "common_start_s": 0.0,
        "common_end_s": 2.0,
        "common_duration_s": 2.0,
        "source_contact_s": 1.0,
        "source_support_s": 1.5,
        "desired_contact_point_s": 0.5,
        "source_contact_transitions": 2,
        "source_support_transitions": 4,
        "desired_contact_transitions": 2,
        "penetration_duration_pct": 10.0,
        "skating_duration_pct": 20.0,
        "floating_duration_pct": 3.0,
        "contact_preservation_pct": 90.0,
        "joint_limit_duration_pct": 0.0,
        "tendon_jump_duration_pct": 0.0,
        "self_collision_duration_pct": 1.0,
        "support_penetration_duration_pct": 2.0,
        "invalid_support_duration_pct": 5.0,
        "penetration_max_depth_mm": 3.0,
        "penetration_frame_depth_sum_mm": 6.0,
        "penetration_frame_depth_sq_sum_mm2": 20.0,
        "penetration_frame_depth_n": 2,
        "skating_max_velocity_m_s": 0.5,
        "skating_frame_velocity_sum_m_s": 1.0,
        "skating_frame_velocity_sq_sum_m2_s2": 0.58,
        "skating_frame_velocity_n": 2,
        "floating_max_height_mm": 20.0,
        "joint_limit_max_excess_deg": 0.0,
        "tendon_max_jump": 0.0,
        "self_collision_max_depth_mm": 2.0,
        "support_penetration_max_depth_mm": 4.0,
        "solver_native_fps": 4.0,
        "t_frame_s": 0.25,
        "solver_solved_frames": 20,
        "solver_seconds_per_motion_second": 2.5,
        "timing_context_json": json.dumps(context, sort_keys=True),
    }


def test_metric_audit_accepts_formula_consistent_rows():
    first = _row()
    second = _row("OmniRetarget")

    assert audit_metric_rows([first, second])["passed"] is True
    assert audit_timing_contexts([first, second])["passed"] is True


def test_metric_audit_rejects_bounded_identity_and_pooling_errors():
    row = _row()
    row["contact_preservation_pct"] = 101.0
    row["invalid_support_duration_pct"] = 4.0
    row["penetration_max_depth_mm"] = 6.0

    report = audit_metric_rows([row])

    assert report["passed"] is False
    assert any("outside [0, 100]" in issue for issue in report["issues"])
    assert any("invalid support" in issue for issue in report["issues"])
    assert any("mean of its pooled" in issue for issue in report["issues"])


def test_metric_audit_allows_only_machine_precision_at_percentage_bounds():
    row = _row()
    row["skating_duration_pct"] = 100.0 + 1e-12
    assert audit_metric_rows([row])["passed"] is True

    row["skating_duration_pct"] = 100.0 + 1e-6
    report = audit_metric_rows([row])
    assert report["passed"] is False
    assert any("outside [0, 100]" in issue for issue in report["issues"])


def test_metric_audit_accepts_undefined_contact_metrics_without_eligible_source_time():
    row = _row()
    row["source_contact_s"] = 0.0
    row["source_support_s"] = 0.0
    row["desired_contact_point_s"] = 0.0
    for key in (
        "skating_duration_pct",
        "skating_max_velocity_m_s",
        "floating_duration_pct",
        "floating_max_height_mm",
        "support_penetration_duration_pct",
        "support_penetration_max_depth_mm",
        "invalid_support_duration_pct",
        "contact_preservation_pct",
    ):
        row[key] = float("nan")
    row["skating_frame_velocity_sum_m_s"] = 0.0
    row["skating_frame_velocity_sq_sum_m2_s2"] = 0.0
    row["skating_frame_velocity_n"] = 0

    assert audit_metric_rows([row])["passed"] is True


def test_metric_audit_rejects_undefined_contact_metric_with_eligible_source_time():
    row = _row()
    row["skating_duration_pct"] = float("nan")

    report = audit_metric_rows([row])

    assert report["passed"] is False
    assert any("skating_duration_pct is not finite" in issue for issue in report["issues"])


def test_cross_method_denominator_tolerance_uses_contact_transition_bound():
    first = _row()
    second = _row("GMR")
    first["fps"] = second["fps"] = 100.0
    second["desired_contact_point_s"] += 0.029
    assert audit_metric_rows([first, second])["passed"] is True

    second["desired_contact_point_s"] += 0.01
    report = audit_metric_rows([first, second])
    assert report["passed"] is False
    assert any("source-derived desired_contact_point_s" in issue for issue in report["issues"])


def test_timing_audit_accepts_final_legacy_producer_timings_with_warning():
    legacy = _row()
    legacy["timing_context_json"] = ""

    report = audit_timing_contexts([legacy])

    assert report["passed"] is True
    assert report["mode"] == "producer_retarget_fps"
    assert report["successful_rows_with_legacy_timing"] == 1
    assert report["warnings"]


def test_timing_audit_rejects_incomparable_contexts():

    first = _row()
    second = _row("GMR")
    context = json.loads(second["timing_context_json"])
    context["cpu_request"] = 16
    second["timing_context_json"] = json.dumps(context)
    report = audit_timing_contexts([first, second])
    assert report["passed"] is False
    assert any("different timing allocations" in issue for issue in report["issues"])


def test_timing_audit_rejects_mixed_uniform_and_legacy_timings():
    uniform = _row()
    uniform["timing_source"] = "uniform_retarget_call"
    legacy = _row("GMR")
    legacy["timing_context_json"] = ""
    legacy["timing_source"] = "producer_retarget_fps"

    report = audit_timing_contexts([uniform, legacy])

    assert report["passed"] is False
    assert any("cannot be mixed" in issue for issue in report["issues"])


@pytest.mark.parametrize("field", ("omp_num_threads", "mkl_num_threads", "openblas_num_threads"))
def test_timing_audit_rejects_hidden_thread_oversubscription(field):
    row = _row()
    context = json.loads(row["timing_context_json"])
    context[field] = 8
    row["timing_context_json"] = json.dumps(context)

    assert audit_timing_contexts([row])["passed"] is False
