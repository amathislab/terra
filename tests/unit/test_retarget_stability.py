from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from terra.stability import apply_stability_policy, retarget_stability, validate_stability_retry


def _result(
    *,
    step: float,
    error_max: float,
    error_mean: float = 0.04,
    native_step: float = 0.01,
    native_fps: float = 100.0,
):
    qpos = np.zeros((3, 7), dtype=float)
    qpos[1:, 0] = step
    error = np.full((3, 2), error_mean, dtype=float)
    error[0, 0] = error_max
    return SimpleNamespace(
        trajectory=SimpleNamespace(
            data=SimpleNamespace(qpos=qpos),
            info=SimpleNamespace(frequency=100.0),
        ),
        analysis={
            "pos_error": error,
            "native_fps": native_fps,
            "numerical_envelope": {"max_root_step_m": native_step},
        },
    )


def test_stability_retry_gate_is_conjunctive() -> None:
    assert retarget_stability(_result(step=0.02, error_max=0.2))["requires_retry"] is False
    assert retarget_stability(_result(step=0.04, error_max=0.2))["requires_retry"] is False
    assert retarget_stability(_result(step=0.02, error_max=0.4))["requires_retry"] is False
    report = retarget_stability(_result(step=0.04, error_max=0.4))
    assert report["requires_retry"] is True
    assert report["max_control_root_speed_m_s"] == pytest.approx(4.0)
    assert "schema_version" not in report


def test_stability_gate_allows_fast_output_when_native_motion_is_equally_fast() -> None:
    report = retarget_stability(_result(step=0.10, error_max=0.4, native_step=0.09, native_fps=100.0))

    assert report["max_control_root_speed_m_s"] == pytest.approx(10.0)
    assert report["native_root_speed_m_s"] == pytest.approx(9.0)
    assert report["requires_retry"] is False


def test_stability_retry_must_clear_gate_and_preserve_mean_tracking() -> None:
    primary = retarget_stability(_result(step=0.04, error_max=0.4, error_mean=0.04))
    retry = retarget_stability(_result(step=0.01, error_max=0.2, error_mean=0.04))
    validate_stability_retry(primary, retry)

    with pytest.raises(ValueError, match="still exceeds"):
        validate_stability_retry(primary, primary)
    regressed = retarget_stability(_result(step=0.01, error_max=0.2, error_mean=0.20))
    with pytest.raises(ValueError, match="regressed mean tracking"):
        validate_stability_retry(primary, regressed)


def test_shared_stability_policy_retries_once_with_conservative_step() -> None:
    primary = _result(step=0.04, error_max=0.4)
    retried = _result(step=0.01, error_max=0.2)
    requests = []

    result = apply_stability_policy(
        primary,
        config={"step_size": 0.2, "other": 7},
        retry=lambda config: requests.append(config) or retried,
    )

    assert result is retried
    assert requests == [{"step_size": 0.1, "other": 7}]
    assert retried.analysis["stability_retry"]["triggered"] is True
    assert retried.analysis["stability_retry"]["primary"]["requires_retry"] is True


def test_shared_stability_policy_does_not_retry_a_passing_result() -> None:
    primary = _result(step=0.01, error_max=0.2)

    result = apply_stability_policy(
        primary,
        config=None,
        retry=lambda _config: pytest.fail("passing output must not be rerun"),
    )

    assert result is primary
    assert primary.analysis["stability_check"]["requires_retry"] is False
