"""Regression tests for motion-local isolation of the native SMPL baseline."""

from __future__ import annotations

import os
import signal
from dataclasses import replace

from terra import retarget_cache_paths
from terra.commands.run import (
    _run_worker_isolated,
    _smpl_needs_isolation,
    _worker_exit_error,
)
from terra.dataset_pipeline import MotionRecord, load_dataset_config
from terra.datasets.config import bundled_dataset_config


def _terminate_native_worker(_config, _record, _overwrite):
    """Model a native library terminating a worker without a Python exception."""
    os.kill(os.getpid(), signal.SIGTERM)


def test_disposable_worker_returns_normal_python_result(tmp_path):
    config = load_dataset_config(bundled_dataset_config("gait120"))
    config = replace(config, name="example", cache_root=tmp_path, method="smpl")
    record = MotionRecord(
        motion="Example/conversion-rejected",
        dataset="example",
        source_path=tmp_path / "missing.npz",
        fit_passed=False,
    )

    result = _run_worker_isolated(config, record, overwrite=False)

    assert result["motion"] == record.motion
    assert result["status"] == "conversion_failed"


def test_disposable_worker_converts_native_signal_to_motion_failure(tmp_path):
    config = load_dataset_config(bundled_dataset_config("gait120"))
    config = replace(config, name="example", cache_root=tmp_path, method="smpl")
    record = MotionRecord(
        motion="Example/native-crash",
        dataset="example",
        source_path=tmp_path / "source.npz",
    )

    result = _run_worker_isolated(
        config,
        record,
        overwrite=False,
        worker=_terminate_native_worker,
    )

    assert result["motion"] == record.motion
    assert result["status"] == "failed"
    assert "SIGTERM" in result["error"]


def test_native_signal_is_reported_explicitly():
    message = _worker_exit_error(-signal.SIGABRT)

    assert "SIGABRT" in message
    assert f"signal {signal.SIGABRT}" in message


def test_smpl_isolation_is_only_used_for_uncached_fits(tmp_path):
    config = load_dataset_config(bundled_dataset_config("gait120"))
    config = replace(config, cache_root=tmp_path, method="smpl")
    record = MotionRecord(
        motion="Example/Trial01",
        dataset="example",
        source_path=tmp_path / "source.npz",
    )

    assert _smpl_needs_isolation(config, record, overwrite=False)

    paths = retarget_cache_paths(
        config.cache_root,
        record.motion,
        method="smpl",
        env_name=config.env_name,
    )
    paths.trajectory_path.parent.mkdir(parents=True)
    paths.trajectory_path.touch()
    paths.analysis_path.touch()

    assert not _smpl_needs_isolation(config, record, overwrite=False)
    assert _smpl_needs_isolation(config, record, overwrite=True)
    assert not _smpl_needs_isolation(
        replace(config, method="terra"),
        record,
        overwrite=False,
    )
    assert not _smpl_needs_isolation(
        config,
        replace(record, fit_passed=False),
        overwrite=False,
    )
