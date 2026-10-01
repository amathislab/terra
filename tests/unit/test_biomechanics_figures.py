from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from terra.biomechanics_figures import render_comparison_figures


def _comparison(path: Path, *, include_grf: bool) -> tuple[Path, Path]:
    phase = np.linspace(0.0, 100.0, 101)
    target = np.stack((np.sin(np.pi * phase / 100.0) ** 2, np.cos(np.pi * phase / 100.0) ** 2), axis=-1)
    generated = np.roll(target, 3, axis=0)
    if include_grf:
        grf_channels = np.asarray(("right",))
        grf_shape = (1, len(phase), 1, 3)
        target_grf = np.zeros(grf_shape, dtype=np.float64)
        target_grf[0, :, 0, 2] = 0.8 + 0.2 * np.sin(2.0 * np.pi * phase / 100.0)
        generated_grf = target_grf * 0.95
        grf_valid = np.ones((1, len(phase), 1), dtype=bool)
        mapped_sides = np.asarray((("right",),))
    else:
        grf_channels = np.asarray((), dtype="U1")
        grf_shape = (1, len(phase), 0, 3)
        target_grf = np.empty(grf_shape, dtype=np.float64)
        generated_grf = np.empty(grf_shape, dtype=np.float64)
        grf_valid = np.empty((1, len(phase), 0), dtype=bool)
        mapped_sides = np.empty((1, 0), dtype="U8")
    traces = path / "comparison_traces.npz"
    np.savez_compressed(
        traces,
        schema_version=np.array(2),
        motion=np.array("Gait120/S001/LevelWalking/Trial01/AllSteps_stageii"),
        dataset=np.array("gait120"),
        subject=np.array("S001"),
        motion_type=np.array("LevelWalking"),
        phase_percent=phase,
        phase_definition=np.array("published gait cycle (0-100%)"),
        stride_positions=np.asarray(("cycle",)),
        emg_channels=np.asarray(("VastusLateralis", "TibialisAnterior")),
        emg_muscles=np.asarray(("VastusLateralis", "TibialisAnterior")),
        target_emg_valid=np.ones((1, len(phase), 2), dtype=bool),
        target_emg_raw=target[None],
        target_emg_std_raw=np.full((1, len(phase), 2), 0.05),
        target_emg_shape_normalized=target[None],
        generated_emg_mean_shape_normalized=generated[None],
        generated_emg_std_shape_normalized=np.full((1, len(phase), 2), 0.04),
        successful_seed=np.arange(10),
        grf_channels=grf_channels,
        grf_axes=np.asarray(("terra_x", "terra_y", "terra_z")),
        grf_mapped_sides=mapped_sides,
        target_grf_valid=grf_valid,
        target_grf_body_weight=target_grf,
        target_grf_std_body_weight=np.full(grf_shape, 0.03),
        generated_grf_mean_body_weight=generated_grf,
        generated_grf_std_body_weight=np.full(grf_shape, 0.02),
    )
    metrics = path / "metrics.csv"
    rows = [
        {
            "signal": "emg",
            "position": "cycle",
            "channel": channel,
            "waveform_correlation": "0.75",
            "peak_phase_error_percent": "3.0",
            "normal_grf_rmse_bw": "",
        }
        for channel in ("VastusLateralis", "TibialisAnterior")
    ]
    if include_grf:
        rows.append(
            {
                "signal": "grf",
                "position": "cycle",
                "channel": "right",
                "waveform_correlation": "0.91",
                "peak_phase_error_percent": "",
                "normal_grf_rmse_bw": "0.05",
            }
        )
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return traces, metrics


def test_render_comparison_figures_writes_emg_png_pdf_and_provenance(tmp_path: Path) -> None:
    traces, metrics = _comparison(tmp_path, include_grf=False)

    report = render_comparison_figures(traces, metrics, output_dir=tmp_path / "figures")

    assert [figure["signal"] for figure in report["figures"]] == ["emg"]
    assert (tmp_path / "figures" / "emg_traces.png").stat().st_size > 1_000
    assert (tmp_path / "figures" / "emg_traces.pdf").stat().st_size > 1_000
    manifest = json.loads((tmp_path / "figures" / "figures.json").read_text())
    assert manifest["inputs"]["comparison_traces"]["sha256"]
    assert manifest["style"]["raster_dpi"] == 300
    assert not (tmp_path / "figures" / "grf_traces.png").exists()


def test_render_comparison_figures_adds_all_grf_directions_when_available(tmp_path: Path) -> None:
    traces, metrics = _comparison(tmp_path, include_grf=True)

    report = render_comparison_figures(traces, metrics, output_dir=tmp_path / "figures")

    assert [figure["signal"] for figure in report["figures"]] == ["emg", "grf"]
    grf = report["figures"][1]
    assert grf["panels"] == 3
    assert Path(grf["png"]).stat().st_size > 1_000
    assert Path(grf["pdf"]).stat().st_size > 1_000


def test_renderer_isolated_from_parent_process_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import terra.biomechanics_figures as module

    traces, metrics = _comparison(tmp_path, include_grf=False)

    def broken_parent(*args, **kwargs):
        raise AssertionError("parent plotting state must not be used")

    monkeypatch.setattr(module, "_render_comparison_figures_in_process", broken_parent)
    report = module.render_comparison_figures(traces, metrics, output_dir=tmp_path / "isolated")
    assert len(report["render_source_sha256"]) == 64
    assert Path(report["figures"][0]["png"]).is_file()
