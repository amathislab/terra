"""Publication figures for policy-to-EMG/GRF comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

FIGURE_SCHEMA_VERSION = 1
MEASURED_COLOR = "#171717"
MEASURED_BAND_COLOR = "#8A8A8A"
GENERATED_COLOR = "#0072B2"
GENERATED_BAND_COLOR = "#56B4E9"


def _mpl() -> Any:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/terra-matplotlib")
    # Matplotlib's PDF backend invokes fontTools for routine font subsetting.
    # Application-level INFO logging should not expand that implementation
    # detail into hundreds of glyph-table messages per figure.
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.0,
            "axes.labelsize": 8.0,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.5,
            "lines.linewidth": 1.6,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def _load_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _scalar(values: Mapping[str, np.ndarray], key: str) -> str:
    return str(np.asarray(values[key]).reshape(()))


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-_.").casefold() or "trace"


def _words(value: str) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return re.sub(r"[_-]+", " ", text).strip()


def _phase_label(definition: str) -> str:
    key = definition.casefold()
    if "gait cycle" in key:
        return "Gait cycle (%)"
    if "contact time" in key or "contact cycle" in key:
        return "Contact cycle (%)"
    return "Normalized phase (%)"


def _title(arrays: Mapping[str, np.ndarray], position: str, signal: str) -> str:
    dataset = _scalar(arrays, "dataset").replace("gait120", "Gait120").replace("darmstadt", "Darmstadt")
    if dataset == "vielemeyer":
        dataset = "Vielemeyer"
    subject = _scalar(arrays, "subject")
    motion = _words(_scalar(arrays, "motion_type"))
    position_text = "" if position.casefold() in {"cycle", "contact"} else f" · {position}"
    return f"{dataset} · {subject} · {motion}{position_text} · {signal}"


def _metric_lookup(rows: Sequence[Mapping[str, str]], signal: str) -> dict[tuple[str, str], Mapping[str, str]]:
    return {(str(row["position"]), str(row["channel"])): row for row in rows if str(row.get("signal", "")) == signal}


def _number(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _masked(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result[~np.asarray(valid, dtype=bool)] = np.nan
    return result


def _target_emg_band(
    mean_normalized: np.ndarray,
    mean_raw: np.ndarray,
    std_raw: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(valid, dtype=bool) & np.isfinite(mean_raw) & np.isfinite(std_raw)
    if not mask.any():
        empty = np.full_like(np.asarray(mean_normalized, dtype=np.float64), np.nan)
        return empty, empty
    span = float(np.max(np.asarray(mean_raw)[mask]) - np.min(np.asarray(mean_raw)[mask]))
    scaled_std = np.zeros_like(np.asarray(mean_normalized, dtype=np.float64))
    if span > 1e-12:
        scaled_std[mask] = np.asarray(std_raw, dtype=np.float64)[mask] / span
    mean = np.asarray(mean_normalized, dtype=np.float64)
    lower = _masked(np.maximum(0.0, mean - scaled_std), mask)
    upper = _masked(mean + scaled_std, mask)
    return lower, upper


def _generated_band(mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(mean) & np.isfinite(std)
    return (
        _masked(np.maximum(0.0, np.asarray(mean) - np.asarray(std)), finite),
        _masked(np.asarray(mean) + np.asarray(std), finite),
    )


def _legend(figure: Any, *, signal: str, trials: int) -> None:
    from matplotlib.lines import Line2D

    units = "EMG" if signal == "EMG" else "GRF"
    handles = [
        Line2D([0], [0], color=MEASURED_COLOR, linestyle="--", label=f"Measured {units} mean ± SD"),
        Line2D(
            [0],
            [0],
            color=GENERATED_COLOR,
            label=f"Policy mean ± SD ({trials} completed gait{'s' if trials != 1 else ''})",
        ),
    ]
    figure.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.955))


def _save_figure(figure: Any, stem: Path) -> tuple[Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, dpi in ((".png", 300), (".pdf", None)):
        destination = stem.with_suffix(suffix)
        temporary = destination.with_name(f".{destination.stem}.tmp-{os.getpid()}{suffix}")
        options: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white", "format": suffix[1:]}
        if dpi is not None:
            options["dpi"] = dpi
        figure.savefig(temporary, **options)
        temporary.replace(destination)
        outputs.append(destination.resolve())
    return outputs[0], outputs[1]


def _render_emg_position(
    arrays: Mapping[str, np.ndarray],
    metrics: Sequence[Mapping[str, str]],
    position_index: int,
    output_stem: Path,
) -> tuple[Path, Path] | None:
    valid_all = np.asarray(arrays["target_emg_valid"], dtype=bool)[position_index]
    generated_all = np.asarray(arrays["generated_emg_mean_shape_normalized"], dtype=np.float64)[position_index]
    eligible = [
        index
        for index in range(valid_all.shape[1])
        if np.any(valid_all[:, index] & np.isfinite(generated_all[:, index]))
    ]
    if not eligible:
        return None
    plt = _mpl()
    columns = min(4, len(eligible))
    rows = math.ceil(len(eligible) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(7.2, 1.75 * rows + 0.85),
        sharex=True,
        squeeze=False,
    )
    phase = np.asarray(arrays["phase_percent"], dtype=np.float64)
    positions = [str(value) for value in np.asarray(arrays["stride_positions"]).reshape(-1)]
    channels = [str(value) for value in np.asarray(arrays["emg_channels"]).reshape(-1)]
    muscles = [str(value) for value in np.asarray(arrays["emg_muscles"]).reshape(-1)]
    metric_by_channel = _metric_lookup(metrics, "emg")
    trials = (
        int(np.max(np.asarray(arrays["generated_emg_sample_count"])[position_index]))
        if "generated_emg_sample_count" in arrays
        else len(np.asarray(arrays["successful_seed"]).reshape(-1))
    )
    ymax = 1.0
    for panel_index, channel_index in enumerate(eligible):
        axis = axes.flat[panel_index]
        valid = valid_all[:, channel_index]
        target = _masked(np.asarray(arrays["target_emg_shape_normalized"])[position_index, :, channel_index], valid)
        target_lower, target_upper = _target_emg_band(
            target,
            np.asarray(arrays["target_emg_raw"])[position_index, :, channel_index],
            np.asarray(arrays["target_emg_std_raw"])[position_index, :, channel_index],
            valid,
        )
        generated = generated_all[:, channel_index]
        generated_lower, generated_upper = _generated_band(
            generated,
            np.asarray(arrays["generated_emg_std_shape_normalized"])[position_index, :, channel_index],
        )
        axis.fill_between(phase, target_lower, target_upper, color=MEASURED_BAND_COLOR, alpha=0.20, linewidth=0)
        axis.fill_between(
            phase,
            generated_lower,
            generated_upper,
            color=GENERATED_BAND_COLOR,
            alpha=0.28,
            linewidth=0,
        )
        axis.plot(phase, target, color=MEASURED_COLOR, linestyle="--", linewidth=1.6)
        axis.plot(phase, generated, color=GENERATED_COLOR, linewidth=1.7)
        axis.set_title(_words(muscles[channel_index]))
        axis.set_xlim(0.0, 100.0)
        axis.set_xticks((0, 25, 50, 75, 100))
        axis.grid(axis="y", color="#D7D7D7", linewidth=0.45, alpha=0.75)
        metric = metric_by_channel.get((positions[position_index], channels[channel_index]))
        if metric is not None:
            correlation = _number(metric.get("waveform_correlation"))
            peak_error = _number(metric.get("peak_phase_error_percent"))
            lines = []
            if correlation is not None:
                lines.append(f"r = {correlation:.2f}")
            if peak_error is not None:
                lines.append(f"Δpeak = {peak_error:.0f}%")
            if lines:
                axis.text(
                    0.97,
                    0.96,
                    "\n".join(lines),
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    color=GENERATED_COLOR,
                    fontsize=6.8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
                )
        upper_values = np.concatenate(
            (target_upper[np.isfinite(target_upper)], generated_upper[np.isfinite(generated_upper)])
        )
        if len(upper_values):
            ymax = max(ymax, float(np.max(upper_values)))
    ymax = math.ceil(ymax * 5.0) / 5.0
    for panel_index, axis in enumerate(axes.flat):
        if panel_index >= len(eligible):
            axis.set_visible(False)
            continue
        axis.set_ylim(0.0, ymax * 1.04)
        row, column = divmod(panel_index, columns)
        if column == 0:
            axis.set_ylabel("Normalized activity")
        if row == rows - 1:
            axis.set_xlabel(_phase_label(_scalar(arrays, "phase_definition")))
    position = positions[position_index]
    figure.suptitle(_title(arrays, position, "EMG and muscle activation"), x=0.5, y=0.995, fontsize=9.5)
    figure.text(0.006, 0.985, "a", fontsize=12, fontweight="bold", va="top")
    _legend(figure, signal="EMG", trials=trials)
    figure.tight_layout(rect=(0.02, 0.015, 1.0, 0.90), h_pad=0.75, w_pad=0.55)
    outputs = _save_figure(figure, output_stem)
    plt.close(figure)
    return outputs


def _axis_title(value: str) -> str:
    key = value.casefold().removeprefix("terra_")
    if key in {"x", "forward", "walking", "walking_direction", "anterior_posterior"}:
        return "Fore-aft"
    if key in {"y", "left", "lateral", "medio_lateral"}:
        return "Medio-lateral"
    if key in {"z", "up", "vertical"}:
        return "Vertical"
    return _words(value)


def _render_grf_position(
    arrays: Mapping[str, np.ndarray],
    metrics: Sequence[Mapping[str, str]],
    position_index: int,
    output_stem: Path,
) -> tuple[Path, Path] | None:
    channels = [str(value) for value in np.asarray(arrays["grf_channels"]).reshape(-1)]
    if not channels:
        return None
    target_valid = np.asarray(arrays["target_grf_valid"], dtype=bool)[position_index]
    generated_mean = np.asarray(arrays["generated_grf_mean_body_weight"], dtype=np.float64)[position_index]
    eligible = [
        index
        for index in range(len(channels))
        if np.any(target_valid[:, index] & np.isfinite(generated_mean[:, index]).any(axis=-1))
    ]
    if not eligible:
        return None
    plt = _mpl()
    axes_names = [str(value) for value in np.asarray(arrays["grf_axes"]).reshape(-1)]
    figure, axes = plt.subplots(
        len(eligible),
        len(axes_names),
        figsize=(7.2, 1.75 * len(eligible) + 0.9),
        sharex=True,
        squeeze=False,
    )
    phase = np.asarray(arrays["phase_percent"], dtype=np.float64)
    positions = [str(value) for value in np.asarray(arrays["stride_positions"]).reshape(-1)]
    mapped_sides = np.asarray(arrays["grf_mapped_sides"])
    metric_by_channel = _metric_lookup(metrics, "grf")
    trials = (
        int(np.max(np.asarray(arrays["generated_grf_sample_count"])[position_index]))
        if "generated_grf_sample_count" in arrays
        else len(np.asarray(arrays["successful_seed"]).reshape(-1))
    )
    normal_axes = [
        index for index, name in enumerate(axes_names) if name.casefold() in {"z", "terra_z", "up", "vertical"}
    ]
    normal_axis = normal_axes[0] if len(normal_axes) == 1 else None
    for row_index, channel_index in enumerate(eligible):
        valid = target_valid[:, channel_index]
        side = str(mapped_sides[position_index, channel_index]) or channels[channel_index]
        metric = metric_by_channel.get((positions[position_index], channels[channel_index]))
        for axis_index, axis_name in enumerate(axes_names):
            axis = axes[row_index, axis_index]
            target = _masked(
                np.asarray(arrays["target_grf_body_weight"])[position_index, :, channel_index, axis_index], valid
            )
            target_std = _masked(
                np.asarray(arrays["target_grf_std_body_weight"])[position_index, :, channel_index, axis_index], valid
            )
            generated = generated_mean[:, channel_index, axis_index]
            generated_std = np.asarray(arrays["generated_grf_std_body_weight"])[
                position_index, :, channel_index, axis_index
            ]
            target_lower = target - target_std
            target_upper = target + target_std
            generated_lower = generated - generated_std
            generated_upper = generated + generated_std
            axis.fill_between(phase, target_lower, target_upper, color=MEASURED_BAND_COLOR, alpha=0.20, linewidth=0)
            axis.fill_between(
                phase,
                generated_lower,
                generated_upper,
                color=GENERATED_BAND_COLOR,
                alpha=0.28,
                linewidth=0,
            )
            axis.plot(phase, target, color=MEASURED_COLOR, linestyle="--", linewidth=1.6)
            axis.plot(phase, generated, color=GENERATED_COLOR, linewidth=1.7)
            axis.axhline(0.0, color="#A8A8A8", linewidth=0.55, zorder=0)
            axis.set_xlim(0.0, 100.0)
            axis.set_xticks((0, 25, 50, 75, 100))
            axis.grid(axis="y", color="#D7D7D7", linewidth=0.45, alpha=0.75)
            if row_index == 0:
                axis.set_title(_axis_title(axis_name))
            if axis_index == 0:
                axis.set_ylabel(f"{_words(side).title()} foot\nGRF (BW)")
            if row_index == len(eligible) - 1:
                axis.set_xlabel(_phase_label(_scalar(arrays, "phase_definition")))
            if metric is not None and normal_axis == axis_index:
                correlation = _number(metric.get("waveform_correlation"))
                rmse = _number(metric.get("normal_grf_rmse_bw"))
                lines = []
                if correlation is not None:
                    lines.append(f"r = {correlation:.2f}")
                if rmse is not None:
                    lines.append(f"RMSE = {rmse:.2f} BW")
                if lines:
                    axis.text(
                        0.97,
                        0.96,
                        "\n".join(lines),
                        transform=axis.transAxes,
                        ha="right",
                        va="top",
                        color=GENERATED_COLOR,
                        fontsize=6.8,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
                    )
    position = positions[position_index]
    figure.suptitle(_title(arrays, position, "ground-reaction forces"), x=0.5, y=0.995, fontsize=9.5)
    figure.text(0.006, 0.985, "b", fontsize=12, fontweight="bold", va="top")
    _legend(figure, signal="GRF", trials=trials)
    figure.tight_layout(rect=(0.02, 0.015, 1.0, 0.89), h_pad=0.75, w_pad=0.65)
    outputs = _save_figure(figure, output_stem)
    plt.close(figure)
    return outputs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _render_comparison_figures_in_process(
    comparison_traces: str | Path,
    metrics_csv: str | Path,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Render EMG and available GRF panels from one completed comparison."""

    traces_path = Path(comparison_traces).expanduser().resolve()
    metrics_path = Path(metrics_csv).expanduser().resolve()
    if not traces_path.is_file():
        raise FileNotFoundError(f"comparison traces not found: {traces_path}")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"comparison metrics not found: {metrics_path}")
    output = Path(output_dir).expanduser().resolve()
    arrays = _load_npz(traces_path)
    metrics = _load_metrics(metrics_path)
    positions = [str(value) for value in np.asarray(arrays["stride_positions"]).reshape(-1)]
    records: list[dict[str, Any]] = []
    multiple_positions = len(positions) > 1
    for position_index, position in enumerate(positions):
        suffix = f"-{_safe_name(position)}" if multiple_positions else ""
        emg = _render_emg_position(arrays, metrics, position_index, output / f"emg_traces{suffix}")
        if emg is not None:
            records.append(
                {
                    "signal": "emg",
                    "position": position,
                    "panels": int(np.asarray(arrays["emg_channels"]).size),
                    "png": str(emg[0]),
                    "pdf": str(emg[1]),
                }
            )
        grf = _render_grf_position(arrays, metrics, position_index, output / f"grf_traces{suffix}")
        if grf is not None:
            records.append(
                {
                    "signal": "grf",
                    "position": position,
                    "panels": int(np.asarray(arrays["grf_channels"]).size * np.asarray(arrays["grf_axes"]).size),
                    "png": str(grf[0]),
                    "pdf": str(grf[1]),
                }
            )
    report = {
        "schema_version": FIGURE_SCHEMA_VERSION,
        "render_source_sha256": _sha256(Path(__file__)),
        "comparison_schema_version": int(np.asarray(arrays["schema_version"]).reshape(())),
        "motion": _scalar(arrays, "motion"),
        "dataset": _scalar(arrays, "dataset"),
        "subject": _scalar(arrays, "subject"),
        "motion_type": _scalar(arrays, "motion_type"),
        "inputs": {
            "comparison_traces": {"path": str(traces_path), "sha256": _sha256(traces_path)},
            "metrics": {"path": str(metrics_path), "sha256": _sha256(metrics_path)},
        },
        "style": {
            "measured": "black dashed mean with gray ±1 SD band",
            "generated": "blue solid mean with blue ±1 SD band across completed policy gait windows",
            "emg_normalization": (
                "measured mean min-max normalized with SD divided by the same mean span; "
                "each generated gait window min-max normalized per position and channel before mean and SD"
            ),
            "grf_normalization": "body weights",
            "raster_dpi": 300,
            "vector_format": "PDF with embedded TrueType fonts",
        },
        "figures": records,
    }
    manifest = output / "figures.json"
    _write_json(manifest, report)
    report["manifest"] = str(manifest.resolve())
    return report


def render_comparison_figures(
    comparison_traces: str | Path,
    metrics_csv: str | Path,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Render in a clean process, isolated from long-running policy state.

    In-process plotting after some policy runs produced invalid tight bounding
    boxes despite valid saved arrays. A clean renderer reproduces the figures
    from the exact artifacts without inheriting policy-library global state.
    """
    traces = Path(comparison_traces).expanduser().resolve()
    metrics = Path(metrics_csv).expanduser().resolve()
    for path in (traces, metrics):
        if not path.is_file():
            raise FileNotFoundError(path)
    code = (
        "import json,sys; "
        "from terra.biomechanics_figures import _render_comparison_figures_in_process; "
        "print(json.dumps(_render_comparison_figures_in_process(sys.argv[1],sys.argv[2],output_dir=sys.argv[3])))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(traces), str(metrics), str(Path(output_dir).expanduser().resolve())],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"isolated biomechanics renderer failed: {result.stderr[-4000:]}")
    return json.loads(result.stdout)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="terra biomechanics plot",
        description="Render manuscript-ready EMG and GRF figures from a completed comparison.",
    )
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="default: COMPARISON_DIR/figures")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    comparison = args.comparison_dir.expanduser().resolve()
    report = render_comparison_figures(
        comparison / "comparison_traces.npz",
        comparison / "metrics.csv",
        output_dir=args.output_dir or comparison / "figures",
    )
    summary_path = comparison / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        artifacts = summary.setdefault("artifacts", {})
        if not isinstance(artifacts, dict):
            raise ValueError(f"comparison summary artifacts must be an object: {summary_path}")
        summary["figures"] = report["figures"]
        artifacts["figures"] = report["manifest"]
        _write_json(summary_path, summary)
        report["comparison_summary"] = str(summary_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "render_comparison_figures"]
