"""Render a deterministic validation panel from a saved policy checkpoint."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from loco_mujoco.core.stateful_object import StatefulObject
from loco_mujoco.task_factories import TaskFactory
from terra.rl import register_components
from terra.rl.backend import install_backend_integrations
from terra.rl.hooks import TerraValidationVideoRecorder


def _parse_motion(value: str) -> dict[str, str]:
    name, separator, path = value.partition("=")
    name = name.strip()
    path = path.strip()
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("motion must use NAME=DATASET_PATH syntax")
    return {"name": name, "path": path}


def _load_panel(path: str | Path) -> tuple[list[dict[str, str]], int, str]:
    panel_path = Path(path).expanduser()
    try:
        document = OmegaConf.to_container(OmegaConf.load(panel_path), resolve=True)
    except Exception as error:
        raise ValueError(f"invalid validation panel spec {panel_path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("validation panel spec must be a mapping")
    unknown = sorted(set(document) - {"schema_version", "name", "description", "length", "motions"})
    if unknown:
        raise ValueError(f"validation panel spec has unknown field(s): {', '.join(unknown)}")
    if document.get("schema_version") != 1:
        raise ValueError("validation panel spec schema_version must be 1")
    name = str(document.get("name", "")).strip()
    if not name:
        raise ValueError("validation panel spec requires a non-empty name")
    length = document.get("length", 616)
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("validation panel spec length must be a positive integer")
    raw_motions = document.get("motions")
    if not isinstance(raw_motions, list) or not raw_motions:
        raise ValueError("validation panel spec motions must be a non-empty list")
    # Reuse the recorder's normalization for required keys and duplicate names.
    normalized = TerraValidationVideoRecorder._normalize_named_motions(raw_motions)
    return [{"name": motion_name, "path": motion_path} for motion_name, motion_path in normalized], length, name


def _checkpoint_timestep(metadata) -> int:
    if isinstance(metadata, dict):
        value = metadata.get("global_timestep", 0)
    else:
        value = getattr(metadata, "global_timestep", 0)
    return max(0, int(value or 0))


def _upgrade_legacy_observation_config(config) -> bool:
    """Restore the implicit future-reference contract of pre-guard checkpoints."""
    goal_params = config.experiment.env_params.get("goal_params", {})
    if "enable_future_reference_observations" in goal_params:
        return False
    if not {"future_reference_stride", "future_reference_horizon"}.issubset(goal_params):
        return False
    goal_params["enable_future_reference_observations"] = True
    print(
        "[ValidationVideo] Upgraded legacy checkpoint observation config: "
        "future reference height/ankle cues are enabled."
    )
    return True


def _checkpoint_observation_dimension(raw_agent_state) -> int | None:
    """Read the policy input width from PPO running-normalization statistics."""
    state = getattr(raw_agent_state, "train_state", raw_agent_state)
    if isinstance(state, Mapping):
        run_stats = state.get("run_stats", {})
    else:
        run_stats = getattr(state, "run_stats", {})

    dimensions = set()

    def visit(value, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key))
            return
        shape = getattr(value, "shape", None)
        if key in {"mean", "var"} and shape is not None and len(shape) == 1 and int(shape[0]) > 1:
            dimensions.add(int(shape[0]))

    visit(run_stats)
    if len(dimensions) > 1:
        raise ValueError(f"checkpoint contains inconsistent observation statistic widths: {sorted(dimensions)}")
    return next(iter(dimensions), None)


def _initialize_checkpoint_policy(checkpoint: str, recorder: TerraValidationVideoRecorder):
    """Restore an inference policy using the same CPU environment as rendering."""
    from musclemimic.runner.engine import pick_algorithm
    from musclemimic.runner.eval_utils import align_agent_state, load_checkpoint

    config, raw_agent_state, metadata = load_checkpoint(checkpoint)
    OmegaConf.set_struct(config, False)
    _upgrade_legacy_observation_config(config)
    algorithm_cls = pick_algorithm(config)
    register_components()
    install_backend_integrations(str(config.experiment.get("algorithm", algorithm_cls.__name__)))

    config_holder = SimpleNamespace(config=config)
    first_motion = recorder.named_motions[0][1] if recorder.named_motions else None
    env_params = recorder._build_env_params(config_holder, "validation_render_bootstrap")
    task_params = recorder._build_task_params(config_holder, first_motion)
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

    saved_instances = StatefulObject._instances.copy()
    StatefulObject._instances.clear()
    env = None
    try:
        env = factory.make(**env_params, **task_params)
        expected_observation_dim = _checkpoint_observation_dimension(raw_agent_state)
        actual_observation_dim = int(env.info.observation_space.shape[0])
        if expected_observation_dim is not None and actual_observation_dim != expected_observation_dim:
            raise RuntimeError(
                "validation environment observation width does not match checkpoint: "
                f"environment={actual_observation_dim}, checkpoint={expected_observation_dim}"
            )
        agent_conf = algorithm_cls.init_agent_conf(env, config)
        agent_state = align_agent_state(raw_agent_state, agent_conf)
    finally:
        if env is not None:
            env.stop()
        StatefulObject._instances = saved_instances
    return config, metadata, algorithm_cls, agent_conf, agent_state


def render_checkpoint(
    checkpoint: str,
    video_dir: str,
    motions: Sequence[dict[str, str]],
    *,
    length: int = 616,
    timestep: int | None = None,
) -> tuple[dict[str, str], int, object]:
    """Render named motions and return their video paths and checkpoint step."""
    recorder = TerraValidationVideoRecorder(
        video_dir=video_dir,
        frequency=1,
        length=length,
        deterministic=True,
        named_motions=motions,
    )
    config, metadata, algorithm_cls, agent_conf, agent_state = _initialize_checkpoint_policy(checkpoint, recorder)
    checkpoint_step = _checkpoint_timestep(metadata) if timestep is None else int(timestep)
    paths = recorder.record_episode(
        agent_conf=agent_conf,
        agent_state=agent_state,
        validation_number=1,
        timestep=checkpoint_step,
        algorithm_cls=algorithm_cls,
    )
    if not isinstance(paths, dict) or not paths:
        raise RuntimeError("no validation videos were produced")
    audit_path = Path(video_dir) / "initialization_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "checkpoint": checkpoint,
                "global_timestep": checkpoint_step,
                "motions": recorder.initialization_audits,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[ValidationVideo] Initialization audit report: {audit_path}")
    return paths, checkpoint_step, config


def _log_to_wandb(paths: dict[str, str], checkpoint: str, timestep: int, config, run_name: str) -> None:
    import wandb

    project = str(config.get("wandb", {}).get("project", "terra"))
    run = wandb.init(
        project=project,
        name=run_name,
        job_type="validation-render",
        tags=["terra", "validation-render", "checkpoint-evaluation"],
        config={"checkpoint": checkpoint, "global_timestep": timestep},
    )
    try:
        run.log(
            {
                f"Validation/Video/{name}": wandb.Video(path, format="mp4")
                for name, path in paths.items()
            },
            step=timestep,
        )
    finally:
        run.finish()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Complete PPO checkpoint directory")
    parser.add_argument("--video-dir", required=True, help="Directory in which MP4 files are written")
    motion_source = parser.add_mutually_exclusive_group(required=True)
    motion_source.add_argument(
        "--motion",
        action="append",
        type=_parse_motion,
        metavar="NAME=DATASET_PATH",
        help="Pinned validation motion; repeat for a multi-video panel",
    )
    motion_source.add_argument("--panel", help="Versioned validation panel YAML")
    parser.add_argument("--length", type=int, default=None, help="Override maximum rollout steps per video")
    parser.add_argument("--wandb", action="store_true", help="Log the completed panel to a separate W&B run")
    parser.add_argument("--wandb-run-name", default=None, help="Optional W&B run name")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.panel:
        try:
            motions, panel_length, panel_name = _load_panel(args.panel)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    else:
        motions = args.motion
        panel_length = 616
        panel_name = "custom"
    length = args.length if args.length is not None else panel_length
    if length <= 0:
        raise SystemExit("--length must be positive")
    checkpoint = str(Path(args.checkpoint).expanduser())
    video_dir = str(Path(args.video_dir).expanduser())
    paths, timestep, config = render_checkpoint(
        checkpoint,
        video_dir,
        motions,
        length=length,
    )
    print(json.dumps({"checkpoint": checkpoint, "global_timestep": timestep, "videos": paths}, indent=2))
    if args.wandb:
        default_name = f"{Path(checkpoint).parent.name}-{Path(checkpoint).name}-{panel_name}-renders"
        _log_to_wandb(paths, checkpoint, timestep, config, args.wandb_run_name or default_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
