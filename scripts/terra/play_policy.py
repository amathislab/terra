#!/usr/bin/env python3
"""Watch a trained PPO policy in native MuJoCo, or save its attempts as MP4 files."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def selected_motions(record: dict, split: str, motions: list[str] | None) -> list[str]:
    """Select cache identifiers from the materialized dataset."""
    available = [row["motion"] for row in record["motions"] if row.get("split", "train") == split]
    selected = available if motions is None else motions
    if not selected:
        raise ValueError(f"the dataset has no {split} motions")
    if len(selected) != len(set(selected)):
        raise ValueError("motion names must be unique")
    missing = set(selected) - set(available)
    if missing:
        raise ValueError(f"motion is absent from the {split} split: {sorted(missing)[0]}")
    return selected


def playback_config(config, record: dict, motion: str, video_dir: Path | None):
    """Keep checkpoint observations and network settings; select one paired motion."""
    from omegaconf import OmegaConf

    from terra.visualization.scene import REFERENCE_RGBA

    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    exp = config.experiment
    if exp.algorithm != "PPOJax":
        raise ValueError("this script supports TERRA PPO checkpoints")
    params = exp.env_params
    # Match validation resets and perturbations without changing policy inputs.
    validation = exp.get("validation", {})
    params.update(validation.get("env_params", {}))
    for name in ("init_state_type", "init_state_params", "terminal_state_type", "terminal_state_params"):
        if name in validation:
            params[name] = validation[name]
    params.env_name = str(params.env_name).removeprefix("Mjx")
    params.headless = video_dir is not None
    for key in list(params):
        if key.startswith("mjx_") or key in {"num_envs", "nconmax", "njmax"}:
            del params[key]
    params.th_params = {**params.get("th_params", {}), "random_start": False, "fixed_start_conf": [0, 0]}
    goal = str(params.goal_type)
    visual_goals = {"TerraGoal": "TerraGoalVisual", "TerraFullBodyTrackingGoal": "TerraFullBodyTrackingGoalVisual"}
    if video_dir is not None:
        params.goal_type = visual_goals.get(goal, goal)
        params.goal_params.visualize_goal = True
        params.goal_params.enable_enhanced_visualization = True
        params.goal_params.target_geom_rgba = list(REFERENCE_RGBA)
        params.viewer_size = [640, 480]
        params.default_camera_mode = "follow"
        params.camera_params = {"follow": {"azimuth": 135.0, "elevation": -15.0, "distance": 4.0}}
        params.recorder_params = {
            "path": str(video_dir),
            "tag": motion,
            "video_name": "policy",
            "fps": round(1 / (float(params.timestep) * int(params.n_substeps))),
            "compress": True,
        }
    factory = exp.task_factory.params
    factory.frame_zero_reset_probability = 1.0
    factory.first_quarter_reset_probability = 0.0
    factory.trajectory_cache_root = None
    factory.trajectory_cache_key = None
    factory.amass_dataset_conf.update(
        {
            "rel_dataset_path": [motion],
            "dataset_group": None,
            "cache_root": record["destination_cache"],
            "retargeting_method": record.get("retargeting_method", "terra"),
            "output_cache_subdir": record.get("retargeting_method", "terra"),
            "load_paired_terrain": True,
            "require_nonflat_terrain": record.get("terrain_mode", "nonflat") == "nonflat",
            "allow_cache_download": False,
        }
    )
    return config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="checkpoint_N directory, or its parent (latest)")
    parser.add_argument(
        "--materialization-record", type=Path, required=True, help="dataset record from train materialize"
    )
    parser.add_argument(
        "--motion", action="append", help="cache identifier; repeatable; default: all motions in the split"
    )
    parser.add_argument("--split", choices=("train", "evaluation"), default="train")
    parser.add_argument("--video-dir", type=Path, help="save MP4 files here instead of opening the native MuJoCo GUI")
    parser.add_argument("--steps", type=int, default=1000, help="maximum control steps per motion")
    parser.add_argument(
        "--repeat", action="store_true", help="retry after episode termination when recording, until --steps"
    )
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of using policy means")
    parser.add_argument(
        "--train-state-seed", type=int, default=0, help="seed index for checkpoints trained with multiple seeds"
    )
    args = parser.parse_args(argv)
    if args.steps < 1 or args.train_state_seed < 0:
        parser.error("--steps must be positive and --train-state-seed must be non-negative")
    try:
        record = json.loads(args.materialization_record.expanduser().read_text())
        motions = selected_motions(record, args.split, args.motion)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    # One software-rendered motion needs a small thread pool.
    for name in ("LP_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if args.video_dir is not None:
        os.environ.setdefault("MUJOCO_GL", "osmesa")
    else:
        os.environ.setdefault("MUJOCO_GL", "glfw")
    # Import simulation and JAX only after choosing the rendering backend.
    import numpy as np
    from omegaconf import OmegaConf

    from musclemimic.algorithms.ppo import PPOJax
    from musclemimic.algorithms.ppo.inference import play_policy
    from musclemimic.runner.eval_utils import load_checkpoint, run_with_mujoco_viewer
    from terra.artifacts import validate_retarget_artifacts
    from terra.rl import register_components
    from terra.rl.task_factory import TerraImitationFactory
    from terra.rl.trajectory import install_trajectory_stability

    register_components()
    install_trajectory_stability()
    config, saved_state, _metadata = load_checkpoint(str(args.checkpoint.expanduser().resolve()))
    if not 0 <= args.train_state_seed < int(config.experiment.n_seeds):
        parser.error("--train-state-seed is outside the checkpoint's seed range")
    video_dir = None if args.video_dir is None else args.video_dir.expanduser().resolve()
    for motion in motions:
        validate_retarget_artifacts(
            record["destination_cache"], motion, method=record.get("retargeting_method", "terra")
        )
        np.random.seed(0)
        run_config = playback_config(config, record, motion, video_dir)
        exp = run_config.experiment
        env = TerraImitationFactory.make(
            **OmegaConf.to_container(exp.env_params, resolve=True),
            **OmegaConf.to_container(exp.task_factory.params, resolve=True),
        )
        try:
            agent_conf = PPOJax.init_agent_conf(env, run_config)
            agent_state = PPOJax.restore_agent_state(saved_state.train_state, agent_conf)
            print(f"Playing {motion} on its paired terrain", flush=True)
            if video_dir is None:
                existing_threads = set(threading.enumerate())
                run_with_mujoco_viewer(
                    env,
                    agent_conf,
                    agent_state,
                    n_steps=args.steps,
                    deterministic=not args.stochastic,
                    train_state_seed=args.train_state_seed,
                )
                # MuJoCo closes its passive window on a daemon thread. Wait for
                # that thread before Python tears down GLFW at process exit.
                for thread in threading.enumerate():
                    if thread not in existing_threads and thread.name.endswith("(_launch_internal)"):
                        thread.join()
            else:
                play_policy(
                    env,
                    agent_conf,
                    agent_state,
                    n_envs=1,
                    n_steps=args.steps,
                    render=True,
                    record=True,
                    deterministic=not args.stochastic,
                    use_mujoco=True,
                    do_wrap_env=False,
                    train_state_seed=args.train_state_seed,
                    stop_after_first_episode=not args.repeat,
                )
                print("Video:", video_dir / motion / "policy.mp4", flush=True)
        finally:
            env.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
