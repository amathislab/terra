"""Adapters from official PRISM takes to TERRA terrain inputs."""

from __future__ import annotations

import pickle
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

# First 24 joints returned by the standard SMPL model. Names follow TERRA's convention.
SMPL_TERRA_JOINTS = (
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
)


class _LegacyCh:
    """Array-only stand-in for the one legacy ``chumpy.Ch`` value in SMPL pickles."""

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        return np.array(self.x, dtype=dtype, copy=copy)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.x, name)

    def __getitem__(self, item: Any) -> Any:
        return self.x[item]


def _create_smpl_model(smplx: Any, model_dir: Path, gender: str, device: str) -> Any:
    """Load legacy SMPL assets without making obsolete chumpy a runtime dependency."""
    try:
        return smplx.create(
            str(Path(model_dir).expanduser()),
            model_type="smpl",
            gender=gender,
            use_pca=False,
        ).to(device)
    except ModuleNotFoundError as exc:
        if exc.name != "chumpy":
            raise

    # The common SMPL .pkl stores shapedirs as a chumpy wrapper around an
    # ordinary ndarray. smplx only converts it back to ndarray while loading. Supplying
    # this narrowly scoped unpickling class avoids installing abandoned chumpy, which is
    # incompatible with current Python/NumPy, without changing the model values.
    ch_module = types.ModuleType("chumpy.ch")
    ch_module.Ch = _LegacyCh
    package = types.ModuleType("chumpy")
    package.ch = ch_module
    previous_package = sys.modules.get("chumpy")
    previous_module = sys.modules.get("chumpy.ch")
    sys.modules["chumpy"] = package
    sys.modules["chumpy.ch"] = ch_module
    try:
        return smplx.create(
            str(Path(model_dir).expanduser()),
            model_type="smpl",
            gender=gender,
            use_pca=False,
        ).to(device)
    finally:
        if previous_package is None:
            sys.modules.pop("chumpy", None)
        else:
            sys.modules["chumpy"] = previous_package
        if previous_module is None:
            sys.modules.pop("chumpy.ch", None)
        else:
            sys.modules["chumpy.ch"] = previous_module


def load_take(path: Path) -> dict[str, Any]:
    """Load an official PRISM pickle.

    Pickle permits code execution. The caller is responsible for using only files
    obtained from the official PRISM release.
    """
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def observed_support_points(take: Mapping[str, Any], *, stride: int = 1) -> np.ndarray:
    """Return world-frame CoP samples for frames with measured insole activity."""
    if stride < 1:
        raise ValueError("stride must be positive")
    points = []
    for side in ("L_Foot", "R_Foot"):
        foot = take["insole"][side]
        contacts = np.asarray(foot["contacts"], dtype=bool)
        active = np.any(contacts, axis=1)
        cop = np.asarray(foot["CoP_world"], dtype=float)
        if cop.shape != (len(active), 3):
            raise ValueError(f"{side} CoP_world must have shape ({len(active)}, 3), got {cop.shape}")
        valid = active & np.all(np.isfinite(cop), axis=1)
        points.append(cop[valid][::stride])
    return np.concatenate(points, axis=0) if points else np.empty((0, 3))


def observed_support_xy(take: Mapping[str, Any], *, stride: int = 1) -> np.ndarray:
    """Return world-frame CoP XY samples for frames with measured insole activity."""
    return observed_support_points(take, stride=stride)[:, :2]


def smpl_world_joints(
    take: Mapping[str, Any],
    model_dir: Path,
    *,
    device: str = "cpu",
    batch_frames: int = 512,
) -> np.ndarray:
    """Evaluate PRISM's optical-MoCap SMPL labels in the dataset world frame.

    ``model_dir`` must contain the SMPL model files. The object meshes are
    deliberately not read here, preserving reconstruction/evaluation separation.
    """
    import smplx
    import torch

    smpl = take["smpl_params"]
    poses = np.asarray(smpl["poses"], dtype=np.float32)
    betas = np.asarray(smpl["betas"], dtype=np.float32)
    trans = np.asarray(smpl["trans"], dtype=np.float32)
    root_offset = np.asarray(smpl["root_offset"], dtype=np.float32).reshape(1, 3)
    if poses.ndim != 2 or poses.shape[1] != 72:
        raise ValueError(f"SMPL poses must have shape (F, 72), got {poses.shape}")
    if trans.shape != (len(poses), 3):
        raise ValueError(f"SMPL trans must have shape ({len(poses)}, 3), got {trans.shape}")
    if betas.ndim == 1:
        betas = np.broadcast_to(betas, (len(poses), len(betas)))
    elif betas.shape[0] == 1:
        betas = np.broadcast_to(betas, (len(poses), betas.shape[1]))
    if betas.shape != (len(poses), 10):
        raise ValueError(f"SMPL betas must have shape (F, 10), got {betas.shape}")
    if batch_frames < 1:
        raise ValueError("batch_frames must be positive")

    gender = str(smpl["gender"]).lower()
    model = _create_smpl_model(smplx, model_dir, gender, device)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(poses), batch_frames):
            end = min(start + batch_frames, len(poses))
            as_tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)  # noqa: E731
            output = model(
                global_orient=as_tensor(poses[start:end, :3]),
                body_pose=as_tensor(poses[start:end, 3:]),
                betas=as_tensor(betas[start:end]),
                transl=as_tensor(trans[start:end] + root_offset),
                return_verts=False,
            )
            chunks.append(output.joints[:, : len(SMPL_TERRA_JOINTS)].detach().cpu().numpy())
    joints = np.concatenate(chunks, axis=0)
    if joints.shape != (len(poses), len(SMPL_TERRA_JOINTS), 3):
        raise RuntimeError(f"unexpected SMPL joint shape {joints.shape}")
    return joints


def cached_smpl_world_joints(
    take: Mapping[str, Any],
    model_dir: Path,
    cache_path: Path,
    *,
    device: str = "cpu",
    batch_frames: int = 512,
) -> np.ndarray:
    """Load cached joints or evaluate and atomically cache them."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            joints = np.asarray(cached["joints"])
        if joints.ndim != 3 or joints.shape[1:] != (len(SMPL_TERRA_JOINTS), 3):
            raise ValueError(f"invalid joint cache at {cache_path}: {joints.shape}")
        return joints

    joints = smpl_world_joints(take, model_dir, device=device, batch_frames=batch_frames)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, joints=joints)
    temporary.replace(cache_path)
    return joints


def selected_take_paths(data_root: Path, take_ids: Sequence[str] = ()) -> list[Path]:
    """Select subject/take paths, accepting IDs as ``subj001/take002``."""
    from .root import resolve_data_root

    root = resolve_data_root(Path(data_root))
    if not take_ids:
        return sorted(root.glob("subj*/take*.pkl"))
    paths = []
    for take_id in take_ids:
        subject, separator, take = take_id.replace("_", "/", 1).partition("/")
        if not separator:
            raise ValueError(f"take ID must be subjNNN/takeNNN, got {take_id!r}")
        path = root / subject / f"{take.removesuffix('.pkl')}.pkl"
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    return paths
