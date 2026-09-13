"""
Shared helpers for turning recorded frames into episode tensors.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


def compute_point_flows(coords: torch.Tensor) -> torch.Tensor:
    flows = torch.zeros_like(coords)
    if coords.shape[0] > 1:
        flows[1:] = coords[1:] - coords[:-1]
    return flows


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def write_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def mask_ids(mask_info_path: Path, label: str) -> List[str]:
    with mask_info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    return [str(mask_id) for mask_id, name in info.items() if str(name).lower() == label.lower()]


def read_mask(mask_root: Path, view_idx: int, mask_ids: Sequence[str], frame_idx: int) -> np.ndarray:
    mask = None
    for mask_id in mask_ids:
        path = mask_root / str(view_idx) / str(mask_id) / f"{frame_idx}.png"
        if not path.exists():
            continue
        current = np.asarray(Image.open(path)) > 0
        mask = current if mask is None else (mask | current)
    if mask is None:
        raise FileNotFoundError(
            f"No mask for view={view_idx}, frame={frame_idx}, mask_ids={list(mask_ids)} under {mask_root}"
        )
    return mask


def read_rgb(color_root: Path, view_idx: int, frame_idx: int) -> np.ndarray:
    return np.asarray(Image.open(color_root / str(view_idx) / f"{frame_idx}.png").convert("RGB"))


def flip_z_points(points: np.ndarray) -> np.ndarray:
    flipped = np.asarray(points, dtype=np.float32).copy()
    flipped[..., 2] *= -1.0
    return flipped


def flip_z_extrinsics(camera_to_world: np.ndarray) -> np.ndarray:
    flipped = np.asarray(camera_to_world, dtype=np.float32).copy()
    flipped[..., 2, :] *= -1.0
    return flipped


def depth_to_world_points(
    depth_mm: np.ndarray,
    object_mask: np.ndarray,
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
    max_points: int,
    frame_idx: int,
    view_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = object_mask & (depth_mm > 0) & (depth_mm < 60000)
    ys, xs = np.nonzero(valid)
    if ys.size == 0:
        return (
            np.zeros((max_points, 3), dtype=np.float32),
            np.zeros((max_points, 2), dtype=np.int32),
            np.zeros((max_points,), dtype=bool),
        )

    if ys.size > max_points:
        rng = np.random.default_rng(seed=(frame_idx + 1) * 1009 + (view_idx + 1) * 9176)
        chosen = rng.choice(ys.size, size=max_points, replace=False)
        ys = ys[chosen]
        xs = xs[chosen]

    z = depth_mm[ys, xs].astype(np.float32) / 1000.0
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    camera_points = np.stack([x, y, z, np.ones_like(z)], axis=1)
    world_points = (camera_to_world @ camera_points.T).T[:, :3].astype(np.float32)

    count = world_points.shape[0]
    padded_points = np.zeros((max_points, 3), dtype=np.float32)
    padded_pixels = np.zeros((max_points, 2), dtype=np.int32)
    padded_valid = np.zeros((max_points,), dtype=bool)
    padded_points[:count] = world_points
    padded_pixels[:count] = np.stack([xs, ys], axis=1).astype(np.int32)
    padded_valid[:count] = True
    return padded_points, padded_pixels, padded_valid


def visible_indices(visibilities: np.ndarray) -> np.ndarray:
    max_visible = int(visibilities.sum(axis=1).max(initial=0))
    indices = np.full((visibilities.shape[0], max_visible), -1, dtype=np.int64)
    for frame_idx, visible in enumerate(visibilities):
        current = np.nonzero(visible)[0].astype(np.int64)
        indices[frame_idx, : current.size] = current
    return indices


def nearest_particle_ids(
    observed_points: np.ndarray,
    observed_valid: np.ndarray,
    particle_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    view_count, frame_count, max_points, _ = observed_points.shape
    particle_ids = np.full((view_count, frame_count, max_points), -1, dtype=np.int64)
    particle_distances = np.full((view_count, frame_count, max_points), np.inf, dtype=np.float32)

    for frame_idx in range(frame_count):
        tree = cKDTree(particle_points[frame_idx])
        for view_idx in range(view_count):
            valid = observed_valid[view_idx, frame_idx]
            if not valid.any():
                continue
            distances, indices = tree.query(observed_points[view_idx, frame_idx, valid], k=1)
            particle_ids[view_idx, frame_idx, valid] = indices.astype(np.int64)
            particle_distances[view_idx, frame_idx, valid] = distances.astype(np.float32)
    return particle_ids, particle_distances


def knn_warp_cache(
    source_points: np.ndarray,
    query_points: np.ndarray,
    k: int,
    power: float,
    eps: float = 1.0e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    source_points = np.asarray(source_points, dtype=np.float32)
    query_points = np.asarray(query_points, dtype=np.float32)
    k = min(max(int(k), 1), int(source_points.shape[0]))
    distances, indices = cKDTree(source_points).query(query_points, k=k)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    zero = distances <= eps
    weights = 1.0 / np.maximum(distances, eps) ** float(power)
    if zero.any():
        weights[...] = np.where(zero, 1.0, weights)
        has_exact = zero.any(axis=1, keepdims=True)
        weights = np.where(has_exact, zero.astype(np.float32), weights)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), eps)
    return indices.astype(np.int64), weights.astype(np.float32)


def warp_query_points(
    source_points: np.ndarray,
    target_points: np.ndarray,
    query_points: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_weights: np.ndarray,
) -> np.ndarray:
    displacement = target_points.astype(np.float32) - source_points.astype(np.float32)
    query_displacement = (
        displacement[neighbor_indices] * neighbor_weights[..., None]
    ).sum(axis=1)
    return (query_points.astype(np.float32) + query_displacement).astype(np.float32)
