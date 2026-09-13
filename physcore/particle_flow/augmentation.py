"""
Augment converted episodes with randomized-material MPM rollouts.

The generated episodes keep the real capture's initial particle coordinates,
controller/contact trajectories, camera calibration, and particle indexing.
Smooth material fields are sampled for each augmented rollout, the object is
simulated for the same number of frames, and noisy depth observations are
rendered back through the saved cameras.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from physcore.particle_flow.dataset import ParticleFlowEpisodeDataset
from physcore.particle_flow.rollout import BatchedDifferentiableRolloutEngine, DifferentiableRolloutEngine
from physcore.particle_flow.episode_runtime import (
    _catmull_rom_window,
    _hold_index_window,
    _load_episode_tensors,
    _real_world_chunk_dt,
    _run_locked_rollout,
    _should_use_contact_kinematic_targets,
    _temporary_rollout_timestep,
)


def _clone_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _write_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _compute_point_flows(coords: torch.Tensor) -> torch.Tensor:
    flows = torch.zeros_like(coords)
    if coords.shape[0] > 1:
        flows[1:] = coords[1:] - coords[:-1]
    return flows


def _shift_camera_extrinsics(camera_extrinsics: torch.Tensor, domain_shift: torch.Tensor) -> torch.Tensor:
    shifted = camera_extrinsics.detach().cpu().clone().float()
    shift = domain_shift.detach().cpu().to(dtype=shifted.dtype).reshape(3)
    if shifted.ndim == 2:
        shifted[:3, 3] = shifted[:3, 3] + shift
    elif shifted.ndim >= 3:
        shifted[..., :3, 3] = shifted[..., :3, 3] + shift.view(*([1] * (shifted.ndim - 2)), 3)
    return shifted


def _episode_config(root: Path) -> Dict:
    config_path = root / "config.yaml"
    if not config_path.exists():
        return {}
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else dict(cfg)


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + t * (b - a)


def _perlin_octave(points: np.ndarray, frequency: float, rng: np.random.Generator) -> np.ndarray:
    scaled = np.asarray(points, dtype=np.float32) * float(frequency)
    base = np.floor(scaled).astype(np.int64)
    frac = (scaled - base).astype(np.float32)
    grid_size = max(int(np.ceil(float(frequency))) + 3, 4)
    gradients = rng.normal(size=(grid_size, grid_size, grid_size, 3)).astype(np.float32)
    gradients /= np.linalg.norm(gradients, axis=-1, keepdims=True).clip(min=1.0e-8)

    def corner(ix: int, iy: int, iz: int) -> np.ndarray:
        indices = (base + np.array([ix, iy, iz], dtype=np.int64)) % grid_size
        grad = gradients[indices[:, 0], indices[:, 1], indices[:, 2]]
        disp = frac - np.array([ix, iy, iz], dtype=np.float32)
        return np.sum(grad * disp, axis=-1)

    n000 = corner(0, 0, 0)
    n100 = corner(1, 0, 0)
    n010 = corner(0, 1, 0)
    n110 = corner(1, 1, 0)
    n001 = corner(0, 0, 1)
    n101 = corner(1, 0, 1)
    n011 = corner(0, 1, 1)
    n111 = corner(1, 1, 1)

    u = _fade(frac[:, 0])
    v = _fade(frac[:, 1])
    w = _fade(frac[:, 2])
    x00 = _lerp(n000, n100, u)
    x10 = _lerp(n010, n110, u)
    x01 = _lerp(n001, n101, u)
    x11 = _lerp(n011, n111, u)
    y0 = _lerp(x00, x10, v)
    y1 = _lerp(x01, x11, v)
    return _lerp(y0, y1, w).astype(np.float32)


def _smooth_noise_field(
    canonical_positions: torch.Tensor,
    args: SimpleNamespace,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict]:
    points = canonical_positions.detach().cpu().numpy().astype(np.float32)
    finite = np.isfinite(points).all(axis=-1)
    if not finite.any():
        points = np.zeros_like(points, dtype=np.float32)
        finite = np.ones(points.shape[0], dtype=bool)
    lo = np.percentile(points[finite], 1.0, axis=0).astype(np.float32)
    hi = np.percentile(points[finite], 99.0, axis=0).astype(np.float32)
    extent = np.maximum(hi - lo, 1.0e-6)
    normalized = np.clip((points - lo[None, :]) / extent[None, :], 0.0, 1.0)

    frequency = max(float(args.material_noise_frequency), 1.0e-4)
    octaves = max(int(args.material_noise_octaves), 1)
    persistence = float(args.material_noise_persistence)
    lacunarity = max(float(args.material_noise_lacunarity), 1.0)
    total = np.zeros(points.shape[0], dtype=np.float32)
    amplitude = 1.0
    amplitude_sum = 0.0
    for octave_idx in range(octaves):
        octave_frequency = frequency * (lacunarity ** octave_idx)
        total += float(amplitude) * _perlin_octave(normalized, octave_frequency, rng)
        amplitude_sum += float(amplitude)
        amplitude *= persistence
    total /= max(amplitude_sum, 1.0e-8)

    q_lo = float(np.percentile(total[finite], 2.0))
    q_hi = float(np.percentile(total[finite], 98.0))
    if abs(q_hi - q_lo) <= 1.0e-8:
        field = np.full(points.shape[0], 0.5, dtype=np.float32)
    else:
        field = np.clip((total - q_lo) / (q_hi - q_lo), 0.0, 1.0).astype(np.float32)
    return field, {
        "frequency": float(frequency),
        "octaves": int(octaves),
        "persistence": float(persistence),
        "lacunarity": float(lacunarity),
        "bbox_min": lo.astype(float).tolist(),
        "bbox_max": hi.astype(float).tolist(),
        "quantile_min": q_lo,
        "quantile_max": q_hi,
    }


def _log_e_to_particle_colors(log_E: torch.Tensor, log_e_range: Sequence[float]) -> torch.Tensor:
    lo, hi = float(log_e_range[0]), float(log_e_range[1])
    denom = max(hi - lo, 1.0e-8)
    t = ((log_E.float() - lo) / denom).clamp(0.0, 1.0)
    stops = torch.tensor(
        [
            [49.0, 54.0, 149.0],
            [69.0, 123.0, 157.0],
            [102.0, 168.0, 122.0],
            [239.0, 196.0, 95.0],
            [204.0, 71.0, 78.0],
        ],
        dtype=torch.float32,
        device=t.device,
    )
    scaled = t * float(stops.shape[0] - 1)
    idx0 = torch.floor(scaled).long().clamp(0, stops.shape[0] - 1)
    idx1 = (idx0 + 1).clamp(0, stops.shape[0] - 1)
    frac = (scaled - idx0.float()).unsqueeze(-1)
    colors = stops[idx0] * (1.0 - frac) + stops[idx1] * frac
    return colors.round().to(dtype=torch.uint8).cpu()


def _camera_for_view(cameras: torch.Tensor, view_idx: int, local_view_idx: int) -> np.ndarray:
    array = cameras.detach().cpu().numpy().astype(np.float32)
    if array.ndim == 2:
        return array
    selected = view_idx if 0 <= int(view_idx) < array.shape[0] else local_view_idx
    if selected < 0 or selected >= array.shape[0]:
        raise IndexError(f"Camera view {view_idx} is out of range for camera array shape {array.shape}")
    return array[selected]


def _image_size_from_episode(data: Dict, cfg: Dict) -> Tuple[int, int]:
    depth_maps = data.get("depth_maps", None)
    if torch.is_tensor(depth_maps) and depth_maps.ndim >= 3:
        return int(depth_maps.shape[-1]), int(depth_maps.shape[-2])
    rgb_images = data.get("rgb_images", None)
    if torch.is_tensor(rgb_images) and rgb_images.ndim >= 4:
        return int(rgb_images.shape[-2]), int(rgb_images.shape[-3])
    image_size = (
        cfg.get("observation", {}).get("image_size_wh", None)
        if isinstance(cfg.get("observation", {}), dict)
        else None
    )
    if image_size is None:
        raise ValueError("Cannot infer image size; missing depth_maps/rgb_images/config observation.image_size_wh")
    return int(image_size[0]), int(image_size[1])


def _render_particle_depth_frame(
    points_world: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    width: int,
    height: int,
    particle_ids: np.ndarray,
    splat_radius: int,
    near: float,
    far: float,
) -> Tuple[np.ndarray, np.ndarray]:
    world_to_camera = np.linalg.inv(camera_to_world).astype(np.float32)
    points_h = np.concatenate(
        [points_world.astype(np.float32), np.ones((points_world.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    camera_points = (world_to_camera @ points_h.T).T[:, :3]
    z = camera_points[:, 2]
    valid = np.isfinite(camera_points).all(axis=1) & (z > float(near)) & (z < float(far))
    if not valid.any():
        return (
            np.full((height, width), np.nan, dtype=np.float32),
            np.full((height, width), -1, dtype=np.int64),
        )

    camera_points = camera_points[valid]
    z = z[valid]
    ids = particle_ids[valid]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    u0 = np.rint(camera_points[:, 0] * fx / z + cx).astype(np.int64)
    v0 = np.rint(camera_points[:, 1] * fy / z + cy).astype(np.int64)

    offsets = [(0, 0)]
    radius = max(int(splat_radius), 0)
    if radius > 0:
        offsets = [
            (dx, dy)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            if dx * dx + dy * dy <= radius * radius
        ]

    all_u, all_v, all_z, all_ids = [], [], [], []
    for dx, dy in offsets:
        all_u.append(u0 + dx)
        all_v.append(v0 + dy)
        all_z.append(z)
        all_ids.append(ids)
    u = np.concatenate(all_u)
    v = np.concatenate(all_v)
    z_rep = np.concatenate(all_z).astype(np.float32)
    ids_rep = np.concatenate(all_ids).astype(np.int64)
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not in_bounds.any():
        return (
            np.full((height, width), np.nan, dtype=np.float32),
            np.full((height, width), -1, dtype=np.int64),
        )

    u = u[in_bounds]
    v = v[in_bounds]
    z_rep = z_rep[in_bounds]
    ids_rep = ids_rep[in_bounds]
    flat = v * width + u
    depth_flat = np.full((height * width,), np.nan, dtype=np.float32)
    id_flat = np.full((height * width,), -1, dtype=np.int64)

    order = np.argsort(z_rep, kind="stable")
    flat_sorted = flat[order]
    _, first_positions = np.unique(flat_sorted, return_index=True)
    chosen = order[first_positions]
    chosen_flat = flat[chosen]
    depth_flat[chosen_flat] = z_rep[chosen]
    id_flat[chosen_flat] = ids_rep[chosen]
    return depth_flat.reshape(height, width), id_flat.reshape(height, width)


def _backproject_pixels(
    xs: np.ndarray,
    ys: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    z = depth.astype(np.float32)
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    camera_points = np.stack([x, y, z, np.ones_like(z)], axis=1)
    return (camera_to_world @ camera_points.T).T[:, :3].astype(np.float32)


def _visible_indices_from_observation_ids(object_particle_ids: np.ndarray) -> np.ndarray:
    view_count, frame_count, _ = object_particle_ids.shape
    counts = [int((object_particle_ids[v, t] >= 0).sum()) for v in range(view_count) for t in range(frame_count)]
    max_visible = max(counts) if counts else 0
    visible = np.full((view_count, frame_count, max_visible), -1, dtype=np.int64)
    for view_idx in range(view_count):
        for frame_idx in range(frame_count):
            ids = np.unique(object_particle_ids[view_idx, frame_idx])
            ids = ids[ids >= 0]
            visible[view_idx, frame_idx, : ids.shape[0]] = ids
    return visible


def _ensure_visible_particle_indices(data: Dict) -> None:
    if "visible_particle_indices" in data:
        return
    tracked_visible = data.get("tracked_visible_particle_indices", None)
    if torch.is_tensor(tracked_visible):
        data["visible_particle_indices"] = tracked_visible.long()
        return
    observation_data = data.get("observation_data", None)
    if observation_data and torch.is_tensor(observation_data.get("object_particle_ids", None)):
        data["visible_particle_indices"] = torch.from_numpy(
            _visible_indices_from_observation_ids(
                observation_data["object_particle_ids"].detach().cpu().numpy().astype(np.int64)
            )
        ).long()
        return
    particle_coords = data.get("particle_coords", None)
    if torch.is_tensor(particle_coords):
        frame_count, num_particles = int(particle_coords.shape[0]), int(particle_coords.shape[1])
        data["visible_particle_indices"] = torch.arange(num_particles, dtype=torch.long).view(1, -1).expand(
            frame_count,
            -1,
        ).clone()


def _render_noisy_observations(
    coords: torch.Tensor,
    source_data: Dict,
    source_cfg: Dict,
    args: SimpleNamespace,
    rng: np.random.Generator,
) -> Dict:
    if "camera_intrinsics" not in source_data or "camera_extrinsics" not in source_data:
        raise ValueError("Source episode must contain camera_intrinsics and camera_extrinsics")
    intrinsics_all = source_data["camera_intrinsics"].float()
    extrinsics_all = source_data["camera_extrinsics"].float()

    observation_data = source_data.get("observation_data", {}) or {}
    if "view_indices" in observation_data:
        views = [int(v) for v in torch.as_tensor(observation_data["view_indices"]).reshape(-1).tolist()]
    else:
        primary_view = int(source_data.get("primary_view", 0))
        views = [primary_view]
    primary_view = int(source_data.get("primary_view", views[0]))
    if primary_view not in views:
        primary_view = views[0]
    primary_local_idx = views.index(primary_view)

    width, height = _image_size_from_episode(source_data, source_cfg)
    frame_count, num_particles, _ = coords.shape
    max_points = int(args.max_observed_points)
    if max_points <= 0:
        existing = observation_data.get("object_points_clean", None)
        max_points = int(existing.shape[-2]) if torch.is_tensor(existing) else num_particles

    render_count = int(num_particles)
    render_mode = str(args.render_particle_subset).lower()
    if render_mode == "tracked":
        render_count = int(source_data.get("tracked_particle_count", num_particles))
    elif render_mode == "shell":
        render_count = int(source_data.get("completed_shell_count", source_data.get("tracked_particle_count", num_particles)))
    elif render_mode != "all":
        raise ValueError(f"Unsupported --render-particle-subset {args.render_particle_subset!r}")
    render_count = max(0, min(render_count, num_particles))
    render_ids = np.arange(render_count, dtype=np.int64)
    coords_np = coords.detach().cpu().numpy().astype(np.float32)

    object_points_clean = np.zeros((len(views), frame_count, max_points, 3), dtype=np.float32)
    object_points_noisy = np.zeros_like(object_points_clean)
    object_pixels = np.zeros((len(views), frame_count, max_points, 2), dtype=np.int64)
    object_valid = np.zeros((len(views), frame_count, max_points), dtype=bool)
    object_particle_ids = np.full((len(views), frame_count, max_points), -1, dtype=np.int64)
    object_particle_distances = np.zeros((len(views), frame_count, max_points), dtype=np.float32)
    primary_depth = np.full((frame_count, height, width), np.nan, dtype=np.float32)
    primary_segmented_depth = np.full_like(primary_depth, np.nan)
    primary_object_masks = np.zeros((frame_count, height, width), dtype=np.uint8)

    for local_view_idx, view_idx in enumerate(views):
        intrinsic = _camera_for_view(intrinsics_all, view_idx, local_view_idx)
        camera_to_world = _camera_for_view(extrinsics_all, view_idx, local_view_idx)
        for frame_idx in range(frame_count):
            clean_depth, id_map = _render_particle_depth_frame(
                coords_np[frame_idx, :render_count],
                intrinsic,
                camera_to_world,
                width=width,
                height=height,
                particle_ids=render_ids,
                splat_radius=int(args.splat_radius),
                near=float(args.near_depth),
                far=float(args.far_depth),
            )
            valid_pixels = np.isfinite(clean_depth) & (id_map >= 0)
            if float(args.depth_dropout) > 0.0 and valid_pixels.any():
                keep = rng.random(size=valid_pixels.shape) >= float(args.depth_dropout)
                valid_pixels = valid_pixels & keep

            noisy_depth = clean_depth.copy()
            if valid_pixels.any():
                depth_values = clean_depth[valid_pixels]
                noise_std = float(args.depth_noise_std) + float(args.depth_noise_relative) * depth_values
                noisy_values = depth_values + rng.normal(0.0, noise_std).astype(np.float32)
                noisy_values = np.clip(noisy_values, float(args.near_depth), float(args.far_depth))
                noisy_depth[valid_pixels] = noisy_values.astype(np.float32)
                noisy_depth[~valid_pixels] = np.nan
            else:
                noisy_depth[:] = np.nan

            ys, xs = np.nonzero(valid_pixels)
            if ys.size > max_points:
                choice = rng.choice(ys.size, size=max_points, replace=False)
                choice.sort()
                ys = ys[choice]
                xs = xs[choice]
            count = int(ys.size)
            if count > 0:
                clean_z = clean_depth[ys, xs]
                noisy_z = noisy_depth[ys, xs]
                ids = id_map[ys, xs].astype(np.int64)
                object_points_clean[local_view_idx, frame_idx, :count] = _backproject_pixels(
                    xs, ys, clean_z, intrinsic, camera_to_world
                )
                object_points_noisy[local_view_idx, frame_idx, :count] = _backproject_pixels(
                    xs, ys, noisy_z, intrinsic, camera_to_world
                )
                object_pixels[local_view_idx, frame_idx, :count] = np.stack([xs, ys], axis=1)
                object_valid[local_view_idx, frame_idx, :count] = True
                object_particle_ids[local_view_idx, frame_idx, :count] = ids
                object_particle_distances[local_view_idx, frame_idx, :count] = np.linalg.norm(
                    object_points_clean[local_view_idx, frame_idx, :count] - coords_np[frame_idx, ids],
                    axis=-1,
                ).astype(np.float32)

            if local_view_idx == primary_local_idx:
                primary_depth[frame_idx] = noisy_depth
                primary_segmented_depth[frame_idx] = noisy_depth
                primary_object_masks[frame_idx] = valid_pixels.astype(np.uint8)

    visible_particle_indices = _visible_indices_from_observation_ids(object_particle_ids)
    return {
        "observation_data": {
            "frame_indices": torch.arange(frame_count, dtype=torch.long),
            "view_indices": torch.tensor(views, dtype=torch.long),
            "object_points_clean": torch.from_numpy(object_points_clean).float(),
            "object_points_noisy": torch.from_numpy(object_points_noisy).float(),
            "object_valid_mask": torch.from_numpy(object_valid).bool(),
            "object_pixels": torch.from_numpy(object_pixels).long(),
            "object_particle_ids": torch.from_numpy(object_particle_ids).long(),
            "object_particle_distances": torch.from_numpy(object_particle_distances).float(),
        },
        "visible_particle_indices": torch.from_numpy(visible_particle_indices).long(),
        "tracked_visible_particle_indices": torch.from_numpy(visible_particle_indices).long(),
        "depth_maps": torch.from_numpy(primary_depth).float(),
        "segmented_depth_maps": torch.from_numpy(primary_segmented_depth).float(),
        "object_masks": torch.from_numpy(primary_object_masks).to(torch.uint8),
        "primary_view": primary_view,
        "depth_render_source": "synthetic_particle_splat",
        "depth_primary_view": primary_view,
        "observation_views": list(views),
        "rendered_particle_count": int(render_count),
        "rendered_depth_noise": {
            "depth_noise_std": float(args.depth_noise_std),
            "depth_noise_relative": float(args.depth_noise_relative),
            "depth_dropout": float(args.depth_dropout),
            "splat_radius": int(args.splat_radius),
            "near_depth": float(args.near_depth),
            "far_depth": float(args.far_depth),
        },
    }


def _simulate_episode(
    episode_data: Dict,
    source_cfg: Dict,
    material_log_E: torch.Tensor,
    material_nu: torch.Tensor,
    rollout_engine: DifferentiableRolloutEngine,
    rollout_steps: int,
    device: torch.device,
    args: SimpleNamespace,
) -> Dict[str, torch.Tensor]:
    dataset_like = _clone_to_cpu(episode_data)
    dataset_like["particle_material_params"] = {
        "log_E": material_log_E.float(),
        "nu": material_nu.float(),
    }
    _ensure_visible_particle_indices(dataset_like)
    if "ground_height" not in dataset_like:
        ground_height = source_cfg.get("simulation", {}).get("ground_height", None)
        if ground_height is not None:
            dataset_like["ground_height"] = float(ground_height)
    episode = _load_episode_tensors(dataset_like, device)
    coords = episode["coords"]
    r_coords = episode["r_coords"]
    particle_v = episode["particle_v"]
    particle_F = episode["particle_F"]
    particle_C = episode["particle_C"]

    frame_count, num_particles, _ = coords.shape
    current_positions = coords[0].contiguous()
    current_velocities = particle_v[0].contiguous()
    current_F = particle_F[0].contiguous()
    current_C = None if particle_C is None else particle_C[0].contiguous()
    material_log_E_device = material_log_E.to(device=device, dtype=coords.dtype)
    material_nu_device = material_nu.to(device=device, dtype=coords.dtype)

    out_coords = torch.empty_like(coords)
    out_velocities = torch.empty_like(particle_v)
    out_F = torch.empty_like(particle_F)
    out_C = torch.empty_like(particle_F) if particle_C is None else torch.empty_like(particle_C)
    out_coords[0] = current_positions
    out_velocities[0] = current_velocities
    out_F[0] = current_F
    out_C[0] = torch.zeros_like(current_F) if current_C is None else current_C

    manipulation_flag = episode.get("manipulation_flag", coords.new_zeros(()))
    manipulation_contact_particle_ids = episode.get("manipulation_contact_particle_ids", None)
    controller_grid_points = episode.get("controller_grid_points", None)
    kinematic_contact_particle_ids = (
        manipulation_contact_particle_ids
        if _should_use_contact_kinematic_targets(
            is_real_world_episode=bool(episode.get("is_real_world", False)),
            manipulation_contact_particle_ids=manipulation_contact_particle_ids,
            controller_grid_points=controller_grid_points,
        )
        else None
    )
    particle_material_models = episode.get("particle_material_models", None)
    # rigid collision config now lives in config.yaml (rigid_body.collision); fall
    # back to it when the source .pt doesn't carry rigid_collision_cfg.
    rigid_collision_cfg = episode.get("rigid_collision_cfg") or source_cfg.get("rigid_body", {}).get("collision", {})
    rigid_body_primitives = episode.get("rigid_body_primitives", [])
    base_ground_height = float(episode.get("ground_height", rollout_engine.ground_height))
    grid_dx = 1.0 / max(int(getattr(rollout_engine, "num_grids", 1)), 1)
    ground_clearance = (
        float(args.ground_clearance)
        if args.ground_clearance is not None
        else float(args.ground_clearance_cells) * grid_dx
    )
    rollout_ground_height = base_ground_height - max(ground_clearance, 0.0)

    # If the engine's plasticity model supports per-particle yield tracking, enable it
    # for the duration of this rollout so the augmenter can record whether the sample
    # actually exhibited plastic flow.
    yield_tracker = rollout_engine.plasticity_model
    track_yield = (
        hasattr(yield_tracker, "enable_yield_tracking")
        and hasattr(yield_tracker, "get_yield_stats")
    )
    if track_yield:
        yield_tracker.enable_yield_tracking(int(coords.shape[1]))
        yield_tracker.reset_yield_tracking()

    progress = tqdm(total=frame_count - 1, desc="simulate", unit="frame", dynamic_ncols=True, leave=False)
    with torch.no_grad():
        for frame_idx in range(frame_count - 1):
            next_frame = frame_idx + 1
            rigid_window = _catmull_rom_window(r_coords, frame_idx, next_frame, rollout_steps)
            controller_window = (
                None if controller_grid_points is None
                else _catmull_rom_window(controller_grid_points, frame_idx, next_frame, rollout_steps)
            )
            contact_ids_window = (
                None if kinematic_contact_particle_ids is None
                else _hold_index_window(kinematic_contact_particle_ids, frame_idx, rollout_steps)
            )
            real_chunk_dt = _real_world_chunk_dt(episode, rollout_engine, rollout_steps)
            with _temporary_rollout_timestep(
                rollout_engine,
                dt=real_chunk_dt,
                ground_height=rollout_ground_height,
            ):
                result = _run_locked_rollout(
                    rollout_engine,
                    current_positions,
                    current_velocities,
                    current_F,
                    current_positions.new_zeros(rollout_steps, num_particles, 3),
                    material_log_E_device,
                    material_nu_device,
                    C=current_C,
                    material_model_info=particle_material_models,
                    rigid_points=rigid_window,
                    rigid_collision_cfg=rigid_collision_cfg,
                    rigid_body_primitives=rigid_body_primitives,
                    manipulation_indicator=manipulation_flag,
                    manipulation_contact_particle_ids=contact_ids_window,
                    controller_grid_points=controller_window,
                )
            current_positions = result["predicted_positions"][-1].detach().contiguous()
            current_velocities = result["final_velocity"].detach().contiguous()
            current_F = result["final_deformation_gradient"].detach().contiguous()
            current_C = None if result["final_C"] is None else result["final_C"].detach().contiguous()

            out_coords[next_frame] = current_positions
            out_velocities[next_frame] = current_velocities
            out_F[next_frame] = current_F
            out_C[next_frame] = torch.zeros_like(current_F) if current_C is None else current_C
            progress.update(1)
    progress.close()

    elasticity_name = type(rollout_engine.elasticity_model).__name__
    plasticity_name = type(rollout_engine.plasticity_model).__name__
    # Top-level mode flag derived from the active plasticity model. "plastic" is
    # any non-Identity plasticity (e.g., VonMisesPlasticity configured for
    # plasticine); "elastic" is IdentityPlasticity. Lets downstream training /
    # filtering code categorize episodes without parsing plasticity_meta.
    plasticity_mode = "elastic" if plasticity_name == "IdentityPlasticity" else "plastic"
    plasticity_meta: Dict[str, object] = {"name": plasticity_name, "mode": plasticity_mode}
    sigma_y_buffer = getattr(rollout_engine.plasticity_model, "sigma_y", None)
    if torch.is_tensor(sigma_y_buffer):
        plasticity_meta["sigma_y"] = float(sigma_y_buffer.detach().reshape(-1)[0].item())

    yield_stats: Dict[str, object] = {}
    per_particle_yield_mask: Optional[torch.Tensor] = None
    if track_yield:
        yield_stats = dict(yield_tracker.get_yield_stats())
        per_particle_yield_mask = yield_tracker.get_per_particle_yield_mask()
        if per_particle_yield_mask is not None:
            per_particle_yield_mask = per_particle_yield_mask.detach().cpu()
        plasticity_meta.update(yield_stats)
        yield_tracker.disable_yield_tracking()

    output = {
        "particle_coords": out_coords.detach().cpu(),
        "particle_flows": _compute_point_flows(out_coords.detach().cpu()),
        "particle_velocities": out_velocities.detach().cpu(),
        "particle_deformation_gradients": out_F.detach().cpu(),
        "particle_affine_velocity_matrices": out_C.detach().cpu(),
        "rigid_body_coords": r_coords.detach().cpu(),
        "rigid_body_flows": _compute_point_flows(r_coords.detach().cpu()),
        "domain_shift": episode.get("domain_shift", coords.new_zeros(3)).detach().cpu(),
        "ground_height": float(rollout_ground_height),
        "original_ground_height": float(base_ground_height),
        "ground_clearance": float(max(ground_clearance, 0.0)),
        "ground_surface": str(rollout_engine.ground_surface),
        "ground_elasticity": float(getattr(rollout_engine, "ground_elasticity", 0.5)),
        "ground_friction": float(getattr(rollout_engine, "ground_friction", 0.3)),
        "grid_resolution": int(getattr(rollout_engine, "num_grids", 0)),
        # Per-particle constitutive declaration so replay is reproducible regardless of
        # whatever the global plasticity default is at load time. All particles share one
        # plasticity model id (homogeneous augmentation).
        "particle_material_models": {
            "elasticity_ids": torch.zeros(int(out_coords.shape[1]), dtype=torch.long),
            "elasticity_names": [elasticity_name],
            "plasticity_ids": torch.zeros(int(out_coords.shape[1]), dtype=torch.long),
            "plasticity_names": [plasticity_name],
        },
        "plasticity_meta": plasticity_meta,
        "plasticity_mode": plasticity_mode,
    }
    if yield_stats:
        output["yield_stats"] = yield_stats
        output["any_particle_yielded"] = bool(yield_stats.get("any_particle_yielded", False))
    if per_particle_yield_mask is not None:
        output["particle_ever_yielded"] = per_particle_yield_mask
    if controller_grid_points is not None:
        output["controller_grid_points"] = controller_grid_points.detach().cpu()
    return output


def _update_sequence_aliases(data: Dict, coords: torch.Tensor) -> None:
    frame_count, num_particles, _ = coords.shape
    for key in (
        "particle_trajectories",
        "completed_full_particle_coords",
    ):
        value = data.get(key, None)
        if torch.is_tensor(value) and tuple(value.shape[:2]) == (frame_count, num_particles):
            data[key] = coords.clone()
    if "particle_motion_valid" in data:
        data["particle_motion_valid"] = torch.ones(frame_count, num_particles, dtype=torch.bool)


def _copy_sidecars(source_root: Path, output_root: Path, source_cfg: Dict, metadata: Dict) -> None:
    cfg = copy.deepcopy(source_cfg)
    cfg["source_dataset"] = "phystwin_augmented"
    cfg["is_real_world"] = True
    cfg["augmentation"] = metadata
    cfg.setdefault("simulation", {})
    cfg["simulation"]["ground_height"] = float(metadata.get("ground_height", cfg["simulation"].get("ground_height", 0.0)))
    cfg["simulation"]["ground_surface"] = str(metadata.get("ground_surface", cfg["simulation"].get("ground_surface", "slip")))
    cfg["simulation"]["ground_elasticity"] = float(metadata.get("ground_elasticity", cfg["simulation"].get("ground_elasticity", 0.5)))
    cfg["simulation"]["ground_friction"] = float(metadata.get("ground_friction", cfg["simulation"].get("ground_friction", 0.3)))
    cfg.setdefault("observation", {})
    cfg["observation"]["views"] = list(metadata.get("observation_views", cfg["observation"].get("views", [])))
    cfg["observation"]["primary_view"] = int(metadata.get("depth_primary_view", cfg["observation"].get("primary_view", 0)))
    cfg["observation"]["depth_render_source"] = str(metadata.get("depth_render_source", "synthetic_particle_splat"))
    cfg["observation"]["depth_tensor_scope"] = "primary_view_only"
    _write_json(output_root / "config.yaml", cfg)
    _write_json(output_root / "source_metadata.json", metadata)


def _write_augmented_episode(
    source_root: Path,
    output_root: Path,
    raw_data: Dict,
    source_cfg: Dict,
    simulation: Dict[str, torch.Tensor],
    material_log_E: torch.Tensor,
    material_nu: torch.Tensor,
    material_meta: Dict,
    render_data: Dict,
    args: SimpleNamespace,
    aug_index: int,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    data = _clone_to_cpu(raw_data)
    data.update(simulation)
    data["particle_material_params"] = {
        "log_E": material_log_E.float().clone(),
        "nu": material_nu.float().clone(),
    }
    data["particle_colors"] = _log_e_to_particle_colors(material_log_E, args.log_e_range)
    _update_sequence_aliases(data, simulation["particle_coords"])

    observation_data = render_data["observation_data"]
    data["observation_data"] = observation_data
    data["visible_particle_indices"] = render_data["visible_particle_indices"]
    data["tracked_visible_particle_indices"] = render_data["tracked_visible_particle_indices"]
    data["depth_available"] = True
    data["depth_key"] = f"augmented_phystwin_view_{render_data['primary_view']}_depth_m"
    data["depth_render_source"] = render_data["depth_render_source"]
    data["depth_primary_view"] = int(render_data["depth_primary_view"])
    data["depth_tensor_scope"] = "primary_view_only"
    if "camera_extrinsics" in data and torch.is_tensor(data["camera_extrinsics"]):
        data["camera_extrinsics"] = _shift_camera_extrinsics(
            data["camera_extrinsics"],
            simulation.get("domain_shift", torch.zeros(3)),
        )
    data["primary_view"] = int(render_data["primary_view"])
    data["source_dataset"] = "phystwin_augmented"
    data["is_real_world"] = True
    if "ground_height" not in data:
        ground_height = source_cfg.get("simulation", {}).get("ground_height", None)
        if ground_height is not None:
            data["ground_height"] = float(ground_height)
    data["source_episode_dir"] = str(source_root)
    data["augmentation_index"] = int(aug_index)
    grid_resolution = simulation.get("grid_resolution", args.grid_resolution)
    data["augmentation_metadata"] = {
        "source_episode_dir": str(source_root),
        "augmentation_index": int(aug_index),
        "material": material_meta,
        "rendering": render_data["rendered_depth_noise"],
        "depth_render_source": render_data["depth_render_source"],
        "depth_primary_view": int(render_data["depth_primary_view"]),
        "depth_tensor_scope": "primary_view_only",
        "observation_views": [int(v) for v in render_data["observation_views"]],
        "rendered_particle_count": int(render_data["rendered_particle_count"]),
        "rollout_steps_per_frame": int(args.rollout_steps),
        "grid_resolution": None if grid_resolution is None else int(grid_resolution),
        "ground_surface": simulation.get("ground_surface", str(args.ground_surface)),
        "ground_elasticity": float(simulation.get("ground_elasticity", args.ground_elasticity)),
        "ground_friction": float(simulation.get("ground_friction", args.ground_friction)),
        "ground_height": float(simulation.get("ground_height", 0.0)),
        "original_ground_height": float(simulation.get("original_ground_height", simulation.get("ground_height", 0.0))),
        "ground_clearance": float(simulation.get("ground_clearance", 0.0)),
        "ground_clearance_cells": float(args.ground_clearance_cells),
        "domain_shift": simulation.get("domain_shift", torch.zeros(3)).float().tolist(),
    }

    # Drop heavy image/depth tensors that no training/validation/visualization path
    # consumes for augmented episodes (~700 MB per episode total). The
    # rgb_images, depth_maps, segmented_depth_maps, object_masks, and hand_masks
    # tensors are either inherited from raw_data or freshly rendered, but every
    # downstream consumer reads them only from the converted real episode.
    for _heavy in (
        "rgb_images",
        "depth_maps",
        "segmented_depth_maps",
        "object_masks",
        "hand_masks",
    ):
        data.pop(_heavy, None)

    torch.save(data, output_root / "episode_data.pt")
    metadata = {
        **data["augmentation_metadata"],
        "output_episode_dir": str(output_root),
        "num_frames": int(simulation["particle_coords"].shape[0]),
        "num_particles": int(simulation["particle_coords"].shape[1]),
        "views": [int(v) for v in observation_data["view_indices"].reshape(-1).tolist()],
        "observation_views": [int(v) for v in observation_data["view_indices"].reshape(-1).tolist()],
        "max_observed_points": int(observation_data["object_points_clean"].shape[-2]),
    }
    _copy_sidecars(source_root, output_root, source_cfg, metadata)


def _knn_roughness(coords: torch.Tensor, values: torch.Tensor, k: int = 8) -> Dict:
    pts = coords.detach().cpu().float()
    vals = values.detach().cpu().float()
    chunks = []
    for start in range(0, pts.shape[0], 1024):
        end = min(start + 1024, pts.shape[0])
        d = torch.cdist(pts[start:end], pts)
        _, nn = torch.topk(d, k=min(k + 1, pts.shape[0]), largest=False, dim=1)
        nn = nn[:, 1:]
        src = torch.arange(start, end).view(-1, 1).expand_as(nn)
        chunks.append((vals[src] - vals[nn]).abs().reshape(-1))
    delta = torch.cat(chunks)
    return {
        "knn_k": int(k),
        "knn_mean_abs_delta": float(delta.mean().item()),
        "knn_p95_abs_delta": float(delta.quantile(0.95).item()),
    }


def _noise_args(group: Dict, rng: np.random.Generator) -> SimpleNamespace:
    noise = group["material"]["noise"]
    return SimpleNamespace(
        material_noise_frequency=float(rng.uniform(noise["freq_range"][0], noise["freq_range"][1])),
        material_noise_octaves=int(noise["octaves"]),
        material_noise_persistence=float(noise["persistence"]),
        material_noise_lacunarity=float(noise["lacunarity"]),
    )


def _sample_standardized_loge(
    canonical: torch.Tensor,
    group: Dict,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, Dict]:
    spec = group["material"]["log_E"]
    field, noise_meta = _smooth_noise_field(canonical, _noise_args(group, rng), rng)
    z = (field - float(field.mean())) / max(float(field.std()), 1.0e-8)
    target_mean = float(rng.uniform(spec["mean_range"][0], spec["mean_range"][1]))
    target_std = float(rng.uniform(spec["std_range"][0], spec["std_range"][1]))
    log_e_np = np.clip(target_mean + target_std * z, spec["clamp"][0], spec["clamp"][1]).astype(np.float32)
    log_e = torch.from_numpy(log_e_np)
    return log_e, {
        "mode": "coordinate_smooth_random_field_standardized",
        "description": "geometry-agnostic smooth random field over normalized particle coordinates; no contact/object segmentation",
        "target_mean_log_E": target_mean,
        "target_std_log_E": target_std,
        "log_E_clamp": [float(spec["clamp"][0]), float(spec["clamp"][1])],
        "mean_log_E": float(log_e.mean().item()),
        "std_log_E": float(log_e.std(unbiased=False).item()),
        "min_log_E": float(log_e.min().item()),
        "max_log_E": float(log_e.max().item()),
        "frac_log_E_le_5p3": float((log_e <= 5.3).float().mean().item()),
        "frac_log_E_ge_10": float((log_e >= 10.0).float().mean().item()),
        "noise": noise_meta,
        "roughness": _knn_roughness(canonical, log_e, k=8),
    }


def _sample_standardized_nu(
    canonical: torch.Tensor,
    group: Dict,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, Dict]:
    spec = group["material"]["nu"]
    field, noise_meta = _smooth_noise_field(canonical, _noise_args(group, rng), rng)
    z = (field - float(field.mean())) / max(float(field.std()), 1.0e-8)
    target_mean = float(rng.uniform(spec["mean_range"][0], spec["mean_range"][1]))
    target_std = float(rng.uniform(spec["std_range"][0], spec["std_range"][1]))
    nu_np = np.clip(target_mean + target_std * z, spec["clamp"][0], spec["clamp"][1]).astype(np.float32)
    nu = torch.from_numpy(nu_np)
    return nu, {
        "nu_mode": "coordinate_smooth_random_field_standardized",
        "nu_description": "geometry-agnostic smooth random field over normalized particle coordinates; no contact/object segmentation",
        "target_mean_nu": target_mean,
        "target_std_nu": target_std,
        "nu_clamp": [float(spec["clamp"][0]), float(spec["clamp"][1])],
        "mean_nu": float(nu.mean().item()),
        "std_nu": float(nu.std(unbiased=False).item()),
        "min_nu": float(nu.min().item()),
        "max_nu": float(nu.max().item()),
        "nu_noise": noise_meta,
        "nu_roughness": _knn_roughness(canonical, nu, k=8),
    }


def _motion_metrics(coords: torch.Tensor) -> Dict:
    coords = coords.detach().cpu().float()
    disp = (coords - coords[:1]).norm(dim=-1)
    lo = coords.amin(dim=(0, 1))
    hi = coords.amax(dim=(0, 1))
    ext = hi - lo
    finite = bool(torch.isfinite(coords).all().item())
    return {
        "finite": finite,
        "bounds_min": [float(x) for x in lo],
        "bounds_max": [float(x) for x in hi],
        "extent": [float(x) for x in ext],
        "mean_displacement": float(disp.mean().item()),
        "final_mean_displacement": float(disp[-1].mean().item()),
        "max_displacement": float(disp.max().item()),
        "final_max_displacement": float(disp[-1].max().item()),
        "stable_heuristic": bool(
            finite
            and float(ext[0]) < 0.7
            and float(ext[1]) < 0.7
            and float(ext[2]) < 0.45
            and float(disp.max()) < 0.45
        ),
    }


def _augmentation_args(group: Dict, device: str, rollout_steps: int) -> SimpleNamespace:
    render = group.get("render", {})
    ground = group.get("ground", {})
    rollout = group.get("rollout", {})
    return SimpleNamespace(
        rollout_steps=int(rollout_steps),
        grid_resolution=rollout.get("num_grids", None),
        ground_surface=str(rollout.get("ground_surface", "coulomb")),
        ground_elasticity=float(ground.get("elasticity", rollout.get("ground_elasticity", 0.0))),
        ground_friction=float(ground.get("friction", rollout.get("ground_friction", 0.5))),
        ground_clearance=ground.get("clearance", None),
        ground_clearance_cells=float(ground.get("clearance_cells", 0.0)),
        device=str(device),
        log_e_range=tuple(float(v) for v in group["material"]["log_E"]["clamp"]),
        nu_range=tuple(float(v) for v in group["material"]["nu"]["clamp"]),
        max_observed_points=int(render.get("max_observed_points", 0)),
        render_particle_subset=str(render.get("particle_subset", "shell")),
        splat_radius=int(render.get("splat_radius", 1)),
        depth_noise_std=float(render.get("depth_noise_std", 0.002)),
        depth_noise_relative=float(render.get("depth_noise_relative", 0.001)),
        depth_dropout=float(render.get("depth_dropout", 0.01)),
        near_depth=float(render.get("near_depth", 0.01)),
        far_depth=float(render.get("far_depth", 60.0)),
    )


def generate_smoothfield_episodes(
    source_root: Path,
    output_root: Path,
    group: Dict,
    indices: Sequence[int],
    device_name: str,
    progress: Optional[Callable[[], None]] = None,
) -> List[Dict]:
    """Generate smooth-field augmentation episodes for one source/group pair."""
    source_root = Path(source_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.create({"dataset": group["dataset"], "rollout": group["rollout"]})
    OmegaConf.set_struct(cfg, False)
    cfg.setdefault("rollout", {})
    rollout_steps = max(int(cfg.get("dataset", {}).get("rollout_steps", 1)), 1)
    aug_args = _augmentation_args(group, device_name, rollout_steps)
    cfg.rollout.ground_surface = str(aug_args.ground_surface)
    cfg.rollout.ground_elasticity = float(aug_args.ground_elasticity)
    cfg.rollout.ground_friction = float(aug_args.ground_friction)

    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    rollout_engine_cls = (
        BatchedDifferentiableRolloutEngine
        if bool(cfg.get("rollout", {}).get("use_batched_mpm", True))
        else DifferentiableRolloutEngine
    )
    rollout_engine = rollout_engine_cls(cfg.get("rollout", {}), device=device)

    raw_data = torch.load(source_root / "episode_data.pt", map_location="cpu")
    source_cfg = _episode_config(source_root)
    normalized_episode = ParticleFlowEpisodeDataset(
        [str(source_root)],
        cache_size=int(cfg.get("dataset", {}).get("cache_size", 2)),
        real_world_domain_center=cfg.get("dataset", {}).get("real_world_domain_center", None),
    )[0]
    canonical = normalized_episode["particle_coords"][0].float()

    summary = []
    for idx in [int(i) for i in indices]:
        rng = np.random.default_rng(int(group["seed_base"]) + idx * 1009)
        episode_dir = output_root / f"episode_{idx:04d}"
        log_E, loge_meta = _sample_standardized_loge(canonical, group, rng)
        nu, nu_meta = _sample_standardized_nu(canonical, group, rng)
        material_meta = {
            **loge_meta,
            **nu_meta,
            "source_episode_dir": str(source_root),
            "seed": int(group["seed_base"]) + idx * 1009,
        }
        print(json.dumps({"episode": idx, "material": material_meta}, indent=2), flush=True)
        simulation = _simulate_episode(
            normalized_episode,
            source_cfg,
            log_E,
            nu,
            rollout_engine,
            int(aug_args.rollout_steps),
            device,
            aug_args,
        )
        material_meta["motion_metrics"] = _motion_metrics(simulation["particle_coords"])
        render_data = _render_noisy_observations(
            simulation["particle_coords"],
            {
                **raw_data,
                "camera_extrinsics": _shift_camera_extrinsics(
                    raw_data["camera_extrinsics"],
                    simulation.get("domain_shift", torch.zeros(3)),
                ),
            },
            source_cfg,
            aug_args,
            rng,
        )
        _write_augmented_episode(
            source_root,
            episode_dir,
            raw_data,
            source_cfg,
            simulation,
            log_E,
            nu,
            material_meta,
            render_data,
            aug_args,
            idx,
        )
        record = {"episode": idx, "episode_dir": str(episode_dir), "material": material_meta}
        summary.append(record)
        print(
            json.dumps({"wrote": str(episode_dir / "episode_data.pt"), "motion": material_meta["motion_metrics"]}, indent=2),
            flush=True,
        )
        if progress is not None:
            progress()

    worker_summary = output_root / f"worker_{min(indices):04d}_{max(indices):04d}_summary.json"
    worker_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {worker_summary}", flush=True)
    return summary
