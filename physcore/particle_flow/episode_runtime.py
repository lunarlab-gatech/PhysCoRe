"""
Per-episode rollout helpers shared by the training and validation entrypoints.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from glob import glob, has_magic
import math
import os
import sys
import time
from pathlib import Path
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm
import matplotlib

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from physcore.particle_flow.dataset import (
    ParticleFlowEpisodeDataset,
    build_dataloaders,
    select_observation_views,
)
from physcore.particle_flow.rollout import (
    BatchedDifferentiableRolloutEngine,
    DifferentiableRolloutEngine,
    RandomMaterialGuessInitializer,
)
from physcore.particle_flow.losses import _compute_real_world_reconstruction_loss
from physcore.particle_flow.rollout_runtime import (
    real_world_chunk_dt as _real_world_chunk_dt,
    temporary_rollout_timestep as _temporary_rollout_timestep,
)
from physcore.particle_flow.runtime import (
    allreduce_gradients as _allreduce_gradients,
    broadcast_parameters as _broadcast_parameters,
    cleanup_distributed as _cleanup_distributed,
    get_rank as _get_rank,
    get_world_size as _get_world_size,
    is_distributed as _is_distributed,
    is_main_process as _is_main_process,
    setup_distributed as _setup_distributed,
    should_use_live_tqdm as _should_use_live_tqdm,
)
from physcore.sim.visualizer import render_comparison_video, render_video, render_frame, scalar_to_heatmap_colors, _write_video_frames

_WARP_ROLLOUT_LOCK = threading.Lock()

def _episode_has_material_gt(episode: Dict[str, object], material_targets_override: Optional[torch.Tensor] = None) -> bool:
    if material_targets_override is not None:
        return True
    source_dataset = str(episode.get('source_dataset', '')).strip().lower()
    if source_dataset in {'phystwin_augmented', 'augmented_phystwin'}:
        return True
    return not bool(episode.get('is_real_world', False))

def _set_equal_3d_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(0.55 * float((maxs - mins).max()), 1.0e-4)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

def _render_material_gt_comparison_video(
    *,
    static_points: torch.Tensor,
    predicted_log_E_history: torch.Tensor,
    gt_log_E: torch.Tensor,
    chunk_indices: Sequence[int],
    output_path: Path,
    vmin: float,
    vmax: float,
    fps: int = 4,
) -> None:
    import imageio.v2 as imageio
    from matplotlib.colors import Normalize

    output_path.parent.mkdir(parents=True, exist_ok=True)
    points_np = static_points.detach().float().cpu().numpy()
    pred_np = predicted_log_E_history.detach().float().cpu().numpy()
    gt_np = gt_log_E.detach().float().cpu().numpy()
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('turbo')
    frames: List[np.ndarray] = []

    for frame_idx in range(pred_np.shape[0]):
        fig = plt.figure(figsize=(10.0, 4.6), dpi=140)
        pred_ax = fig.add_subplot(121, projection='3d')
        gt_ax = fig.add_subplot(122, projection='3d')
        pred_colors = cmap(norm(pred_np[frame_idx]))
        gt_colors = cmap(norm(gt_np))
        for ax, colors, title in (
            (pred_ax, pred_colors, f'Predicted log_E | update chunk {int(chunk_indices[frame_idx])}'),
            (gt_ax, gt_colors, 'GT log_E'),
        ):
            ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=colors, s=2.0, linewidths=0)
            ax.view_init(elev=22.0, azim=225.0)
            _set_equal_3d_axes(ax, points_np)
            ax.set_axis_off()
            ax.set_title(title, fontsize=9)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=[pred_ax, gt_ax], fraction=0.030, pad=0.02, label='log_E')
        fig.tight_layout()
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        frames.append(frame)

    if frames:
        imageio.mimsave(output_path, frames, fps=max(int(fps), 1), macro_block_size=16)

def _save_material_gt_comparison_video(
    *,
    model: torch.nn.Module,
    cfg,
    train_dataset,
    episode_index: int = 0,
    material_guess_initializer: RandomMaterialGuessInitializer,
    device: torch.device,
    output_dir: str,
    epoch: int,
    history_steps: int,
    rollout_steps: int,
    geometry_frame: str,
    material_update_chunk_interval: int,
    graph_history_chunk_window: Optional[int] = None,
    graph_history_reset_on_update: Optional[bool] = None,
) -> Optional[str]:
    try:
        if train_dataset is None or len(train_dataset) == 0:
            return None
        episode_index = int(episode_index)
        if episode_index < 0 or episode_index >= len(train_dataset):
            return None
        raw_episode = train_dataset[episode_index]
        episode = _load_episode_tensors(raw_episode, device)
        if not _episode_has_material_gt(episode):
            return None
        coords = episode['coords']
        r_coords = episode['r_coords']
        particle_v = episode['particle_v']
        particle_F = episode['particle_F']
        total_frames = int(coords.shape[0])
        start_frame = max(int(history_steps) - 1, 0)
        current_frame = start_frame
        current_positions = coords[current_frame].contiguous()
        previous_chunk_positions = current_positions.detach().contiguous()
        current_velocities = particle_v[current_frame].contiguous()
        current_F = particle_F[current_frame].contiguous()
        current_correction_displacement = torch.zeros_like(current_positions)
        current_correction_mask = torch.zeros(
            current_positions.shape[:-1],
            device=current_positions.device,
            dtype=torch.bool,
        )
        current_material = _sample_material_guess(
            material_guess_initializer,
            current_positions,
            episode['gt_log_E'],
            episode['gt_nu'],
        )
        canonical_points = current_positions.unsqueeze(0).contiguous()
        canonical_bbox_min, canonical_bbox_max = _canonical_bbox(canonical_points)
        base_model = _base_model(model)
        was_training = base_model.training
        base_model.eval()
        cached_context = base_model.build_cached_context(canonical_points, geometry_frame=geometry_frame)
        temporal_state: Optional[Dict[str, torch.Tensor]] = None
        material_update_chunk_interval = max(int(material_update_chunk_interval), 1)
        if graph_history_chunk_window is None:
            graph_history_chunk_window = material_update_chunk_interval
        graph_history_chunk_window = max(int(graph_history_chunk_window), 1)
        if graph_history_reset_on_update is None:
            graph_history_reset_on_update = graph_history_chunk_window <= material_update_chunk_interval
        graph_history_reset_on_update = bool(graph_history_reset_on_update)
        material_update_offset = 0
        use_sparse_graph_input_aggregation = _uses_graph_input_aggregation(base_model)
        graph_input_buffer: List[Dict[str, Optional[torch.Tensor]]] = []
        material_history = [current_material.squeeze(0).detach().cpu()]
        update_chunks = [0]
        chunk_idx = 0
        max_chunks = max(1, total_frames - start_frame - 1)
        material_attention_total_chunks = max(int(math.ceil(max_chunks / max(int(rollout_steps), 1))), 1)

        while current_frame < total_frames - 1:
            chunk_idx += 1
            current_frame += 1
            current_positions = coords[current_frame].contiguous()
            current_velocities = particle_v[current_frame].contiguous()
            current_F = particle_F[current_frame].contiguous()
            should_predict_material = chunk_idx > 0
            is_material_update_chunk = should_predict_material and (
                material_update_chunk_interval <= 1
                or (chunk_idx % material_update_chunk_interval) == material_update_offset
            )
            graph_input_snapshots = None
            if should_predict_material and use_sparse_graph_input_aggregation:
                graph_input_buffer.append(
                    _make_graph_input_snapshot(
                        points=current_positions.unsqueeze(0),
                        rigid_points=r_coords[current_frame].unsqueeze(0),
                        material_properties=current_material,
                        point_velocity=current_velocities.unsqueeze(0),
                        point_F=current_F.unsqueeze(0),
                        correction_displacement=current_correction_displacement.unsqueeze(0),
                        correction_mask=current_correction_mask.unsqueeze(0),
                        canonical_points=canonical_points,
                        previous_points=previous_chunk_positions.unsqueeze(0),
                        manipulation_indicator=episode.get('manipulation_flag', coords.new_zeros(())),
                    )
                )
                if len(graph_input_buffer) > graph_history_chunk_window:
                    graph_input_buffer = graph_input_buffer[-graph_history_chunk_window:]
                if is_material_update_chunk:
                    graph_input_snapshots = list(graph_input_buffer)

            if should_predict_material and (is_material_update_chunk or not use_sparse_graph_input_aggregation):
                pred = base_model(
                    current_positions.unsqueeze(0),
                    r_coords[current_frame].unsqueeze(0),
                    current_material,
                    current_velocities.unsqueeze(0),
                    current_F.unsqueeze(0),
                    correction_displacement=current_correction_displacement.unsqueeze(0),
                    correction_mask=current_correction_mask.unsqueeze(0),
                    temporal_state=temporal_state,
                    episode_id=episode.get('episode_index', None),
                    cached_context=cached_context,
                    canonical_points=canonical_points,
                    canonical_bbox_min=canonical_bbox_min,
                    canonical_bbox_max=canonical_bbox_max,
                    geometry_frame=geometry_frame,
                    material_attention_total_chunks=material_attention_total_chunks,
                    manipulation_indicator=episode.get('manipulation_flag', coords.new_zeros(())),
                    previous_points=previous_chunk_positions.unsqueeze(0),
                    graph_input_snapshots=graph_input_snapshots,
                    chunk_idx=chunk_idx,
                )
                temporal_state = pred['temporal_state']
                if is_material_update_chunk:
                    current_material = pred['refined_material']
                    material_history.append(current_material.squeeze(0).detach().cpu())
                    update_chunks.append(chunk_idx)
                    if use_sparse_graph_input_aggregation and graph_history_reset_on_update:
                        graph_input_buffer = []
            previous_chunk_positions = current_positions.detach().contiguous()

        if was_training:
            base_model.train()
        material_history_tensor = torch.stack(material_history, dim=0)
        raw_episode_index = raw_episode.get('episode_index', episode_index) if isinstance(raw_episode, dict) else episode_index
        try:
            raw_episode_index = int(raw_episode_index)
        except Exception:
            raw_episode_index = episode_index
        video_path = (
            Path(output_dir)
            / 'visualization'
            / f'material_pred_vs_gt_episode_{raw_episode_index:04d}_epoch_{int(epoch):04d}.mp4'
        )
        _render_material_gt_comparison_video(
            static_points=coords[start_frame].detach().cpu(),
            predicted_log_E_history=material_history_tensor[:, :, 0],
            gt_log_E=episode['gt_log_E'].detach().cpu(),
            chunk_indices=update_chunks,
            output_path=video_path,
            vmin=float(cfg.model.get('material_log_E_min', 5.0)),
            vmax=float(cfg.model.get('material_log_E_max', 12.0)),
            fps=int((cfg.get('wandb', {}) or {}).get('material_video_fps', 4) or 4),
        )
        torch.save(
            {
                'predicted_material_history': material_history_tensor,
                'gt_material': torch.stack([
                    episode['gt_log_E'].detach().cpu(),
                    episode['gt_nu'].detach().cpu(),
                ], dim=-1),
                'update_chunks': torch.tensor(update_chunks, dtype=torch.long),
            },
            video_path.with_suffix('.pt'),
        )
        return str(video_path)
    except Exception as exc:
        print(f'Material comparison visualization failed: {exc}')
        return None

def _canonical_bbox(canonical_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if canonical_points.ndim == 2:
        canonical_points = canonical_points.unsqueeze(0)
    return (
        canonical_points.amin(dim=1, keepdim=True).contiguous(),
        canonical_points.amax(dim=1, keepdim=True).contiguous(),
    )

def _base_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, 'module') else model

def _uses_graph_input_aggregation(model: torch.nn.Module) -> bool:
    return bool(getattr(_base_model(model), 'uses_graph_input_aggregation', False))

def _detach_snapshot_tensor(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if not torch.is_tensor(value):
        return value
    return value.detach().contiguous()

def _make_graph_input_snapshot(
    *,
    points: torch.Tensor,
    rigid_points: torch.Tensor,
    material_properties: torch.Tensor,
    point_velocity: torch.Tensor,
    point_F: torch.Tensor,
    correction_displacement: Optional[torch.Tensor],
    correction_mask: Optional[torch.Tensor],
    canonical_points: torch.Tensor,
    previous_points: torch.Tensor,
    manipulation_indicator: Optional[torch.Tensor],
) -> Dict[str, Optional[torch.Tensor]]:
    return {
        'points': _detach_snapshot_tensor(points),
        'rigid_points': _detach_snapshot_tensor(rigid_points),
        'material_properties': _detach_snapshot_tensor(material_properties),
        'point_velocity': _detach_snapshot_tensor(point_velocity),
        'point_F': _detach_snapshot_tensor(point_F),
        'correction_displacement': _detach_snapshot_tensor(correction_displacement),
        'correction_mask': _detach_snapshot_tensor(correction_mask),
        'canonical_points': _detach_snapshot_tensor(canonical_points),
        'previous_points': _detach_snapshot_tensor(previous_points),
        'manipulation_indicator': _detach_snapshot_tensor(manipulation_indicator),
    }

def _run_locked_rollout(
    rollout_engine: DifferentiableRolloutEngine,
    *args,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    with _WARP_ROLLOUT_LOCK:
        return rollout_engine(*args, **kwargs)

def _expand_simulation_batch_roots(roots: Sequence[str]) -> List[str]:
    expanded_roots: List[str] = []
    seen = set()

    for root in roots:
        root_str = str(root)
        if has_magic(root_str):
            if root_str not in seen:
                expanded_roots.append(root_str)
                seen.add(root_str)
            continue

        root_path = Path(root_str)
        if root_path.is_dir() and not (root_path / 'episode_data.pt').exists():
            simulation_dirs = sorted(
                child for child in root_path.iterdir()
                if child.is_dir() and child.name.startswith('simulation_')
            )
            if simulation_dirs:
                for simulation_dir in simulation_dirs:
                    simulation_dir_str = str(simulation_dir)
                    if simulation_dir_str in seen:
                        continue
                    expanded_roots.append(simulation_dir_str)
                    seen.add(simulation_dir_str)
                continue

        if root_str not in seen:
            expanded_roots.append(root_str)
            seen.add(root_str)

    return expanded_roots

def _load_episode_tensors(ep: Dict, device: torch.device) -> Dict[str, object]:
    coords = ep['particle_coords'].to(device)
    flows = ep['particle_flows'].to(device)
    is_real_world = bool(ep.get('is_real_world', False))

    particle_v = ep.get('particle_velocities', None)
    if particle_v is None:
        frame_dt = float(ep.get('frame_dt', 0.0))
        particle_v = flows / max(frame_dt, 1.0e-8) if frame_dt > 0.0 else flows

    particle_F = ep.get('particle_deformation_gradients', None)
    if particle_F is None:
        particle_F = torch.eye(3, device=device, dtype=coords.dtype).view(1, 1, 3, 3).expand(
            coords.shape[0],
            coords.shape[1],
            3,
            3,
        ).clone()

    particle_C = ep.get('particle_affine_velocity_matrices', None)
    if particle_C is None and bool(ep.get('is_real_world', False)):
        particle_C = torch.zeros(
            coords.shape[0],
            coords.shape[1],
            3,
            3,
            device=device,
            dtype=coords.dtype,
        )

    observation_data = ep.get('observation_data', None)
    if observation_data:
        observation_data = {
            key: (value.to(device) if torch.is_tensor(value) else value)
            for key, value in observation_data.items()
        }
    manipulation_contact_particle_ids = ep.get(
        'manipulation_contact_particle_ids',
        ep.get('contact_object_particle_ids', None),
    )
    if torch.is_tensor(manipulation_contact_particle_ids):
        manipulation_contact_particle_ids = manipulation_contact_particle_ids.to(device=device, dtype=torch.long)
    controller_grid_points = None
    for controller_key in (
        'controller_grid_points',
        'controller_points',
        'manipulation_contact_points',
        'contact_object_points',
    ):
        value = ep.get(controller_key, None)
        if torch.is_tensor(value):
            controller_grid_points = value.to(device=device, dtype=coords.dtype)
            break
    r_coords = ep['rigid_body_coords'].to(device)
    ground_height = float(ep.get('ground_height', 0.0))
    domain_shift = coords.new_zeros(3)
    domain_center = ep.get('real_world_domain_center', None)
    if is_real_world and domain_center is not None:
        target_center = torch.as_tensor(domain_center, device=device, dtype=coords.dtype)
        if target_center.numel() != 3:
            raise ValueError(
                'real_world_domain_center must contain exactly three coordinates; '
                f'got shape {tuple(target_center.shape)}'
            )
        target_center = target_center.reshape(3)
        frame0_min = coords[0].amin(dim=0)
        frame0_max = coords[0].amax(dim=0)
        frame0_center = 0.5 * (frame0_min + frame0_max)
        domain_shift = target_center - frame0_center
        shift = domain_shift.view(1, 1, 3)
        coords = coords + shift
        r_coords = r_coords + shift
        if controller_grid_points is not None:
            controller_grid_points = controller_grid_points + shift
        ground_height += float(domain_shift[2].item())
        if observation_data:
            shifted_observation_data = {}
            for key, value in observation_data.items():
                if (
                    torch.is_tensor(value)
                    and value.is_floating_point()
                    and value.ndim >= 1
                    and value.shape[-1] == 3
                    and 'points' in str(key)
                ):
                    shifted_observation_data[key] = value + domain_shift.to(
                        device=value.device,
                        dtype=value.dtype,
                    ).view(*([1] * (value.ndim - 1)), 3)
                else:
                    shifted_observation_data[key] = value
            observation_data = shifted_observation_data

    return {
        'coords': coords,
        'flows': flows,
        'particle_v': particle_v.to(device),
        'particle_F': particle_F.to(device),
        'particle_C': None if particle_C is None else particle_C.to(device),
        'r_coords': r_coords,
        'gt_log_E': ep['particle_material_params']['log_E'].to(device),
        'gt_nu': ep['particle_material_params']['nu'].to(device),
        'interaction_start_frame': int(ep.get('interaction_start_frame', 0)),
        **({
            'particle_material_models': {
                'elasticity_ids': ep['particle_material_models']['elasticity_ids'].to(device),
                'plasticity_ids': ep['particle_material_models']['plasticity_ids'].to(device),
                'elasticity_names': list(ep['particle_material_models']['elasticity_names']),
                'plasticity_names': list(ep['particle_material_models']['plasticity_names']),
            }
        } if 'particle_material_models' in ep else {}),
        'vis_idx': ep['visible_particle_indices'].to(device),
        'rigid_collision_cfg': ep.get('rigid_collision_cfg', {}),
        'rigid_body_primitives': ep.get('rigid_body_primitives', []),
        'source_dataset': str(ep.get('source_dataset', 'synthetic')).lower(),
        'is_real_world': is_real_world,
        'observation_views': ep.get('observation_views', ep.get('observation_view_index', 0)),
        'frame_dt': float(ep.get('frame_dt', 0.0)),
        'sim_dt': float(ep.get('sim_dt', 0.0)),
        'steps_per_frame': int(ep.get('steps_per_frame', 0)),
        'ground_height': ground_height,
        'domain_shift': domain_shift,
        'manipulation_flag': torch.as_tensor(
            ep.get('manipulation_flag', 1.0 if bool(ep.get('is_manipulation', False)) else 0.0),
            device=device,
            dtype=coords.dtype,
        ),
        'is_manipulation': bool(ep.get('is_manipulation', False)),
        **({
            'manipulation_contact_particle_ids': manipulation_contact_particle_ids,
        } if manipulation_contact_particle_ids is not None else {}),
        **({
            'controller_grid_points': controller_grid_points,
        } if controller_grid_points is not None else {}),
        **({'observation_data': observation_data} if observation_data else {}),
        # Persistent-tracks fields used by train.observed_features() to take
        # the cotracker-direct displacement path instead of the noisy
        # depth-observation fallback.
        **({'tracked_particle_count': int(ep['tracked_particle_count'])}
           if 'tracked_particle_count' in ep else {}),
        **({'include_tracked_surface_points': bool(ep['include_tracked_surface_points'])}
           if 'include_tracked_surface_points' in ep else {}),
        **({'particle_motion_valid': ep['particle_motion_valid'].to(device)}
           if 'particle_motion_valid' in ep and torch.is_tensor(ep['particle_motion_valid']) else {}),
        **({'completed_shell_count': int(ep['completed_shell_count'])}
           if 'completed_shell_count' in ep else {}),
        'episode_index': ep.get('episode_index', None),
    }

def _get_visible_indices(vis_idx: torch.Tensor, frame_idx: int, num_particles: int, device: torch.device) -> torch.Tensor:
    if vis_idx.ndim == 3:
        vis = vis_idx[:, frame_idx].reshape(-1)
    else:
        vis = vis_idx[frame_idx]
    vis = vis[vis >= 0].unique()
    if vis.numel() == 0:
        vis = torch.arange(min(num_particles, 10), device=device)
    return vis

def _sample_material_guess(
    initializer: RandomMaterialGuessInitializer,
    current_particle_positions: torch.Tensor,
    gt_log_E: torch.Tensor,
    gt_nu: torch.Tensor,
    seed: Optional[int] = None,
) -> torch.Tensor:
    sample = {
        'current_particle_positions': current_particle_positions,
        'gt_material_log_E': gt_log_E,
        'gt_material_nu': gt_nu,
    }
    if seed is None:
        material_guess = initializer(sample)
    else:
        fork_devices: List[int] = []
        if current_particle_positions.device.type == 'cuda' and current_particle_positions.device.index is not None:
            fork_devices = [current_particle_positions.device.index]
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(int(seed))
            if current_particle_positions.device.type == 'cuda':
                torch.cuda.manual_seed_all(int(seed))
            material_guess = initializer(sample)

    return torch.stack([
        material_guess['log_E'],
        material_guess['nu'],
    ], dim=-1).unsqueeze(0)

def _build_correction_targets(
    predicted_positions: torch.Tensor,
    observed_positions: torch.Tensor,
    visible_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_particles = int(predicted_positions.shape[0])
    target_positions = torch.zeros_like(predicted_positions)
    target_mask = torch.zeros(num_particles, dtype=torch.bool, device=predicted_positions.device)

    if observed_positions.numel() == 0:
        return target_positions, target_mask

    if visible_indices is not None:
        corr_idx = visible_indices.reshape(-1).to(device=predicted_positions.device, dtype=torch.long)
        num_pairs = min(int(corr_idx.numel()), int(observed_positions.shape[0]))
        corr_idx = corr_idx[:num_pairs]
        observed_positions = observed_positions[:num_pairs]
        valid = (corr_idx >= 0) & (corr_idx < num_particles)
        if valid.any():
            corr_idx = corr_idx[valid]
            observed_positions = observed_positions[valid]
            target_position_sum = torch.zeros_like(predicted_positions)
            target_position_count = torch.zeros(
                num_particles,
                1,
                device=predicted_positions.device,
                dtype=predicted_positions.dtype,
            )
            target_position_sum.index_add_(0, corr_idx, observed_positions)
            target_position_count.index_add_(
                0,
                corr_idx,
                torch.ones(
                    observed_positions.shape[0],
                    1,
                    device=predicted_positions.device,
                    dtype=predicted_positions.dtype,
                ),
            )
            target_mask = target_position_count.squeeze(-1) > 0
            target_positions[target_mask] = (
                target_position_sum[target_mask] / target_position_count[target_mask]
            )
            return target_positions, target_mask

    distances = torch.cdist(observed_positions.unsqueeze(0), predicted_positions.unsqueeze(0)).squeeze(0)
    matched_particle_idx = distances.argmin(dim=1)
    target_position_sum = torch.zeros_like(predicted_positions)
    target_position_count = torch.zeros(
        num_particles, 1, device=predicted_positions.device, dtype=predicted_positions.dtype,
    )
    target_position_sum.index_add_(0, matched_particle_idx, observed_positions)
    target_position_count.index_add_(
        0,
        matched_particle_idx,
        torch.ones(observed_positions.shape[0], 1, device=predicted_positions.device, dtype=predicted_positions.dtype),
    )
    target_mask = target_position_count.squeeze(-1) > 0
    if target_mask.any():
        target_positions[target_mask] = (
            target_position_sum[target_mask] / target_position_count[target_mask]
        )
    return target_positions, target_mask

def _mean_masked_displacement_norm(
    displacement: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if mask is None:
        return displacement.norm(dim=-1).mean()
    valid_mask = mask.to(device=displacement.device, dtype=torch.bool)
    if not valid_mask.any():
        return displacement.new_zeros(())
    return displacement[valid_mask].norm(dim=-1).mean()

def _frame_observation_local_index(observation_data: Dict[str, torch.Tensor], frame_idx: int) -> Optional[int]:
    frame_indices = observation_data.get('frame_indices', None)
    if frame_indices is None:
        return int(frame_idx)
    frame_indices = frame_indices.long().reshape(-1)
    matches = torch.nonzero(frame_indices == int(frame_idx), as_tuple=False).reshape(-1)
    if matches.numel() == 0:
        return None
    return int(matches[0].item())

def _get_correction_observation(
    episode: Dict[str, object],
    coords: torch.Tensor,
    vis_idx: torch.Tensor,
    frame_idx: int,
    num_particles: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    observation_data = episode.get('observation_data', None)
    if observation_data:
        local_idx = _frame_observation_local_index(observation_data, int(frame_idx))
        if local_idx is not None:
            observation_views = episode.get('observation_views', episode.get('observation_view_index', 0))
            points = select_observation_views(observation_data['object_points_clean'], observation_views)
            valid_mask = select_observation_views(observation_data['object_valid_mask'], observation_views)
            particle_ids = select_observation_views(observation_data['object_particle_ids'], observation_views)
            if 0 <= local_idx < points.shape[0]:
                valid = valid_mask[local_idx].bool()
                ids = particle_ids[local_idx].long()
                valid = valid & (ids >= 0) & (ids < int(num_particles))
                if valid.any():
                    return points[local_idx, valid], ids[valid]

    visible_indices = _get_visible_indices(vis_idx, frame_idx, num_particles, coords.device)
    return coords[frame_idx, visible_indices], visible_indices

def _build_real_world_endpoint_loss_sample(
    episode: Dict[str, object],
    target_frame: int,
    reference: torch.Tensor,
) -> Optional[Dict[str, torch.Tensor]]:
    if not bool(episode.get('is_real_world', False)):
        return None
    observation_data = episode.get('observation_data', None)
    if not observation_data:
        return None

    required_keys = ('object_points_clean', 'object_valid_mask', 'object_particle_ids')
    if any(key not in observation_data for key in required_keys):
        return None

    local_idx = _frame_observation_local_index(observation_data, int(target_frame))
    if local_idx is None:
        return None

    observation_views = episode.get('observation_views', episode.get('observation_view_index', 0))
    points = select_observation_views(observation_data['object_points_clean'], observation_views)
    valid_mask = select_observation_views(observation_data['object_valid_mask'], observation_views)
    particle_ids = select_observation_views(observation_data['object_particle_ids'], observation_views)
    if local_idx < 0 or local_idx >= points.shape[0]:
        return None

    return {
        'is_real_world': True,
        'future_observation_points_clean': points[local_idx:local_idx + 1].to(
            device=reference.device,
            dtype=reference.dtype,
        ),
        'future_observation_valid_mask': valid_mask[local_idx:local_idx + 1].to(
            device=reference.device,
            dtype=torch.bool,
        ),
        'future_observation_particle_ids': particle_ids[local_idx:local_idx + 1].to(
            device=reference.device,
            dtype=torch.long,
        ),
    }

def _catmull_rom_window(sequence: Optional[torch.Tensor], start_frame: int, end_frame: int, steps: int) -> Optional[torch.Tensor]:
    if sequence is None:
        return None
    if sequence.shape[0] == 0:
        return sequence
    steps = max(int(steps), 1)
    max_frame = int(sequence.shape[0]) - 1
    i1 = max(0, min(int(start_frame), max_frame))
    i2 = max(0, min(int(end_frame), max_frame))
    i0 = max(0, min(i1 - 1, max_frame))
    i3 = max(0, min(i2 + 1, max_frame))

    p0 = sequence[i0]
    p1 = sequence[i1]
    p2 = sequence[i2]
    p3 = sequence[i3]
    u = torch.linspace(
        0.0,
        1.0,
        steps=steps + 1,
        device=sequence.device,
        dtype=sequence.dtype,
    )
    view_shape = (steps + 1,) + (1,) * p1.ndim
    u = u.view(view_shape)
    u2 = u * u
    u3 = u2 * u
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
    ).contiguous()

def _piecewise_catmull_rom_window(
    sequence: Optional[torch.Tensor],
    start_frame: int,
    end_frame: int,
    steps_per_frame: int,
) -> Optional[torch.Tensor]:
    """Per-frame Catmull-Rom, concatenated. Respects every intermediate frame."""
    if sequence is None:
        return None
    if sequence.shape[0] == 0:
        return sequence
    n_frames = end_frame - start_frame
    if n_frames <= 1:
        return _catmull_rom_window(sequence, start_frame, end_frame,
                                   max(n_frames, 1) * max(int(steps_per_frame), 1))
    segments = []
    for f in range(start_frame, end_frame):
        seg = _catmull_rom_window(sequence, f, f + 1, steps_per_frame)
        segments.append(seg[:-1] if f < end_frame - 1 else seg)
    return torch.cat(segments, dim=0).contiguous()

def _hold_index_window(sequence: Optional[torch.Tensor], frame_idx: int, steps: int) -> Optional[torch.Tensor]:
    if sequence is None:
        return None
    if sequence.shape[0] == 0:
        return sequence
    frame_idx = max(0, min(int(frame_idx), int(sequence.shape[0]) - 1))
    return sequence[frame_idx].unsqueeze(0).expand(int(steps) + 1, *sequence.shape[1:]).contiguous()

def _contact_row_stability(contact_particle_ids: Optional[torch.Tensor]) -> float:
    if not torch.is_tensor(contact_particle_ids) or contact_particle_ids.ndim != 2:
        return 1.0
    if contact_particle_ids.shape[0] < 2 or contact_particle_ids.shape[1] == 0:
        return 1.0
    valid = (contact_particle_ids[1:] >= 0) & (contact_particle_ids[:-1] >= 0)
    if not bool(valid.any().item()):
        return 1.0
    same = (contact_particle_ids[1:] == contact_particle_ids[:-1]) & valid
    return float(same.float().sum().item() / max(float(valid.float().sum().item()), 1.0))

def _should_use_contact_kinematic_targets(
    *,
    is_real_world_episode: bool,
    manipulation_contact_particle_ids: Optional[torch.Tensor],
    controller_grid_points: Optional[torch.Tensor],
) -> bool:
    if manipulation_contact_particle_ids is None:
        return False
    if not bool(is_real_world_episode):
        return True
    if controller_grid_points is None:
        return True
    return _contact_row_stability(manipulation_contact_particle_ids) >= 0.95

def _masked_position_l1(
    predicted_positions: torch.Tensor,
    target_positions: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    if target_mask.numel() == 0 or not bool(target_mask.any()):
        return predicted_positions.new_zeros(())
    return (predicted_positions[target_mask] - target_positions[target_mask]).abs().mean()

def _run_chunk_rollout_and_correction(
    rollout_engine: DifferentiableRolloutEngine,
    current_positions: torch.Tensor,
    current_velocities: torch.Tensor,
    current_F: torch.Tensor,
    current_C: Optional[torch.Tensor],
    material_state: torch.Tensor,
    chunk_steps: int,
    current_frame: int,
    next_chunk_frame: int,
    next_chunk_observed: torch.Tensor,
    next_chunk_vis: torch.Tensor,
    coords: torch.Tensor,
    r_coords: torch.Tensor,
    particle_material_models: Optional[Dict],
    rigid_collision_cfg: Optional[Dict],
    rigid_body_primitives: Optional[List[Dict]],
    correction_steps_count: int,
    run_correction: bool,
    manipulation_flag: Optional[torch.Tensor] = None,
    manipulation_contact_particle_ids: Optional[torch.Tensor] = None,
    controller_grid_points: Optional[torch.Tensor] = None,
    rigid_points_window: Optional[torch.Tensor] = None,
    manipulation_contact_particle_ids_window: Optional[torch.Tensor] = None,
    controller_grid_points_window: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    num_particles = int(current_positions.shape[0])
    zero_delta_v = torch.zeros(chunk_steps, num_particles, 3, device=current_positions.device)
    rollout_rigid_points = (
        rigid_points_window
        if rigid_points_window is not None
        else r_coords[current_frame : current_frame + chunk_steps + 1]
    )
    rollout_manipulation_ids = (
        manipulation_contact_particle_ids_window
        if manipulation_contact_particle_ids_window is not None
        else (
            None
            if manipulation_contact_particle_ids is None
            else manipulation_contact_particle_ids[current_frame : current_frame + chunk_steps + 1]
        )
    )
    rollout_controller_grid_points = (
        controller_grid_points_window
        if controller_grid_points_window is not None
        else (
            None
            if controller_grid_points is None
            else controller_grid_points[current_frame : current_frame + chunk_steps + 1]
        )
    )
    pure_mpm_result = _run_locked_rollout(
        rollout_engine,
        current_positions,
        current_velocities,
        current_F,
        zero_delta_v,
        material_state.squeeze(0)[:, 0],
        material_state.squeeze(0)[:, 1],
        C=current_C,
        material_model_info=particle_material_models,
        rigid_points=rollout_rigid_points,
        rigid_collision_cfg=rigid_collision_cfg,
        rigid_body_primitives=rigid_body_primitives,
        manipulation_indicator=manipulation_flag,
        manipulation_contact_particle_ids=rollout_manipulation_ids,
        controller_grid_points=rollout_controller_grid_points,
    )
    chunk_positions = pure_mpm_result['predicted_positions']
    chunk_flows = pure_mpm_result['predicted_flows']
    correction_targets, correction_target_mask = _build_correction_targets(
        chunk_positions[-1].detach(),
        next_chunk_observed,
        visible_indices=next_chunk_vis,
    )
    correction_target_displacement = (
        (correction_targets - chunk_positions[-1])
        * correction_target_mask.to(device=chunk_positions.device, dtype=chunk_positions.dtype).unsqueeze(-1)
    )
    surface_correspondence_cost = _mean_masked_displacement_norm(
        correction_targets - chunk_positions[-1],
        correction_target_mask,
    )

    mpm_endpoint_loss = _masked_position_l1(
        chunk_positions[-1],
        correction_targets,
        correction_target_mask,
    )
    correction_delta_v = chunk_positions[-1].new_zeros(correction_steps_count, num_particles, 3)
    correction_rigid = r_coords[next_chunk_frame].unsqueeze(0).expand(
        correction_steps_count + 1, -1, -1,
    ).contiguous()

    with torch.no_grad():
        if run_correction:
            correction_init_velocity = torch.zeros_like(pure_mpm_result['final_velocity'])
            correction_init_C = (
                None if pure_mpm_result['final_C'] is None
                else torch.zeros_like(pure_mpm_result['final_C'])
            )
            correction_result = _run_locked_rollout(
                rollout_engine,
                chunk_positions[-1],
                correction_init_velocity,
                pure_mpm_result['final_deformation_gradient'],
                correction_delta_v,
                material_state.squeeze(0)[:, 0],
                material_state.squeeze(0)[:, 1],
                C=correction_init_C,
                material_model_info=particle_material_models,
                rigid_points=correction_rigid,
                rigid_collision_cfg=rigid_collision_cfg,
                rigid_body_primitives=rigid_body_primitives,
                manipulation_indicator=manipulation_flag,
                manipulation_contact_particle_ids=(
                    None
                    if manipulation_contact_particle_ids is None
                    else manipulation_contact_particle_ids[next_chunk_frame].unsqueeze(0).expand(
                        correction_steps_count + 1, -1,
                    ).contiguous()
                ),
                controller_grid_points=(
                    None
                    if controller_grid_points is None
                    else controller_grid_points[next_chunk_frame].unsqueeze(0).expand(
                        correction_steps_count + 1, -1, -1,
                    ).contiguous()
                ),
                direct_velocity=False,
                correction_targets=correction_targets,
                correction_target_mask=correction_target_mask,
            )
        else:
            correction_result = {
                'predicted_positions': chunk_positions[-1:].detach(),
                'final_velocity': pure_mpm_result['final_velocity'],
                'final_deformation_gradient': pure_mpm_result['final_deformation_gradient'],
                'final_C': pure_mpm_result['final_C'],
                'mean_target_delta_velocity_norm': chunk_positions.new_zeros(()),
            }

    correction_displacement = correction_result['predicted_positions'][-1] - chunk_positions[-1]
    correction_loss = _masked_position_l1(
        correction_result['predicted_positions'][-1],
        correction_targets,
        correction_target_mask,
    )
    surface_correction_loss = correction_loss

    return {
        'chunk_positions': chunk_positions,
        'chunk_flows': chunk_flows,
        'pure_mpm_result': pure_mpm_result,
        'correction_targets': correction_targets,
        'correction_target_mask': correction_target_mask,
        'correction_target_displacement': correction_target_displacement,
        'surface_correspondence_cost': surface_correspondence_cost,
        'correction_result': correction_result,
        'correction_displacement': correction_displacement,
        'correction_displacement_score': _mean_masked_displacement_norm(
            correction_displacement,
            correction_target_mask,
        ),
        'mpm_endpoint_loss': mpm_endpoint_loss,
        'correction_loss': correction_loss,
        'surface_correction_loss': surface_correction_loss,
    }

def _zero_real_loss_terms(reference: torch.Tensor) -> Dict[str, torch.Tensor]:
    zero = reference.sum() * 0.0
    return {
        'real_correspondence_loss': zero,
        'real_obs_to_pred_loss': zero,
        'real_pred_visible_to_obs_loss': zero,
        'real_temporal_smoothness_loss': zero,
    }

def _compute_loss(
    predicted_positions: torch.Tensor,
    predicted_flows: torch.Tensor,
    gt_positions: torch.Tensor,
    gt_flows: torch.Tensor,
    position_weight: float,
    flow_weight: float,
    episode: Optional[Dict[str, object]] = None,
    start_frame: Optional[object] = None,
    target_frame: Optional[object] = None,
    loss_config: Optional[object] = None,
    return_terms: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    del gt_flows, flow_weight

    if episode is not None and start_frame is not None and bool(episode.get('is_real_world', False)):
        cfg = {} if loss_config is None else loss_config
        real_cfg = cfg.get('real_world_reconstruction', {})
        if not bool(real_cfg.get('enabled', True)):
            endpoint_error = (predicted_positions[-1] - gt_positions[-1]).abs()
            weighted = endpoint_error.mean() * position_weight
            raw = endpoint_error.mean()
            if return_terms:
                return weighted, raw, _zero_real_loss_terms(predicted_positions)
            return weighted, raw
        if predicted_positions.ndim == 4:
            if torch.is_tensor(start_frame):
                start_frames = [int(frame.item()) for frame in start_frame.reshape(-1)]
            elif isinstance(start_frame, (list, tuple)):
                start_frames = [int(frame) for frame in start_frame]
            else:
                raise ValueError('Batched real-world loss requires a sequence of start frames')
            if target_frame is None:
                target_frames = [frame + 1 for frame in start_frames]
            elif torch.is_tensor(target_frame):
                target_frames = [int(frame.item()) for frame in target_frame.reshape(-1)]
            elif isinstance(target_frame, (list, tuple)):
                target_frames = [int(frame) for frame in target_frame]
            else:
                target_frames = [int(target_frame)] * len(start_frames)
            real_losses = []
            real_term_values: Dict[str, List[torch.Tensor]] = {}
            for batch_idx, frame_idx in enumerate(start_frames):
                del frame_idx
                sample = _build_real_world_endpoint_loss_sample(
                    episode,
                    int(target_frames[batch_idx]),
                    predicted_positions[:, batch_idx],
                )
                if sample is None:
                    continue
                loss_terms = _compute_real_world_reconstruction_loss(
                    sample,
                    predicted_positions[-1:, batch_idx],
                    predicted_flows[-1:, batch_idx],
                    cfg,
                    str(cfg.get('position_loss_type', cfg.get('reconstruction_loss_type', 'l1'))).lower(),
                    float(cfg.get('smooth_l1_beta', 1.0)),
                )
                real_losses.append(loss_terms['real_recon_loss'])
                for key in (
                    'real_correspondence_loss',
                    'real_obs_to_pred_loss',
                    'real_pred_visible_to_obs_loss',
                    'real_temporal_smoothness_loss',
                ):
                    real_term_values.setdefault(key, []).append(loss_terms[key])
            if real_losses:
                raw_real_loss = torch.stack(real_losses).mean()
                if return_terms:
                    averaged_terms = {
                        key: torch.stack(values).mean()
                        for key, values in real_term_values.items()
                    }
                    return raw_real_loss * position_weight, raw_real_loss, averaged_terms
                return raw_real_loss * position_weight, raw_real_loss
            zero = predicted_positions.sum() * 0.0
            if return_terms:
                return zero, zero, _zero_real_loss_terms(predicted_positions)
            return zero, zero
        else:
            real_target_frame = int(target_frame) if target_frame is not None else int(start_frame) + 1
            sample = _build_real_world_endpoint_loss_sample(
                episode,
                real_target_frame,
                predicted_positions,
            )
            if sample is not None:
                loss_terms = _compute_real_world_reconstruction_loss(
                    sample,
                    predicted_positions[-1:],
                    predicted_flows[-1:],
                    cfg,
                    str(cfg.get('position_loss_type', cfg.get('reconstruction_loss_type', 'l1'))).lower(),
                    float(cfg.get('smooth_l1_beta', 1.0)),
                )
                raw_real_loss = loss_terms['real_recon_loss']
                if return_terms:
                    return raw_real_loss * position_weight, raw_real_loss, {
                        'real_correspondence_loss': loss_terms['real_correspondence_loss'],
                        'real_obs_to_pred_loss': loss_terms['real_obs_to_pred_loss'],
                        'real_pred_visible_to_obs_loss': loss_terms['real_pred_visible_to_obs_loss'],
                        'real_temporal_smoothness_loss': loss_terms['real_temporal_smoothness_loss'],
                    }
                return raw_real_loss * position_weight, raw_real_loss
            zero = predicted_positions.sum() * 0.0
            if return_terms:
                return zero, zero, _zero_real_loss_terms(predicted_positions)
            return zero, zero

    endpoint_error = (predicted_positions[-1] - gt_positions[-1]).abs()
    weighted = endpoint_error.mean() * position_weight
    raw = endpoint_error.mean()
    if return_terms:
        return weighted, raw, _zero_real_loss_terms(predicted_positions)
    return weighted, raw

def _normalize_geometry_frame(point_frame: object, *, field_name: str) -> str:
    normalized = str(point_frame).strip().lower()
    if normalized == 'canonical':
        return 'canonical'
    if normalized in {'current_coordinate', 'current', 'dynamic'}:
        return 'current_coordinate'
    raise ValueError(
        f'Unknown {field_name}={point_frame!r} '
        '(expected "canonical" or "current_coordinate")'
    )



# --- Episode-level helpers shared by the MfM and RfD scripts ---


def _kinematic_contact_ids(ep: Dict) -> Optional[torch.Tensor]:
    """Only use ``manipulation_contact_particle_ids`` as MPM kinematic targets
    when the row identities are stable enough (>=95%). Otherwise rely on
    ``controller_grid_points`` for soft grid-velocity coupling."""
    raw = ep.get('manipulation_contact_particle_ids')
    if not torch.is_tensor(raw):
        return None
    return raw if _should_use_contact_kinematic_targets(
        is_real_world_episode=bool(ep.get('is_real_world', False)),
        manipulation_contact_particle_ids=raw,
        controller_grid_points=ep.get('controller_grid_points'),
    ) else None


def _pull_tracked_visible_indices(episode_root: str, device: torch.device) -> Optional[torch.Tensor]:
    """Load ``tracked_visible_particle_indices`` from the converted
    ``episode_data.pt``. Returns (T, K) int64 on ``device``, or None if absent."""
    path = Path(episode_root) / 'episode_data.pt'
    if not path.exists():
        return None
    data = torch.load(path, map_location='cpu', weights_only=False)
    tvis = data.get('tracked_visible_particle_indices', None)
    if not torch.is_tensor(tvis):
        return None
    return tvis.to(device=device, dtype=torch.long)
