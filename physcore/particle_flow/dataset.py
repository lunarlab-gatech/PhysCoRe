"""
Episode datasets and sliding-window sampling for training.
"""

from __future__ import annotations

import glob
import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset


@dataclass
class EpisodeInfo:
    root_dir: str
    trajectories_path: str
    config_path: Optional[str]
    rigid_body_path: Optional[str]
    n_frames: int
    n_particles: int
    interaction_start_frame: int
    frame_dt: float
    sim_dt: float
    steps_per_frame: int
    ground_height: float
    rigid_friction: float
    rigid_surface_fps: int
    valid_start_frames: Tuple[int, ...]
    observation_frame_to_local: Dict[int, int]
    rigid_collision_cfg: Dict = field(default_factory=dict)
    rigid_body_primitives: List[Dict] = field(default_factory=list)
    episode_flags: Dict = field(default_factory=dict)


@dataclass
class WindowInfo:
    episode_index: int
    current_frame: int
    future_flow_l2_mean: float
    future_flow_l2_max: float


@dataclass
class EpisodeDataSource:
    root_dir: str
    trajectories_path: str
    config_path: Optional[str]
    rigid_collision_cfg: Dict = field(default_factory=dict)
    rigid_body_primitives: List[Dict] = field(default_factory=list)
    object_primitives: List[Dict] = field(default_factory=list)
    episode_flags: Dict = field(default_factory=dict)


def _point_cloud_bbox_center(points: torch.Tensor) -> torch.Tensor:
    if points.numel() == 0:
        return torch.zeros(3, dtype=torch.float32)
    return 0.5 * (points.min(dim=0).values + points.max(dim=0).values)


def _compute_point_flows(point_coords: torch.Tensor) -> torch.Tensor:
    flows = torch.zeros_like(point_coords)
    if point_coords.shape[0] > 1:
        flows[1:] = point_coords[1:] - point_coords[:-1]
    return flows


def _resolve_particle_coords(data: Dict) -> torch.Tensor:
    if 'particle_coords' in data:
        return data['particle_coords']
    return data['particle_trajectories']


def normalize_observation_views(
    observation_views: object = None,
    observation_view_index: int = 0,
) -> object:
    if observation_views is None:
        return (int(observation_view_index),)
    if isinstance(observation_views, str):
        value = observation_views.strip().lower()
        if value in {'all', '*'}:
            return 'all'
        if not value:
            return (int(observation_view_index),)
        return tuple(int(part.strip()) for part in value.split(',') if part.strip())
    if torch.is_tensor(observation_views):
        values = observation_views.reshape(-1).tolist()
        return tuple(int(value) for value in values)
    try:
        values = list(observation_views)  # type: ignore[arg-type]
    except TypeError:
        return (int(observation_views),)
    if not values:
        return (int(observation_view_index),)
    return tuple(int(value) for value in values)


def _has_observation_view_dim(tensor: torch.Tensor) -> bool:
    if tensor.ndim >= 4:
        return True
    return tensor.ndim >= 3 and int(tensor.shape[-1]) != 3


def _observation_view_indices(num_views: int, observation_views: object) -> torch.Tensor:
    if str(observation_views).lower() == 'all':
        indices = list(range(int(num_views)))
    else:
        indices = [int(value) for value in observation_views]  # type: ignore[arg-type]
    if not indices:
        indices = [0]
    clamped = [min(max(index, 0), int(num_views) - 1) for index in indices]
    return torch.tensor(clamped, dtype=torch.long)


def select_observation_views(tensor: torch.Tensor, observation_views: object) -> torch.Tensor:
    observation_views = normalize_observation_views(observation_views)
    if not _has_observation_view_dim(tensor):
        return tensor
    num_views = int(tensor.shape[0])
    view_indices = _observation_view_indices(num_views, observation_views)
    selected = tensor.index_select(0, view_indices.to(device=tensor.device))
    use_multi_view = str(observation_views).lower() == 'all' or selected.shape[0] > 1
    if not use_multi_view:
        return selected[0]
    if selected.ndim == 3:
        return selected.permute(1, 0, 2).reshape(selected.shape[1], selected.shape[0] * selected.shape[2])
    if selected.ndim >= 4:
        trailing_shape = tuple(selected.shape[3:])
        return selected.permute(1, 0, 2, *range(3, selected.ndim)).reshape(
            selected.shape[1],
            selected.shape[0] * selected.shape[2],
            *trailing_shape,
        )
    raise ValueError(f'Unsupported observation tensor shape: {tuple(tensor.shape)}')


def _resolve_particle_flows(data: Dict, particle_coords: Optional[torch.Tensor] = None) -> torch.Tensor:
    if 'particle_flows' in data:
        return data['particle_flows']
    if particle_coords is None:
        particle_coords = _resolve_particle_coords(data)
    return _compute_point_flows(particle_coords)


def _resolve_particle_velocities(data: Dict) -> Optional[torch.Tensor]:
    if 'particle_velocities' in data:
        return data['particle_velocities']
    return None


def _resolve_particle_deformation_gradients(data: Dict) -> Optional[torch.Tensor]:
    if 'particle_deformation_gradients' in data:
        return data['particle_deformation_gradients']
    if 'particle_F' in data:
        return data['particle_F']
    return None


def _resolve_particle_affine_velocity_matrices(data: Dict) -> Optional[torch.Tensor]:
    for key in (
        'particle_affine_velocity_matrices',
        'particle_affine_matrices',
        'particle_affine_velocity_matrix',
        'particle_C',
    ):
        if key in data:
            return data[key]
    return None


def _resolve_rigid_coords(data: Dict) -> Optional[torch.Tensor]:
    if 'rigid_body_coords' in data:
        return data['rigid_body_coords']
    return data.get('rigid_body_trajectories', None)


def _resolve_controller_grid_points(data: Dict) -> Optional[torch.Tensor]:
    for key in (
        'controller_grid_points',
        'controller_points',
        'manipulation_contact_points',
        'contact_object_points',
    ):
        value = data.get(key, None)
        if torch.is_tensor(value):
            return value
    return None


def _resolve_rigid_flows(data: Dict, rigid_coords: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
    if 'rigid_body_flows' in data:
        return data['rigid_body_flows']
    if rigid_coords is None:
        rigid_coords = _resolve_rigid_coords(data)
    if rigid_coords is None:
        return None
    return _compute_point_flows(rigid_coords)


def _resolve_material_params(data: Dict) -> Dict:
    if 'particle_material_params' in data:
        return data['particle_material_params']
    return data['material_info']


def _infer_particle_material_ids(log_E: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    material_pairs = torch.stack([log_E.float(), nu.float()], dim=-1)
    _, inverse = torch.unique(material_pairs, dim=0, return_inverse=True)
    return inverse.long()


def _resolve_visible_particle_indices(data: Dict, observation_views: object) -> Optional[torch.Tensor]:
    visible_particle_indices = data.get('visible_particle_indices', None)
    if visible_particle_indices is None:
        return None
    visible_particle_indices = visible_particle_indices.long()
    if visible_particle_indices.ndim == 2:
        return visible_particle_indices
    if visible_particle_indices.ndim == 3:
        return select_observation_views(visible_particle_indices, observation_views)
    raise ValueError(
        'visible_particle_indices must have shape (T, K) or (V, T, K); '
        f'got {tuple(visible_particle_indices.shape)}'
    )


def _resolve_all_visible_particle_indices(
    data: Dict,
    total_frames: int,
    observation_views: object = None,
) -> Optional[torch.Tensor]:
    visible_particle_indices = data.get('visible_particle_indices', None)
    if visible_particle_indices is not None:
        visible_particle_indices = visible_particle_indices.long()
        if observation_views is not None and visible_particle_indices.ndim == 3:
            return select_observation_views(visible_particle_indices, observation_views)
        return visible_particle_indices

    observation_data = data.get('observation_data', None)
    if not observation_data:
        return None

    object_particle_ids = observation_data.get('object_particle_ids', None)
    if object_particle_ids is None:
        return None
    object_particle_ids = object_particle_ids.long()
    if observation_views is not None and object_particle_ids.ndim == 3:
        object_particle_ids = select_observation_views(object_particle_ids, observation_views)
    if object_particle_ids.ndim == 2:
        return object_particle_ids

    frame_indices = observation_data.get('frame_indices', None)
    if frame_indices is None:
        return object_particle_ids

    frame_indices = torch.as_tensor(frame_indices, dtype=torch.long)
    full_visible = torch.full(
        (object_particle_ids.shape[0], total_frames, object_particle_ids.shape[-1]),
        -1,
        dtype=torch.long,
    )
    valid_frames = (frame_indices >= 0) & (frame_indices < total_frames)
    if valid_frames.any():
        full_visible[:, frame_indices[valid_frames]] = object_particle_ids[:, valid_frames]
    return full_visible


def _resolve_episode_roots(patterns: Sequence[str]) -> List[str]:
    episode_roots: Dict[str, str] = {}
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches and os.path.exists(pattern):
            matches = [pattern]
        for match in matches:
            path = Path(match)
            if path.is_file() and path.name == 'episode_data.pt':
                episode_roots[str(path.resolve().parent)] = str(path.resolve().parent)
                continue
            if not path.is_dir():
                continue
            direct_file = path / 'episode_data.pt'
            if direct_file.exists():
                episode_roots[str(path.resolve())] = str(path.resolve())
                continue
            for found in path.glob('**/episode_data.pt'):
                episode_roots[str(found.resolve().parent)] = str(found.resolve().parent)
    return list(dict.fromkeys(episode_roots.values()))


def rollout_collate(batch: List[Dict]) -> List[Dict]:
    return batch


def _load_episode_flags_from_config(config_path: Optional[str]) -> Dict:
    """Episode-level flags now stored in config.yaml ("episode" section) instead of
    episode_data.pt. Defaults preserve prior loader behavior when no config is present
    (non-manipulation), so only episodes whose config marks them manipulation get flagged."""
    flags = {
        "is_manipulation": False,
        "manipulation_flag": None,
        "include_tracked_surface_points": False,
        "interaction_start_frame": 0,
        "depth_available": False,
    }
    if config_path and Path(config_path).exists():
        ep_cfg = OmegaConf.load(config_path).get("episode", {}) or {}
        for key in flags:
            if key in ep_cfg:
                value = ep_cfg[key]
                flags[key] = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    if flags["manipulation_flag"] is None:
        flags["manipulation_flag"] = 1.0 if flags["is_manipulation"] else 0.0
    return flags


def _load_rigid_collision_cfg_from_config(config_path: Optional[str]) -> Dict:
    if not config_path:
        return {}
    config_file = Path(config_path)
    if not config_file.exists():
        return {}

    episode_cfg = OmegaConf.load(config_file)
    rigid_cfg = episode_cfg.get('rigid_body', {})
    collision_cfg = rigid_cfg.get('collision', {})
    if OmegaConf.is_config(collision_cfg):
        return OmegaConf.to_container(collision_cfg, resolve=True)
    if collision_cfg:
        return dict(collision_cfg)
    return {}


def _load_rigid_body_primitives(root_dir: str) -> List[Dict]:
    rigid_body_path = Path(root_dir) / 'rigid_body.json'
    if not rigid_body_path.exists():
        return []

    with open(rigid_body_path, 'r', encoding='utf-8') as f:
        rigid_info = json.load(f)
    return list(rigid_info.get('rigid_body_primitives', []))


def _load_object_primitives(root_dir: str) -> List[Dict]:
    primitives_path = Path(root_dir) / 'primitives.json'
    if not primitives_path.exists():
        return []

    with open(primitives_path, 'r', encoding='utf-8') as f:
        primitives = json.load(f)
    return list(primitives)


def _infer_particle_material_models(material_params: Dict, object_primitives: Sequence[Dict]) -> Optional[Dict]:
    if not object_primitives:
        return None

    log_E = material_params['log_E'].float()
    nu = material_params['nu'].float()
    if log_E.numel() == 0:
        return None

    primitive_specs = []
    for primitive in object_primitives:
        material = dict(primitive.get('material', {}))
        if not material:
            return None
        primitive_specs.append({
            'log_E': float(material['log_E']),
            'nu': float(material['nu']),
            'elasticity': str(material['elasticity']),
            'plasticity': str(material['plasticity']),
        })

    primitive_pairs = torch.tensor(
        [[spec['log_E'], spec['nu']] for spec in primitive_specs],
        dtype=torch.float32,
    )
    particle_pairs = torch.stack([log_E, nu], dim=-1)
    unique_pairs, inverse = torch.unique(particle_pairs, dim=0, return_inverse=True)

    elasticity_names_by_pair: List[str] = []
    plasticity_names_by_pair: List[str] = []
    tolerance = 1.0e-5
    for pair in unique_pairs:
        pair_diff = (primitive_pairs - pair.unsqueeze(0)).abs().amax(dim=-1)
        matches = (pair_diff <= tolerance).nonzero(as_tuple=False).squeeze(1)
        if matches.numel() == 0:
            return None

        elasticity_candidates = {primitive_specs[int(match_idx)]['elasticity'] for match_idx in matches.tolist()}
        plasticity_candidates = {primitive_specs[int(match_idx)]['plasticity'] for match_idx in matches.tolist()}
        if len(elasticity_candidates) != 1 or len(plasticity_candidates) != 1:
            return None

        elasticity_names_by_pair.append(next(iter(elasticity_candidates)))
        plasticity_names_by_pair.append(next(iter(plasticity_candidates)))

    elasticity_names = sorted(set(elasticity_names_by_pair))
    plasticity_names = sorted(set(plasticity_names_by_pair))
    elasticity_to_id = {name: idx for idx, name in enumerate(elasticity_names)}
    plasticity_to_id = {name: idx for idx, name in enumerate(plasticity_names)}

    pair_elasticity_ids = torch.tensor(
        [elasticity_to_id[name] for name in elasticity_names_by_pair],
        dtype=torch.long,
    )
    pair_plasticity_ids = torch.tensor(
        [plasticity_to_id[name] for name in plasticity_names_by_pair],
        dtype=torch.long,
    )

    return {
        'elasticity_ids': pair_elasticity_ids[inverse].long(),
        'plasticity_ids': pair_plasticity_ids[inverse].long(),
        'elasticity_names': elasticity_names,
        'plasticity_names': plasticity_names,
    }


class ParticleFlowEpisodeDataset(Dataset):
    def __init__(
        self,
        episode_roots: Sequence[str],
        cache_size: int = 2,
        real_world_domain_center: Optional[Sequence[float]] = None,
        observation_views: object = None,
        observation_view_index: int = 0,
    ) -> None:
        super().__init__()
        self._cache_size = max(int(cache_size), 1)
        self._episode_cache: OrderedDict[str, Dict] = OrderedDict()
        self.episodes: List[EpisodeDataSource] = []
        self.real_world_domain_center = (
            tuple(float(value) for value in real_world_domain_center)
            if real_world_domain_center is not None else None
        )
        self.observation_views = normalize_observation_views(
            observation_views,
            observation_view_index=observation_view_index,
        )

        for root_dir in episode_roots:
            root = Path(root_dir)
            trajectories_path = root / 'episode_data.pt'
            if not trajectories_path.exists():
                continue
            config_path = root / 'config.yaml'
            self.episodes.append(EpisodeDataSource(
                root_dir=str(root.resolve()),
                trajectories_path=str(trajectories_path.resolve()),
                config_path=str(config_path.resolve()) if config_path.exists() else None,
                rigid_collision_cfg=_load_rigid_collision_cfg_from_config(str(config_path.resolve()) if config_path.exists() else None),
                rigid_body_primitives=_load_rigid_body_primitives(str(root.resolve())),
                object_primitives=_load_object_primitives(str(root.resolve())),
                episode_flags=_load_episode_flags_from_config(str(config_path.resolve()) if config_path.exists() else None),
            ))

        if not self.episodes:
            raise ValueError('No episode trajectories were found for the provided trajectory roots.')

    def _load_episode(self, trajectories_path: str) -> Dict:
        cached = self._episode_cache.get(trajectories_path)
        if cached is not None:
            self._episode_cache.move_to_end(trajectories_path)
            return cached

        data = torch.load(trajectories_path, map_location='cpu')
        self._episode_cache[trajectories_path] = data
        while len(self._episode_cache) > self._cache_size:
            self._episode_cache.popitem(last=False)
        return data

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> Dict:
        info = self.episodes[index]
        self._last_episode_index = index
        data = self._load_episode(info.trajectories_path)

        particle_coords = _resolve_particle_coords(data).float()
        particle_flows = _resolve_particle_flows(data, particle_coords).float()
        particle_velocities = _resolve_particle_velocities(data)
        if particle_velocities is not None:
            particle_velocities = particle_velocities.float()
        particle_deformation_gradients = _resolve_particle_deformation_gradients(data)
        if particle_deformation_gradients is not None:
            particle_deformation_gradients = particle_deformation_gradients.float()
        particle_affine_velocity_matrices = _resolve_particle_affine_velocity_matrices(data)
        if particle_affine_velocity_matrices is not None:
            particle_affine_velocity_matrices = particle_affine_velocity_matrices.float()

        rigid_body_coords = _resolve_rigid_coords(data)
        if rigid_body_coords is None:
            rigid_body_coords = torch.zeros(particle_coords.shape[0], 0, 3, dtype=particle_coords.dtype)
            rigid_body_flows = torch.zeros_like(rigid_body_coords)
        else:
            rigid_body_coords = rigid_body_coords.float()
            rigid_body_flows = _resolve_rigid_flows(data, rigid_body_coords)
            rigid_body_flows = rigid_body_flows.float() if rigid_body_flows is not None else _compute_point_flows(rigid_body_coords).float()

        visible_particle_indices = _resolve_all_visible_particle_indices(
            data,
            total_frames=int(particle_coords.shape[0]),
            observation_views=self.observation_views,
        )
        if visible_particle_indices is None:
            visible_particle_indices = torch.full((particle_coords.shape[0], 0), -1, dtype=torch.long)

        material_params = _resolve_material_params(data)
        particle_material_models = _infer_particle_material_models(material_params, info.object_primitives)
        manipulation_contact_particle_ids = data.get(
            'manipulation_contact_particle_ids',
            data.get('contact_object_particle_ids', None),
        )
        if torch.is_tensor(manipulation_contact_particle_ids):
            manipulation_contact_particle_ids = manipulation_contact_particle_ids.long()
        controller_grid_points = _resolve_controller_grid_points(data)
        if torch.is_tensor(controller_grid_points):
            controller_grid_points = controller_grid_points.float()

        return {
            'particle_coords': particle_coords,
            'particle_flows': particle_flows,
            **({'particle_velocities': particle_velocities} if particle_velocities is not None else {}),
            **({'particle_deformation_gradients': particle_deformation_gradients} if particle_deformation_gradients is not None else {}),
            **({'particle_affine_velocity_matrices': particle_affine_velocity_matrices} if particle_affine_velocity_matrices is not None else {}),
            'rigid_body_coords': rigid_body_coords,
            'rigid_body_flows': rigid_body_flows,
            'visible_particle_indices': visible_particle_indices.long(),
            'observation_views': self.observation_views,
            'interaction_start_frame': int(data.get('interaction_start_frame', info.episode_flags['interaction_start_frame'])),
            'particle_material_params': {
                'log_E': material_params['log_E'].float(),
                'nu': material_params['nu'].float(),
            },
            **({'particle_material_models': particle_material_models} if particle_material_models is not None else {}),
            'rigid_collision_cfg': dict(info.rigid_collision_cfg),
            'rigid_body_primitives': list(info.rigid_body_primitives),
            'source_dataset': str(data.get('source_dataset', 'synthetic')).lower(),
            'is_real_world': bool(data.get('is_real_world', str(data.get('source_dataset', '')).lower() in {'phystwin', 'real', 'real_world'})),
            'manipulation_flag': torch.as_tensor(
                data.get('manipulation_flag', info.episode_flags['manipulation_flag']),
                dtype=torch.float32,
            ),
            'is_manipulation': bool(data.get('is_manipulation', info.episode_flags['is_manipulation'])),
            **({
                'manipulation_contact_particle_ids': manipulation_contact_particle_ids,
            } if manipulation_contact_particle_ids is not None else {}),
            **({
                'controller_grid_points': controller_grid_points,
            } if controller_grid_points is not None else {}),
            'frame_dt': float(data.get('frame_dt', 0.0)),
            'ground_height': float(data.get('ground_height', 0.0)),
            **({
                'real_world_domain_center': torch.tensor(
                    self.real_world_domain_center,
                    dtype=torch.float32,
                ),
            } if self.real_world_domain_center is not None else {}),
            **({'observation_data': data['observation_data']} if data.get('observation_data', None) else {}),
            # Propagate persistent-tracks fields so observed_features() can take
            # the cotracker-direct path (coords[F]-coords[F-1] for tracked
            # particles) instead of the noisy depth-observation fallback.
            **({'tracked_particle_count': int(data['tracked_particle_count'])}
               if 'tracked_particle_count' in data else {}),
            'include_tracked_surface_points': bool(data.get('include_tracked_surface_points', info.episode_flags['include_tracked_surface_points'])),
            **({'particle_motion_valid': data['particle_motion_valid'].bool()}
               if 'particle_motion_valid' in data and torch.is_tensor(data['particle_motion_valid']) else {}),
            # completed_shell_count splits particles into shell [0, count) vs
            # interior [count, N) for per-class metric tracking in train_MfM.py.
            **({'completed_shell_count': int(data['completed_shell_count'])}
               if 'completed_shell_count' in data else {}),
            'episode_index': index,
        }


class ParticleFlowTrajectoryDataset(Dataset):
    def __init__(
        self,
        episode_roots: Sequence[str],
        history_steps: int,
        rollout_steps: int,
        sample_stride: int = 1,
        observation_view_index: int = 0,
        observation_views: object = None,
        use_noisy_observation: bool = True,
        require_observation: bool = True,
        interaction_only: bool = True,
        cache_size: int = 2,
        max_windows: Optional[int] = None,
        default_ground_height: float = 0.02,
        min_interaction_offset: int = 0,
        min_future_flow_l2_mean: float = 0.0,
        min_future_flow_l2_max: float = 0.0,
        sort_by_motion: str = 'none',
        real_world_domain_center: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.history_steps = int(history_steps)
        self.rollout_steps = int(rollout_steps)
        self.sample_stride = max(int(sample_stride), 1)
        self.observation_view_index = int(observation_view_index)
        self.observation_views = normalize_observation_views(
            observation_views,
            observation_view_index=self.observation_view_index,
        )
        self.use_noisy_observation = bool(use_noisy_observation)
        self.require_observation = bool(require_observation)
        self.interaction_only = bool(interaction_only)
        self.default_ground_height = float(default_ground_height)
        self.min_interaction_offset = max(int(min_interaction_offset), 0)
        self.min_future_flow_l2_mean = float(min_future_flow_l2_mean)
        self.min_future_flow_l2_max = float(min_future_flow_l2_max)
        self.sort_by_motion = str(sort_by_motion).lower()
        self.real_world_domain_center = (
            tuple(float(value) for value in real_world_domain_center)
            if real_world_domain_center is not None else None
        )
        self._cache_size = max(int(cache_size), 1)
        self._episode_cache: OrderedDict[str, Dict] = OrderedDict()

        self.episodes: List[EpisodeInfo] = []
        self.samples: List[WindowInfo] = []

        for root_dir in episode_roots:
            info, windows = self._build_episode_info(root_dir)
            if not windows:
                continue
            episode_idx = len(self.episodes)
            self.episodes.append(info)
            for window in windows:
                window.episode_index = episode_idx
            self.samples.extend(windows)

        if self.sort_by_motion == 'descending':
            self.samples.sort(key=lambda item: (item.future_flow_l2_max, item.future_flow_l2_mean), reverse=True)
        elif self.sort_by_motion == 'ascending':
            self.samples.sort(key=lambda item: (item.future_flow_l2_max, item.future_flow_l2_mean))

        if max_windows is not None:
            self.samples = self.samples[:int(max_windows)]

        if not self.samples:
            raise ValueError('No rollout windows were found for the provided trajectory roots.')

    def get_episode_index_for_sample(self, sample_index: int) -> int:
        if sample_index < 0 or sample_index >= len(self.samples):
            raise IndexError(f'Sample index out of range: {sample_index}')
        return int(self.samples[sample_index].episode_index)

    def get_episode_rollout_schedule(self, episode_index: int) -> List[Tuple[int, int]]:
        info = self.episodes[episode_index]
        start_frame = max(self.history_steps, 1)
        if self.interaction_only:
            start_frame = max(start_frame, info.interaction_start_frame)
        if info.valid_start_frames:
            start_frame = max(start_frame, int(info.valid_start_frames[0]))

        final_frame = info.n_frames - 1
        if start_frame >= final_frame:
            return []

        observation_frames = sorted(info.observation_frame_to_local.keys())
        schedule: List[Tuple[int, int]] = []
        current_frame = start_frame
        while current_frame < final_frame:
            if self.require_observation and current_frame not in info.observation_frame_to_local:
                next_observation_frame = next((frame for frame in observation_frames if frame > current_frame), None)
                if next_observation_frame is None:
                    break
                current_frame = next_observation_frame
                continue

            future_steps = min(self.rollout_steps, final_frame - current_frame)
            if future_steps <= 0:
                break
            schedule.append((current_frame, future_steps))
            current_frame += future_steps

        return schedule

    def _load_episode(self, trajectories_path: str) -> Dict:
        cached = self._episode_cache.get(trajectories_path)
        if cached is not None:
            self._episode_cache.move_to_end(trajectories_path)
            return cached

        data = torch.load(trajectories_path, map_location='cpu')
        self._episode_cache[trajectories_path] = data
        while len(self._episode_cache) > self._cache_size:
            self._episode_cache.popitem(last=False)
        return data

    def _build_episode_info(self, root_dir: str) -> Tuple[EpisodeInfo, List[WindowInfo]]:
        root = Path(root_dir)
        trajectories_path = root / 'episode_data.pt'
        if not trajectories_path.exists():
            raise FileNotFoundError(f'Missing episode_data.pt under {root_dir}')

        data = torch.load(trajectories_path, map_location='cpu')
        particle_trajectory = _resolve_particle_coords(data)
        n_frames = int(particle_trajectory.shape[0])
        n_particles = int(particle_trajectory.shape[1])
        interaction_start_frame = int(data.get('interaction_start_frame', 0))
        frame_dt = float(data.get('frame_dt', 0.0))

        config_path = root / 'config.yaml'
        rigid_body_path = root / 'rigid_body.json'
        sim_dt = frame_dt if frame_dt > 0 else 2.0e-4
        steps_per_frame = 1
        ground_height = self.default_ground_height
        rigid_friction = 0.5
        rigid_surface_fps = 256
        rigid_collision_cfg: Dict = {}  # populated from config.yaml below
        if config_path.exists():
            episode_cfg = OmegaConf.load(config_path)
            sim_cfg = episode_cfg.get('simulation', {})
            sim_dt = float(sim_cfg.get('dt', frame_dt))
            cfg_steps_per_frame = int(sim_cfg.get('steps_per_frame', 0))
            if frame_dt > 0:
                inferred_steps = int(round(frame_dt / max(sim_dt, 1e-8)))
                steps_per_frame = cfg_steps_per_frame if cfg_steps_per_frame > 0 else max(inferred_steps, 1)
            else:
                steps_per_frame = max(cfg_steps_per_frame, 1)
                frame_dt = sim_dt * steps_per_frame
            ground_height = float(sim_cfg.get('ground_height', ground_height))
            rigid_cfg = episode_cfg.get('rigid_body', {})
            rigid_friction = float(rigid_cfg.get('friction', rigid_friction))
            rigid_surface_fps = int(rigid_cfg.get('n_surface_fps', rigid_surface_fps))
            collision_cfg = rigid_cfg.get('collision', {})
            if OmegaConf.is_config(collision_cfg):
                rigid_collision_cfg = OmegaConf.to_container(collision_cfg, resolve=True)
            elif collision_cfg:
                rigid_collision_cfg = dict(collision_cfg)
        elif frame_dt <= 0:
            frame_dt = sim_dt * steps_per_frame

        rigid_body_primitives: List[Dict] = []
        if rigid_body_path.exists():
            with open(rigid_body_path, 'r') as f:
                rigid_body_meta = json.load(f)
            rigid_body_primitives = list(rigid_body_meta.get('rigid_body_primitives', []))

        observation_frame_to_local: Dict[int, int] = {}
        visible_particle_indices = _resolve_visible_particle_indices(data, self.observation_views)
        if visible_particle_indices is not None:
            valid_frames = torch.nonzero((visible_particle_indices >= 0).any(dim=-1), as_tuple=False).squeeze(-1).tolist()
            observation_frame_to_local = {int(frame_idx): int(frame_idx) for frame_idx in valid_frames}
        elif 'observation_data' in data and data['observation_data']:
            frame_indices = data['observation_data']['frame_indices'].tolist()
            observation_frame_to_local = {int(frame_idx): local_idx for local_idx, frame_idx in enumerate(frame_indices)}

        start_min = max(self.history_steps, 1)
        if self.interaction_only:
            start_min = max(start_min, interaction_start_frame + self.min_interaction_offset)
        start_max = n_frames - self.rollout_steps - 1

        candidate_windows: List[WindowInfo] = []
        for frame_idx in range(start_min, start_max + 1, self.sample_stride):
            if self.require_observation and frame_idx not in observation_frame_to_local:
                continue
            future_flows = particle_trajectory[frame_idx + 1:frame_idx + 1 + self.rollout_steps] - particle_trajectory[frame_idx:frame_idx + self.rollout_steps]
            future_flow_l2 = future_flows.norm(dim=-1)
            candidate_windows.append(
                WindowInfo(
                    episode_index=-1,
                    current_frame=frame_idx,
                    future_flow_l2_mean=float(future_flow_l2.mean().item()),
                    future_flow_l2_max=float(future_flow_l2.max().item()),
                )
            )

        filtered_windows = [
            window for window in candidate_windows
            if window.future_flow_l2_mean >= self.min_future_flow_l2_mean
            and window.future_flow_l2_max >= self.min_future_flow_l2_max
        ]
        if not filtered_windows and candidate_windows and (
            self.min_future_flow_l2_mean > 0.0 or self.min_future_flow_l2_max > 0.0
        ):
            candidate_windows.sort(key=lambda item: (item.future_flow_l2_max, item.future_flow_l2_mean), reverse=True)
            fallback_count = min(len(candidate_windows), 32)
            filtered_windows = candidate_windows[:fallback_count]
            print(
                f'Warning: no windows in {root_dir} satisfied motion thresholds; '
                f'falling back to the top {fallback_count} highest-motion windows.'
            )

        valid_start_frames = tuple(window.current_frame for window in filtered_windows)

        episode_info = EpisodeInfo(
            root_dir=str(root.resolve()),
            trajectories_path=str(trajectories_path.resolve()),
            config_path=str(config_path.resolve()) if config_path.exists() else None,
            rigid_body_path=str(rigid_body_path.resolve()) if rigid_body_path.exists() else None,
            n_frames=n_frames,
            n_particles=n_particles,
            interaction_start_frame=interaction_start_frame,
            frame_dt=frame_dt,
            sim_dt=sim_dt,
            steps_per_frame=steps_per_frame,
            ground_height=ground_height,
            rigid_friction=rigid_friction,
            rigid_surface_fps=rigid_surface_fps,
            rigid_collision_cfg=rigid_collision_cfg,
            rigid_body_primitives=rigid_body_primitives,
            episode_flags=_load_episode_flags_from_config(str(config_path.resolve()) if config_path.exists() else None),
            valid_start_frames=valid_start_frames,
            observation_frame_to_local=observation_frame_to_local,
        )
        return episode_info, filtered_windows

    def __len__(self) -> int:
        return len(self.samples)

    def _compute_window_motion_stats(
        self,
        particle_trajectory: torch.Tensor,
        current_frame: int,
        future_steps: int,
    ) -> Tuple[float, float]:
        future_flows = (
            particle_trajectory[current_frame + 1:current_frame + 1 + future_steps]
            - particle_trajectory[current_frame:current_frame + future_steps]
        )
        future_flow_l2 = future_flows.norm(dim=-1)
        return float(future_flow_l2.mean().item()), float(future_flow_l2.max().item())

    def _build_observation(
        self,
        data: Dict,
        particle_trajectory: torch.Tensor,
        info: EpisodeInfo,
        current_frame: int,
        domain_shift: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        visible_particle_indices = _resolve_visible_particle_indices(data, self.observation_views)
        if visible_particle_indices is not None:
            if current_frame < 0 or current_frame >= visible_particle_indices.shape[0]:
                num_points = int(visible_particle_indices.shape[-1]) if visible_particle_indices.ndim >= 1 else 1
                return {
                    'observation_points': torch.zeros(num_points, 3, dtype=torch.float32),
                    'observation_points_clean': torch.zeros(num_points, 3, dtype=torch.float32),
                    'observation_flow_from_prev': torch.zeros(num_points, 3, dtype=torch.float32),
                    'observation_valid_mask': torch.zeros(num_points, dtype=torch.bool),
                    'observation_particle_ids': torch.full((num_points,), -1, dtype=torch.long),
                }

            observation_particle_ids = visible_particle_indices[current_frame].long()
            observation_valid_mask = observation_particle_ids >= 0
            observation_points_clean = torch.zeros(observation_particle_ids.shape[0], 3, dtype=torch.float32)
            observation_flow_from_prev = torch.zeros_like(observation_points_clean)
            if observation_valid_mask.any():
                particle_ids = observation_particle_ids[observation_valid_mask]
                observation_points_clean[observation_valid_mask] = particle_trajectory[current_frame, particle_ids]
                if current_frame > 0:
                    observation_flow_from_prev[observation_valid_mask] = (
                        particle_trajectory[current_frame, particle_ids]
                        - particle_trajectory[current_frame - 1, particle_ids]
                    )

            return {
                'observation_points': observation_points_clean.clone(),
                'observation_points_clean': observation_points_clean,
                'observation_flow_from_prev': observation_flow_from_prev.float(),
                'observation_valid_mask': observation_valid_mask,
                'observation_particle_ids': observation_particle_ids,
            }

        if (
            'observation_data' not in data
            or not data['observation_data']
            or current_frame not in info.observation_frame_to_local
        ):
            num_points = 1
            return {
                'observation_points': torch.zeros(num_points, 3, dtype=torch.float32),
                'observation_points_clean': torch.zeros(num_points, 3, dtype=torch.float32),
                'observation_flow_from_prev': torch.zeros(num_points, 3, dtype=torch.float32),
                'observation_valid_mask': torch.zeros(num_points, dtype=torch.bool),
                'observation_particle_ids': torch.full((num_points,), -1, dtype=torch.long),
            }

        observation_data = data['observation_data']
        local_idx = info.observation_frame_to_local[current_frame]
        point_key = 'object_points_noisy' if self.use_noisy_observation else 'object_points_clean'
        observation_points_all = select_observation_views(
            observation_data[point_key],
            self.observation_views,
        ).float()
        observation_points_clean_all = select_observation_views(
            observation_data['object_points_clean'],
            self.observation_views,
        ).float()
        observation_points = observation_points_all[local_idx]
        observation_points_clean = observation_points_clean_all[local_idx]
        if domain_shift is not None:
            shift = domain_shift.to(dtype=observation_points.dtype).view(1, 3)
            observation_points = observation_points + shift
            observation_points_clean = observation_points_clean + shift
        observation_valid_mask = select_observation_views(
            observation_data['object_valid_mask'],
            self.observation_views,
        )[local_idx].bool()
        observation_particle_ids = select_observation_views(
            observation_data['object_particle_ids'],
            self.observation_views,
        )[local_idx].long()

        observation_flow_from_prev = torch.zeros_like(observation_points_clean)
        valid_mask = observation_valid_mask & (observation_particle_ids >= 0)
        if valid_mask.any():
            particle_ids = observation_particle_ids[valid_mask]
            observation_flow_from_prev[valid_mask] = particle_trajectory[current_frame, particle_ids] - particle_trajectory[current_frame - 1, particle_ids]

        return {
            'observation_points': observation_points,
            'observation_points_clean': observation_points_clean,
            'observation_flow_from_prev': observation_flow_from_prev.float(),
            'observation_valid_mask': observation_valid_mask,
            'observation_particle_ids': observation_particle_ids,
        }

    def _build_future_observations(
        self,
        data: Dict,
        particle_trajectory: torch.Tensor,
        info: EpisodeInfo,
        frame_indices: Sequence[int],
        domain_shift: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        observations = [
            self._build_observation(data, particle_trajectory, info, int(frame_idx), domain_shift=domain_shift)
            for frame_idx in frame_indices
        ]
        if not observations:
            return {
                'future_observation_points': torch.zeros(0, 1, 3, dtype=torch.float32),
                'future_observation_points_clean': torch.zeros(0, 1, 3, dtype=torch.float32),
                'future_observation_flow_from_prev': torch.zeros(0, 1, 3, dtype=torch.float32),
                'future_observation_valid_mask': torch.zeros(0, 1, dtype=torch.bool),
                'future_observation_particle_ids': torch.full((0, 1), -1, dtype=torch.long),
            }

        return {
            'future_observation_points': torch.stack([obs['observation_points'] for obs in observations], dim=0),
            'future_observation_points_clean': torch.stack([obs['observation_points_clean'] for obs in observations], dim=0),
            'future_observation_flow_from_prev': torch.stack([obs['observation_flow_from_prev'] for obs in observations], dim=0),
            'future_observation_valid_mask': torch.stack([obs['observation_valid_mask'] for obs in observations], dim=0),
            'future_observation_particle_ids': torch.stack([obs['observation_particle_ids'] for obs in observations], dim=0),
        }

    def build_sample_for_frame(
        self,
        episode_index: int,
        current_frame: int,
        future_steps: Optional[int] = None,
    ) -> Dict:
        info = self.episodes[episode_index]
        data = self._load_episode(info.trajectories_path)
        source_dataset = str(data.get('source_dataset', 'synthetic')).lower()
        is_real_world = bool(data.get('is_real_world', source_dataset in {'phystwin', 'real', 'real_world'}))
        particle_trajectory = _resolve_particle_coords(data).float()
        particle_flows = _resolve_particle_flows(data, particle_trajectory).float()
        domain_shift = torch.zeros(3, dtype=particle_trajectory.dtype)
        if is_real_world and self.real_world_domain_center is not None:
            target_center = torch.tensor(self.real_world_domain_center, dtype=particle_trajectory.dtype)
            frame0_center = _point_cloud_bbox_center(particle_trajectory[0]).to(dtype=particle_trajectory.dtype)
            domain_shift = target_center - frame0_center
            particle_trajectory = particle_trajectory + domain_shift.view(1, 1, 3)
        if future_steps is None:
            future_steps = self.rollout_steps
        future_steps = max(int(future_steps), 1)

        current_particle_positions = particle_trajectory[current_frame].float()
        history_particle_flows = particle_flows[current_frame - self.history_steps + 1:current_frame + 1].float()
        current_particle_velocity = (history_particle_flows[-1] / max(info.frame_dt, 1e-8)).float()
        future_particle_positions = particle_trajectory[current_frame + 1:current_frame + 1 + future_steps].float()
        future_particle_flows = particle_flows[current_frame + 1:current_frame + 1 + future_steps].float()

        rigid_body_trajectory = _resolve_rigid_coords(data)
        rigid_body_flows = _resolve_rigid_flows(data, rigid_body_trajectory)
        if rigid_body_trajectory is not None:
            rigid_body_trajectory = rigid_body_trajectory.float() + domain_shift.view(1, 1, 3)
            rigid_body_flows = rigid_body_flows.float() if rigid_body_flows is not None else _compute_point_flows(rigid_body_trajectory).float()
            current_rigid_points = rigid_body_trajectory[current_frame].float()
            future_rigid_points = rigid_body_trajectory[current_frame + 1:current_frame + 1 + future_steps].float()
            future_rigid_flows = rigid_body_flows[current_frame + 1:current_frame + 1 + future_steps].float()
            current_rigid_center = _point_cloud_bbox_center(current_rigid_points).float()
            future_rigid_centers = torch.stack(
                [_point_cloud_bbox_center(points).float() for points in future_rigid_points],
                dim=0,
            )
        else:
            current_rigid_points = torch.zeros(1, 3, dtype=torch.float32)
            future_rigid_points = torch.zeros(future_steps, 1, 3, dtype=torch.float32)
            future_rigid_flows = torch.zeros(future_steps, 1, 3, dtype=torch.float32)
            current_rigid_center = torch.zeros(3, dtype=torch.float32)
            future_rigid_centers = torch.zeros(future_steps, 3, dtype=torch.float32)

        window_future_flow_l2_mean, window_future_flow_l2_max = self._compute_window_motion_stats(
            particle_trajectory,
            current_frame=current_frame,
            future_steps=future_steps,
        )
        material_info = _resolve_material_params(data)
        particle_material_ids = data['particle_material_ids'].long() if 'particle_material_ids' in data else _infer_particle_material_ids(
            material_info['log_E'],
            material_info['nu'],
        )
        observation = self._build_observation(
            data,
            particle_trajectory,
            info,
            current_frame,
            domain_shift=domain_shift if is_real_world else None,
        )
        future_frame_indices = list(range(current_frame + 1, current_frame + 1 + future_steps))
        future_observations = self._build_future_observations(
            data,
            particle_trajectory,
            info,
            future_frame_indices,
            domain_shift=domain_shift if is_real_world else None,
        )
        manipulation_flag = torch.as_tensor(
            data.get('manipulation_flag', info.episode_flags['manipulation_flag']),
            dtype=torch.float32,
        )
        manipulation_contact_particle_ids = data.get(
            'manipulation_contact_particle_ids',
            data.get('contact_object_particle_ids', None),
        )
        if torch.is_tensor(manipulation_contact_particle_ids):
            manipulation_contact_particle_ids = manipulation_contact_particle_ids[
                current_frame : current_frame + future_steps + 1
            ].long()
        controller_grid_points = _resolve_controller_grid_points(data)
        if torch.is_tensor(controller_grid_points):
            controller_grid_points = controller_grid_points[
                current_frame : current_frame + future_steps + 1
            ].float() + domain_shift.view(1, 1, 3)

        return {
            'episode_root': info.root_dir,
            'episode_index': episode_index,
            'current_frame': int(current_frame),
            'source_dataset': source_dataset,
            'is_real_world': is_real_world,
            'manipulation_flag': manipulation_flag,
            'is_manipulation': bool(data.get('is_manipulation', info.episode_flags['is_manipulation'])),
            **({
                'manipulation_contact_particle_ids': manipulation_contact_particle_ids,
            } if manipulation_contact_particle_ids is not None else {}),
            **({
                'controller_grid_points': controller_grid_points,
            } if controller_grid_points is not None else {}),
            'window_future_flow_l2_mean': window_future_flow_l2_mean,
            'window_future_flow_l2_max': window_future_flow_l2_max,
            'history_particle_flows': history_particle_flows,
            'current_particle_positions': current_particle_positions,
            'current_particle_velocity': current_particle_velocity,
            'future_particle_positions': future_particle_positions,
            'future_particle_flows': future_particle_flows,
            'current_rigid_points': current_rigid_points,
            'future_rigid_points': future_rigid_points,
            'future_rigid_flows': future_rigid_flows,
            'current_rigid_center': current_rigid_center,
            'future_rigid_centers': future_rigid_centers,
            'rigid_body_primitives': info.rigid_body_primitives,
            'rigid_friction': info.rigid_friction,
            'rigid_surface_fps': info.rigid_surface_fps,
            'rigid_collision_cfg': dict(info.rigid_collision_cfg),
            'gt_material_log_E': material_info['log_E'].float(),
            'gt_material_nu': material_info['nu'].float(),
            'particle_material_ids': particle_material_ids,
            'frame_dt': info.frame_dt,
            'sim_dt': info.sim_dt,
            'steps_per_frame': info.steps_per_frame,
            'ground_height': info.ground_height + (float(domain_shift[2].item()) if is_real_world else 0.0),
            'domain_shift': domain_shift,
            **observation,
            **future_observations,
        }

    def __getitem__(self, index: int) -> Dict:
        window = self.samples[index]
        sample = self.build_sample_for_frame(
            episode_index=window.episode_index,
            current_frame=window.current_frame,
            future_steps=self.rollout_steps,
        )
        sample['window_future_flow_l2_mean'] = float(window.future_flow_l2_mean)
        sample['window_future_flow_l2_max'] = float(window.future_flow_l2_max)
        return sample


def _build_dataset_from_roots(
    roots: Sequence[str],
    dataset_cfg: DictConfig,
    max_windows: Optional[int] = None,
) -> Optional[ParticleFlowTrajectoryDataset]:
    episode_roots = _resolve_episode_roots(roots)
    if not episode_roots:
        return None
    return ParticleFlowTrajectoryDataset(
        episode_roots=episode_roots,
        history_steps=int(dataset_cfg.history_steps),
        rollout_steps=int(dataset_cfg.rollout_steps),
        sample_stride=int(dataset_cfg.get('sample_stride', 1)),
        observation_view_index=int(dataset_cfg.get('observation_view_index', 0)),
        observation_views=dataset_cfg.get('observation_views', None),
        use_noisy_observation=bool(dataset_cfg.get('use_noisy_observation', True)),
        require_observation=bool(dataset_cfg.get('require_observation', True)),
        interaction_only=bool(dataset_cfg.get('interaction_only', True)),
        cache_size=int(dataset_cfg.get('cache_size', 2)),
        max_windows=max_windows,
        default_ground_height=float(dataset_cfg.get('default_ground_height', 0.02)),
        min_interaction_offset=int(dataset_cfg.get('min_interaction_offset', 0)),
        min_future_flow_l2_mean=float(dataset_cfg.get('min_future_flow_l2_mean', 0.0)),
        min_future_flow_l2_max=float(dataset_cfg.get('min_future_flow_l2_max', 0.0)),
        sort_by_motion=str(dataset_cfg.get('sort_by_motion', 'none')),
        real_world_domain_center=dataset_cfg.get('real_world_domain_center', None),
    )


def _build_episode_dataset_from_roots(
    roots: Sequence[str],
    dataset_cfg: DictConfig,
) -> Optional[ParticleFlowEpisodeDataset]:
    episode_roots = _resolve_episode_roots(roots)
    if not episode_roots:
        return None
    return ParticleFlowEpisodeDataset(
        episode_roots=episode_roots,
        cache_size=int(dataset_cfg.get('cache_size', 2)),
        real_world_domain_center=dataset_cfg.get('real_world_domain_center', None),
        observation_views=dataset_cfg.get('observation_views', None),
        observation_view_index=int(dataset_cfg.get('observation_view_index', 0)),
    )


def build_rollout_dataloaders(cfg: DictConfig) -> Tuple[DataLoader, Optional[DataLoader]]:
    dataset_cfg = cfg.dataset
    train_roots = list(dataset_cfg.get('train_roots', []))
    val_roots = list(dataset_cfg.get('val_roots', []))
    if not train_roots:
        raise ValueError('dataset.train_roots must contain at least one trajectory directory or glob pattern.')

    train_dataset = _build_dataset_from_roots(
        roots=train_roots,
        dataset_cfg=dataset_cfg,
        max_windows=dataset_cfg.get('max_train_windows', None),
    )
    if train_dataset is None:
        raise ValueError('No training trajectories were found from dataset.train_roots.')

    val_dataset = _build_dataset_from_roots(
        roots=val_roots,
        dataset_cfg=dataset_cfg,
        max_windows=dataset_cfg.get('max_val_windows', None),
    ) if val_roots else None

    batch_size = int(cfg.train.get('batch_size', 1))
    num_workers = int(cfg.train.get('num_workers', 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=bool(cfg.train.get('shuffle', True)),
        num_workers=num_workers,
        collate_fn=rollout_collate,
        pin_memory=bool(cfg.train.get('pin_memory', False)),
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=rollout_collate,
            pin_memory=bool(cfg.train.get('pin_memory', False)),
        )

    return train_loader, val_loader


def build_dataloaders(cfg: DictConfig) -> Tuple[DataLoader, Optional[DataLoader]]:
    dataset_cfg = cfg.dataset
    train_roots = list(dataset_cfg.get('train_roots', []))
    val_roots = list(dataset_cfg.get('val_roots', []))
    if not train_roots:
        raise ValueError('dataset.train_roots must contain at least one trajectory directory or glob pattern.')

    train_dataset = _build_episode_dataset_from_roots(
        roots=train_roots,
        dataset_cfg=dataset_cfg,
    )
    if train_dataset is None:
        raise ValueError('No training trajectories were found from dataset.train_roots.')

    val_dataset = _build_episode_dataset_from_roots(
        roots=val_roots,
        dataset_cfg=dataset_cfg,
    ) if val_roots else None

    batch_size = int(cfg.train.get('batch_size', 1))
    num_workers = int(cfg.train.get('num_workers', 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=bool(cfg.train.get('shuffle', True)),
        num_workers=num_workers,
        collate_fn=rollout_collate,
        pin_memory=bool(cfg.train.get('pin_memory', False)),
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=rollout_collate,
            pin_memory=bool(cfg.train.get('pin_memory', False)),
        )

    return train_loader, val_loader
