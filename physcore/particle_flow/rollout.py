"""
Differentiable MPM rollout engines, in single and batched forms.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from physcore.sim.materials import get_elasticity_model, get_plasticity_model
from physcore.sim.mpm import (
    BatchedMPMSolver,
    MPMSolver,
    add_batched_rigid_body_collider,
    add_batched_surface_collider,
    add_rigid_body_collider,
    add_surface_collider,
)
from physcore.sim.primitives import Primitive
from physcore.sim.rigid_body import BatchedRigidBody, RigidBody, sample_surface_points


class RandomMaterialGuessInitializer:
    def __init__(self, material_cfg: DictConfig) -> None:
        self.mode = str(material_cfg.get('mode', 'uniform')).lower()
        self.log_E_range = tuple(float(v) for v in material_cfg.get('log_E_range', [8.2, 8.6]))
        self.nu_range = tuple(float(v) for v in material_cfg.get('nu_range', [0.15, 0.45]))
        self.log_E_noise_std = float(material_cfg.get('log_E_noise_std', 0.15))
        self.nu_noise_std = float(material_cfg.get('nu_noise_std', 0.03))
        # Per-particle noise added on top of the base uniform/jitter sample.
        # Clamp range can be wider than log_E_range to allow the noise to spread.
        self.per_particle_noise = bool(material_cfg.get('per_particle_noise', False))
        self.per_particle_log_E_noise_std = float(material_cfg.get('per_particle_log_E_noise_std', 0.5))
        self.per_particle_nu_noise_std = float(material_cfg.get('per_particle_nu_noise_std', 0.05))
        log_E_clamp = material_cfg.get('log_E_clamp_range', None)
        nu_clamp = material_cfg.get('nu_clamp_range', None)
        self.log_E_clamp_range = (
            tuple(float(v) for v in log_E_clamp) if log_E_clamp is not None
            else self.log_E_range
        )
        self.nu_clamp_range = (
            tuple(float(v) for v in nu_clamp) if nu_clamp is not None
            else self.nu_range
        )

    def clamp_materials(self, log_E: Tensor, nu: Tensor) -> Tuple[Tensor, Tensor]:
        return (
            log_E.clamp(self.log_E_clamp_range[0], self.log_E_clamp_range[1]),
            nu.clamp(self.nu_clamp_range[0], self.nu_clamp_range[1]),
        )

    def encode_material_features(self, log_E: Tensor, nu: Tensor) -> Tensor:
        return torch.stack([
            (log_E - self.log_E_clamp_range[0]) / max(self.log_E_clamp_range[1] - self.log_E_clamp_range[0], 1e-8),
            (nu - self.nu_clamp_range[0]) / max(self.nu_clamp_range[1] - self.nu_clamp_range[0], 1e-8),
        ], dim=-1)

    def clone_material_state(self, material_state: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {
            'log_E': material_state['log_E'].clone(),
            'nu': material_state['nu'].clone(),
            'material_features': material_state['material_features'].clone(),
        }

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        num_particles = sample['current_particle_positions'].shape[0]
        device = sample['current_particle_positions'].device

        if self.mode == 'ground_truth_jitter':
            log_E = sample['gt_material_log_E'] + self.log_E_noise_std * torch.randn(num_particles, device=device)
            nu = sample['gt_material_nu'] + self.nu_noise_std * torch.randn(num_particles, device=device)
            log_E = log_E.clamp(self.log_E_range[0], self.log_E_range[1])
            nu = nu.clamp(self.nu_range[0], self.nu_range[1])
        else:
            log_E = torch.empty(num_particles, device=device).uniform_(self.log_E_range[0], self.log_E_range[1])
            nu = torch.empty(num_particles, device=device).uniform_(self.nu_range[0], self.nu_range[1])

        # Add per-particle Gaussian noise to break symmetry across particles
        if self.per_particle_noise:
            log_E = log_E + self.per_particle_log_E_noise_std * torch.randn(num_particles, device=device)
            nu = nu + self.per_particle_nu_noise_std * torch.randn(num_particles, device=device)

        log_E, nu = self.clamp_materials(log_E, nu)
        material_features = self.encode_material_features(log_E, nu)
        return {
            'log_E': log_E,
            'nu': nu,
            'material_features': material_features,
        }


# class DifferentiableRolloutEngine(nn.Module):
#     def __init__(self, rollout_cfg: DictConfig, material_cfg: DictConfig, rollout_steps: int, device: torch.device) -> None:
#         super().__init__()
#         self.rollout_steps = int(rollout_steps)
#         self.device = device
#         self.rollout_cfg = rollout_cfg
#         self.material_guess_initializer = RandomMaterialGuessInitializer(material_cfg)
#         self.use_episode_sim_dt = bool(rollout_cfg.get('use_episode_sim_dt', True))
#         self.default_dt = float(rollout_cfg.get('default_dt', 2.0e-4))
#         self.default_steps_per_frame = int(rollout_cfg.get('default_steps_per_frame', 10))
#         self.num_grids = int(rollout_cfg.get('num_grids', 40))
#         self.clip_bound = float(rollout_cfg.get('clip_bound', 0.5))
#         self.gravity = list(rollout_cfg.get('gravity', [0.0, 0.0, -9.8]))
#         self.damping = float(rollout_cfg.get('damping', 0.999))
#         self.rho = float(rollout_cfg.get('rho', 1000.0))
#         self.ground_height = float(rollout_cfg.get('ground_height', 0.02))
#         self.ground_surface = str(rollout_cfg.get('ground_surface', 'sticky'))
#         self.use_explicit_rigid_contact = bool(rollout_cfg.get('use_explicit_rigid_contact', True))
#         self.use_episode_rigid_contact = bool(rollout_cfg.get('use_episode_rigid_contact', True))
#         self.rigid_surface_fps = int(rollout_cfg.get('rigid_surface_fps', 256))
#         self.rigid_friction = float(rollout_cfg.get('rigid_friction', 0.5))
#         self.rigid_contact_margin = float(rollout_cfg.get('rigid_contact_margin', 0.012))
#         self.rigid_velocity_blend = float(rollout_cfg.get('rigid_velocity_blend', 0.90))
#         self.rigid_stickiness = float(rollout_cfg.get('rigid_stickiness', 0.80))
#         self.rigid_strict_nonpenetration = bool(rollout_cfg.get('rigid_strict_nonpenetration', True))
#         self.rigid_projection_iterations = int(rollout_cfg.get('rigid_projection_iterations', 2))
#         self.rigid_projection_margin = float(rollout_cfg.get('rigid_projection_margin', 1.0e-5))
#         self.max_force_norm = rollout_cfg.get('max_force_norm', None)
#         self.apply_force_every_substep = bool(rollout_cfg.get('apply_force_every_substep', True))

#         elasticity_name = str(rollout_cfg.get('elasticity_model', 'CorotatedElasticity'))
#         plasticity_name = str(rollout_cfg.get('plasticity_model', 'IdentityPlasticity'))
#         self.elasticity_model = get_elasticity_model(elasticity_name).to(device)
#         self.plasticity_model = get_plasticity_model(plasticity_name).to(device)
#         self.elasticity_model.requires_grad_(False)
#         self.plasticity_model.requires_grad_(False)
#         self.elasticity_model.eval()
#         self.plasticity_model.eval()

#     def _resolve_sim_timestep(self, sample: Dict) -> Tuple[float, int]:
#         if self.use_episode_sim_dt:
#             sim_dt = float(sample.get('sim_dt', self.default_dt))
#             steps_per_frame = int(sample.get('steps_per_frame', self.default_steps_per_frame))
#             return sim_dt, max(steps_per_frame, 1)
#         return self.default_dt, max(self.default_steps_per_frame, 1)

#     def _build_solver(self, sample: Dict, init_pos: Tensor, sim_dt: float) -> Tuple[MPMSolver, Dict[str, Tensor]]:
#         solver = MPMSolver(
#             num_grids=self.num_grids,
#             dt=sim_dt,
#             gravity=self.gravity,
#             damping=self.damping,
#             clip_bound_factor=self.clip_bound,
#             device=str(self.device),
#         )
#         state = solver.init_particles(init_pos, rho=self.rho)
#         add_surface_collider(
#             solver,
#             point=[1.0, 1.0, float(sample.get('ground_height', self.ground_height))],
#             normal=[0.0, 0.0, 1.0],
#             surface=self.ground_surface,
#         )
#         return solver, state

#     def _deserialize_primitives(self, primitive_dicts: List[Dict]) -> List[Primitive]:
#         primitives: List[Primitive] = []
#         for item in primitive_dicts:
#             primitives.append(
#                 Primitive(
#                     ptype=str(item['type']),
#                     params=dict(item['params']),
#                     scale=np.asarray(item['scale'], dtype=np.float64),
#                     rotation=np.asarray(item['rotation'], dtype=np.float64),
#                     translation=np.asarray(item['translation'], dtype=np.float64),
#                     material=dict(item.get('material', {})),
#                 )
#             )
#         return primitives

#     def _build_rigid_body(self, sample: Dict, frame_dt: float) -> RigidBody | None:
#         primitive_dicts = sample.get('rigid_body_primitives', None)
#         if not primitive_dicts:
#             return None
#         if sample['current_rigid_points'].numel() == 0:
#             return None

#         primitives = self._deserialize_primitives(primitive_dicts)
#         surface_points = sample_surface_points(
#             primitives,
#             n_fps=int(sample.get('rigid_surface_fps', self.rigid_surface_fps)),
#             device=str(self.device),
#         )
#         rigid_body = RigidBody(primitives, surface_points, device=str(self.device))

#         current_center = sample['current_rigid_center'].detach().cpu().tolist()
#         future_centers = sample['future_rigid_centers'].detach().cpu().tolist()
#         waypoints = [(0.0, current_center)]
#         for step_idx, center in enumerate(future_centers):
#             waypoints.append(((step_idx + 1) * float(frame_dt), center))
#         if len(waypoints) < 2:
#             return None

#         rigid_body.set_waypoint_trajectory(waypoints, z_min=None)
#         return rigid_body

#     def _get_rigid_contact_kwargs(self, sample: Dict) -> Dict:
#         contact_cfg = dict(sample.get('rigid_collision_cfg', {})) if self.use_episode_rigid_contact else {}
#         return {
#             'friction': float(sample.get('rigid_friction', self.rigid_friction)) if self.use_episode_rigid_contact else self.rigid_friction,
#             'contact_margin': float(contact_cfg.get('contact_margin', self.rigid_contact_margin)),
#             'velocity_blend': float(contact_cfg.get('velocity_blend', self.rigid_velocity_blend)),
#             'stickiness': float(contact_cfg.get('stickiness', self.rigid_stickiness)),
#             'strict_nonpenetration': bool(contact_cfg.get('strict_nonpenetration', self.rigid_strict_nonpenetration)),
#             'projection_iterations': int(contact_cfg.get('projection_iterations', self.rigid_projection_iterations)),
#             'projection_margin': float(contact_cfg.get('projection_margin', self.rigid_projection_margin)),
#         }

#     def _clamp_force(self, force: Tensor) -> Tensor:
#         if self.max_force_norm is None:
#             return force
#         max_force_norm = float(self.max_force_norm)
#         force_norm = force.norm(dim=-1, keepdim=True).clamp(min=1e-8)
#         scale = torch.clamp(max_force_norm / force_norm, max=1.0)
#         return force * scale

#     def _clamp_materials(self, log_E: Tensor, nu: Tensor) -> Tuple[Tensor, Tensor]:
#         return self.material_guess_initializer.clamp_materials(log_E, nu)

#     def _material_features(self, log_E: Tensor, nu: Tensor) -> Tensor:
#         return self.material_guess_initializer.encode_material_features(log_E, nu)

#     def _clone_material_state(self, material_state: Dict[str, Tensor]) -> Dict[str, Tensor]:
#         return self.material_guess_initializer.clone_material_state(material_state)

#     def _sanitize_material_state(self, material_state: Dict[str, Tensor]) -> Dict[str, Tensor]:
#         log_E, nu = self._clamp_materials(material_state['log_E'], material_state['nu'])
#         return {
#             'log_E': log_E,
#             'nu': nu,
#             'material_features': self._material_features(log_E, nu),
#         }

#     def _apply_material_delta(self, material_state: Dict[str, Tensor], delta_material_step: Tensor) -> Dict[str, Tensor]:
#         log_E, nu = self._clamp_materials(
#             material_state['log_E'] + delta_material_step[..., 0],
#             material_state['nu'] + delta_material_step[..., 1],
#         )
#         return {
#             'log_E': log_E,
#             'nu': nu,
#             'material_features': self._material_features(log_E, nu),
#         }

#     def forward(
#         self,
#         sample: Dict,
#         material_guess: Dict[str, Tensor],
#         delta_external_forces: Tensor,
#         delta_material_properties: Tensor | None = None,
#     ) -> Dict[str, Tensor]:
#         sim_dt, steps_per_frame = self._resolve_sim_timestep(sample)
#         frame_dt = float(sample.get('frame_dt', sim_dt * steps_per_frame))
#         x = sample['current_particle_positions'].clone()
#         v = torch.zeros_like(sample['current_particle_velocity'])
#         num_particles = x.shape[0]
#         material_state = self._sanitize_material_state(self._clone_material_state(material_guess))
#         solver, state = self._build_solver(sample, init_pos=x, sim_dt=sim_dt)
#         state['v'] = v
#         state['C'] = torch.zeros(num_particles, 3, 3, device=self.device, dtype=x.dtype)
#         state['F'] = torch.eye(3, device=self.device, dtype=x.dtype).unsqueeze(0).repeat(num_particles, 1, 1)

#         if self.use_explicit_rigid_contact:
#             rigid_body = self._build_rigid_body(sample, frame_dt=frame_dt)
#             if rigid_body is not None:
#                 rigid_contact_kwargs = self._get_rigid_contact_kwargs(sample)
#                 add_rigid_body_collider(
#                     solver,
#                     sdf_fn=rigid_body.sdf,
#                     velocity_fn=rigid_body.get_velocity,
#                     bounds_fn=rigid_body.get_world_bounds,
#                     start_time=0.0,
#                     **rigid_contact_kwargs,
#                 )

#         max_steps = min(self.rollout_steps, delta_external_forces.shape[0])
#         delta_external_forces = delta_external_forces[:max_steps]
#         if delta_material_properties is None:
#             delta_material_properties = torch.zeros(max_steps, num_particles, 2, device=x.device, dtype=x.dtype)
#         else:
#             max_steps = min(max_steps, delta_material_properties.shape[0])
#             delta_external_forces = delta_external_forces[:max_steps]
#             delta_material_properties = delta_material_properties[:max_steps]

#         predicted_positions = []
#         predicted_flows = []
#         predicted_material_log_E = []
#         predicted_material_nu = []
#         previous_x = state['x']
#         for frame_idx in range(max_steps):
#             frame_force = self._clamp_force(delta_external_forces[frame_idx])
#             material_state = self._apply_material_delta(material_state, delta_material_properties[frame_idx])
#             current_log_E = material_state['log_E']
#             current_nu = material_state['nu']
#             for _ in range(steps_per_frame):
#                 if self.apply_force_every_substep:
#                     state['v'] = state['v'] + sim_dt * frame_force / solver.model.p_mass
#                 stress = self.elasticity_model(state['F'], current_log_E, current_nu)
#                 state = solver.step(state, stress)
#                 state['F'] = self.plasticity_model(state['F'], current_log_E, current_nu)
#             predicted_positions.append(state['x'])
#             predicted_flows.append(state['x'] - previous_x)
#             predicted_material_log_E.append(current_log_E)
#             predicted_material_nu.append(current_nu)
#             previous_x = state['x']

#         return {
#             'predicted_positions': torch.stack(predicted_positions, dim=0),
#             'predicted_flows': torch.stack(predicted_flows, dim=0),
#             'delta_external_forces': delta_external_forces,
#             'delta_material_properties': delta_material_properties,
#             'predicted_material_log_E': torch.stack(predicted_material_log_E, dim=0),
#             'predicted_material_nu': torch.stack(predicted_material_nu, dim=0),
#             'material_guess': material_guess,
#             'final_material_state': material_state,
#         }


def _point_cloud_bbox_center(points: Tensor) -> Tensor:
    if points.numel() == 0:
        return torch.zeros(3, device=points.device, dtype=points.dtype)
    return 0.5 * (points.min(dim=0).values + points.max(dim=0).values)


class DifferentiableRolloutEngine(nn.Module):
    """Differentiable MPM rollout with optional rigid-body contact replay."""

    def __init__(
        self,
        rollout_cfg: Optional[DictConfig] = None,
        device: str | torch.device = 'cuda',
        dt: float = 2e-4,
        steps_per_frame: int = 10,
        num_grids: int = 40,
        gravity: Tuple[float, float, float] = (0.0, 0.0, -9.8),
        damping: float = 0.999,
        rho: float = 1000.0,
        clip_bound: float = 0.5,
        ground_height: float = 0.02,
        ground_surface: str = 'sticky',
        ground_elasticity: float = 0.5,
        ground_friction: float = 0.3,
        ground_tangent_damping: float = 0.0,
        ground_static_velocity_threshold: float = 0.0,
        elasticity: str = 'CorotatedElasticity',
        plasticity: str = 'IdentityPlasticity',
        use_explicit_rigid_contact: bool = True,
        rigid_surface_fps: int = 256,
        rigid_friction: float = 0.5,
        rigid_contact_margin: float = 0.012,
        rigid_velocity_blend: float = 0.90,
        rigid_stickiness: float = 0.80,
        rigid_strict_nonpenetration: bool = True,
        rigid_projection_iterations: int = 2,
        rigid_projection_margin: float = 1.0e-5,
        max_delta_velocity_norm: Optional[float] = None,
    ) -> None:
        super().__init__()
        cfg = rollout_cfg or {}
        self.device = torch.device(device)
        self.dt = float(cfg.get('default_dt', dt))
        self.steps_per_frame = max(int(cfg.get('default_steps_per_frame', steps_per_frame)), 1)
        self.num_grids = int(cfg.get('num_grids', num_grids))
        self.gravity = list(cfg.get('gravity', gravity))
        self.damping = float(cfg.get('damping', damping))
        self.rho = float(cfg.get('rho', rho))
        self.clip_bound = float(cfg.get('clip_bound', clip_bound))
        self.ground_height = float(cfg.get('ground_height', ground_height))
        self.ground_surface = str(cfg.get('ground_surface', ground_surface))
        self.ground_elasticity = float(cfg.get('ground_elasticity', cfg.get('collide_elas', ground_elasticity)))
        self.ground_friction = float(cfg.get('ground_friction', cfg.get('collide_fric', ground_friction)))
        self.ground_tangent_damping = float(cfg.get('ground_tangent_damping', ground_tangent_damping))
        self.ground_static_velocity_threshold = float(
            cfg.get('ground_static_velocity_threshold', ground_static_velocity_threshold)
        )
        self.use_explicit_rigid_contact = bool(cfg.get('use_explicit_rigid_contact', use_explicit_rigid_contact))
        self.rigid_surface_fps = int(cfg.get('rigid_surface_fps', rigid_surface_fps))
        self.rigid_friction = float(cfg.get('rigid_friction', rigid_friction))
        self.rigid_contact_margin = float(cfg.get('rigid_contact_margin', rigid_contact_margin))
        self.rigid_velocity_blend = float(cfg.get('rigid_velocity_blend', rigid_velocity_blend))
        self.rigid_stickiness = float(cfg.get('rigid_stickiness', rigid_stickiness))
        self.rigid_strict_nonpenetration = bool(cfg.get('rigid_strict_nonpenetration', rigid_strict_nonpenetration))
        self.rigid_projection_iterations = int(cfg.get('rigid_projection_iterations', rigid_projection_iterations))
        self.rigid_projection_margin = float(cfg.get('rigid_projection_margin', rigid_projection_margin))
        self.manipulation_contact_radius = float(cfg.get('manipulation_contact_radius', 0.025))
        self.manipulation_position_blend = float(cfg.get('manipulation_position_blend', 1.0))
        self.manipulation_velocity_blend = float(cfg.get('manipulation_velocity_blend', 1.0))
        self.manipulation_grid_velocity_blend = float(cfg.get('manipulation_grid_velocity_blend', 1.0))
        self.manipulation_controller_grid_contact_radius = float(
            cfg.get('manipulation_controller_grid_contact_radius', 0.0)
        )
        self.manipulation_controller_grid_velocity_blend = float(
            cfg.get('manipulation_controller_grid_velocity_blend', 1.0)
        )
        self.max_delta_velocity_norm = cfg.get('max_delta_velocity_norm', max_delta_velocity_norm)
        legacy_correction_spring_stiffness = cfg.get('correction_spring_stiffness', None)
        legacy_correction_dashpot_damping = cfg.get('correction_dashpot_damping', None)
        self.correction_target_velocity_gain = float(
            cfg.get(
                'correction_target_velocity_gain',
                20.0 if legacy_correction_spring_stiffness is None
                else float(legacy_correction_spring_stiffness) * self.dt,
            )
        )
        self.correction_target_velocity_damping = float(
            cfg.get(
                'correction_target_velocity_damping',
                0.5 if legacy_correction_dashpot_damping is None
                else float(legacy_correction_dashpot_damping) * self.dt,
            )
        )
        self.correction_target_velocity_integral_gain = float(
            cfg.get('correction_target_velocity_integral_gain', 0.0)
        )

        # Correction-rollout-specific overrides (applied only when
        # correction_targets is not None).
        correction_cfg = cfg.get('correction', {}) or {}
        self.correction_mode = str(correction_cfg.get('mode', 'pid'))
        self.correction_dt_scale = float(correction_cfg.get('dt_scale', 1.0))
        self.correction_steps_per_frame_scale = max(
            int(correction_cfg.get('steps_per_frame_scale', 1)), 1,
        )
        _cap = correction_cfg.get('velocity_cap', None)
        self.correction_velocity_cap = None if _cap is None else float(_cap)
        self.correction_spring_omega_n = float(correction_cfg.get('omega_n', 1000.0))
        self.correction_spring_zeta = float(correction_cfg.get('zeta', 2.0))
        self.corotated_rotation_backward = str(
            cfg.get('corotated_rotation_backward', cfg.get('corotated_svd_backward', 'exact'))
        )
        self.corotated_volume_j_backward = str(cfg.get('corotated_volume_j_backward', 'svd'))
        j_clamp = cfg.get('corotated_j_clamp_range', [-1.0e4, 1.0e4])
        self.corotated_j_clamp_min = float(j_clamp[0])
        self.corotated_j_clamp_max = float(j_clamp[1])

        elasticity_name = str(cfg.get('elasticity_model', elasticity))
        plasticity_name = str(cfg.get('plasticity_model', plasticity))
        # Kwargs applied to every plasticity instance the engine builds (by name).
        # Lets the YAML configure parameters like VonMisesPlasticity's `sigma_y` without
        # editing the model class. Cached per-name so a fresh instance created for a
        # per-particle override (via _get_plasticity_model) picks up the same config.
        self._plasticity_kwargs: Dict[str, Dict] = {
            str(k): dict(v) for k, v in dict(cfg.get('plasticity_kwargs', {}) or {}).items()
        }
        self._plasticity_default_name = plasticity_name
        self._elasticity_kwargs: Dict[str, Dict] = {
            str(k): dict(v) for k, v in dict(cfg.get('elasticity_kwargs', {}) or {}).items()
        }
        self.elasticity_model = get_elasticity_model(
            elasticity_name, **self._elasticity_kwargs.get(elasticity_name, {}),
        ).to(self.device)
        self._configure_elasticity_backward(self.elasticity_model)
        self.plasticity_model = get_plasticity_model(
            plasticity_name, **self._plasticity_kwargs.get(plasticity_name, {}),
        ).to(self.device)
        self.elasticity_model.requires_grad_(False)
        self.plasticity_model.requires_grad_(False)
        self.elasticity_model.eval()
        self.plasticity_model.eval()
        self._elasticity_model_cache: Dict[str, nn.Module] = {elasticity_name: self.elasticity_model}
        self._plasticity_model_cache: Dict[str, nn.Module] = {plasticity_name: self.plasticity_model}

    def _configure_elasticity_backward(self, model: nn.Module) -> None:
        configure = getattr(model, 'configure_backward_stabilization', None)
        if configure is None:
            return
        configure(
            rotation_backward_mode=self.corotated_rotation_backward,
            volume_j_backward_mode=self.corotated_volume_j_backward,
            j_clamp_min=self.corotated_j_clamp_min,
            j_clamp_max=self.corotated_j_clamp_max,
        )

    def _get_elasticity_model(self, name: str) -> nn.Module:
        model = self._elasticity_model_cache.get(name)
        if model is None:
            model = get_elasticity_model(name, **self._elasticity_kwargs.get(name, {})).to(self.device)
            self._configure_elasticity_backward(model)
            model.requires_grad_(False)
            model.eval()
            self._elasticity_model_cache[name] = model
        return model

    def _get_plasticity_model(self, name: str) -> nn.Module:
        model = self._plasticity_model_cache.get(name)
        if model is None:
            model = get_plasticity_model(name, **self._plasticity_kwargs.get(name, {})).to(self.device)
            model.requires_grad_(False)
            model.eval()
            self._plasticity_model_cache[name] = model
        return model

    def _compute_stress(
        self,
        F: Tensor,
        log_E: Tensor,
        nu: Tensor,
        material_model_info: Optional[Dict],
    ) -> Tensor:
        if material_model_info is None:
            return self.elasticity_model(F, log_E, nu)

        elasticity_ids = material_model_info.get('elasticity_ids', None)
        elasticity_names = material_model_info.get('elasticity_names', None)
        if elasticity_ids is None or not elasticity_names:
            return self.elasticity_model(F, log_E, nu)

        stress = torch.zeros_like(F)
        for idx, name in enumerate(elasticity_names):
            mask = elasticity_ids == idx
            if not mask.any():
                continue
            model = self._get_elasticity_model(str(name))
            stress[mask] = model(F[mask], log_E[mask], nu[mask])
        return stress

    def _apply_plasticity(
        self,
        F: Tensor,
        log_E: Tensor,
        nu: Tensor,
        material_model_info: Optional[Dict],
    ) -> Tensor:
        if material_model_info is None:
            return self.plasticity_model(F, log_E, nu)

        plasticity_ids = material_model_info.get('plasticity_ids', None)
        plasticity_names = material_model_info.get('plasticity_names', None)
        if plasticity_ids is None or not plasticity_names:
            return self.plasticity_model(F, log_E, nu)

        F_out = F.clone()
        for idx, name in enumerate(plasticity_names):
            mask = plasticity_ids == idx
            if not mask.any():
                continue
            model = self._get_plasticity_model(str(name))
            F_out[mask] = model(F[mask], log_E[mask], nu[mask])
        return F_out

    def _make_solver(self, init_pos: Tensor) -> Tuple[MPMSolver, dict]:
        solver = MPMSolver(
            num_grids=self.num_grids,
            dt=self.dt,
            gravity=self.gravity,
            damping=self.damping,
            clip_bound_factor=self.clip_bound,
            device=str(self.device),
        )
        state = solver.init_particles(init_pos, rho=self.rho)
        add_surface_collider(
            solver,
            point=[1.0, 1.0, self.ground_height],
            normal=[0.0, 0.0, 1.0],
            surface=self.ground_surface,
            friction=self.ground_friction,
            elasticity=self.ground_elasticity,
            tangent_damping=self.ground_tangent_damping,
            static_velocity_threshold=self.ground_static_velocity_threshold,
        )
        return solver, state

    def _deserialize_primitives(self, primitive_dicts: List[Dict]) -> List[Primitive]:
        primitives: List[Primitive] = []
        for item in primitive_dicts:
            primitives.append(
                Primitive(
                    ptype=str(item['type']),
                    params=dict(item['params']),
                    scale=np.asarray(item['scale'], dtype=np.float64),
                    rotation=np.asarray(item['rotation'], dtype=np.float64),
                    translation=np.asarray(item['translation'], dtype=np.float64),
                    material=dict(item.get('material', {})),
                )
            )
        return primitives

    def _build_rigid_body(self, rigid_body_primitives: Optional[List[Dict]], rigid_points: Optional[Tensor]) -> Optional[RigidBody]:
        if not self.use_explicit_rigid_contact or not rigid_body_primitives:
            return None
        if rigid_points is None or rigid_points.numel() == 0 or rigid_points.shape[0] < 2:
            return None

        primitives = self._deserialize_primitives(rigid_body_primitives)
        surface_points = sample_surface_points(
            primitives,
            n_fps=self.rigid_surface_fps,
            device=str(self.device),
        )
        rigid_body = RigidBody(primitives, surface_points, device=str(self.device))

        frame_dt = self.dt * self.steps_per_frame
        waypoints = [(0.0, _point_cloud_bbox_center(rigid_points[0]).detach().cpu().tolist())]
        for step_idx in range(1, rigid_points.shape[0]):
            center = _point_cloud_bbox_center(rigid_points[step_idx]).detach().cpu().tolist()
            waypoints.append((step_idx * frame_dt, center))

        if len(waypoints) < 2:
            return None
        rigid_body.set_waypoint_trajectory(waypoints, z_min=None)
        return rigid_body

    def _get_rigid_contact_kwargs(self, rigid_collision_cfg: Optional[Dict]) -> Dict:
        contact_cfg = dict(rigid_collision_cfg or {})
        return {
            'friction': float(contact_cfg.get('friction', self.rigid_friction)),
            'contact_margin': float(contact_cfg.get('contact_margin', self.rigid_contact_margin)),
            'velocity_blend': float(contact_cfg.get('velocity_blend', self.rigid_velocity_blend)),
            'stickiness': float(contact_cfg.get('stickiness', self.rigid_stickiness)),
            'strict_nonpenetration': bool(contact_cfg.get('strict_nonpenetration', self.rigid_strict_nonpenetration)),
            'projection_iterations': int(contact_cfg.get('projection_iterations', self.rigid_projection_iterations)),
            'projection_margin': float(contact_cfg.get('projection_margin', self.rigid_projection_margin)),
        }

    def _get_manipulation_contact_kwargs(self, rigid_collision_cfg: Optional[Dict]) -> Dict:
        contact_cfg = dict(rigid_collision_cfg or {})
        return {
            'radius': max(float(contact_cfg.get('manipulation_contact_radius', self.manipulation_contact_radius)), 0.0),
            'position_blend': float(np.clip(
                contact_cfg.get('manipulation_position_blend', self.manipulation_position_blend),
                0.0,
                1.0,
            )),
            'velocity_blend': float(np.clip(
                contact_cfg.get('manipulation_velocity_blend', self.manipulation_velocity_blend),
                0.0,
                1.0,
            )),
        }

    def _manipulation_enabled(self, manipulation_indicator: Optional[Tensor]) -> bool:
        if manipulation_indicator is None:
            return False
        if not torch.is_tensor(manipulation_indicator):
            return bool(manipulation_indicator)
        return bool((manipulation_indicator.detach() > 0.5).any().item())

    def _normalize_kinematic_particle_ids(
        self,
        particle_ids: Optional[Tensor],
        device: torch.device,
    ) -> Optional[Tensor]:
        if particle_ids is None:
            return None
        particle_ids = particle_ids.to(device=device, dtype=torch.long)
        if particle_ids.ndim == 1:
            particle_ids = particle_ids.view(1, -1)
        if particle_ids.ndim != 2:
            return None
        if particle_ids.shape[0] > 1:
            particle_ids = particle_ids[:1].expand_as(particle_ids)
        return particle_ids

    def _kinematic_targets_for_substep(
        self,
        state: Dict[str, Tensor],
        rigid_points: Optional[Tensor],
        particle_ids: Optional[Tensor],
        step: int,
        substep: int,
        alpha_offset: int,
    ) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
        if rigid_points is None or particle_ids is None or rigid_points.ndim != 3:
            return None
        if rigid_points.shape[0] <= step + 1 or particle_ids.shape[0] <= step:
            return None
        if rigid_points.shape[1] == 0 or particle_ids.shape[1] == 0:
            return None

        ids = particle_ids[step].to(device=state['x'].device, dtype=torch.long)
        valid = (ids >= 0) & (ids < state['x'].shape[0])
        if not valid.any():
            return None

        contact_start = rigid_points[step].to(device=state['x'].device, dtype=state['x'].dtype)
        contact_end = rigid_points[step + 1].to(device=state['x'].device, dtype=state['x'].dtype)
        ids = ids[valid]
        contact_start = contact_start[valid]
        contact_end = contact_end[valid]

        alpha = float(substep + alpha_offset) / float(max(int(self.steps_per_frame), 1))
        target_x = contact_start + alpha * (contact_end - contact_start)
        frame_dt = max(self.dt * self.steps_per_frame, 1.0e-8)
        target_v = (contact_end - contact_start) / frame_dt

        unique_ids, inverse = torch.unique(ids, sorted=False, return_inverse=True)
        if unique_ids.numel() != ids.numel():
            target_x_accum = target_x.new_zeros(unique_ids.shape[0], 3)
            target_v_accum = target_v.new_zeros(unique_ids.shape[0], 3)
            counts = target_x.new_zeros(unique_ids.shape[0], 1)
            target_x_accum.index_add_(0, inverse, target_x)
            target_v_accum.index_add_(0, inverse, target_v)
            counts.index_add_(0, inverse, torch.ones(ids.shape[0], 1, device=target_x.device, dtype=target_x.dtype))
            counts = counts.clamp_min(1.0)
            target_x = target_x_accum / counts
            target_v = target_v_accum / counts
            ids = unique_ids
        return ids, target_x, target_v

    def _controller_grid_targets_for_substep(
        self,
        controller_grid_points: Optional[Tensor],
        step: int,
        substep: int,
        alpha_offset: int,
    ) -> Optional[Tuple[Tensor, Tensor]]:
        if controller_grid_points is None or controller_grid_points.ndim != 3:
            return None
        if controller_grid_points.shape[0] <= step + 1 or controller_grid_points.shape[1] == 0:
            return None

        contact_start = controller_grid_points[step].to(device=self.device)
        contact_end = controller_grid_points[step + 1].to(device=self.device)
        alpha = float(substep + alpha_offset) / float(max(int(self.steps_per_frame), 1))
        target_x = contact_start + alpha * (contact_end - contact_start)
        frame_dt = max(self.dt * self.steps_per_frame, 1.0e-8)
        target_v = (contact_end - contact_start) / frame_dt
        return target_x, target_v

    def _batched_controller_grid_targets_for_substep(
        self,
        controller_grid_points: Optional[Tensor],
        manipulation_indicator: Optional[Tensor],
        step: int,
        substep: int,
        alpha_offset: int,
        batch_size: int,
    ) -> Optional[Tuple[Tensor, Tensor]]:
        if controller_grid_points is None:
            return None
        if controller_grid_points.ndim == 3:
            controller_grid_points = controller_grid_points.unsqueeze(0).expand(batch_size, -1, -1, -1)
        if controller_grid_points.ndim != 4:
            return None
        if controller_grid_points.shape[1] <= step + 1 or controller_grid_points.shape[2] == 0:
            return None
        if controller_grid_points.shape[0] != batch_size:
            if controller_grid_points.shape[0] == 1:
                controller_grid_points = controller_grid_points.expand(batch_size, -1, -1, -1)
            else:
                return None

        contact_start = controller_grid_points[:, step].to(device=self.device)
        contact_end = controller_grid_points[:, step + 1].to(device=self.device)
        alpha = float(substep + alpha_offset) / float(max(int(self.steps_per_frame), 1))
        target_x = contact_start + alpha * (contact_end - contact_start)
        frame_dt = max(self.dt * self.steps_per_frame, 1.0e-8)
        target_v = (contact_end - contact_start) / frame_dt

        if manipulation_indicator is not None:
            enabled_value = (
                manipulation_indicator.to(device=target_x.device)
                if torch.is_tensor(manipulation_indicator)
                else torch.as_tensor(manipulation_indicator, device=target_x.device)
            )
            enabled = enabled_value.reshape(-1) > 0.5
            if enabled.numel() == 1:
                enabled = enabled.expand(batch_size)
            elif enabled.numel() != batch_size:
                enabled = enabled[:1].expand(batch_size)
            target_v = target_v * enabled.to(dtype=target_v.dtype).view(batch_size, 1, 1)
        return target_x, target_v

    def _apply_kinematic_particle_targets(
        self,
        state: Dict[str, Tensor],
        rigid_points: Optional[Tensor],
        particle_ids: Optional[Tensor],
        step: int,
        substep: int,
        *,
        alpha_offset: int,
    ) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
        targets = self._kinematic_targets_for_substep(
            state,
            rigid_points,
            particle_ids,
            step=step,
            substep=substep,
            alpha_offset=alpha_offset,
        )
        if targets is None:
            return None
        ids, target_x, target_v = targets
        state['x'] = state['x'].clone()
        state['v'] = state['v'].clone()
        state['x'][ids] = target_x
        state['v'][ids] = target_v
        if state.get('C', None) is not None:
            state['C'] = state['C'].clone()
            state['C'][ids] = 0.0
        if state.get('F', None) is not None:
            state['F'] = state['F'].clone()
            eye = torch.eye(3, device=state['F'].device, dtype=state['F'].dtype)
            state['F'][ids] = eye
        return ids, target_x, target_v

    def _zero_stress_for_kinematic_particles(self, stress: Tensor, particle_ids: Optional[Tensor]) -> Tensor:
        if particle_ids is None or particle_ids.numel() == 0:
            return stress
        stress = stress.clone()
        stress[particle_ids] = 0.0
        return stress

    def _normalize_batched_kinematic_particle_ids(
        self,
        particle_ids: Optional[Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[Tensor]:
        if particle_ids is None:
            return None
        particle_ids = particle_ids.to(device=device, dtype=torch.long)
        if particle_ids.ndim == 2:
            if particle_ids.shape[0] > 1:
                particle_ids = particle_ids[:1].expand_as(particle_ids)
            particle_ids = particle_ids.unsqueeze(0).expand(batch_size, -1, -1)
        if particle_ids.ndim != 3:
            return None
        if particle_ids.shape[0] != batch_size:
            if particle_ids.shape[0] == 1:
                particle_ids = particle_ids.expand(batch_size, -1, -1)
            else:
                return None
        if particle_ids.shape[1] > 1:
            particle_ids = particle_ids[:, :1, :].expand_as(particle_ids)
        return particle_ids

    def _apply_batched_kinematic_particle_targets(
        self,
        state: Dict[str, Tensor],
        rigid_points: Optional[Tensor],
        particle_ids: Optional[Tensor],
        manipulation_indicator: Optional[Tensor],
        step: int,
        substep: int,
        *,
        alpha_offset: int,
    ) -> Optional[Tensor]:
        if rigid_points is None or particle_ids is None or rigid_points.ndim != 4 or particle_ids.ndim != 3:
            return None
        if rigid_points.shape[1] <= step + 1 or particle_ids.shape[1] <= step:
            return None
        batch_size, num_particles = state['x'].shape[:2]

        if manipulation_indicator is None:
            enabled = torch.ones(batch_size, dtype=torch.bool, device=state['x'].device)
        else:
            enabled_value = (
                manipulation_indicator.to(device=state['x'].device)
                if torch.is_tensor(manipulation_indicator)
                else torch.as_tensor(manipulation_indicator, device=state['x'].device)
            )
            enabled = enabled_value.reshape(-1) > 0.5
            if enabled.numel() == 1:
                enabled = enabled.expand(batch_size)
            elif enabled.numel() != batch_size:
                enabled = enabled[:1].expand(batch_size)
        if not enabled.any():
            return None

        all_ids = []
        state['x'] = state['x'].clone()
        state['v'] = state['v'].clone()
        if state.get('C', None) is not None:
            state['C'] = state['C'].clone()
        if state.get('F', None) is not None:
            state['F'] = state['F'].clone()
            eye = torch.eye(3, device=state['F'].device, dtype=state['F'].dtype)
        else:
            eye = None

        alpha = float(substep + alpha_offset) / float(max(int(self.steps_per_frame), 1))
        frame_dt = max(self.dt * self.steps_per_frame, 1.0e-8)
        for batch_idx in range(batch_size):
            if not bool(enabled[batch_idx].item()):
                continue
            ids = particle_ids[batch_idx, step].to(device=state['x'].device, dtype=torch.long)
            valid = (ids >= 0) & (ids < num_particles)
            if not valid.any():
                continue
            ids = ids[valid]
            start = rigid_points[batch_idx, step].to(device=state['x'].device, dtype=state['x'].dtype)[valid]
            end = rigid_points[batch_idx, step + 1].to(device=state['x'].device, dtype=state['x'].dtype)[valid]
            target_x = start + alpha * (end - start)
            target_v = (end - start) / frame_dt
            unique_ids, inverse = torch.unique(ids, sorted=False, return_inverse=True)
            if unique_ids.numel() != ids.numel():
                target_x_accum = target_x.new_zeros(unique_ids.shape[0], 3)
                target_v_accum = target_v.new_zeros(unique_ids.shape[0], 3)
                counts = target_x.new_zeros(unique_ids.shape[0], 1)
                target_x_accum.index_add_(0, inverse, target_x)
                target_v_accum.index_add_(0, inverse, target_v)
                counts.index_add_(0, inverse, torch.ones(ids.shape[0], 1, device=target_x.device, dtype=target_x.dtype))
                target_x = target_x_accum / counts.clamp_min(1.0)
                target_v = target_v_accum / counts.clamp_min(1.0)
                ids = unique_ids
            state['x'][batch_idx, ids] = target_x
            state['v'][batch_idx, ids] = target_v
            if state.get('C', None) is not None:
                state['C'][batch_idx, ids] = 0.0
            if state.get('F', None) is not None and eye is not None:
                state['F'][batch_idx, ids] = eye
            all_ids.append((batch_idx, ids))
        if not all_ids:
            return None
        mask = torch.zeros(batch_size, num_particles, dtype=torch.bool, device=state['x'].device)
        for batch_idx, ids in all_ids:
            mask[batch_idx, ids] = True
        return mask

    def _zero_batched_stress_for_kinematic_particles(self, stress: Tensor, mask: Optional[Tensor]) -> Tensor:
        if mask is None or not mask.any():
            return stress
        stress = stress.clone()
        stress[mask] = 0.0
        return stress

    def _add_manipulation_grid_contacts(
        self,
        solver: MPMSolver,
        rigid_points: Optional[Tensor],
        *,
        radius: float,
        position_blend: float,
        velocity_blend: float,
    ) -> None:
        del position_blend
        if rigid_points is None or radius <= 0.0 or rigid_points.ndim != 3:
            return
        if rigid_points.shape[0] < 2 or rigid_points.shape[1] == 0:
            return

        frame_dt = max(self.dt * self.steps_per_frame, 1.0e-8)
        radius = float(radius)
        velocity_blend = float(velocity_blend)
        rigid_points = rigid_points.detach()

        def force_contact_grid(model_ref) -> None:
            substep_index = int(round(float(model_ref.time) / max(float(model_ref.dt), 1.0e-12)))
            frame_step = substep_index // max(int(self.steps_per_frame), 1)
            substep = substep_index % max(int(self.steps_per_frame), 1)
            if frame_step < 0 or frame_step + 1 >= rigid_points.shape[0]:
                return

            contact_start = rigid_points[frame_step].to(device=model_ref.grid_mv.device, dtype=model_ref.grid_mv.dtype)
            contact_end = rigid_points[frame_step + 1].to(device=model_ref.grid_mv.device, dtype=model_ref.grid_mv.dtype)
            alpha = float(substep + 1) / float(max(int(self.steps_per_frame), 1))
            contact_interp = contact_start + alpha * (contact_end - contact_start)
            contact_velocity = (contact_end - contact_start) / frame_dt

            grid_pos = model_ref.grid_x.to(device=model_ref.grid_mv.device, dtype=model_ref.grid_mv.dtype) * float(model_ref.dx)
            effective_radius = radius + 2.0 * float(model_ref.dx)
            distances = torch.cdist(grid_pos.unsqueeze(0), contact_interp.unsqueeze(0)).squeeze(0)
            nearest_dist, nearest_idx = distances.min(dim=-1)
            active = (nearest_dist <= effective_radius) & (model_ref.grid_m > 1.0e-15)
            if not active.any():
                return
            target_velocity = contact_velocity.index_select(0, nearest_idx[active])
            model_ref.grid_mv[active] = (
                (1.0 - velocity_blend) * model_ref.grid_mv[active]
                + velocity_blend * target_velocity
            )

        solver.post_grid_process.append(force_contact_grid)

    def _add_batched_manipulation_grid_contacts(
        self,
        solver: BatchedMPMSolver,
        rigid_points: Optional[Tensor],
        manipulation_indicator: Optional[Tensor],
        *,
        radius: float,
        position_blend: float,
        velocity_blend: float,
    ) -> None:
        del position_blend
        if rigid_points is None or radius <= 0.0 or rigid_points.ndim != 4:
            return
        if rigid_points.shape[1] < 2 or rigid_points.shape[2] == 0:
            return

        batch_size = int(rigid_points.shape[0])
        if manipulation_indicator is None:
            enabled_template = torch.zeros(batch_size, dtype=torch.bool, device=rigid_points.device)
        else:
            enabled_value = (
                manipulation_indicator.to(device=rigid_points.device)
                if torch.is_tensor(manipulation_indicator)
                else torch.as_tensor(manipulation_indicator, device=rigid_points.device)
            )
            enabled_template = enabled_value.reshape(-1) > 0.5
            if enabled_template.numel() == 1:
                enabled_template = enabled_template.expand(batch_size)
            elif enabled_template.numel() != batch_size:
                enabled_template = enabled_template[:1].expand(batch_size)
        if not enabled_template.any():
            return

        frame_dt = max(self.dt * self.steps_per_frame, 1.0e-8)
        radius = float(radius)
        velocity_blend = float(velocity_blend)
        rigid_points = rigid_points.detach()

        def force_contact_grid(model_ref) -> None:
            substep_index = int(round(float(model_ref.time) / max(float(model_ref.dt), 1.0e-12)))
            frame_step = substep_index // max(int(self.steps_per_frame), 1)
            substep = substep_index % max(int(self.steps_per_frame), 1)
            if frame_step < 0 or frame_step + 1 >= rigid_points.shape[1]:
                return

            enabled = enabled_template.to(device=model_ref.grid_mv.device)
            if not enabled.any():
                return

            contact_start = rigid_points[:, frame_step].to(device=model_ref.grid_mv.device, dtype=model_ref.grid_mv.dtype)
            contact_end = rigid_points[:, frame_step + 1].to(device=model_ref.grid_mv.device, dtype=model_ref.grid_mv.dtype)
            alpha = float(substep + 1) / float(max(int(self.steps_per_frame), 1))
            contact_interp = contact_start + alpha * (contact_end - contact_start)
            contact_velocity = (contact_end - contact_start) / frame_dt

            grid_pos = (
                model_ref.batched_grid_x.to(device=model_ref.grid_mv.device, dtype=model_ref.grid_mv.dtype)
                * float(model_ref.dx)
            ).reshape(model_ref.batch_size, model_ref.num_grid_nodes, 3)
            effective_radius = radius + 2.0 * float(model_ref.dx)
            distances = torch.cdist(grid_pos, contact_interp)
            nearest_dist, nearest_idx = distances.min(dim=-1)
            active = (
                (nearest_dist <= effective_radius)
                & (model_ref.grid_m.reshape(model_ref.batch_size, model_ref.num_grid_nodes) > 1.0e-15)
                & enabled.unsqueeze(-1)
            )
            if not active.any():
                return
            target_velocity = contact_velocity.gather(
                1,
                nearest_idx.unsqueeze(-1).expand(-1, -1, 3),
            )
            flat_active = active.reshape(-1)
            flat_target = target_velocity.reshape(-1, 3)
            model_ref.grid_mv[flat_active] = (
                (1.0 - velocity_blend) * model_ref.grid_mv[flat_active]
                + velocity_blend * flat_target[flat_active]
            )

        solver.post_grid_process.append(force_contact_grid)

    def _clamp_delta_velocity(self, delta_velocity: Tensor) -> Tensor:
        # if self.max_delta_velocity_norm is None:
        #     return delta_velocity
        # max_delta_velocity_norm = float(self.max_delta_velocity_norm)
        # delta_velocity_norm = delta_velocity.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        # scale = torch.clamp(max_delta_velocity_norm / delta_velocity_norm, max=1.0)
        # return delta_velocity * scale
        return delta_velocity

    def _compute_target_velocity_command(
        self,
        positions: Tensor,
        velocities: Tensor,
        correction_targets: Optional[Tensor],
        correction_target_mask: Optional[Tensor],
        correction_error_integral: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if correction_targets is None:
            return torch.zeros_like(positions), correction_error_integral

        position_error = correction_targets - positions

        if self.correction_mode == 'spring':
            # Force-based spring-dashpot: additive velocity update.
            # dv = dt · (ω_n²·err − 2·ω_n·ζ·v)
            k_eff = self.correction_spring_omega_n * self.correction_spring_omega_n
            c_eff = 2.0 * self.correction_spring_omega_n * self.correction_spring_zeta
            dv = (k_eff * position_error - c_eff * velocities) * self.dt
            commanded_velocity = velocities + dv
        else:
            # PID (velocity replacement).
            if correction_error_integral is None:
                correction_error_integral = torch.zeros_like(position_error)
            correction_error_integral = correction_error_integral + position_error * self.dt
            commanded_velocity = (
                self.correction_target_velocity_gain * position_error
                + self.correction_target_velocity_integral_gain * correction_error_integral
                - self.correction_target_velocity_damping * velocities
            )

        if self.correction_velocity_cap is not None:
            norm = commanded_velocity.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = torch.clamp(self.correction_velocity_cap / norm, max=1.0)
            commanded_velocity = commanded_velocity * scale

        if correction_target_mask is not None:
            mask = correction_target_mask.unsqueeze(-1).to(dtype=positions.dtype)
            commanded_velocity = commanded_velocity * mask
            if correction_error_integral is not None:
                correction_error_integral = correction_error_integral * mask
        return self._clamp_delta_velocity(commanded_velocity), correction_error_integral

    def forward(
        self,
        positions: Tensor,
        velocities: Tensor,
        F: Tensor,
        delta_velocities: Tensor,
        log_E: Tensor,
        nu: Tensor,
        C: Optional[Tensor] = None,
        material_model_info: Optional[Dict] = None,
        rigid_points: Optional[Tensor] = None,
        rigid_collision_cfg: Optional[Dict] = None,
        rigid_body_primitives: Optional[List[Dict]] = None,
        direct_velocity: bool = False,
        correction_targets: Optional[Tensor] = None,
        correction_target_mask: Optional[Tensor] = None,
        manipulation_indicator: Optional[Tensor] = None,
        manipulation_contact_particle_ids: Optional[Tensor] = None,
        controller_grid_points: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        # When correction_targets is set, optionally run with finer dt / more
        # substeps per frame to raise the CFL velocity ceiling during the
        # corrective pass. Pure-MPM calls (correction_targets=None) keep the
        # configured simulation timing.
        _saved_dt = self.dt
        _saved_spf = self.steps_per_frame
        if correction_targets is not None and (
            self.correction_dt_scale != 1.0 or self.correction_steps_per_frame_scale != 1
        ):
            self.dt = _saved_dt * self.correction_dt_scale
            self.steps_per_frame = max(int(_saved_spf * self.correction_steps_per_frame_scale), 1)
        try:
            return self._forward_impl(
                positions, velocities, F, delta_velocities, log_E, nu,
                C=C,
                material_model_info=material_model_info,
                rigid_points=rigid_points,
                rigid_collision_cfg=rigid_collision_cfg,
                rigid_body_primitives=rigid_body_primitives,
                direct_velocity=direct_velocity,
                correction_targets=correction_targets,
                correction_target_mask=correction_target_mask,
                manipulation_indicator=manipulation_indicator,
                manipulation_contact_particle_ids=manipulation_contact_particle_ids,
                controller_grid_points=controller_grid_points,
            )
        finally:
            self.dt = _saved_dt
            self.steps_per_frame = _saved_spf

    def _forward_impl(
        self,
        positions: Tensor,
        velocities: Tensor,
        F: Tensor,
        delta_velocities: Tensor,
        log_E: Tensor,
        nu: Tensor,
        C: Optional[Tensor] = None,
        material_model_info: Optional[Dict] = None,
        rigid_points: Optional[Tensor] = None,
        rigid_collision_cfg: Optional[Dict] = None,
        rigid_body_primitives: Optional[List[Dict]] = None,
        direct_velocity: bool = False,
        correction_targets: Optional[Tensor] = None,
        correction_target_mask: Optional[Tensor] = None,
        manipulation_indicator: Optional[Tensor] = None,
        manipulation_contact_particle_ids: Optional[Tensor] = None,
        controller_grid_points: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        solver, state = self._make_solver(positions)
        state['v'] = velocities
        state['C'] = torch.zeros_like(F) if C is None else C
        state['F'] = F

        if correction_targets is not None:
            correction_targets = correction_targets.to(device=positions.device, dtype=positions.dtype)
            if correction_target_mask is None:
                correction_target_mask = torch.ones(
                    positions.shape[0], dtype=torch.bool, device=positions.device,
                )
            else:
                correction_target_mask = correction_target_mask.to(device=positions.device).bool()
        if controller_grid_points is not None:
            controller_grid_points = controller_grid_points.to(device=positions.device, dtype=positions.dtype)

        rigid_body = self._build_rigid_body(rigid_body_primitives, rigid_points)
        if rigid_body is not None:
            add_rigid_body_collider(
                solver,
                sdf_fn=rigid_body.sdf,
                velocity_fn=rigid_body.get_velocity,
                bounds_fn=rigid_body.get_world_bounds,
                start_time=0.0,
                **self._get_rigid_contact_kwargs(rigid_collision_cfg),
            )

        total_steps = delta_velocities.shape[0]
        manipulation_active = self._manipulation_enabled(manipulation_indicator)
        kinematic_particle_ids = (
            self._normalize_kinematic_particle_ids(manipulation_contact_particle_ids, positions.device)
            if manipulation_active
            else None
        )
        predicted_positions = []
        predicted_flows = []
        prev_x = state['x']
        target_delta_velocity_norm_sum = positions.new_zeros(())
        target_delta_velocity_count = 0
        correction_error_integral = None if correction_targets is None else torch.zeros_like(positions)

        for step in range(total_steps):
            frame_delta_velocity = self._clamp_delta_velocity(delta_velocities[step])
            if direct_velocity:
                state['v'] = frame_delta_velocity.clone()
            else:
                state['v'] = state['v'] + frame_delta_velocity
            for substep_idx in range(self.steps_per_frame):
                if correction_targets is not None:
                    target_velocity_command, correction_error_integral = self._compute_target_velocity_command(
                        state['x'],
                        state['v'],
                        correction_targets,
                        correction_target_mask,
                        correction_error_integral,
                    )
                    if correction_target_mask is None:
                        state['v'] = target_velocity_command
                    else:
                        state['v'] = torch.where(
                            correction_target_mask.unsqueeze(-1),
                            target_velocity_command,
                            state['v'],
                        )
                    target_delta_velocity_norm_sum = (
                        target_delta_velocity_norm_sum + target_velocity_command.norm(dim=-1).mean()
                    )
                    target_delta_velocity_count += 1
                kinematic_ids = None
                kinematic_target_v = None
                controller_grid_x = None
                controller_grid_v = None
                if kinematic_particle_ids is not None:
                    kinematic_targets = self._apply_kinematic_particle_targets(
                        state,
                        rigid_points,
                        kinematic_particle_ids,
                        step,
                        substep_idx,
                        alpha_offset=0,
                    )
                    if kinematic_targets is not None:
                        kinematic_ids, _, kinematic_target_v = kinematic_targets
                controller_grid_targets = self._controller_grid_targets_for_substep(
                    controller_grid_points,
                    step,
                    substep_idx,
                    alpha_offset=1,
                )
                if controller_grid_targets is not None:
                    controller_grid_x, controller_grid_v = controller_grid_targets
                stress = self._compute_stress(state['F'], log_E, nu, material_model_info)
                stress = self._zero_stress_for_kinematic_particles(stress, kinematic_ids)
                x, v, C, F_new = solver.model.p2g2p(
                    state['x'],
                    state['v'],
                    state['C'],
                    state['F'],
                    stress,
                    handle_particle_ids=kinematic_ids,
                    handle_target_v=kinematic_target_v,
                    handle_grid_velocity_blend=self.manipulation_grid_velocity_blend if kinematic_ids is not None else 0.0,
                    controller_grid_points=controller_grid_x,
                    controller_grid_target_v=controller_grid_v,
                    controller_grid_contact_radius=self.manipulation_controller_grid_contact_radius,
                    controller_grid_velocity_blend=self.manipulation_controller_grid_velocity_blend,
                )
                state = {'x': x, 'v': v, 'C': C, 'F': F_new}
                state['F'] = self._apply_plasticity(state['F'], log_E, nu, material_model_info)
                for operation in solver.post_step_process:
                    operation(solver, state)
                if kinematic_particle_ids is not None:
                    self._apply_kinematic_particle_targets(
                        state,
                        rigid_points,
                        kinematic_particle_ids,
                        step,
                        substep_idx,
                        alpha_offset=1,
                    )
            predicted_positions.append(state['x'])
            predicted_flows.append(state['x'] - prev_x)
            prev_x = state['x']

        return {
            'predicted_positions': torch.stack(predicted_positions, dim=0),
            'predicted_flows': torch.stack(predicted_flows, dim=0),
            'final_velocity': state['v'],
            'final_C': state['C'],
            'final_deformation_gradient': state['F'],
            'mean_target_delta_velocity_norm': (
                target_delta_velocity_norm_sum / max(target_delta_velocity_count, 1)
            ),
        }


class BatchedDifferentiableRolloutEngine(DifferentiableRolloutEngine):
    """Differentiable rollout engine with a batched MPM backend.

    Inputs with shape ``(N, 3)`` keep the exact original behavior by delegating
    to :class:`DifferentiableRolloutEngine`.  Inputs with shape ``(B, N, 3)``
    run one independent grid per batch element through :class:`BatchedMPMSolver`
    and return tensors shaped ``(T, B, N, ...)``.
    """

    def _make_batched_solver(self, init_pos: Tensor) -> Tuple[BatchedMPMSolver, dict]:
        solver = BatchedMPMSolver(
            num_grids=self.num_grids,
            dt=self.dt,
            gravity=self.gravity,
            damping=self.damping,
            clip_bound_factor=self.clip_bound,
            device=str(self.device),
        )
        state = solver.init_particles(init_pos, rho=self.rho)
        add_batched_surface_collider(
            solver,
            point=[1.0, 1.0, self.ground_height],
            normal=[0.0, 0.0, 1.0],
            surface=self.ground_surface,
            friction=self.ground_friction,
            elasticity=self.ground_elasticity,
            tangent_damping=self.ground_tangent_damping,
            static_velocity_threshold=self.ground_static_velocity_threshold,
        )
        return solver, state

    def _build_batched_rigid_body(
        self,
        rigid_body_primitives: Optional[List[Dict]],
        rigid_points: Optional[Tensor],
    ) -> Optional[BatchedRigidBody]:
        if not self.use_explicit_rigid_contact or not rigid_body_primitives:
            return None
        if rigid_points is None or rigid_points.numel() == 0 or rigid_points.ndim != 4:
            return None
        if rigid_points.shape[2] < 2:
            return None

        primitives = self._deserialize_primitives(rigid_body_primitives)
        surface_points = sample_surface_points(
            primitives,
            n_fps=self.rigid_surface_fps,
            device=str(self.device),
        )
        rigid_body = BatchedRigidBody(primitives, surface_points, device=str(self.device))

        frame_dt = self.dt * self.steps_per_frame
        centers = 0.5 * (
            rigid_points.amin(dim=2).detach()
            + rigid_points.amax(dim=2).detach()
        )
        times = torch.arange(
            rigid_points.shape[1],
            device=rigid_points.device,
            dtype=rigid_points.dtype,
        ) * float(frame_dt)
        rigid_body.set_waypoint_trajectory(times, centers, z_min=None)
        return rigid_body

    def _slice_material_model_info(self, material_model_info: Optional[Dict], index: int) -> Optional[Dict]:
        if material_model_info is None:
            return None
        sliced = dict(material_model_info)
        for key in ('elasticity_ids', 'plasticity_ids'):
            value = sliced.get(key, None)
            if torch.is_tensor(value) and value.ndim >= 2:
                sliced[key] = value[index]
        return sliced

    def _flatten_material_model_info(self, material_model_info: Optional[Dict], batch_size: int) -> Optional[Dict]:
        if material_model_info is None:
            return None
        flattened = dict(material_model_info)
        for key in ('elasticity_ids', 'plasticity_ids'):
            value = flattened.get(key, None)
            if torch.is_tensor(value):
                if value.ndim == 1:
                    flattened[key] = value.unsqueeze(0).expand(batch_size, -1).reshape(-1)
                elif value.ndim >= 2:
                    flattened[key] = value.reshape(-1)
        return flattened

    def _compute_batched_stress(
        self,
        F: Tensor,
        log_E: Tensor,
        nu: Tensor,
        material_model_info: Optional[Dict],
    ) -> Tensor:
        batch_size, num_particles = F.shape[:2]
        stress = self._compute_stress(
            F.reshape(batch_size * num_particles, 3, 3),
            log_E.reshape(batch_size * num_particles),
            nu.reshape(batch_size * num_particles),
            self._flatten_material_model_info(material_model_info, batch_size),
        )
        return stress.reshape(batch_size, num_particles, 3, 3)

    def _apply_batched_plasticity(
        self,
        F: Tensor,
        log_E: Tensor,
        nu: Tensor,
        material_model_info: Optional[Dict],
    ) -> Tensor:
        batch_size, num_particles = F.shape[:2]
        F_out = self._apply_plasticity(
            F.reshape(batch_size * num_particles, 3, 3),
            log_E.reshape(batch_size * num_particles),
            nu.reshape(batch_size * num_particles),
            self._flatten_material_model_info(material_model_info, batch_size),
        )
        return F_out.reshape(batch_size, num_particles, 3, 3)

    def _should_fallback_to_unbatched(
        self,
        rigid_points: Optional[Tensor],
        rigid_body_primitives: Optional[List[Dict]],
    ) -> bool:
        if not self.use_explicit_rigid_contact:
            return False
        if rigid_points is None:
            return False
        if rigid_points.numel() == 0:
            return False
        if not rigid_body_primitives:
            return False
        primitives_is_batched = (
            isinstance(rigid_body_primitives, (list, tuple))
            and len(rigid_body_primitives) > 0
            and isinstance(rigid_body_primitives[0], (list, tuple))
        )
        return primitives_is_batched or rigid_points.ndim != 4

    def _forward_batched_by_loop(
        self,
        positions: Tensor,
        velocities: Tensor,
        F: Tensor,
        delta_velocities: Tensor,
        log_E: Tensor,
        nu: Tensor,
        C: Optional[Tensor],
        material_model_info: Optional[Dict],
        rigid_points: Optional[Tensor],
        rigid_collision_cfg: Optional[Dict],
        rigid_body_primitives: Optional[List[Dict]],
        direct_velocity: bool,
        correction_targets: Optional[Tensor],
        correction_target_mask: Optional[Tensor],
        manipulation_indicator: Optional[Tensor],
        manipulation_contact_particle_ids: Optional[Tensor],
        controller_grid_points: Optional[Tensor],
    ) -> Dict[str, Tensor]:
        outputs: List[Dict[str, Tensor]] = []
        batch_size = int(positions.shape[0])
        primitives_is_batched = (
            isinstance(rigid_body_primitives, (list, tuple))
            and len(rigid_body_primitives) == batch_size
            and (batch_size == 0 or isinstance(rigid_body_primitives[0], (list, tuple)))
        )
        collision_is_batched = (
            isinstance(rigid_collision_cfg, (list, tuple))
            and len(rigid_collision_cfg) == batch_size
        )
        for batch_idx in range(batch_size):
            outputs.append(
                super().forward(
                    positions[batch_idx],
                    velocities[batch_idx],
                    F[batch_idx],
                    delta_velocities[:, batch_idx],
                    log_E[batch_idx],
                    nu[batch_idx],
                    C=None if C is None else C[batch_idx],
                    material_model_info=self._slice_material_model_info(material_model_info, batch_idx),
                    rigid_points=None if rigid_points is None else rigid_points[batch_idx],
                    rigid_collision_cfg=(
                        rigid_collision_cfg[batch_idx] if collision_is_batched else rigid_collision_cfg
                    ),
                    rigid_body_primitives=(
                        list(rigid_body_primitives[batch_idx]) if primitives_is_batched else rigid_body_primitives
                    ),
                    direct_velocity=direct_velocity,
                    correction_targets=None if correction_targets is None else correction_targets[batch_idx],
                    correction_target_mask=None if correction_target_mask is None else correction_target_mask[batch_idx],
                    manipulation_indicator=(
                        None if manipulation_indicator is None
                        else manipulation_indicator.reshape(-1)[batch_idx]
                        if torch.is_tensor(manipulation_indicator) and manipulation_indicator.numel() > 1
                        else manipulation_indicator
                    ),
                    manipulation_contact_particle_ids=(
                        None
                        if manipulation_contact_particle_ids is None
                        else manipulation_contact_particle_ids[batch_idx]
                        if manipulation_contact_particle_ids.ndim == 3
                        else manipulation_contact_particle_ids
                    ),
                    controller_grid_points=(
                        None
                        if controller_grid_points is None
                        else controller_grid_points[batch_idx]
                        if controller_grid_points.ndim == 4
                        else controller_grid_points
                    ),
                )
            )
        return {
            'predicted_positions': torch.stack([item['predicted_positions'] for item in outputs], dim=1),
            'predicted_flows': torch.stack([item['predicted_flows'] for item in outputs], dim=1),
            'final_velocity': torch.stack([item['final_velocity'] for item in outputs], dim=0),
            'final_C': torch.stack([item['final_C'] for item in outputs], dim=0),
            'final_deformation_gradient': torch.stack(
                [item['final_deformation_gradient'] for item in outputs],
                dim=0,
            ),
            'mean_target_delta_velocity_norm': torch.stack(
                [item['mean_target_delta_velocity_norm'] for item in outputs]
            ).mean(),
        }

    def forward(
        self,
        positions: Tensor,
        velocities: Tensor,
        F: Tensor,
        delta_velocities: Tensor,
        log_E: Tensor,
        nu: Tensor,
        C: Optional[Tensor] = None,
        material_model_info: Optional[Dict] = None,
        rigid_points: Optional[Tensor] = None,
        rigid_collision_cfg: Optional[Dict] = None,
        rigid_body_primitives: Optional[List[Dict]] = None,
        direct_velocity: bool = False,
        correction_targets: Optional[Tensor] = None,
        correction_target_mask: Optional[Tensor] = None,
        manipulation_indicator: Optional[Tensor] = None,
        manipulation_contact_particle_ids: Optional[Tensor] = None,
        controller_grid_points: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if positions.ndim == 2:
            return super().forward(
                positions,
                velocities,
                F,
                delta_velocities,
                log_E,
                nu,
                C=C,
                material_model_info=material_model_info,
                rigid_points=rigid_points,
                rigid_collision_cfg=rigid_collision_cfg,
                rigid_body_primitives=rigid_body_primitives,
                direct_velocity=direct_velocity,
                correction_targets=correction_targets,
                correction_target_mask=correction_target_mask,
                manipulation_indicator=manipulation_indicator,
                manipulation_contact_particle_ids=manipulation_contact_particle_ids,
                controller_grid_points=controller_grid_points,
            )
        if positions.ndim != 3:
            raise ValueError(f'positions must have shape (N, 3) or (B, N, 3), got {tuple(positions.shape)}')

        _saved_dt = self.dt
        _saved_spf = self.steps_per_frame
        if correction_targets is not None and (
            self.correction_dt_scale != 1.0 or self.correction_steps_per_frame_scale != 1
        ):
            self.dt = _saved_dt * self.correction_dt_scale
            self.steps_per_frame = max(int(_saved_spf * self.correction_steps_per_frame_scale), 1)
        try:
            if self._should_fallback_to_unbatched(rigid_points, rigid_body_primitives):
                return self._forward_batched_by_loop(
                    positions,
                    velocities,
                    F,
                    delta_velocities,
                    log_E,
                    nu,
                    C,
                    material_model_info,
                    rigid_points,
                    rigid_collision_cfg,
                    rigid_body_primitives,
                    direct_velocity,
                    correction_targets,
                    correction_target_mask,
                    manipulation_indicator,
                    manipulation_contact_particle_ids,
                    controller_grid_points,
                )
            return self._forward_batched_impl(
                positions,
                velocities,
                F,
                delta_velocities,
                log_E,
                nu,
                C=C,
                material_model_info=material_model_info,
                rigid_points=rigid_points,
                rigid_collision_cfg=rigid_collision_cfg,
                rigid_body_primitives=rigid_body_primitives,
                direct_velocity=direct_velocity,
                correction_targets=correction_targets,
                correction_target_mask=correction_target_mask,
                manipulation_indicator=manipulation_indicator,
                manipulation_contact_particle_ids=manipulation_contact_particle_ids,
                controller_grid_points=controller_grid_points,
            )
        finally:
            self.dt = _saved_dt
            self.steps_per_frame = _saved_spf

    def _forward_batched_impl(
        self,
        positions: Tensor,
        velocities: Tensor,
        F: Tensor,
        delta_velocities: Tensor,
        log_E: Tensor,
        nu: Tensor,
        C: Optional[Tensor] = None,
        material_model_info: Optional[Dict] = None,
        rigid_points: Optional[Tensor] = None,
        rigid_collision_cfg: Optional[Dict] = None,
        rigid_body_primitives: Optional[List[Dict]] = None,
        direct_velocity: bool = False,
        correction_targets: Optional[Tensor] = None,
        correction_target_mask: Optional[Tensor] = None,
        manipulation_indicator: Optional[Tensor] = None,
        manipulation_contact_particle_ids: Optional[Tensor] = None,
        controller_grid_points: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        batch_size = int(positions.shape[0])
        solver, state = self._make_batched_solver(positions)
        state['v'] = velocities
        state['C'] = torch.zeros_like(F) if C is None else C
        state['F'] = F

        rigid_body = self._build_batched_rigid_body(rigid_body_primitives, rigid_points)
        if rigid_body is not None:
            add_batched_rigid_body_collider(
                solver,
                rigid_body,
                start_time=0.0,
                **self._get_rigid_contact_kwargs(rigid_collision_cfg),
            )

        if correction_targets is not None:
            correction_targets = correction_targets.to(device=positions.device, dtype=positions.dtype)
            if correction_target_mask is None:
                correction_target_mask = torch.ones(
                    positions.shape[:2], dtype=torch.bool, device=positions.device,
                )
            else:
                correction_target_mask = correction_target_mask.to(device=positions.device).bool()
        if controller_grid_points is not None:
            controller_grid_points = controller_grid_points.to(device=positions.device, dtype=positions.dtype)

        total_steps = delta_velocities.shape[0]
        kinematic_particle_ids = self._normalize_batched_kinematic_particle_ids(
            manipulation_contact_particle_ids,
            batch_size=batch_size,
            device=positions.device,
        )
        predicted_positions = []
        predicted_flows = []
        prev_x = state['x']
        target_delta_velocity_norm_sum = positions.new_zeros(())
        target_delta_velocity_count = 0
        correction_error_integral = None if correction_targets is None else torch.zeros_like(positions)

        for step in range(total_steps):
            frame_delta_velocity = self._clamp_delta_velocity(delta_velocities[step])
            if direct_velocity:
                state['v'] = frame_delta_velocity.clone()
            else:
                state['v'] = state['v'] + frame_delta_velocity
            for substep_idx in range(self.steps_per_frame):
                if correction_targets is not None:
                    target_velocity_command, correction_error_integral = self._compute_target_velocity_command(
                        state['x'],
                        state['v'],
                        correction_targets,
                        correction_target_mask,
                        correction_error_integral,
                    )
                    if correction_target_mask is None:
                        state['v'] = target_velocity_command
                    else:
                        state['v'] = torch.where(
                            correction_target_mask.unsqueeze(-1),
                            target_velocity_command,
                            state['v'],
                        )
                    target_delta_velocity_norm_sum = (
                        target_delta_velocity_norm_sum + target_velocity_command.norm(dim=-1).mean()
                    )
                    target_delta_velocity_count += 1
                kinematic_mask = self._apply_batched_kinematic_particle_targets(
                    state,
                    rigid_points,
                    kinematic_particle_ids,
                    manipulation_indicator,
                    step,
                    substep_idx,
                    alpha_offset=0,
                )
                stress = self._compute_batched_stress(state['F'], log_E, nu, material_model_info)
                stress = self._zero_batched_stress_for_kinematic_particles(stress, kinematic_mask)
                controller_grid_x = None
                controller_grid_v = None
                controller_grid_targets = self._batched_controller_grid_targets_for_substep(
                    controller_grid_points,
                    manipulation_indicator,
                    step,
                    substep_idx,
                    alpha_offset=1,
                    batch_size=batch_size,
                )
                if controller_grid_targets is not None:
                    controller_grid_x, controller_grid_v = controller_grid_targets
                x, v, C_next, F_new = solver.model.p2g2p(
                    state['x'], state['v'], state['C'], state['F'], stress,
                    controller_grid_points=controller_grid_x,
                    controller_grid_target_v=controller_grid_v,
                    controller_grid_contact_radius=self.manipulation_controller_grid_contact_radius,
                    controller_grid_velocity_blend=self.manipulation_controller_grid_velocity_blend,
                )
                state = {'x': x, 'v': v, 'C': C_next, 'F': F_new}
                state['F'] = self._apply_batched_plasticity(state['F'], log_E, nu, material_model_info)
                for operation in solver.post_step_process:
                    operation(solver, state)
                self._apply_batched_kinematic_particle_targets(
                    state,
                    rigid_points,
                    kinematic_particle_ids,
                    manipulation_indicator,
                    step,
                    substep_idx,
                    alpha_offset=1,
                )
            predicted_positions.append(state['x'])
            predicted_flows.append(state['x'] - prev_x)
            prev_x = state['x']

        return {
            'predicted_positions': torch.stack(predicted_positions, dim=0),
            'predicted_flows': torch.stack(predicted_flows, dim=0),
            'final_velocity': state['v'],
            'final_C': state['C'],
            'final_deformation_gradient': state['F'],
            'mean_target_delta_velocity_norm': (
                target_delta_velocity_norm_sum / max(target_delta_velocity_count, 1)
            ),
        }
