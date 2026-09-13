"""
MPM solver — thin wrapper around omniphysgs MPMModel.

Wraps src/mpm_core/mpm_model.py with a dict-based state interface for
compatibility with the simulation pipeline. Keeps add_rigid_body_collider
(not in original) for manipulator interaction.
"""

import numpy as np
import torch
from torch import Tensor
from typing import Callable, List, Optional
from omegaconf import OmegaConf

from physcore.mpm.mpm_model import BatchedMPMModel, MPMModel
from physcore.mpm.set_boundary_conditions import (
    add_surface_collider as _orig_add_surface_collider,
)


class MPMSolver:
    """Wrapper around omniphysgs MPMModel with dict-based state interface."""

    def __init__(
        self,
        num_grids: int = 50,
        dt: float = 2e-4,
        gravity: List[float] = None,
        damping: float = 1.0,
        clip_bound_factor: float = 0.5,
        device: str = 'cuda',
    ):
        if gravity is None:
            gravity = [0.0, 0.0, -9.8]

        self._num_grids = num_grids
        self._dt = dt
        self._gravity = gravity
        self._damping = damping
        self._clip_bound_factor = clip_bound_factor
        self._device = device

        self._model: Optional[MPMModel] = None
        self._post_step_process: List[Callable] = []

    @property
    def model(self) -> MPMModel:
        assert self._model is not None, "Call init_particles() first"
        return self._model

    @property
    def time(self):
        return self.model.time

    @time.setter
    def time(self, val):
        self.model.time = val

    @property
    def damping(self):
        if self._model is not None:
            return self._model.damping
        return self._damping

    @damping.setter
    def damping(self, val):
        self._damping = val
        if self._model is not None:
            self._model.damping = val

    @property
    def device(self):
        return self._device

    @property
    def dx(self):
        return self.model.dx

    @property
    def pre_particle_process(self):
        return self.model.pre_particle_process

    @property
    def post_grid_process(self):
        return self.model.post_grid_process

    @property
    def post_step_process(self):
        return self._post_step_process

    def init_particles(
        self, positions: Tensor, rho: float = 1000.0,
        vol_override: Optional[float] = None,
    ) -> dict:
        """Initialize particle state. Returns dict of state tensors."""
        N = positions.shape[0]
        positions_detached = positions.detach()

        # Compute bounding box for material_params (center/size)
        bb_min = positions_detached.min(dim=0).values.cpu().numpy()
        bb_max = positions_detached.max(dim=0).values.cpu().numpy()
        center = ((bb_min + bb_max) / 2).tolist()
        size = ((bb_max - bb_min) / 2).tolist()
        size = [max(s, 1e-6) for s in size]

        sim_params = OmegaConf.create({
            'num_grids': self._num_grids,
            'dt': self._dt,
            'gravity': self._gravity,
            'damping': self._damping,
            'clip_bound': self._clip_bound_factor,
        })

        material_params = OmegaConf.create({
            'center': center,
            'size': size,
            'rho': float(rho),
        })

        self._model = MPMModel(
            sim_params=sim_params,
            material_params=material_params,
            init_pos=positions,
            enable_train=False,
            device=self._device,
        )

        if vol_override is not None:
            self._model.vol = vol_override
            self._model.p_mass = rho * vol_override

        state = {
            'x': positions.clone(),
            'v': torch.zeros(N, 3, device=self._device),
            'C': torch.zeros(N, 3, 3, device=self._device),
            'F': torch.eye(3, device=self._device).unsqueeze(0).expand(N, -1, -1).clone(),
        }
        return state

    def reset(self):
        self.model.reset()

    @torch.no_grad()
    def step(
        self,
        state: dict,
        stress: Tensor,
        handle_particle_ids: Optional[Tensor] = None,
        handle_target_v: Optional[Tensor] = None,
        handle_particle_weights: Optional[Tensor] = None,
        handle_grid_velocity_blend: float = 0.0,
        controller_grid_points: Optional[Tensor] = None,
        controller_grid_target_v: Optional[Tensor] = None,
        controller_grid_contact_radius: float = 0.0,
        controller_grid_velocity_blend: float = 0.0,
    ) -> dict:
        """Single P2G2P step delegated to MPMModel."""
        x, v, C, F = self.model.p2g2p(
            state['x'], state['v'], state['C'], state['F'], stress,
            handle_particle_ids=handle_particle_ids,
            handle_target_v=handle_target_v,
            handle_particle_weights=handle_particle_weights,
            handle_grid_velocity_blend=handle_grid_velocity_blend,
            controller_grid_points=controller_grid_points,
            controller_grid_target_v=controller_grid_target_v,
            controller_grid_contact_radius=controller_grid_contact_radius,
            controller_grid_velocity_blend=controller_grid_velocity_blend,
        )
        state['x'] = x
        state['v'] = v
        state['C'] = C
        state['F'] = F
        for operation in self._post_step_process:
            operation(self, state)
        return state


class BatchedMPMSolver:
    """Dict-state wrapper around BatchedMPMModel.

    Batch elements must have the same particle count.  Each element owns an
    independent grid; batching only fuses tensor work.
    """

    def __init__(
        self,
        num_grids: int = 50,
        dt: float = 2e-4,
        gravity: List[float] = None,
        damping: float = 1.0,
        clip_bound_factor: float = 0.5,
        device: str = 'cuda',
    ):
        if gravity is None:
            gravity = [0.0, 0.0, -9.8]

        self._num_grids = num_grids
        self._dt = dt
        self._gravity = gravity
        self._damping = damping
        self._clip_bound_factor = clip_bound_factor
        self._device = device

        self._model: Optional[BatchedMPMModel] = None
        self._post_step_process: List[Callable] = []

    @property
    def model(self) -> BatchedMPMModel:
        assert self._model is not None, "Call init_particles() first"
        return self._model

    @property
    def time(self):
        return self.model.time

    @time.setter
    def time(self, val):
        self.model.time = val

    @property
    def damping(self):
        if self._model is not None:
            return self._model.damping
        return self._damping

    @damping.setter
    def damping(self, val):
        self._damping = val
        if self._model is not None:
            self._model.damping = val

    @property
    def device(self):
        return self._device

    @property
    def dx(self):
        return self.model.dx

    @property
    def pre_particle_process(self):
        return self.model.pre_particle_process

    @property
    def post_grid_process(self):
        return self.model.post_grid_process

    @property
    def post_step_process(self):
        return self._post_step_process

    def init_particles(
        self,
        positions: Tensor,
        rho: float = 1000.0,
        vol_override: Optional[Tensor] = None,
    ) -> dict:
        if positions.ndim != 3:
            raise ValueError(f'BatchedMPMSolver expects positions shape (B, N, 3), got {tuple(positions.shape)}')
        batch_size, n_particles, _ = positions.shape
        positions_detached = positions.detach()

        bb_min = positions_detached.amin(dim=1).cpu()
        bb_max = positions_detached.amax(dim=1).cpu()
        center = ((bb_min + bb_max) / 2).tolist()
        size = ((bb_max - bb_min) / 2).clamp_min(1e-6).tolist()

        sim_params = OmegaConf.create({
            'num_grids': self._num_grids,
            'dt': self._dt,
            'gravity': self._gravity,
            'damping': self._damping,
            'clip_bound': self._clip_bound_factor,
        })

        material_params = {
            'center': center,
            'size': size,
            'rho': float(rho),
        }

        self._model = BatchedMPMModel(
            sim_params=sim_params,
            material_params=material_params,
            init_pos=positions,
            enable_train=False,
            device=self._device,
        )

        if vol_override is not None:
            vol_override = vol_override.to(device=self._device, dtype=positions.dtype)
            if vol_override.ndim == 0:
                vol_override = vol_override.expand(batch_size)
            if vol_override.shape != (batch_size,):
                raise ValueError(f'vol_override must have shape (B,), got {tuple(vol_override.shape)}')
            self._model.vol = vol_override
            self._model.p_mass = rho * vol_override

        state = {
            'x': positions.clone(),
            'v': torch.zeros(batch_size, n_particles, 3, device=self._device, dtype=positions.dtype),
            'C': torch.zeros(batch_size, n_particles, 3, 3, device=self._device, dtype=positions.dtype),
            'F': torch.eye(3, device=self._device, dtype=positions.dtype)
            .view(1, 1, 3, 3)
            .expand(batch_size, n_particles, -1, -1)
            .clone(),
        }
        return state

    def reset(self):
        self.model.reset()

    @torch.no_grad()
    def step(
        self,
        state: dict,
        stress: Tensor,
        controller_grid_points: Optional[Tensor] = None,
        controller_grid_target_v: Optional[Tensor] = None,
        controller_grid_contact_radius: float = 0.0,
        controller_grid_velocity_blend: float = 0.0,
    ) -> dict:
        """Single batched P2G2P step delegated to BatchedMPMModel."""
        x, v, C, F = self.model.p2g2p(
            state['x'], state['v'], state['C'], state['F'], stress,
            controller_grid_points=controller_grid_points,
            controller_grid_target_v=controller_grid_target_v,
            controller_grid_contact_radius=controller_grid_contact_radius,
            controller_grid_velocity_blend=controller_grid_velocity_blend,
        )
        state['x'] = x
        state['v'] = v
        state['C'] = C
        state['F'] = F
        for operation in self._post_step_process:
            operation(self, state)
        return state


def _coulomb_contact_velocity(
    v: Tensor,
    normal: Tensor,
    gravity: Tensor,
    dt: float,
    friction: float,
    elasticity: float,
    static_velocity_threshold: float,
) -> Tensor:
    """Velocity-level Coulomb contact projection.

    The friction budget is per unit mass: collision normal impulse plus the
    gravity support impulse for resting contact. This gives static friction a
    nonzero budget even when normal velocity is near zero.
    """
    n = normal.to(device=v.device, dtype=v.dtype)
    while n.ndim < v.ndim:
        n = n.view(*([1] * (v.ndim - 1)), 3)
    vn = torch.sum(v * n, dim=-1, keepdim=True)
    v_normal = vn * n
    v_tangent = v - v_normal
    tangent_speed = torch.linalg.norm(v_tangent, dim=-1, keepdim=True)
    mu = max(0.0, float(friction))
    e = max(0.0, min(float(elasticity), 1.0))
    static_thresh = max(0.0, float(static_velocity_threshold))
    g = gravity.to(device=v.device, dtype=v.dtype)
    while g.ndim < v.ndim:
        g = g.view(*([1] * (v.ndim - 1)), 3)
    support_impulse = (-torch.sum(g * n, dim=-1, keepdim=True)).clamp_min(0.0) * float(dt)
    collision_impulse = (1.0 + e) * (-vn).clamp_min(0.0)
    friction_budget = mu * (collision_impulse + support_impulse)
    tangent_scale = ((tangent_speed - friction_budget).clamp_min(0.0) / tangent_speed.clamp_min(1.0e-8))
    v_tangent_after = v_tangent * tangent_scale
    v_tangent_after = torch.where(
        tangent_speed <= friction_budget + static_thresh,
        torch.zeros_like(v_tangent_after),
        v_tangent_after,
    )
    v_normal_after = torch.where(vn < 0.0, -e * v_normal, v_normal)
    return v_normal_after + v_tangent_after


# ─────────────── Boundary Conditions ──────────────────────────────────────────

def add_surface_collider(
    solver: MPMSolver,
    point: List[float],
    normal: List[float],
    surface: str = "sticky",
    friction: float = 0.0,
    elasticity: float = 0.5,
    tangent_damping: float = 0.0,
    static_velocity_threshold: float = 0.0,
    start_time: float = 0.0,
    end_time: float = 1e6,
):
    """Add a planar surface collider via the original omniphysgs implementation."""
    point_t = torch.tensor(point, device=solver.model.device).float()
    normal_t = torch.tensor(normal, device=solver.model.device).float()
    normal_t = normal_t / torch.norm(normal_t)
    _orig_add_surface_collider(
        solver.model,
        point=point,
        normal=normal,
        surface=surface,
        friction=friction,
        elasticity=elasticity,
        tangent_damping=tangent_damping,
        static_velocity_threshold=static_velocity_threshold,
        start_time=start_time,
        end_time=end_time,
    )
    if surface not in {"phystwin", "coulomb"}:
        return

    collide_elas = float(max(0.0, min(float(elasticity), 1.0)))
    collide_fric = float(max(0.0, min(float(friction), 2.0)))
    rest_damp = float(max(0.0, min(float(tangent_damping), 1.0)))
    static_thresh = float(max(0.0, float(static_velocity_threshold)))

    def phystwin_project_post_step(_solver: MPMSolver, state: dict):
        x = state['x']
        v = state['v']
        n = normal_t.to(device=x.device, dtype=x.dtype)
        p = point_t.to(device=x.device, dtype=x.dtype)
        signed = torch.sum((x - p) * n, dim=-1, keepdim=True)
        below = signed < 0.0
        # No `below.any()` early-out: the torch.where below is already a no-op
        # when nothing is in contact, and the check costs a host sync per substep.
        if surface == "coulomb":
            v_after = _coulomb_contact_velocity(
                v, n, _solver.model.gravity, _solver.model.dt, friction, elasticity, static_velocity_threshold
            )
            state['v'] = torch.where(below, v_after, v)
        else:
            v_normal = n * torch.sum(v * n, dim=-1, keepdim=True)
            v_tangent = v - v_normal
            moving_into_surface = torch.sum(v * n, dim=-1, keepdim=True) < -1.0e-4
            normal_speed = torch.linalg.norm(v_normal, dim=-1, keepdim=True)
            tangent_speed_raw = torch.linalg.norm(v_tangent, dim=-1, keepdim=True)
            tangent_speed = tangent_speed_raw.clamp_min(1.0e-6)
            tangent_scale = (
                1.0
                - collide_fric
                * (1.0 + collide_elas)
                * normal_speed
                / tangent_speed
            ).clamp_min(0.0)
            collision_tangent = tangent_scale * v_tangent
            collision_tangent = torch.where(
                tangent_speed_raw < static_thresh,
                torch.zeros_like(collision_tangent),
                collision_tangent,
            )
            resting_tangent = (1.0 - rest_damp) * v_tangent
            resting_tangent = torch.where(
                tangent_speed_raw < static_thresh,
                torch.zeros_like(resting_tangent),
                resting_tangent,
            )
            v_after_collision = -collide_elas * v_normal + collision_tangent
            v_after_resting = v_normal + resting_tangent
            state['v'] = torch.where(below & moving_into_surface, v_after_collision, v)
            state['v'] = torch.where(below & ~moving_into_surface, v_after_resting, state['v'])
        state['x'] = torch.where(below, x - signed * n, x)

    solver.post_step_process.append(phystwin_project_post_step)


def add_batched_surface_collider(
    solver: BatchedMPMSolver,
    point: List[float],
    normal: List[float],
    surface: str = "sticky",
    friction: float = 0.0,
    elasticity: float = 0.5,
    tangent_damping: float = 0.0,
    static_velocity_threshold: float = 0.0,
    start_time: float = 0.0,
    end_time: float = 1e6,
):
    model = solver.model
    point_t = torch.tensor(point, device=model.device).float()
    normal_t = torch.tensor(normal, device=model.device).float()
    normal_t = normal_t / torch.norm(normal_t)
    offset = model.grid_x * model.dx - point_t
    dotproduct = torch.sum(offset * normal_t, dim=1)
    local_target = dotproduct < 0.0
    target = local_target.unsqueeze(0).expand(model.batch_size, -1).reshape(-1)

    def collide(model_ref: BatchedMPMModel):
        time = model_ref.time
        if time < start_time or time >= end_time:
            return
        # See the unbatched collider: torch.where over the whole grid instead of
        # bool-mask indexing, which would run nonzero() every substep.
        target_mask = target.unsqueeze(1)
        if surface == "sticky":
            model_ref.grid_mv = torch.where(
                target_mask, torch.zeros_like(model_ref.grid_mv), model_ref.grid_mv,
            )
        elif surface == "slip":
            v = model_ref.grid_mv
            model_ref.grid_mv = torch.where(
                target_mask, v - normal_t * torch.sum(v * normal_t, dim=1, keepdim=True), v,
            )
        elif surface == "collide":
            v = model_ref.grid_mv
            model_ref.grid_mv = torch.where(
                target_mask, v - normal_t * 2.0 * torch.sum(v * normal_t, dim=1, keepdim=True), v,
            )
        elif surface == "phystwin":
            v = model_ref.grid_mv
            v_normal = normal_t * torch.sum(v * normal_t, dim=1, keepdim=True)
            v_tangent = v - v_normal
            moving_into_surface = torch.sum(v * normal_t, dim=1, keepdim=True) < -1.0e-4
            normal_speed = torch.linalg.norm(v_normal, dim=1, keepdim=True)
            tangent_speed_raw = torch.linalg.norm(v_tangent, dim=1, keepdim=True)
            tangent_speed = tangent_speed_raw.clamp_min(1.0e-6)
            collide_elas = float(max(0.0, min(float(elasticity), 1.0)))
            collide_fric = float(max(0.0, min(float(friction), 2.0)))
            rest_damp = float(max(0.0, min(float(tangent_damping), 1.0)))
            static_thresh = float(max(0.0, float(static_velocity_threshold)))
            tangent_scale = (
                1.0
                - collide_fric
                * (1.0 + collide_elas)
                * normal_speed
                / tangent_speed
            ).clamp_min(0.0)
            collision_tangent = tangent_scale * v_tangent
            collision_tangent = torch.where(
                tangent_speed_raw < static_thresh,
                torch.zeros_like(collision_tangent),
                collision_tangent,
            )
            resting_tangent = (1.0 - rest_damp) * v_tangent
            resting_tangent = torch.where(
                tangent_speed_raw < static_thresh,
                torch.zeros_like(resting_tangent),
                resting_tangent,
            )
            v_after_collision = -collide_elas * v_normal + collision_tangent
            v_after_resting = v_normal + resting_tangent
            model_ref.grid_mv = torch.where(
                target_mask & moving_into_surface,
                v_after_collision,
                torch.where(target_mask, v_after_resting, v),
            )
        elif surface == "coulomb":
            v = model_ref.grid_mv
            model_ref.grid_mv = torch.where(
                target_mask,
                _coulomb_contact_velocity(
                    v, normal_t, model_ref.gravity, model_ref.dt, friction, elasticity, static_velocity_threshold
                ),
                v,
            )
        else:
            raise TypeError("Undefined surface type")

    model.post_grid_process.append(collide)
    if surface not in {"phystwin", "coulomb"}:
        return

    collide_elas = float(max(0.0, min(float(elasticity), 1.0)))
    collide_fric = float(max(0.0, min(float(friction), 2.0)))
    rest_damp = float(max(0.0, min(float(tangent_damping), 1.0)))
    static_thresh = float(max(0.0, float(static_velocity_threshold)))

    def phystwin_project_post_step(_solver: BatchedMPMSolver, state: dict):
        x = state['x']
        v = state['v']
        n = normal_t.to(device=x.device, dtype=x.dtype)
        p = point_t.to(device=x.device, dtype=x.dtype)
        signed = torch.sum((x - p.view(1, 1, 3)) * n.view(1, 1, 3), dim=-1, keepdim=True)
        below = signed < 0.0
        # No `below.any()` early-out — see the unbatched collider.
        if surface == "coulomb":
            v_after = _coulomb_contact_velocity(
                v, n, _solver.model.gravity, _solver.model.dt, friction, elasticity, static_velocity_threshold
            )
            state['v'] = torch.where(below, v_after, v)
        else:
            v_normal = n.view(1, 1, 3) * torch.sum(v * n.view(1, 1, 3), dim=-1, keepdim=True)
            v_tangent = v - v_normal
            moving_into_surface = torch.sum(v * n.view(1, 1, 3), dim=-1, keepdim=True) < -1.0e-4
            normal_speed = torch.linalg.norm(v_normal, dim=-1, keepdim=True)
            tangent_speed_raw = torch.linalg.norm(v_tangent, dim=-1, keepdim=True)
            tangent_speed = tangent_speed_raw.clamp_min(1.0e-6)
            tangent_scale = (
                1.0
                - collide_fric
                * (1.0 + collide_elas)
                * normal_speed
                / tangent_speed
            ).clamp_min(0.0)
            collision_tangent = tangent_scale * v_tangent
            collision_tangent = torch.where(
                tangent_speed_raw < static_thresh,
                torch.zeros_like(collision_tangent),
                collision_tangent,
            )
            resting_tangent = (1.0 - rest_damp) * v_tangent
            resting_tangent = torch.where(
                tangent_speed_raw < static_thresh,
                torch.zeros_like(resting_tangent),
                resting_tangent,
            )
            v_after_collision = -collide_elas * v_normal + collision_tangent
            v_after_resting = v_normal + resting_tangent
            state['v'] = torch.where(below & moving_into_surface, v_after_collision, v)
            state['v'] = torch.where(below & ~moving_into_surface, v_after_resting, state['v'])
        state['x'] = torch.where(below, x - signed * n.view(1, 1, 3), x)

    solver.post_step_process.append(phystwin_project_post_step)


def add_batched_rigid_body_collider(
    solver: BatchedMPMSolver,
    rigid_body,
    start_time: float = 0.0,
    end_time: float = 1e6,
    friction: float = 0.3,
    contact_margin: float = 0.0,
    velocity_blend: float = 0.85,
    stickiness: float = 0.75,
    strict_nonpenetration: bool = True,
    projection_iterations: int = 4,
    projection_margin: float = 1e-5,
):
    """Add batched rigid-body SDF contact for shared geometry, batched motion."""
    model = solver.model
    eps = model.dx * 0.5
    shell_margin = max(float(contact_margin), 0.0)
    velocity_blend = float(np.clip(velocity_blend, 0.0, 1.0))
    stickiness = float(np.clip(stickiness, 0.0, 1.0))
    strict_nonpenetration = bool(strict_nonpenetration)
    projection_iterations = max(int(projection_iterations), 1)
    projection_margin = max(float(projection_margin), 0.0)

    def _flat_batch_indices(num_items_per_batch: int, device: torch.device) -> Tensor:
        return torch.arange(model.batch_size, device=device).repeat_interleave(num_items_per_batch)

    def _candidate_mask(points: Tensor, t: float, padding: float, mass_mask: Optional[Tensor] = None) -> Tensor:
        bounds_min, bounds_max = rigid_body.get_world_bounds(t, padding)
        bounds_min = bounds_min.to(device=points.device, dtype=points.dtype)
        bounds_max = bounds_max.to(device=points.device, dtype=points.dtype)
        if points.ndim == 3:
            mask = torch.all((points >= bounds_min.unsqueeze(1)) & (points <= bounds_max.unsqueeze(1)), dim=-1)
        else:
            batch_idx = _flat_batch_indices(model.num_grid_nodes, points.device)
            mask = torch.all((points >= bounds_min[batch_idx]) & (points <= bounds_max[batch_idx]), dim=-1)
        if mass_mask is not None:
            mask = mask & mass_mask
        return mask

    def compute_normals(points: Tensor, batch_indices: Tensor, t: float) -> Tensor:
        grad = torch.zeros_like(points)
        for dim in range(3):
            dx_pos = points.clone()
            dx_neg = points.clone()
            dx_pos[:, dim] += eps
            dx_neg[:, dim] -= eps
            grad[:, dim] = (
                rigid_body.sdf_indexed(dx_pos, batch_indices, t)
                - rigid_body.sdf_indexed(dx_neg, batch_indices, t)
            ) / (2 * eps)

        grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return grad / grad_norm

    def remove_inward_velocity(
        x_flat: Tensor,
        v_flat: Tensor,
        flat_indices: Tensor,
        batch_indices: Tensor,
        t: float,
    ) -> None:
        if flat_indices.numel() == 0:
            return

        x_sel = x_flat.index_select(0, flat_indices)
        v_sel = v_flat.index_select(0, flat_indices)
        n = compute_normals(x_sel, batch_indices, t)
        rb_vel_all = rigid_body.get_velocity(t).to(device=x_flat.device, dtype=x_flat.dtype)
        rb_vel = rb_vel_all.index_select(0, batch_indices)
        v_rel = v_sel - rb_vel
        vn_scalar = (v_rel * n).sum(dim=-1, keepdim=True)

        v_normal = vn_scalar * n
        v_tangent = v_rel - v_normal
        v_tan_norm = v_tangent.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        friction_reduction = (friction * (-vn_scalar).clamp(min=0) / v_tan_norm).clamp(max=1.0)
        v_tangent_corrected = v_tangent * (1.0 - friction_reduction)
        v_corrected = torch.where(vn_scalar < 0, v_tangent_corrected, v_rel)
        v_flat[flat_indices] = rb_vel + v_corrected

    def rigid_collide(model_ref: BatchedMPMModel, x: Tensor, v: Tensor):
        t = model_ref.time
        if t < start_time or t >= end_time:
            return

        candidate_mask = _candidate_mask(x, t, max(shell_margin, 0.0))
        if not candidate_mask.any():
            return

        batch_idx_2d, particle_idx_2d = candidate_mask.nonzero(as_tuple=True)
        flat_indices = batch_idx_2d * model_ref.n_particles + particle_idx_2d
        x_flat = x.reshape(-1, 3)
        v_flat = v.reshape(-1, 3)
        x_candidate = x_flat.index_select(0, flat_indices)
        v_candidate = v_flat.index_select(0, flat_indices)
        sdf_candidate = rigid_body.sdf_indexed(x_candidate, batch_idx_2d, t)
        inside = sdf_candidate < 0.0
        active = sdf_candidate < shell_margin if shell_margin > 0.0 else inside

        if not active.any():
            return

        active_flat_indices = flat_indices[active]
        active_batch_indices = batch_idx_2d[active]
        x_active = x_candidate[active]
        v_active = v_candidate[active]
        sdf_active = sdf_candidate[active]
        n = compute_normals(x_active, active_batch_indices, t)

        if inside.any():
            inside_active = inside[active]
            if inside_active.any():
                x_active = x_active.clone()
                x_active[inside_active] = (
                    x_active[inside_active]
                    - sdf_active[inside_active].unsqueeze(1) * n[inside_active]
                )
                x_flat[active_flat_indices] = x_active

        rb_vel_all = rigid_body.get_velocity(t).to(device=x.device, dtype=x.dtype)
        rb_vel = rb_vel_all.index_select(0, active_batch_indices)
        v_rel = v_active - rb_vel
        vn_scalar = (v_rel * n).sum(dim=-1, keepdim=True)

        penetrating = vn_scalar < 0
        v_normal = vn_scalar * n
        v_tangent = v_rel - v_normal

        v_tan_norm = v_tangent.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        friction_reduction = (friction * (-vn_scalar).clamp(min=0) / v_tan_norm).clamp(max=1.0)
        v_tangent_corrected = v_tangent * (1.0 - friction_reduction)
        v_corrected = torch.where(penetrating, v_tangent_corrected, v_rel)

        if shell_margin > 0.0:
            proximity = ((shell_margin - sdf_active) / max(shell_margin, 1e-8)).unsqueeze(1).clamp(0.0, 1.0)
        else:
            proximity = torch.ones_like(vn_scalar)

        rb_vn = (rb_vel * n).sum(dim=-1, keepdim=True)
        pushing_in = rb_vn < 0
        effective_stickiness = torch.where(pushing_in, stickiness, 0.0)

        coupled_world_velocity = rb_vel + (1.0 - effective_stickiness) * v_corrected
        blend = (velocity_blend * proximity).clamp(0.0, 1.0)
        v_flat[active_flat_indices] = (1.0 - blend) * v_active + blend * coupled_world_velocity

    def rigid_project_post_step(solver_ref: BatchedMPMSolver, state: dict):
        if not strict_nonpenetration:
            return

        t = solver_ref.time
        if t < start_time or t >= end_time:
            return

        x = state['x']
        v = state['v']
        candidate_mask = _candidate_mask(x, t, projection_margin)
        if not candidate_mask.any():
            return

        batch_idx_2d, particle_idx_2d = candidate_mask.nonzero(as_tuple=True)
        flat_indices = batch_idx_2d * solver_ref.model.n_particles + particle_idx_2d
        x_flat = x.reshape(-1, 3)
        v_flat = v.reshape(-1, 3)
        x_candidate = x_flat.index_select(0, flat_indices)
        projected_mask = torch.zeros(flat_indices.shape[0], dtype=torch.bool, device=x.device)

        sdf_val = rigid_body.sdf_indexed(x_candidate, batch_idx_2d, t)
        inside_indices = (sdf_val < projection_margin).nonzero(as_tuple=False).squeeze(1)

        for _ in range(projection_iterations):
            if inside_indices.numel() == 0:
                break

            x_inside = x_candidate.index_select(0, inside_indices)
            batch_inside = batch_idx_2d.index_select(0, inside_indices)
            sdf_inside = rigid_body.sdf_indexed(x_inside, batch_inside, t)
            n_inside = compute_normals(x_inside, batch_inside, t)
            correction = (projection_margin - sdf_inside).unsqueeze(1).clamp(min=0.0)
            x_candidate[inside_indices] = x_inside + correction * n_inside
            projected_mask[inside_indices] = True
            sdf_inside = rigid_body.sdf_indexed(x_candidate.index_select(0, inside_indices), batch_inside, t)
            still_inside = sdf_inside < projection_margin
            inside_indices = inside_indices[still_inside]

        if projected_mask.any():
            x_flat[flat_indices] = x_candidate
            remove_inward_velocity(
                x_flat,
                v_flat,
                flat_indices[projected_mask],
                batch_idx_2d[projected_mask],
                t,
            )

    def rigid_collide_grid(model_ref: BatchedMPMModel):
        t = model_ref.time
        if t < start_time or t >= end_time:
            return

        grid_pos = model_ref.batched_grid_x * model_ref.dx
        grid_batch_idx = _flat_batch_indices(model_ref.num_grid_nodes, grid_pos.device)
        cand_mask = _candidate_mask(
            grid_pos,
            t,
            shell_margin,
            mass_mask=(model_ref.grid_m > 1e-15),
        )
        if not cand_mask.any():
            return

        cand_idx = cand_mask.nonzero(as_tuple=False).squeeze(1)
        gp = grid_pos.index_select(0, cand_idx)
        cand_batch_idx = grid_batch_idx.index_select(0, cand_idx)
        sdf = rigid_body.sdf_indexed(gp, cand_batch_idx, t)
        active = sdf < shell_margin
        if not active.any():
            return

        act_idx = cand_idx[active]
        gp_act = gp[active]
        act_batch_idx = cand_batch_idx[active]

        n = compute_normals(gp_act, act_batch_idx, t)
        rb_vel_all = rigid_body.get_velocity(t).to(device=grid_pos.device, dtype=grid_pos.dtype)
        rb_vel = rb_vel_all.index_select(0, act_batch_idx)
        gv = model_ref.grid_mv[act_idx]

        v_rel = gv - rb_vel
        vn_scalar = (v_rel * n).sum(dim=-1, keepdim=True)
        approaching = vn_scalar < 0

        v_normal = vn_scalar * n
        v_tangent = v_rel - v_normal
        v_tan_norm = v_tangent.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        friction_red = (
            friction * (-vn_scalar).clamp(min=0) / v_tan_norm
        ).clamp(max=1.0)
        v_tangent_corrected = v_tangent * (1.0 - friction_red)

        v_corrected = torch.where(approaching, v_tangent_corrected, v_rel)
        model_ref.grid_mv[act_idx] = rb_vel + v_corrected

    model.pre_particle_process.append(rigid_collide)
    model.post_grid_process.insert(0, rigid_collide_grid)
    solver.post_step_process.append(rigid_project_post_step)


def add_rigid_body_collider(
    solver: MPMSolver,
    sdf_fn,  # Callable: (Tensor(N,3), float_time) -> Tensor(N,)
    velocity_fn,  # Callable: (float_time) -> Tensor(3,)
    bounds_fn=None,  # Callable: (float_time, padding) -> (Tensor(3,), Tensor(3,))
    start_time: float = 0.0,
    end_time: float = 1e6,
    friction: float = 0.3,
    contact_margin: float = 0.0,
    velocity_blend: float = 0.85,
    stickiness: float = 0.75,
    strict_nonpenetration: bool = True,
    projection_iterations: int = 4,
    projection_margin: float = 1e-5,
):
    """Add a rigid body SDF-based collider as pre-particle process.

    The collider operates on a thin shell around the tool instead of only after
    deep penetration. This transfers manipulator motion earlier and produces
    visibly stronger local deformation at moderate grid resolutions.
    """
    model = solver.model
    eps = model.dx * 0.5  # finite difference step for gradient
    shell_margin = max(float(contact_margin), 0.0)
    velocity_blend = float(np.clip(velocity_blend, 0.0, 1.0))
    stickiness = float(np.clip(stickiness, 0.0, 1.0))
    strict_nonpenetration = bool(strict_nonpenetration)
    projection_iterations = max(int(projection_iterations), 1)
    projection_margin = max(float(projection_margin), 0.0)

    def get_candidate_indices(x: Tensor, t: float, padding: float) -> Tensor:
        if bounds_fn is None:
            return torch.arange(x.shape[0], device=x.device)

        bounds_min, bounds_max = bounds_fn(t, padding)
        bounds_min = bounds_min.to(device=x.device, dtype=x.dtype)
        bounds_max = bounds_max.to(device=x.device, dtype=x.dtype)
        candidate_mask = torch.all((x >= bounds_min) & (x <= bounds_max), dim=1)
        return candidate_mask.nonzero(as_tuple=False).squeeze(1)

    def compute_normals(points: Tensor, t: float) -> Tensor:
        grad = torch.zeros_like(points)
        for dim in range(3):
            dx_pos = points.clone()
            dx_neg = points.clone()
            dx_pos[:, dim] += eps
            dx_neg[:, dim] -= eps
            grad[:, dim] = (sdf_fn(dx_pos, t) - sdf_fn(dx_neg, t)) / (2 * eps)

        grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return grad / grad_norm

    def remove_inward_velocity(x: Tensor, v: Tensor, indices: Tensor, t: float) -> None:
        if indices.numel() == 0:
            return

        x_sel = x.index_select(0, indices)
        v_sel = v.index_select(0, indices)
        n = compute_normals(x_sel, t)
        rb_vel = velocity_fn(t)
        v_rel = v_sel - rb_vel
        vn_scalar = (v_rel * n).sum(dim=-1, keepdim=True)

        v_normal = vn_scalar * n
        v_tangent = v_rel - v_normal
        v_tan_norm = v_tangent.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        friction_reduction = (friction * (-vn_scalar).clamp(min=0) / v_tan_norm).clamp(max=1.0)
        v_tangent_corrected = v_tangent * (1.0 - friction_reduction)
        v_corrected = torch.where(vn_scalar < 0, v_tangent_corrected, v_rel)
        v[indices] = rb_vel + v_corrected

    def rigid_collide(model_ref: MPMModel, x: Tensor, v: Tensor):
        t = model_ref.time
        if t < start_time or t >= end_time:
            return

        candidate_indices = get_candidate_indices(x, t, max(shell_margin, 0.0))
        if candidate_indices.numel() == 0:
            return

        x_candidate = x.index_select(0, candidate_indices)
        v_candidate = v.index_select(0, candidate_indices)
        sdf_candidate = sdf_fn(x_candidate, t)  # (M,)
        inside = sdf_candidate < 0.0
        active = sdf_candidate < shell_margin if shell_margin > 0.0 else inside

        if not active.any():
            return

        # Numerical gradient via central differences
        active_indices = candidate_indices[active]
        x_active = x_candidate[active]
        v_active = v_candidate[active]
        sdf_active = sdf_candidate[active]
        n = compute_normals(x_active, t)

        # 1. Project only truly penetrating particles back to the tool surface.
        if inside.any():
            inside_active = inside[active]
            if inside_active.any():
                x_active[inside_active] = (
                    x_active[inside_active]
                    - sdf_active[inside_active].unsqueeze(1) * n[inside_active]
                )
                x[active_indices] = x_active

        # 2. Velocity correction in the rigid-body frame
        rb_vel = velocity_fn(t)  # (3,)
        v_rel = v_active - rb_vel  # relative velocity
        vn_scalar = (v_rel * n).sum(dim=-1, keepdim=True)  # normal component

        # Remove inward normal motion relative to the tool.
        penetrating = vn_scalar < 0

        v_normal = vn_scalar * n
        v_tangent = v_rel - v_normal

        v_tan_norm = v_tangent.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        friction_reduction = (friction * (-vn_scalar).clamp(min=0) / v_tan_norm).clamp(max=1.0)
        v_tangent_corrected = v_tangent * (1.0 - friction_reduction)

        v_corrected = torch.where(penetrating, v_tangent_corrected, v_rel)

        # Blend nearby particles toward the body velocity to create a stronger,
        # more stable poke signal for data generation.
        if shell_margin > 0.0:
            proximity = ((shell_margin - sdf_active) / max(shell_margin, 1e-8)).unsqueeze(1).clamp(0.0, 1.0)
        else:
            proximity = torch.ones_like(vn_scalar)

        # Only apply stickiness when the rigid body is pushing inward;
        # when retracting, let particles decouple so they don't get pulled out.
        rb_vn = (rb_vel * n).sum(dim=-1, keepdim=True)
        pushing_in = rb_vn < 0
        effective_stickiness = torch.where(pushing_in, stickiness, 0.0)

        coupled_world_velocity = rb_vel + (1.0 - effective_stickiness) * v_corrected
        blend = (velocity_blend * proximity).clamp(0.0, 1.0)
        v[active_indices] = (1.0 - blend) * v_active + blend * coupled_world_velocity

    def rigid_project_post_step(solver_ref: MPMSolver, state: dict):
        if not strict_nonpenetration:
            return

        t = solver_ref.time
        if t < start_time or t >= end_time:
            return

        x = state['x']
        v = state['v']
        candidate_indices = get_candidate_indices(x, t, projection_margin)
        if candidate_indices.numel() == 0:
            return

        x_candidate = x.index_select(0, candidate_indices)
        projected_mask = torch.zeros(candidate_indices.shape[0], dtype=torch.bool, device=x.device)

        sdf_val = sdf_fn(x_candidate, t)
        inside_indices = (sdf_val < projection_margin).nonzero(as_tuple=False).squeeze(1)

        for _ in range(projection_iterations):
            if inside_indices.numel() == 0:
                break

            x_inside = x_candidate.index_select(0, inside_indices)
            sdf_inside = sdf_val.index_select(0, inside_indices)
            n_inside = compute_normals(x_inside, t)
            correction = (projection_margin - sdf_inside).unsqueeze(1).clamp(min=0.0)
            x_candidate[inside_indices] = x_inside + correction * n_inside
            projected_mask[inside_indices] = True
            sdf_inside = sdf_fn(x_candidate.index_select(0, inside_indices), t)
            still_inside = sdf_inside < projection_margin
            inside_indices = inside_indices[still_inside]

        if projected_mask.any():
            x[candidate_indices] = x_candidate
            remove_inward_velocity(x, v, candidate_indices[projected_mask], t)

    # ── Grid-level rigid body collision (post_grid_process) ──────────
    # Standard MPM approach: enforce a no-penetration condition directly
    # on grid velocities.  Only grid nodes with material mass are touched
    # to avoid velocity leakage through empty-space nodes.  The tool
    # pushes material by being impenetrable — material accumulates on
    # the tool surface and the elastic stress transmits the force.
    def rigid_collide_grid(model_ref: MPMModel):
        t = model_ref.time
        if t < start_time or t >= end_time:
            return

        grid_pos = model_ref.grid_x * model_ref.dx  # (N_grid, 3)

        # Pre-filter: bounding box AND non-zero mass
        if bounds_fn is not None:
            bmin, bmax = bounds_fn(t, shell_margin)
            bmin = bmin.to(device=grid_pos.device, dtype=grid_pos.dtype)
            bmax = bmax.to(device=grid_pos.device, dtype=grid_pos.dtype)
            cand_mask = torch.all(
                (grid_pos >= bmin) & (grid_pos <= bmax), dim=1
            )
        else:
            cand_mask = torch.ones(
                grid_pos.shape[0], dtype=torch.bool, device=grid_pos.device
            )
        cand_mask = cand_mask & (model_ref.grid_m > 1e-15)

        if not cand_mask.any():
            return

        cand_idx = cand_mask.nonzero(as_tuple=False).squeeze(1)
        if cand_idx.dim() == 0:
            cand_idx = cand_idx.unsqueeze(0)

        gp = grid_pos[cand_idx]
        sdf = sdf_fn(gp, t)
        active = sdf < shell_margin
        if not active.any():
            return

        act_idx = cand_idx[active]
        sdf_act = sdf[active]
        gp_act = gp[active]

        n = compute_normals(gp_act, t)
        rb_vel = velocity_fn(t)
        gv = model_ref.grid_mv[act_idx]

        # Relative velocity in the tool frame
        v_rel = gv - rb_vel
        vn_scalar = (v_rel * n).sum(dim=-1, keepdim=True)  # normal comp.

        # Only correct when approaching the tool surface (inward)
        approaching = vn_scalar < 0

        # Remove the inward normal component; apply Coulomb friction
        v_normal = vn_scalar * n
        v_tangent = v_rel - v_normal
        v_tan_norm = v_tangent.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        friction_red = (
            friction * (-vn_scalar).clamp(min=0) / v_tan_norm
        ).clamp(max=1.0)
        v_tangent_corrected = v_tangent * (1.0 - friction_red)

        # Slip condition: remove inward normal velocity, keep tangential
        v_corrected = torch.where(approaching, v_tangent_corrected, v_rel)
        model_ref.grid_mv[act_idx] = rb_vel + v_corrected

    model.pre_particle_process.append(rigid_collide)
    # Insert grid collider BEFORE any existing post_grid_process entries
    # (e.g. the ground surface collider) so that the ground always runs
    # last and dominates at boundary nodes.
    model.post_grid_process.insert(0, rigid_collide_grid)
    solver.post_step_process.append(rigid_project_post_step)
