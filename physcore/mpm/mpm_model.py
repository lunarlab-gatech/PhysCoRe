"""
The MPM solver, in single-episode and batched forms.
"""

from typing import *

import numpy as np
import torch
from torch import Tensor
from omegaconf import DictConfig

# Packed (distance, point-id) sentinel for the controller-Dirichlet scatter-min:
# +inf in the high 32 bits so untouched grid nodes never look "near".
_DIRICHLET_EMPTY = (0x7F800000 << 32) | 0xFFFFFFFF


def _pack_nearest(distance: Tensor, point_id: Tensor) -> Tensor:
    """Pack (float32 distance, int32 point id) into one sortable int64.

    Distances are non-negative, so their bit pattern is monotonic and an int64
    ``amin`` over the packed value reproduces ``min(dim=-1)`` exactly, ties
    broken toward the lower point id — but with a static output shape.
    """
    return (distance.view(torch.int32).long() << 32) | point_id.to(torch.long)


def _unpack_nearest(packed: Tensor) -> Tuple[Tensor, Tensor]:
    return (packed >> 32).int().view(torch.float32), packed & 0xFFFFFFFF


def _controller_candidate_offsets(radius: float, inv_dx: float, device: torch.device) -> Tensor:
    cells = max(int(np.ceil(radius * float(inv_dx))), 0)
    rng = torch.arange(-cells, cells + 1, device=device, dtype=torch.long)
    return torch.stack(torch.meshgrid(rng, rng, rng, indexing='ij'), dim=-1).reshape(-1, 3)

class MPMModel:
    def __init__(
        self, 
        sim_params: DictConfig,
        material_params: DictConfig, 
        init_pos: Tensor, 
        enable_train: bool=False,
        device: torch.device='cuda',
    ):
        # save simulation parameters
        self.num_grids: int = sim_params['num_grids']
        self.dt: float = sim_params['dt']
        self.gravity: Tensor = torch.tensor(sim_params['gravity'], device=device)
        self.boundary_condition: Optional[DictConfig] = sim_params.get('boundary_condition', None)
        
        self.dx: float = 1 / self.num_grids
        self.inv_dx: float = float(self.num_grids)
        
        self.clip_bound: float = sim_params.get('clip_bound', 0.5) * self.dx
        self.damping = sim_params.get('damping', 1.0)
        assert self.clip_bound >= 0.0
        assert self.damping >= 0.0 and self.damping <= 1.0
        
        self.n_particles: int = init_pos.shape[0]
        self.init_pos: Tensor = init_pos.detach()
        
        self.center: np.ndarray = np.array(material_params['center'])
        self.size: np.ndarray = np.array(material_params['size'])
        self.vol: float = np.prod(self.size) / self.n_particles
        self.p_mass: float = material_params['rho'] * self.vol  # TODO: the mass can be non-constant.

        self.enable_train: bool = enable_train
        self.device: torch.device = device
        
        # init tensors
        num_grids = self.num_grids
        n_dim = 3 # 3D
        self.grid_mv = torch.empty((num_grids ** n_dim, n_dim), device=device)
        self.grid_m = torch.empty((num_grids ** n_dim,), device=device)
        grid_ranges = torch.arange(num_grids, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(grid_ranges, grid_ranges, grid_ranges, indexing='ij')
        self.grid_x = torch.stack((grid_x, grid_y, grid_z), dim=-1).reshape(-1, 3).float() # (n_grid * n_grid * n_grid, 3)
        
        self.offset = torch.tensor([[i, j, k] for i in range(3) for j in range(3) for k in range(3)], device=device).float() # (27, 3)

        # bc
        self.pre_particle_process = []
        self.post_grid_process = []

        self.time = 0.0        
        
    def reset(self) -> None:
        self.time = 0.0
    
    def __call__(self, x: Tensor, v: Tensor, C: Tensor, F: Tensor, stress: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.p2g2p(x, v, C, F, stress)

    def _apply_handle_grid_velocity_constraint(
        self,
        index: Tensor,
        weight: Tensor,
        handle_particle_ids: Optional[Tensor],
        handle_target_v: Optional[Tensor],
        handle_particle_weights: Optional[Tensor],
        blend: float,
    ) -> None:
        if handle_particle_ids is None or handle_target_v is None:
            return
        if handle_particle_ids.numel() == 0 or handle_target_v.numel() == 0:
            return
        local_blend = min(float(blend), 1.0)
        if local_blend <= 0.0:
            return

        # Out-of-range ids are zero-weighted rather than filtered out, so the
        # shapes stay static and no nonzero()/any() sync is needed.
        handle_particle_ids = handle_particle_ids.to(device=self.grid_mv.device, dtype=torch.long)
        valid = (handle_particle_ids >= 0) & (handle_particle_ids < weight.shape[0])
        handle_particle_ids = handle_particle_ids.clamp(0, weight.shape[0] - 1)
        handle_target_v = handle_target_v.to(device=self.grid_mv.device, dtype=self.grid_mv.dtype)
        if handle_particle_weights is None:
            handle_particle_weights = torch.ones(
                handle_particle_ids.shape[0],
                device=self.grid_mv.device,
                dtype=self.grid_mv.dtype,
            )
        else:
            handle_particle_weights = handle_particle_weights.to(
                device=self.grid_mv.device,
                dtype=self.grid_mv.dtype,
            ).clamp(0.0, 1.0)
        handle_particle_weights = handle_particle_weights * valid.to(dtype=self.grid_mv.dtype)

        handle_index = index.view(weight.shape[0], 27)[handle_particle_ids].reshape(-1)
        handle_weight = (weight[handle_particle_ids] * handle_particle_weights[:, None]).reshape(-1)
        handle_target = handle_target_v[:, None, :].expand(-1, 27, -1).reshape(-1, 3)

        grid_w = self.grid_m.new_zeros(self.grid_m.shape)
        grid_target_v = self.grid_mv.new_zeros(self.grid_mv.shape)
        grid_w = grid_w.index_add(dim=0, index=handle_index, source=handle_weight)
        grid_target_v = grid_target_v.index_add(
            dim=0,
            index=handle_index,
            source=handle_weight.unsqueeze(1) * handle_target,
        )
        active = (grid_w > 1.0e-12).unsqueeze(1)
        grid_target_v = torch.where(active, grid_target_v / grid_w.unsqueeze(1).clamp_min(1.0e-12), grid_target_v)
        self.grid_mv = torch.where(
            active,
            (1.0 - local_blend) * self.grid_mv + local_blend * grid_target_v,
            self.grid_mv,
        )

    def _apply_controller_grid_dirichlet(
        self,
        controller_points: Optional[Tensor],
        controller_target_v: Optional[Tensor],
        radius: float,
        blend: float,
    ) -> None:
        if controller_points is None or controller_target_v is None:
            return
        if controller_points.numel() == 0 or controller_target_v.numel() == 0:
            return
        radius = float(radius)
        blend = min(max(float(blend), 0.0), 1.0)
        if radius <= 0.0 or blend <= 0.0:
            return

        controller_points = controller_points.to(device=self.grid_mv.device, dtype=self.grid_mv.dtype)
        controller_target_v = controller_target_v.to(device=self.grid_mv.device, dtype=self.grid_mv.dtype)
        offsets = _controller_candidate_offsets(radius, self.inv_dx, self.grid_mv.device)

        # Every node within `radius` of a controller point lies in that point's
        # candidate box, so a scatter-min over the (point, offset) candidates is
        # equivalent to the old unique()+cdist over the candidate union — but with
        # static shapes, so nothing here forces a host sync.
        center = torch.round(controller_points * float(self.inv_dx)).long()
        candidate_xyz = center[:, None, :] + offsets[None, :, :]
        in_bounds = ((candidate_xyz >= 0) & (candidate_xyz < int(self.num_grids))).all(dim=-1).reshape(-1)
        candidate_xyz = candidate_xyz.clamp(0, int(self.num_grids) - 1)
        candidate_idx = (
            candidate_xyz[..., 0] * self.num_grids * self.num_grids
            + candidate_xyz[..., 1] * self.num_grids
            + candidate_xyz[..., 2]
        ).reshape(-1)

        grid_pos = (
            self.grid_x.index_select(0, candidate_idx)
            .to(device=self.grid_mv.device, dtype=self.grid_mv.dtype)
            * float(self.dx)
        )
        distances = torch.cdist(grid_pos.unsqueeze(0), controller_points.unsqueeze(0)).squeeze(0)
        nearest_dist, nearest_idx = distances.min(dim=-1)
        nearest_dist = torch.where(in_bounds, nearest_dist, nearest_dist.new_full((), float('inf')))
        packed = torch.full(
            (self.grid_m.shape[0],), _DIRICHLET_EMPTY, device=self.grid_mv.device, dtype=torch.long,
        ).index_reduce(0, candidate_idx, _pack_nearest(nearest_dist, nearest_idx), 'amin', include_self=True)
        node_dist, node_point = _unpack_nearest(packed)

        active = ((node_dist <= radius) & (self.grid_m > 1.0e-15)).unsqueeze(1)
        target_v = controller_target_v.index_select(0, node_point.clamp(max=controller_points.shape[0] - 1))
        self.grid_mv = torch.where(
            active, (1.0 - blend) * self.grid_mv + blend * target_v, self.grid_mv,
        )
    
    def p2g2p(
        self,
        x: Tensor,
        v: Tensor,
        C: Tensor,
        F: Tensor,
        stress: Tensor,
        handle_particle_ids: Optional[Tensor] = None,
        handle_target_v: Optional[Tensor] = None,
        handle_particle_weights: Optional[Tensor] = None,
        handle_grid_velocity_blend: float = 0.0,
        controller_grid_points: Optional[Tensor] = None,
        controller_grid_target_v: Optional[Tensor] = None,
        controller_grid_contact_radius: float = 0.0,
        controller_grid_velocity_blend: float = 0.0,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        
        # prepare constants
        dt = self.dt
        vol = self.vol
        p_mass = self.p_mass 
        dx = self.dx
        inv_dx = self.inv_dx 
        n_grids = self.num_grids
        n_particles = self.n_particles
        clip_bound = self.clip_bound
        
        # calculate temporary variables for both p2g and g2p (weight, dpos, index)
        px = x * inv_dx
        base = (px - 0.5).long() # (n_particles, 3)
        fx = px - base.float() # (n_particles, 3)
        
        w = [
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1) ** 2,
                0.5 * (fx - 0.5) ** 2
        ]
        w = torch.stack(w, dim=-1) # (n_particles, 3, 3)
        w_e = torch.einsum('bi, bj, bk -> bijk', w[:, 0], w[:, 1], w[:, 2]) # (n_particles, 3, 3, 3)
        weight = w_e.reshape(-1, 27) # (n_particles, 27)
        
        dw = [
            fx - 1.5,
            -2.0 * (fx - 1.0),
            fx - 0.5
        ]
        dw = torch.stack(dw, dim=-1) # (n_particles, 3, 3)
        dweight = [
            torch.einsum('pi,pj,pk->pijk', dw[:, 0], w[:, 1], w[:, 2]),
            torch.einsum('pi,pj,pk->pijk', w[:, 0], dw[:, 1], w[:, 2]),
            torch.einsum('pi,pj,pk->pijk', w[:, 0], w[:, 1], dw[:, 2])
        ]
        dweight = inv_dx * torch.stack(dweight, dim=-1).reshape(-1, 27, 3) # (n_particles, 3, 3, 3, 3) -> (n_particles, 27, 3)
        
        dpos = (self.offset - fx.unsqueeze(1)) * dx # (n_particles, 27, 3)
        
        index = base.unsqueeze(1) + self.offset.unsqueeze(0).long() # (n_particles, 27, 3)
        index = (index[:, :, 0] * n_grids * n_grids + index[:, :, 1] * n_grids + index[:, :, 2]).reshape(-1) # (n_particles * 27)
        index = index.clamp(0, n_grids ** 3 - 1) # (n_particles * 27) TODO: simple clipping leads to some numerical problems, but it's acceptable for now.
        
        # zero grid
        self.grid_mv = torch.zeros_like(self.grid_mv)
        self.grid_m = torch.zeros_like(self.grid_m)
        
        # pre-particle operation
        for operation in self.pre_particle_process:
            operation(self, x, v)
        
        # p2g
        mv = -dt * vol * torch.einsum('bij, bkj -> bki', stress, dweight) +\
            p_mass * weight.unsqueeze(2) * (v.unsqueeze(1) + torch.einsum('bij, bkj -> bki', C, dpos)) # (n_particles, 3, 3), (n_particles, 27, 3) -> (n_particles, 27, 3)
        mv = mv.reshape(-1, 3) # (n_particles * 27, 3)
        
        m = weight * p_mass # (n_particles, 27)
        m = m.reshape(-1) # (n_particles * 27)
        
        self.grid_mv = self.grid_mv.index_add(dim=0, index=index, source=mv) # (n_grid * n_grid * n_grid, 3)
        self.grid_m = self.grid_m.index_add(dim=0, index=index, source=m) # (n_grid * n_grid * n_grid)        
        
        # grid update
        self.grid_update()
        self._apply_handle_grid_velocity_constraint(
            index=index,
            weight=weight,
            handle_particle_ids=handle_particle_ids,
            handle_target_v=handle_target_v,
            handle_particle_weights=handle_particle_weights,
            blend=handle_grid_velocity_blend,
        )
        self._apply_controller_grid_dirichlet(
            controller_points=controller_grid_points,
            controller_target_v=controller_grid_target_v,
            radius=controller_grid_contact_radius,
            blend=controller_grid_velocity_blend,
        )
        
        # post-grid operation
        for operation in self.post_grid_process:
            operation(self)
        
        # g2p
        v = self.grid_mv.index_select(dim=0, index=index).reshape(-1, 27, 3) # (n_particles, 27, 3)
        C = torch.einsum('bij, bik -> bijk', v, dpos) # (n_particles, 27, 3), (n_particles, 27, 3) -> (n_particles, 27, 3, 3)
        new_F = torch.einsum('bij, bik -> bijk', v, dweight) # (n_particles, 27, 3), (n_particles, 27, 3) -> (n_particles, 27, 3, 3)
        
        v = (weight.unsqueeze(2) * v).sum(dim=1) # (n_particles, 3)
        C = (4.0 * inv_dx * inv_dx * weight.unsqueeze(2).unsqueeze(3) * C).sum(dim=1)# (n_particles, 3, 3)
        new_F = dt * new_F.sum(dim=1) # (n_particles, 3, 3)
        
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        x = x + v * dt
        x = torch.nan_to_num(
            x,
            nan=clip_bound,
            posinf=1.0 - clip_bound,
            neginf=clip_bound,
        ).clamp(clip_bound, 1.0 - clip_bound)
        F = F + torch.bmm(new_F, F)
        F = torch.nan_to_num(F, nan=0.0, posinf=2.0, neginf=-2.0).clamp(-2.0, 2.0)
        self.time += dt
        
        return x, v, C, F
    
    def grid_update(self) -> None:
        # torch.where, not bool-mask indexing: same result without the nonzero() sync.
        mass = self.grid_m.unsqueeze(1)
        self.grid_mv = torch.where(mass > 1e-15, self.grid_mv / mass.clamp_min(1e-15), self.grid_mv)
        self.grid_mv = self.damping * (self.grid_mv + self.dt * self.gravity)

    def pre_p2g_operation(self) -> None:
        pass
    
    def post_grid_operation(self) -> None:
        pass


class BatchedMPMModel:
    """Batched variant of :class:`MPMModel` for same-size particle batches.

    The original MPMModel intentionally keeps one simulation grid per object.
    This class keeps the same math, but stores B independent grids in one
    flattened tensor and offsets P2G indices by batch id.  That lets the heavy
    scatter/gather work run as one larger torch operation without mixing mass
    or momentum between batch elements.
    """

    def __init__(
        self,
        sim_params: DictConfig,
        material_params: DictConfig,
        init_pos: Tensor,
        enable_train: bool = False,
        device: torch.device = 'cuda',
    ):
        if init_pos.ndim != 3:
            raise ValueError(f'BatchedMPMModel expects init_pos shape (B, N, 3), got {tuple(init_pos.shape)}')

        self.num_grids: int = sim_params['num_grids']
        self.dt: float = sim_params['dt']
        self.gravity: Tensor = torch.tensor(sim_params['gravity'], device=device)
        self.boundary_condition: Optional[DictConfig] = sim_params.get('boundary_condition', None)

        self.dx: float = 1 / self.num_grids
        self.inv_dx: float = float(self.num_grids)

        self.clip_bound: float = sim_params.get('clip_bound', 0.5) * self.dx
        self.damping = sim_params.get('damping', 1.0)
        assert self.clip_bound >= 0.0
        assert self.damping >= 0.0 and self.damping <= 1.0

        self.batch_size: int = init_pos.shape[0]
        self.n_particles: int = init_pos.shape[1]
        self.init_pos: Tensor = init_pos.detach()

        size = material_params['size']
        if isinstance(size, Tensor):
            size_tensor = size.to(device=device, dtype=init_pos.dtype)
        else:
            size_tensor = torch.as_tensor(size, device=device, dtype=init_pos.dtype)
        if size_tensor.ndim == 1:
            size_tensor = size_tensor.unsqueeze(0).expand(self.batch_size, -1)
        if size_tensor.shape != (self.batch_size, 3):
            raise ValueError(
                'Batched material size must have shape (B, 3), '
                f'got {tuple(size_tensor.shape)} for batch_size={self.batch_size}'
            )
        self.size: Tensor = size_tensor
        self.center = material_params.get('center', None)
        self.vol: Tensor = torch.prod(self.size, dim=-1) / self.n_particles
        rho = float(material_params['rho'])
        self.p_mass: Tensor = rho * self.vol

        self.enable_train: bool = enable_train
        self.device: torch.device = device

        num_grids = self.num_grids
        n_dim = 3
        num_grid_nodes = num_grids ** n_dim
        self.num_grid_nodes = num_grid_nodes
        self.grid_mv = torch.empty((self.batch_size * num_grid_nodes, n_dim), device=device)
        self.grid_m = torch.empty((self.batch_size * num_grid_nodes,), device=device)
        grid_ranges = torch.arange(num_grids, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(grid_ranges, grid_ranges, grid_ranges, indexing='ij')
        self.grid_x = torch.stack((grid_x, grid_y, grid_z), dim=-1).reshape(-1, 3).float()
        self.batched_grid_x = self.grid_x.unsqueeze(0).expand(self.batch_size, -1, -1).reshape(-1, 3)

        self.offset = torch.tensor(
            [[i, j, k] for i in range(3) for j in range(3) for k in range(3)],
            device=device,
        ).float()
        self.batch_grid_offsets = (
            torch.arange(self.batch_size, device=device, dtype=torch.long) * num_grid_nodes
        ).view(self.batch_size, 1, 1)

        self.pre_particle_process = []
        self.post_grid_process = []

        self.time = 0.0

    def reset(self) -> None:
        self.time = 0.0

    def __call__(self, x: Tensor, v: Tensor, C: Tensor, F: Tensor, stress: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.p2g2p(x, v, C, F, stress)

    def _apply_controller_grid_dirichlet(
        self,
        controller_points: Optional[Tensor],
        controller_target_v: Optional[Tensor],
        radius: float,
        blend: float,
    ) -> None:
        if controller_points is None or controller_target_v is None:
            return
        if controller_points.numel() == 0 or controller_target_v.numel() == 0:
            return
        radius = float(radius)
        blend = min(max(float(blend), 0.0), 1.0)
        if radius <= 0.0 or blend <= 0.0:
            return

        controller_points = controller_points.to(device=self.grid_mv.device, dtype=self.grid_mv.dtype)
        controller_target_v = controller_target_v.to(device=self.grid_mv.device, dtype=self.grid_mv.dtype)
        if controller_points.ndim == 2:
            controller_points = controller_points.unsqueeze(0).expand(self.batch_size, -1, -1)
        if controller_target_v.ndim == 2:
            controller_target_v = controller_target_v.unsqueeze(0).expand(self.batch_size, -1, -1)
        if controller_points.ndim != 3 or controller_target_v.ndim != 3:
            return
        if controller_points.shape[0] != self.batch_size or controller_target_v.shape[0] != self.batch_size:
            return

        if controller_points.shape[1] == 0:
            return
        offsets = _controller_candidate_offsets(radius, self.inv_dx, self.grid_mv.device)
        grid_x = self.grid_x.to(device=self.grid_mv.device)
        num_points = int(controller_points.shape[1])

        # Same scatter-min as MPMModel, with the per-batch grid offset folded into
        # the scatter index so the batch loop disappears too.
        center = torch.round(controller_points * float(self.inv_dx)).long()
        candidate_xyz = center[:, :, None, :] + offsets[None, None, :, :]
        in_bounds = ((candidate_xyz >= 0) & (candidate_xyz < int(self.num_grids))).all(dim=-1).reshape(-1)
        candidate_xyz = candidate_xyz.clamp(0, int(self.num_grids) - 1)
        local_idx = (
            candidate_xyz[..., 0] * self.num_grids * self.num_grids
            + candidate_xyz[..., 1] * self.num_grids
            + candidate_xyz[..., 2]
        )
        global_idx = (local_idx + self.batch_grid_offsets.view(self.batch_size, 1, 1)).reshape(-1)

        grid_pos = grid_x.index_select(0, local_idx.reshape(-1)).to(dtype=self.grid_mv.dtype) * float(self.dx)
        grid_pos = grid_pos.view(self.batch_size, -1, 3)
        distances = torch.cdist(grid_pos, controller_points)
        nearest_dist, nearest_idx = distances.min(dim=-1)
        nearest_dist = nearest_dist.reshape(-1)
        nearest_dist = torch.where(in_bounds, nearest_dist, nearest_dist.new_full((), float('inf')))
        # Offset the point id by batch so a single index_select recovers the velocity.
        point_id = nearest_idx.reshape(-1) + (
            torch.arange(self.batch_size, device=self.grid_mv.device) * num_points
        ).repeat_interleave(nearest_idx.shape[1])
        packed = torch.full(
            (self.grid_m.shape[0],), _DIRICHLET_EMPTY, device=self.grid_mv.device, dtype=torch.long,
        ).index_reduce(0, global_idx, _pack_nearest(nearest_dist, point_id), 'amin', include_self=True)
        node_dist, node_point = _unpack_nearest(packed)

        active = ((node_dist <= radius) & (self.grid_m > 1.0e-15)).unsqueeze(1)
        target_v = controller_target_v.reshape(-1, 3).index_select(
            0, node_point.clamp(max=self.batch_size * num_points - 1),
        )
        self.grid_mv = torch.where(
            active, (1.0 - blend) * self.grid_mv + blend * target_v, self.grid_mv,
        )

    def p2g2p(
        self,
        x: Tensor,
        v: Tensor,
        C: Tensor,
        F: Tensor,
        stress: Tensor,
        controller_grid_points: Optional[Tensor] = None,
        controller_grid_target_v: Optional[Tensor] = None,
        controller_grid_contact_radius: float = 0.0,
        controller_grid_velocity_blend: float = 0.0,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if x.ndim != 3:
            raise ValueError(f'BatchedMPMModel.p2g2p expects x shape (B, N, 3), got {tuple(x.shape)}')

        dt = self.dt
        dx = self.dx
        inv_dx = self.inv_dx
        n_grids = self.num_grids
        clip_bound = self.clip_bound
        batch_size, n_particles, _ = x.shape
        if batch_size != self.batch_size or n_particles != self.n_particles:
            raise ValueError(
                'BatchedMPMModel was initialized for '
                f'(B={self.batch_size}, N={self.n_particles}), got {tuple(x.shape)}'
            )

        vol = self.vol.view(batch_size, 1, 1)
        p_mass = self.p_mass.view(batch_size, 1, 1)

        px = x * inv_dx
        base = (px - 0.5).long()
        fx = px - base.float()

        w = [
            0.5 * (1.5 - fx) ** 2,
            0.75 - (fx - 1) ** 2,
            0.5 * (fx - 0.5) ** 2,
        ]
        w = torch.stack(w, dim=-1)
        w_e = torch.einsum('bni,bnj,bnk->bnijk', w[:, :, 0], w[:, :, 1], w[:, :, 2])
        weight = w_e.reshape(batch_size, n_particles, 27)

        dw = [
            fx - 1.5,
            -2.0 * (fx - 1.0),
            fx - 0.5,
        ]
        dw = torch.stack(dw, dim=-1)
        dweight = [
            torch.einsum('bni,bnj,bnk->bnijk', dw[:, :, 0], w[:, :, 1], w[:, :, 2]),
            torch.einsum('bni,bnj,bnk->bnijk', w[:, :, 0], dw[:, :, 1], w[:, :, 2]),
            torch.einsum('bni,bnj,bnk->bnijk', w[:, :, 0], w[:, :, 1], dw[:, :, 2]),
        ]
        dweight = inv_dx * torch.stack(dweight, dim=-1).reshape(batch_size, n_particles, 27, 3)

        dpos = (self.offset.view(1, 1, 27, 3) - fx.unsqueeze(2)) * dx

        index_xyz = base.unsqueeze(2) + self.offset.view(1, 1, 27, 3).long()
        local_index = (
            index_xyz[:, :, :, 0] * n_grids * n_grids
            + index_xyz[:, :, :, 1] * n_grids
            + index_xyz[:, :, :, 2]
        ).clamp(0, self.num_grid_nodes - 1)
        index = (local_index + self.batch_grid_offsets).reshape(-1)

        self.grid_mv = torch.zeros_like(self.grid_mv)
        self.grid_m = torch.zeros_like(self.grid_m)

        for operation in self.pre_particle_process:
            operation(self, x, v)

        mv = (
            -dt * vol.unsqueeze(-1) * torch.einsum('bnij,bnkj->bnki', stress, dweight)
            + p_mass.unsqueeze(-1)
            * weight.unsqueeze(-1)
            * (v.unsqueeze(2) + torch.einsum('bnij,bnkj->bnki', C, dpos))
        )
        mv = mv.reshape(-1, 3)

        m = (weight * self.p_mass.view(batch_size, 1, 1)).reshape(-1)

        self.grid_mv = self.grid_mv.index_add(dim=0, index=index, source=mv)
        self.grid_m = self.grid_m.index_add(dim=0, index=index, source=m)

        self.grid_update()
        self._apply_controller_grid_dirichlet(
            controller_points=controller_grid_points,
            controller_target_v=controller_grid_target_v,
            radius=controller_grid_contact_radius,
            blend=controller_grid_velocity_blend,
        )

        for operation in self.post_grid_process:
            operation(self)

        gathered_v = self.grid_mv.index_select(dim=0, index=index).reshape(batch_size, n_particles, 27, 3)
        v = (weight.unsqueeze(-1) * gathered_v).sum(dim=2)
        C = 4.0 * inv_dx * inv_dx * torch.einsum(
            'bnq,bnqj,bnqk->bnjk',
            weight,
            gathered_v,
            dpos,
        )
        F_delta = dt * torch.einsum('bnqj,bnqk->bnjk', gathered_v, dweight)

        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        x = x + v * dt
        x = torch.nan_to_num(
            x,
            nan=clip_bound,
            posinf=1.0 - clip_bound,
            neginf=clip_bound,
        ).clamp(clip_bound, 1.0 - clip_bound)
        F = F + torch.matmul(F_delta, F)
        F = torch.nan_to_num(F, nan=0.0, posinf=2.0, neginf=-2.0).clamp(-2.0, 2.0)
        self.time += dt

        return x, v, C, F

    def grid_update(self) -> None:
        # torch.where, not bool-mask indexing: same result without the nonzero() sync.
        mass = self.grid_m.unsqueeze(1)
        self.grid_mv = torch.where(mass > 1e-15, self.grid_mv / mass.clamp_min(1e-15), self.grid_mv)
        self.grid_mv = self.damping * (self.grid_mv + self.dt * self.gravity)
