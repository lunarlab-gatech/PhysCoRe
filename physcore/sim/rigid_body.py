"""
Rigid body generation, SDF evaluation, and surface point sampling.
"""

import torch
import numpy as np
from torch import Tensor
from typing import List, Tuple, Optional, Callable
from collections import deque

from .primitives import Primitive, evaluate_union_sdf
from .zeroverse_composition import (
    estimate_primitives_bounds,
    generate_zeroverse_composite_primitives,
)
from .fps import farthest_point_sampling


def check_connectivity(
    primitives: List[Primitive],
    resolution: int = 64,
    bounds: Tuple[float, float] = (0.0, 1.0),
    device: str = 'cuda',
) -> bool:
    """Check if the union of primitives forms a single connected component
    via flood fill on a voxel grid.
    """
    lo, hi = bounds
    lin = torch.linspace(lo, hi, resolution, device=device)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing='ij')
    grid_points = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

    sdf = evaluate_union_sdf(grid_points, primitives, device)
    inside = (sdf <= 0).cpu().numpy()
    inside_3d = inside.reshape(resolution, resolution, resolution)

    if not inside.any():
        return False

    # BFS flood fill
    visited = np.zeros_like(inside_3d, dtype=bool)
    start = tuple(np.argwhere(inside_3d)[0])
    queue = deque([start])
    visited[start] = True
    count = 1

    while queue:
        ci, cj, ck = queue.popleft()
        for di, dj, dk in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            ni, nj, nk = ci+di, cj+dj, ck+dk
            if (0 <= ni < resolution and 0 <= nj < resolution and
                0 <= nk < resolution and inside_3d[ni, nj, nk] and
                not visited[ni, nj, nk]):
                visited[ni, nj, nk] = True
                queue.append((ni, nj, nk))
                count += 1

    total_inside = inside.sum()
    return count >= 0.9 * total_inside  # Allow small disconnected voxels


def generate_connected_rigid_body(
    rng: np.random.Generator,
    max_attempts: int = 50,
    num_primitives_probs: Optional[List[float]] = None,
    scale_range: Tuple[float, float] = (0.02, 0.08),
    center: Optional[List[float]] = None,
    spread: float = 0.1,
    device: str = 'cuda',
    sub_object_numbers: Optional[List[int]] = None,
    primitive_weights: Optional[List[float]] = None,
    axis_range: Tuple[float, float] = (0.25, 1.2),
    translate_range_rate: Tuple[float, float] = (0.0, 0.2),
    rotate_range: Tuple[float, float] = (0.0, 55.0),
    max_dim_range: Tuple[float, float] = (0.04, 0.10),
) -> List[Primitive]:
    """Generate a rigid body composed of random primitives that forms a
    single connected component using a Zeroverse-style chained composition.
    """
    if center is None:
        center = [0.5, 0.5, 0.7]

    center = np.array(center)

    for attempt in range(max_attempts):
        primitives = generate_zeroverse_composite_primitives(
            rng=rng,
            sub_object_numbers=sub_object_numbers or [1, 2, 3, 4],
            sub_object_weights=num_primitives_probs or [3, 3, 2, 1],
            primitive_weights=primitive_weights or [0.8, 0.8, 2.5],
            axis_range=axis_range,
            translate_range_rate=translate_range_rate,
            rotate_range=rotate_range,
            max_dim_range=max_dim_range,
            center=center.tolist(),
        )

        if check_connectivity(primitives, resolution=48, device=device):
            return primitives

    print(
        f"Warning: Could not generate connected rigid body in {max_attempts} attempts. "
        "Using a compact spherical poking tool."
    )
    return [Primitive(
        ptype='sphere',
        params={'radius': 0.5},
        scale=np.array([max_dim_range[1] * 0.5] * 3),
        rotation=np.eye(3),
        translation=center,
    )]


def sample_surface_points(
    primitives: List[Primitive],
    n_surface: int = 2048,
    n_fps: int = 1024,
    resolution: int = 64,
    bounds: Tuple[float, float] = (0.0, 1.0),
    device: str = 'cuda',
) -> Tensor:
    """Sample points on the surface of the union SDF using voxel-based detection + FPS.

    Returns: (n_fps, 3) tensor of surface points.
    """
    lo, hi = bounds
    dx = (hi - lo) / resolution
    lin = torch.linspace(lo, hi, resolution, device=device)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing='ij')
    grid_points = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

    sdf = evaluate_union_sdf(grid_points, primitives, device)

    # Surface voxels: SDF close to zero (within one voxel width)
    surface_mask = sdf.abs() < dx * 1.5
    surface_points = grid_points[surface_mask]

    if surface_points.shape[0] == 0:
        # Fallback: use points with smallest |SDF|
        _, indices = sdf.abs().topk(n_surface, largest=False)
        surface_points = grid_points[indices]

    if surface_points.shape[0] > n_surface:
        perm = torch.randperm(surface_points.shape[0], device=device)[:n_surface]
        surface_points = surface_points[perm]

    # FPS to get well-distributed subset
    n_out = min(n_fps, surface_points.shape[0])
    fps_indices = farthest_point_sampling(surface_points, n_out)
    return surface_points[fps_indices]


class RigidBody:
    """Animated rigid body with SDF collision and surface tracking.

    Supports piecewise-linear waypoint trajectories for diverse manipulation.
    """

    def __init__(
        self,
        primitives: List[Primitive],
        surface_points: Tensor,
        device: str = 'cuda',
    ):
        self.primitives = primitives
        self.surface_points_local = surface_points  # (M, 3) in initial frame
        self.device = device
        self._wp_times: Optional[List[float]] = None
        self._wp_pos: Optional[List[Tensor]] = None
        self.z_min: Optional[float] = None
        self.initial_center = self._compute_initial_center()
        self.local_bounds_min, self.local_bounds_max = self._compute_local_bounds()

    def _compute_initial_center(self) -> Tensor:
        """Use the rigid body's bounding-box center as the motion reference.

        Zeroverse composition places the tool by its bbox center, not by the
        mean of primitive translations. Using the bbox center keeps the visual
        surface, the SDF, and the waypoint trajectory in the same frame.
        """
        if self.surface_points_local.numel() > 0:
            bounds_min = self.surface_points_local.min(dim=0).values
            bounds_max = self.surface_points_local.max(dim=0).values
            return 0.5 * (bounds_min + bounds_max)

        centers = np.array([p.translation for p in self.primitives])
        return torch.tensor(centers, device=self.device, dtype=torch.float32).mean(dim=0)

    def _compute_local_bounds(self) -> Tuple[Tensor, Tensor]:
        """Compute the rigid body's axis-aligned bounds in its initial pose."""
        if self.primitives:
            bounds_min, bounds_max = estimate_primitives_bounds(self.primitives)
            return (
                torch.tensor(bounds_min, device=self.device, dtype=torch.float32),
                torch.tensor(bounds_max, device=self.device, dtype=torch.float32),
            )

        if self.surface_points_local.numel() > 0:
            return (
                self.surface_points_local.min(dim=0).values,
                self.surface_points_local.max(dim=0).values,
            )

        zeros = torch.zeros(3, device=self.device, dtype=torch.float32)
        return zeros.clone(), zeros.clone()

    def set_waypoint_trajectory(
        self,
        waypoints: List[Tuple[float, List[float]]],
        z_min: Optional[float] = None,
    ):
        """Set piecewise-linear trajectory from (time, [x,y,z]) waypoints.

        Args:
            waypoints: List of (time, [x,y,z]) sorted by ascending time.
            z_min: Minimum z for the body center (ground clamp).
        """
        assert len(waypoints) >= 2, "Need at least 2 waypoints"
        self._wp_times = [float(w[0]) for w in waypoints]
        self._wp_pos = [
            torch.tensor(w[1], device=self.device, dtype=torch.float32)
            for w in waypoints
        ]
        self.z_min = z_min

    def _find_segment(self, t: float) -> int:
        """Return segment index i where times[i] <= t < times[i+1].
        Returns -1 if before first, len(times) if after last."""
        times = self._wp_times
        if times is None or t <= times[0]:
            return -1
        if t >= times[-1]:
            return len(times)
        for i in range(len(times) - 1):
            if times[i] <= t < times[i + 1]:
                return i
        return len(times) - 2

    def get_displacement(self, t: float) -> Tensor:
        """Get rigid body center position at time t via waypoint interpolation."""
        seg = self._find_segment(t)
        times = self._wp_times
        positions = self._wp_pos

        if seg < 0:
            pos = positions[0].clone()
        elif seg >= len(times) - 1:
            pos = positions[-1].clone()
        else:
            dt_seg = times[seg + 1] - times[seg]
            alpha = (t - times[seg]) / max(dt_seg, 1e-10)
            pos = (1.0 - alpha) * positions[seg] + alpha * positions[seg + 1]

        if self.z_min is not None and pos[2].item() < self.z_min:
            pos = pos.clone()
            pos[2] = self.z_min
        return pos

    def get_velocity(self, t: float) -> Tensor:
        """Get rigid body velocity at time t (piecewise constant per segment)."""
        seg = self._find_segment(t)
        times = self._wp_times
        positions = self._wp_pos

        if seg < 0 or seg >= len(times) - 1:
            return torch.zeros(3, device=self.device)

        dt_seg = times[seg + 1] - times[seg]
        vel = (positions[seg + 1] - positions[seg]) / max(dt_seg, 1e-10)

        # If clamped at z_min, zero out downward z but keep lateral velocity
        if self.z_min is not None:
            alpha = (t - times[seg]) / max(dt_seg, 1e-10)
            interp_z = ((1.0 - alpha) * positions[seg][2]
                        + alpha * positions[seg + 1][2]).item()
            if interp_z <= self.z_min and vel[2].item() < 0:
                vel = vel.clone()
                vel[2] = 0.0

        return vel

    def sdf(self, points: Tensor, t: float) -> Tensor:
        """Evaluate rigid body SDF at world-space points at time t."""
        displacement = self.get_displacement(t)
        # Transform points to rigid body's local frame
        # For pure translation: local = world - displacement + initial_center
        # The primitives are defined with their own transforms already.
        # We apply an additional global translation offset.
        offset = displacement - self.initial_center
        p_local = points - offset
        return evaluate_union_sdf(p_local, self.primitives, self.device)

    def get_surface_points(self, t: float) -> Tensor:
        """Get transformed surface points at time t."""
        displacement = self.get_displacement(t)
        offset = displacement - self.initial_center
        return self.surface_points_local + offset

    def get_world_bounds(self, t: float, padding: float = 0.0) -> Tuple[Tensor, Tensor]:
        """Get the rigid body's world-space axis-aligned bounds at time t."""
        displacement = self.get_displacement(t)
        offset = displacement - self.initial_center
        padding_tensor = torch.full((3,), float(padding), device=self.device, dtype=torch.float32)
        return (
            self.local_bounds_min + offset - padding_tensor,
            self.local_bounds_max + offset + padding_tensor,
        )

    def _compute_center(self) -> Tensor:
        """Return the cached rigid-body reference center."""
        return self.initial_center


class BatchedRigidBody:
    """Batched rigid body for shared primitive geometry and batched motion.

    This covers the common training case where samples come from the same
    episode/tool but start at different frames.  The primitive SDF is shared;
    only the per-batch translational trajectory differs.
    """

    def __init__(
        self,
        primitives: List[Primitive],
        surface_points: Tensor,
        device: str = 'cuda',
    ):
        self.primitives = primitives
        self.surface_points_local = surface_points
        self.device = device
        self._wp_times: Optional[Tensor] = None
        self._wp_pos: Optional[Tensor] = None
        self.z_min: Optional[float] = None
        self.initial_center = self._compute_initial_center()
        self.local_bounds_min, self.local_bounds_max = self._compute_local_bounds()

    def _compute_initial_center(self) -> Tensor:
        if self.surface_points_local.numel() > 0:
            bounds_min = self.surface_points_local.min(dim=0).values
            bounds_max = self.surface_points_local.max(dim=0).values
            return 0.5 * (bounds_min + bounds_max)

        centers = np.array([p.translation for p in self.primitives])
        return torch.tensor(centers, device=self.device, dtype=torch.float32).mean(dim=0)

    def _compute_local_bounds(self) -> Tuple[Tensor, Tensor]:
        if self.primitives:
            bounds_min, bounds_max = estimate_primitives_bounds(self.primitives)
            return (
                torch.tensor(bounds_min, device=self.device, dtype=torch.float32),
                torch.tensor(bounds_max, device=self.device, dtype=torch.float32),
            )

        if self.surface_points_local.numel() > 0:
            return (
                self.surface_points_local.min(dim=0).values,
                self.surface_points_local.max(dim=0).values,
            )

        zeros = torch.zeros(3, device=self.device, dtype=torch.float32)
        return zeros.clone(), zeros.clone()

    def set_waypoint_trajectory(
        self,
        times: Tensor,
        centers: Tensor,
        z_min: Optional[float] = None,
    ) -> None:
        """Set batched piecewise-linear center trajectories.

        Args:
            times: ``(T,)`` waypoint times, shared across batch.
            centers: ``(B, T, 3)`` rigid-body centers.
        """
        if centers.ndim != 3 or centers.shape[-1] != 3:
            raise ValueError(f'centers must have shape (B, T, 3), got {tuple(centers.shape)}')
        if times.ndim != 1 or int(times.shape[0]) != int(centers.shape[1]):
            raise ValueError(
                f'times must have shape (T,) matching centers, got times={tuple(times.shape)} '
                f'centers={tuple(centers.shape)}'
            )
        if int(times.shape[0]) < 2:
            raise ValueError('Need at least 2 batched rigid-body waypoints')
        self._wp_times = times.to(device=self.device, dtype=torch.float32)
        self._wp_pos = centers.to(device=self.device, dtype=torch.float32)
        self.z_min = z_min

    @property
    def batch_size(self) -> int:
        if self._wp_pos is None:
            raise RuntimeError('Call set_waypoint_trajectory() before using BatchedRigidBody')
        return int(self._wp_pos.shape[0])

    def _find_segment(self, t: float) -> int:
        times = self._wp_times
        if times is None:
            raise RuntimeError('Call set_waypoint_trajectory() before querying BatchedRigidBody')
        t_float = float(t)
        if t_float <= float(times[0].item()):
            return -1
        if t_float >= float(times[-1].item()):
            return int(times.shape[0])
        # Shared scalar simulation time, so one Python segment lookup is enough.
        for i in range(int(times.shape[0]) - 1):
            if float(times[i].item()) <= t_float < float(times[i + 1].item()):
                return i
        return int(times.shape[0]) - 2

    def get_displacement(self, t: float) -> Tensor:
        seg = self._find_segment(t)
        times = self._wp_times
        positions = self._wp_pos
        assert times is not None and positions is not None

        if seg < 0:
            pos = positions[:, 0].clone()
        elif seg >= int(times.shape[0]) - 1:
            pos = positions[:, -1].clone()
        else:
            dt_seg = float((times[seg + 1] - times[seg]).item())
            alpha = (float(t) - float(times[seg].item())) / max(dt_seg, 1e-10)
            pos = (1.0 - alpha) * positions[:, seg] + alpha * positions[:, seg + 1]

        if self.z_min is not None:
            pos = pos.clone()
            pos[:, 2] = pos[:, 2].clamp_min(float(self.z_min))
        return pos

    def get_velocity(self, t: float) -> Tensor:
        seg = self._find_segment(t)
        times = self._wp_times
        positions = self._wp_pos
        assert times is not None and positions is not None

        if seg < 0 or seg >= int(times.shape[0]) - 1:
            return torch.zeros(positions.shape[0], 3, device=self.device, dtype=positions.dtype)

        dt_seg = float((times[seg + 1] - times[seg]).item())
        vel = (positions[:, seg + 1] - positions[:, seg]) / max(dt_seg, 1e-10)

        if self.z_min is not None:
            alpha = (float(t) - float(times[seg].item())) / max(dt_seg, 1e-10)
            interp_z = (1.0 - alpha) * positions[:, seg, 2] + alpha * positions[:, seg + 1, 2]
            clamp_mask = (interp_z <= float(self.z_min)) & (vel[:, 2] < 0)
            if clamp_mask.any():
                vel = vel.clone()
                vel[clamp_mask, 2] = 0.0
        return vel

    def sdf(self, points: Tensor, t: float) -> Tensor:
        if points.ndim != 3:
            raise ValueError(f'BatchedRigidBody.sdf expects points shape (B, N, 3), got {tuple(points.shape)}')
        displacement = self.get_displacement(t).to(device=points.device, dtype=points.dtype)
        offset = displacement - self.initial_center.to(device=points.device, dtype=points.dtype)
        p_local = points - offset.unsqueeze(1)
        sdf = evaluate_union_sdf(p_local.reshape(-1, 3), self.primitives, self.device)
        return sdf.reshape(points.shape[:-1])

    def sdf_indexed(self, points: Tensor, batch_indices: Tensor, t: float) -> Tensor:
        if points.numel() == 0:
            return points.new_zeros(points.shape[:-1])
        displacement = self.get_displacement(t).to(device=points.device, dtype=points.dtype)
        initial_center = self.initial_center.to(device=points.device, dtype=points.dtype)
        offset = displacement.index_select(0, batch_indices.to(device=points.device, dtype=torch.long)) - initial_center
        p_local = points - offset
        return evaluate_union_sdf(p_local, self.primitives, self.device)

    def get_world_bounds(self, t: float, padding: float = 0.0) -> Tuple[Tensor, Tensor]:
        displacement = self.get_displacement(t)
        offset = displacement - self.initial_center.to(device=displacement.device, dtype=displacement.dtype)
        padding_tensor = torch.full((3,), float(padding), device=displacement.device, dtype=displacement.dtype)
        return (
            self.local_bounds_min.to(device=displacement.device, dtype=displacement.dtype).unsqueeze(0)
            + offset
            - padding_tensor,
            self.local_bounds_max.to(device=displacement.device, dtype=displacement.dtype).unsqueeze(0)
            + offset
            + padding_tensor,
        )
