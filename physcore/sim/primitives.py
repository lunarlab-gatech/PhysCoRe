"""
SDF primitives and random shape generation.

Supports 5 basic shapes: cube, sphere, cylinder, cone, torus.
All SDFs are evaluated in batched PyTorch tensors.
"""

import torch
import numpy as np
from torch import Tensor
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from scipy.spatial.transform import Rotation


# ─────────────────── SDF Functions (origin-centered, unit-ish) ───────────────

def sdf_box(p: Tensor, half_extents: Tensor) -> Tensor:
    """Exact SDF for axis-aligned box centered at origin.
    p: (N, 3), half_extents: (3,)
    """
    q = p.abs() - half_extents
    return q.clamp(min=0).norm(dim=-1) + q.max(dim=-1).values.clamp(max=0)


def sdf_sphere(p: Tensor, radius: float) -> Tensor:
    """p: (N, 3)"""
    return p.norm(dim=-1) - radius


def sdf_cylinder(p: Tensor, radius: float, half_height: float) -> Tensor:
    """Capped cylinder along z-axis. p: (N, 3)"""
    d_radial = p[:, :2].norm(dim=-1) - radius
    d_height = p[:, 2].abs() - half_height
    d = torch.stack([d_radial, d_height], dim=-1)
    return d.clamp(min=0).norm(dim=-1) + d.max(dim=-1).values.clamp(max=0)


def sdf_cone(p: Tensor, radius: float, half_height: float) -> Tensor:
    """Capped cone along z-axis. Base (radius) at z=-half_height, apex at z=+half_height.
    p: (N, 3)
    """
    height = 2.0 * half_height
    # Shift so base at z=0, apex at z=height
    pz = p[:, 2] + half_height
    pr = p[:, :2].norm(dim=-1)

    # 2D problem in (r, z) space
    q = torch.stack([pr, pz], dim=-1)  # (N, 2)

    # Cone tip at (0, height), base edge at (radius, 0)
    # Using iq's sdCappedCone formula with r1=radius (bottom), r2=0 (top)
    r1, r2, h = radius, 0.0, height

    k1 = torch.tensor([r2, h], device=p.device, dtype=p.dtype)
    k2 = torch.tensor([r2 - r1, 2.0 * h], device=p.device, dtype=p.dtype)

    # ca vector
    ca_x_min = torch.where(pz < 0,
                           torch.clamp(q[:, 0], max=r1),
                           torch.clamp(q[:, 0], max=max(r2, 1e-8)))
    ca_x = q[:, 0] - ca_x_min
    ca_y = pz.abs() - h
    ca = torch.stack([ca_x, ca_y], dim=-1)

    # cb vector
    k1_q = k1.unsqueeze(0) - q
    t = (k1_q * k2.unsqueeze(0)).sum(dim=-1) / ((k2 * k2).sum() + 1e-10)
    t = t.clamp(0, 1)
    cb = q - k1.unsqueeze(0) + t.unsqueeze(-1) * k2.unsqueeze(0)

    s_cond = (cb[:, 0] < 0) & (ca[:, 1] < 0)
    s = torch.where(s_cond, torch.tensor(-1.0, device=p.device), torch.tensor(1.0, device=p.device))

    d2 = torch.min((ca * ca).sum(dim=-1), (cb * cb).sum(dim=-1))
    return s * torch.sqrt(d2 + 1e-10)


def sdf_torus(p: Tensor, major_radius: float, minor_radius: float) -> Tensor:
    """Torus in xz-plane (ring around y-axis). p: (N, 3)"""
    q_r = torch.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2) - major_radius
    q = torch.stack([q_r, p[:, 1]], dim=-1)
    return q.norm(dim=-1) - minor_radius


# ─────────────────── Primitive Dataclass ─────────────────────────────────────

PRIMITIVE_TYPES = ['cube', 'sphere', 'cylinder', 'cone', 'torus']


@dataclass
class Primitive:
    ptype: str                    # One of PRIMITIVE_TYPES
    params: Dict[str, float]      # Shape-specific parameters
    scale: np.ndarray             # (3,) per-axis scale
    rotation: np.ndarray          # (3, 3) rotation matrix
    translation: np.ndarray       # (3,) world-space translation
    material: Dict = field(default_factory=dict)  # Material properties


def evaluate_primitive_sdf(p: Tensor, prim: Primitive, device='cuda') -> Tensor:
    """Evaluate SDF of a single transformed primitive at query points.
    p: (N, 3) world-space points.
    Returns: (N,) SDF values.
    """
    R = torch.tensor(prim.rotation, device=device, dtype=torch.float32)
    t = torch.tensor(prim.translation, device=device, dtype=torch.float32)
    s = torch.tensor(prim.scale, device=device, dtype=torch.float32)

    # Transform to local frame: p_local = R^T @ (p - t) / s
    p_local = (p - t) @ R  # R is orthogonal, so R^T @ x = x @ R
    p_local = p_local / s

    # Evaluate SDF in local frame
    if prim.ptype == 'cube':
        sdf = sdf_box(p_local, torch.tensor(prim.params['half_extents'], device=device, dtype=torch.float32))
    elif prim.ptype == 'sphere':
        sdf = sdf_sphere(p_local, prim.params['radius'])
    elif prim.ptype == 'cylinder':
        sdf = sdf_cylinder(p_local, prim.params['radius'], prim.params['half_height'])
    elif prim.ptype == 'cone':
        sdf = sdf_cone(p_local, prim.params['radius'], prim.params['half_height'])
    elif prim.ptype == 'torus':
        sdf = sdf_torus(p_local, prim.params['major_radius'], prim.params['minor_radius'])
    else:
        raise ValueError(f"Unknown primitive type: {prim.ptype}")

    # Approximate world-space SDF by scaling with minimum scale component
    sdf = sdf * s.min().item()
    return sdf


def evaluate_union_sdf(p: Tensor, primitives: List[Primitive], device='cuda') -> Tensor:
    """Evaluate union SDF (min over all primitives). p: (N, 3). Returns: (N,)."""
    sdfs = torch.stack([evaluate_primitive_sdf(p, prim, device) for prim in primitives], dim=0)
    return sdfs.min(dim=0).values


def evaluate_per_primitive_sdf(p: Tensor, primitives: List[Primitive], device='cuda') -> Tuple[Tensor, Tensor]:
    """Evaluate SDF for each primitive and return closest primitive index.
    Returns: (sdf_values (N,), closest_prim_idx (N,))
    """
    sdfs = torch.stack([evaluate_primitive_sdf(p, prim, device) for prim in primitives], dim=0)  # (K, N)
    sdf_min, idx = sdfs.min(dim=0)
    return sdf_min, idx


# ─────────────────── Random Shape Generation ─────────────────────────────────

def _random_primitive_params(ptype: str, rng: np.random.Generator,
                             scale_range: Tuple[float, float] = (0.5, 1.5)) -> Dict:
    """Generate random shape parameters for a primitive (in unit local space)."""
    if ptype == 'cube':
        half = rng.uniform(0.3, 0.7, size=3)
        return {'half_extents': half.tolist()}
    elif ptype == 'sphere':
        return {'radius': rng.uniform(0.4, 0.8)}
    elif ptype == 'cylinder':
        return {'radius': rng.uniform(0.3, 0.7), 'half_height': rng.uniform(0.3, 0.8)}
    elif ptype == 'cone':
        return {'radius': rng.uniform(0.3, 0.7), 'half_height': rng.uniform(0.3, 0.8)}
    elif ptype == 'torus':
        major = rng.uniform(0.4, 0.7)
        minor = rng.uniform(0.15, min(0.35, major * 0.6))
        return {'major_radius': major, 'minor_radius': minor}
    else:
        raise ValueError(f"Unknown primitive type: {ptype}")


def generate_random_primitives(
    rng: np.random.Generator,
    num_primitives_probs: Optional[List[float]] = None,
    primitive_type_probs: Optional[List[float]] = None,
    scale_range: Tuple[float, float] = (0.06, 0.18),
    translation_range: Tuple[List[float], List[float]] = ([0.2, 0.2, 0.35], [0.8, 0.8, 0.70]),
) -> List[Primitive]:
    """Generate a random collection of affine-transformed primitives.

    Args:
        rng: numpy random generator
        num_primitives_probs: probability weights for 1..9 primitives (length 9).
            If None, uniform.
        primitive_type_probs: probability weights for each type in PRIMITIVE_TYPES.
        scale_range: (min_scale, max_scale) for each axis.
        translation_range: ([min_x,y,z], [max_x,y,z]) in MPM space [0,1]^3.
    """
    # Number of primitives
    if num_primitives_probs is None:
        num_primitives_probs = [1.0] * 9
    probs = np.array(num_primitives_probs, dtype=np.float64)
    probs /= probs.sum()
    n_prims = rng.choice(range(1, 10), p=probs)

    # Primitive type probabilities
    if primitive_type_probs is None:
        primitive_type_probs = [1.0] * len(PRIMITIVE_TYPES)
    type_probs = np.array(primitive_type_probs, dtype=np.float64)
    type_probs /= type_probs.sum()

    primitives = []
    t_lo = np.array(translation_range[0])
    t_hi = np.array(translation_range[1])

    for _ in range(n_prims):
        ptype = rng.choice(PRIMITIVE_TYPES, p=type_probs)
        params = _random_primitive_params(ptype, rng)

        # Random non-uniform scale
        scale = rng.uniform(scale_range[0], scale_range[1], size=3)

        # Random rotation (uniform SO(3))
        rotation = Rotation.random(random_state=rng).as_matrix()

        # Random translation within specified range
        translation = rng.uniform(t_lo, t_hi)

        primitives.append(Primitive(
            ptype=ptype,
            params=params,
            scale=scale,
            rotation=rotation,
            translation=translation,
        ))

    return primitives
