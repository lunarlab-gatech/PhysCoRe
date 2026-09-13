"""
Zeroverse-style primitive composition for MPM-friendly shapes.

This ports the chained sub-object construction pattern used by
zeroverse/create_shapes.py::MultiShape.genShape into the lightweight primitive
representation used by the simulation pipeline.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from .primitives import Primitive


ZEROVERSE_PRIMITIVE_IDS = (0, 1, 2)


def _normalize_weights(weights: Optional[Sequence[float]], n_items: int) -> np.ndarray:
    if weights is None:
        values = np.ones(n_items, dtype=np.float64)
    else:
        values = np.asarray(weights, dtype=np.float64)
        if values.shape[0] != n_items:
            raise ValueError(f"Expected {n_items} weights, got {values.shape[0]}")
        values = np.clip(values, 0.0, None)

    if values.sum() <= 0:
        values = np.ones(n_items, dtype=np.float64)
    return values / values.sum()


def _rotation_matrix_xyz(degrees_xyz: Sequence[float]) -> np.ndarray:
    return Rotation.from_euler("xyz", degrees_xyz, degrees=True).as_matrix().astype(np.float64)


def _make_zeroverse_primitive(primitive_id: int, axis_vals: np.ndarray) -> Primitive:
    axis_vals = np.asarray(axis_vals, dtype=np.float64)

    if primitive_id == 0:
        return Primitive(
            ptype="sphere",
            params={"radius": 1.0},
            scale=axis_vals.copy(),
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
        )
    if primitive_id == 1:
        return Primitive(
            ptype="cube",
            params={"half_extents": [1.0, 1.0, 1.0]},
            scale=axis_vals.copy(),
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
        )
    if primitive_id == 2:
        return Primitive(
            ptype="cylinder",
            params={"radius": 1.0, "half_height": 1.0},
            scale=axis_vals.copy(),
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
        )
    raise ValueError(f"Unsupported Zeroverse primitive id: {primitive_id}")


def _transform_existing_primitives(
    primitives: List[Primitive],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> None:
    for prim in primitives:
        prim.translation = prim.translation @ rotation.T + translation
        prim.rotation = rotation @ prim.rotation


def _primitive_local_half_extents(prim: Primitive) -> np.ndarray:
    scale = np.abs(np.asarray(prim.scale, dtype=np.float64))

    if prim.ptype == "cube":
        return np.asarray(prim.params["half_extents"], dtype=np.float64) * scale
    if prim.ptype == "sphere":
        radius = float(prim.params["radius"])
        return np.full(3, radius, dtype=np.float64) * scale
    if prim.ptype in {"cylinder", "cone"}:
        radius = float(prim.params["radius"])
        half_height = float(prim.params["half_height"])
        return np.asarray([radius, radius, half_height], dtype=np.float64) * scale
    if prim.ptype == "torus":
        major = float(prim.params["major_radius"])
        minor = float(prim.params["minor_radius"])
        return np.asarray([major + minor, minor, major + minor], dtype=np.float64) * scale
    raise ValueError(f"Unsupported primitive type: {prim.ptype}")


def estimate_primitives_bounds(primitives: List[Primitive]) -> Tuple[np.ndarray, np.ndarray]:
    mins = []
    maxs = []
    for prim in primitives:
        local_half_extents = _primitive_local_half_extents(prim)
        world_half_extents = np.abs(np.asarray(prim.rotation, dtype=np.float64)) @ local_half_extents
        translation = np.asarray(prim.translation, dtype=np.float64)
        mins.append(translation - world_half_extents)
        maxs.append(translation + world_half_extents)

    return np.min(np.stack(mins, axis=0), axis=0), np.max(np.stack(maxs, axis=0), axis=0)


def _recenter_and_rescale(primitives: List[Primitive], target_max_dim: float) -> None:
    bounds_min, bounds_max = estimate_primitives_bounds(primitives)
    center = 0.5 * (bounds_min + bounds_max)
    current_max_dim = max(float(np.max(bounds_max - bounds_min)), 1e-6)
    scale_factor = float(target_max_dim) / current_max_dim

    for prim in primitives:
        prim.translation = (np.asarray(prim.translation, dtype=np.float64) - center) * scale_factor
        prim.scale = np.asarray(prim.scale, dtype=np.float64) * scale_factor


def _place_primitives(
    primitives: List[Primitive],
    rng: np.random.Generator,
    translation_range: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    center: Optional[Sequence[float]] = None,
) -> None:
    bounds_min, bounds_max = estimate_primitives_bounds(primitives)

    if center is not None:
        center = np.asarray(center, dtype=np.float64)
        offset = center - 0.5 * (bounds_min + bounds_max)
    elif translation_range is not None:
        lo = np.asarray(translation_range[0], dtype=np.float64)
        hi = np.asarray(translation_range[1], dtype=np.float64)
        offset_lo = lo - bounds_min
        offset_hi = hi - bounds_max
        if np.any(offset_lo > offset_hi):
            offset = 0.5 * (lo + hi) - 0.5 * (bounds_min + bounds_max)
        else:
            offset = rng.uniform(offset_lo, offset_hi)
    else:
        offset = np.zeros(3, dtype=np.float64)

    for prim in primitives:
        prim.translation = np.asarray(prim.translation, dtype=np.float64) + offset


def generate_zeroverse_composite_primitives(
    rng: np.random.Generator,
    sub_object_numbers: Optional[Sequence[int]] = None,
    sub_object_weights: Optional[Sequence[float]] = None,
    primitive_weights: Optional[Sequence[float]] = None,
    axis_range: Tuple[float, float] = (0.25, 2.0),
    translate_range_rate: Tuple[float, float] = (0.0, 0.5),
    rotate_range: Tuple[float, float] = (0.0, 180.0),
    max_dim_range: Tuple[float, float] = (0.18, 0.34),
    translation_range: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    center: Optional[Sequence[float]] = None,
) -> List[Primitive]:
    """Generate a chained composite using the Zeroverse MultiShape recipe."""
    if sub_object_numbers is None:
        sub_object_numbers = list(range(1, 10))

    sub_object_numbers = [int(value) for value in sub_object_numbers]
    sub_object_probs = _normalize_weights(sub_object_weights, len(sub_object_numbers))
    primitive_probs = _normalize_weights(primitive_weights, len(ZEROVERSE_PRIMITIVE_IDS))
    n_sub_objects = int(rng.choice(sub_object_numbers, p=sub_object_probs))

    primitives: List[Primitive] = []
    for sub_object_idx in range(n_sub_objects):
        primitive_id = int(rng.choice(ZEROVERSE_PRIMITIVE_IDS, p=primitive_probs))
        axis_vals = rng.uniform(axis_range[0], axis_range[1], size=3)
        max_axis_diameter = float(axis_vals.max()) * 2.0

        translation = rng.uniform(
            translate_range_rate[0] * max_axis_diameter,
            translate_range_rate[1] * max_axis_diameter,
            size=3,
        )
        translation1 = rng.uniform(
            translate_range_rate[0] * max_axis_diameter,
            translate_range_rate[1] * max_axis_diameter,
            size=3,
        )
        rotation = rng.uniform(rotate_range[0], rotate_range[1], size=3)
        rotation1 = rng.uniform(rotate_range[0], rotate_range[1], size=3)

        if sub_object_idx != 0:
            _transform_existing_primitives(
                primitives,
                rotation=_rotation_matrix_xyz(rotation1),
                translation=translation1.astype(np.float64),
            )

        primitive = _make_zeroverse_primitive(primitive_id, axis_vals)
        primitive.rotation = _rotation_matrix_xyz(rotation)
        primitive.translation = translation.astype(np.float64)
        primitives.append(primitive)

    target_max_dim = rng.uniform(max_dim_range[0], max_dim_range[1])
    _recenter_and_rescale(primitives, target_max_dim=target_max_dim)
    _place_primitives(primitives, rng=rng, translation_range=translation_range, center=center)
    return primitives
