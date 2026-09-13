"""
Convert one recorded sequence into a PhysCoRe training episode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conversion_utils import (
    compute_point_flows as _compute_point_flows,
    depth_to_world_points as _depth_to_world_points,
    flip_z_extrinsics as _flip_z_extrinsics,
    flip_z_points as _flip_z_points,
    knn_warp_cache as _knn_warp_cache,
    load_pickle as _load_pickle,
    mask_ids as _mask_ids,
    nearest_particle_ids as _nearest_particle_ids,
    read_mask as _read_mask,
    read_rgb as _read_rgb,
    warp_query_points as _warp_query_points,
    write_json as _write_json,
)


def _mask_ids_from_info(mask_info_path: Path, label: str, exclude_labels: Tuple[str, ...] = ("hand",)) -> List[str]:
    label = str(label).strip()
    if label and label.lower() != "auto":
        return _mask_ids(mask_info_path, label)
    with mask_info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    excluded = {str(item).lower() for item in exclude_labels}
    return [
        str(mask_id)
        for mask_id, name in info.items()
        if str(name).lower() not in excluded
    ]


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] == 0 or voxel_size <= 0.0:
        return points
    keys = np.floor(points / float(voxel_size)).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(unique_idx)]


def _estimate_oriented_normals_knn(
    points: np.ndarray,
    camera_centers: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float32)
    camera_centers = np.asarray(camera_centers, dtype=np.float32)
    query_k = min(max(int(k), 4), int(points.shape[0]))
    _, indices = cKDTree(points).query(points, k=query_k)
    if query_k == 1:
        indices = indices[:, None]
    neighbors = points[indices]
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(query_k - 1, 1)
    _, eigvecs = np.linalg.eigh(cov)
    normals = eigvecs[:, :, 0].astype(np.float32)
    to_camera = camera_centers - points
    flip = np.sum(normals * to_camera, axis=-1) < 0.0
    normals[flip] *= -1.0
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1.0e-8)
    return normals.astype(np.float32)


def _dpsr_reconstruct_mesh(
    points: np.ndarray,
    camera_centers: np.ndarray,
    *,
    res: int,
    sigma: float,
    normal_knn: int,
    bbox_padding: float,
    level: float,
    device: str,
    seed: int,
):
    """Run DPSR on a dense surface point cloud and return a trimesh.Trimesh.
    """
    import trimesh
    from skimage import measure

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from physcore.dpsr import DPSR

    points = np.asarray(points, dtype=np.float32)
    camera_centers = np.asarray(camera_centers, dtype=np.float32)
    valid = np.isfinite(points).all(axis=-1)
    points = points[valid]
    camera_centers = camera_centers[valid]
    if points.shape[0] < max(32, int(normal_knn)):
        raise RuntimeError(f"DPSR needs at least {max(32, int(normal_knn))} finite points; got {points.shape[0]}")

    normals = _estimate_oriented_normals_knn(points, camera_centers, k=int(normal_knn))

    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = 0.5 * (lo + hi)
    side = float(np.max(np.maximum(hi - lo, 1.0e-6)) * (1.0 + 2.0 * float(bbox_padding)))
    bbox_min = (center - 0.5 * side).astype(np.float32)
    normalized = (points - bbox_min[None, :]) / max(side, 1.0e-8)
    normalized = np.clip(normalized, 1.0e-4, 1.0 - 1.0e-4).astype(np.float32)

    torch_device = torch.device(device if str(device) != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DPSR(res=(int(res), int(res), int(res)), sig=float(sigma)).to(torch_device)
    V = torch.from_numpy(normalized).to(torch_device).unsqueeze(0)
    N = torch.from_numpy(normals).to(torch_device).unsqueeze(0)
    with torch.no_grad():
        phi = model(V, N)[0].detach().cpu().numpy().astype(np.float32)

    field_min = float(np.nanmin(phi))
    field_max = float(np.nanmax(phi))
    if not (field_min <= float(level) <= field_max):
        raise RuntimeError(
            f"DPSR level {float(level)} outside field range [{field_min}, {field_max}]"
        )

    verts, faces, _, _ = measure.marching_cubes(
        phi,
        level=float(level),
        spacing=tuple([1.0 / max(int(res) - 1, 1)] * 3),
    )
    verts_world = bbox_min[None, :] + verts.astype(np.float32) * side
    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)
    meta = {
        "method": "dpsr_frame0_marching_cubes",
        "input_points": int(points.shape[0]),
        "normal_knn": int(normal_knn),
        "res": int(res),
        "sigma": float(sigma),
        "bbox_padding": float(bbox_padding),
        "level": float(level),
        "field_min": field_min,
        "field_max": field_max,
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "device": str(torch_device),
    }
    return mesh, meta


def _voxelize_mesh_pool(mesh, voxel_size: float) -> Tuple[np.ndarray, Dict]:
    """Return all voxel centers covering `mesh` (surface + interior) at uniform
    `voxel_size` spacing.

    For thin objects (limb thickness < a few voxels), every voxel inside the
    mesh also crosses the surface, so a strict surface-vs-interior split is
    degenerate.  This function just returns the full pool of "voxels inside
    the mesh"; the caller decides which voxels become shell-identity vs
    interior-fill based on proximity to cotracker / depth anchors.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    if verts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), {
            "voxel_size": float(voxel_size),
            "pool_count": 0,
        }
    filled_grid = mesh.voxelized(pitch=float(voxel_size)).fill()
    pool_pts = np.asarray(filled_grid.points, dtype=np.float32)
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    return pool_pts, {
        "voxel_size": float(voxel_size),
        "pool_count": int(pool_pts.shape[0]),
        "voxel_grid_shape": [int(s) for s in filled_grid.shape],
        "bbox_min": lo.astype(float).tolist(),
        "bbox_max": hi.astype(float).tolist(),
    }


def _drop_pool_near_anchors(
    pool_pts: np.ndarray, anchors: np.ndarray, voxel_size: float
) -> np.ndarray:
    """Return pool points whose nearest anchor is farther than voxel_size/2."""
    from scipy.spatial import cKDTree

    if pool_pts.shape[0] == 0 or anchors.shape[0] == 0:
        return pool_pts.astype(np.float32)
    tree = cKDTree(anchors.astype(np.float32))
    distances, _ = tree.query(pool_pts.astype(np.float32), k=1)
    keep = distances > (0.5 * float(voxel_size))
    return pool_pts[keep].astype(np.float32)


def _warp_with_cotracker(
    cotracker_seq: np.ndarray,
    query_points: np.ndarray,
    *,
    knn_k: int,
    knn_power: float,
) -> np.ndarray:
    """Warp `query_points` (frame 0 positions) through all frames using cotracker.

    For each query point, find K nearest cotracker points at frame 0 and blend
    their per-frame displacements with inverse-distance^power weights.

    Returns (T, M, 3).
    """
    cotracker_seq = np.asarray(cotracker_seq, dtype=np.float32)
    query_points = np.asarray(query_points, dtype=np.float32)
    frame_count = int(cotracker_seq.shape[0])
    out = np.zeros((frame_count, query_points.shape[0], 3), dtype=np.float32)
    if query_points.shape[0] == 0:
        return out
    neighbor_indices, neighbor_weights = _knn_warp_cache(
        source_points=cotracker_seq[0],
        query_points=query_points,
        k=int(knn_k),
        power=float(knn_power),
    )
    for frame_idx in range(frame_count):
        out[frame_idx] = _warp_query_points(
            source_points=cotracker_seq[0],
            target_points=cotracker_seq[frame_idx],
            query_points=query_points,
            neighbor_indices=neighbor_indices,
            neighbor_weights=neighbor_weights,
        )
    return out


def _extend_particle_colors(
    object_colors: np.ndarray,
    object_points_initial: np.ndarray,
    query_points: np.ndarray,
    full_count: int,
    frame_count: int,
) -> np.ndarray:
    """Replicate cotracker colors and propagate them to non-cotracker particles
    via nearest-neighbor on the initial frame.  Cotracker block keeps its own
    per-frame colors; the rest gets the color of its nearest cotracker anchor.
    """
    from scipy.spatial import cKDTree

    colors = np.asarray(object_colors, dtype=np.float32)
    if colors.ndim == 2:
        colors = colors[None].repeat(int(frame_count), axis=0)
    frame_count, tracked_count = colors.shape[:2]
    if query_points.shape[0] == 0:
        return colors[:, :full_count].astype(np.float32)
    nearest = cKDTree(object_points_initial.astype(np.float32)).query(
        query_points.astype(np.float32), k=1
    )[1]
    query_colors = colors[:, nearest]
    if tracked_count + query_colors.shape[1] != full_count:
        # Truncate or zero-pad to match full_count.
        combined = np.concatenate([colors, query_colors], axis=1)
        return combined[:, :full_count].astype(np.float32)
    return np.concatenate([colors, query_colors], axis=1).astype(np.float32)


def _colors_from_nearest_cotracker(
    object_colors: np.ndarray,
    object_points_initial: np.ndarray,
    query_points: np.ndarray,
    frame_count: int,
) -> np.ndarray:
    """Color arbitrary query particles from their nearest cotracker anchor."""
    from scipy.spatial import cKDTree

    colors = np.asarray(object_colors, dtype=np.float32)
    if colors.ndim == 2:
        colors = colors[None].repeat(int(frame_count), axis=0)
    frame_count = int(colors.shape[0])
    query_points = np.asarray(query_points, dtype=np.float32)
    if query_points.shape[0] == 0:
        return np.zeros((frame_count, 0, 3), dtype=np.float32)
    nearest = cKDTree(object_points_initial.astype(np.float32)).query(query_points, k=1)[1]
    return colors[:, nearest].astype(np.float32)


def _nearest_sequence_particle_ids(query_points: np.ndarray, particle_points: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    ids = np.full(query_points.shape[:2], -1, dtype=np.int64)
    for frame_idx in range(query_points.shape[0]):
        _, indices = cKDTree(particle_points[frame_idx]).query(query_points[frame_idx], k=1)
        ids[frame_idx] = indices.astype(np.int64)
    return ids


def _visible_indices_from_observation_ids(observed_particle_ids: np.ndarray) -> np.ndarray:
    frame_count = int(observed_particle_ids.shape[1])
    per_frame = []
    max_count = 0
    for frame_idx in range(frame_count):
        ids = observed_particle_ids[:, frame_idx].reshape(-1)
        ids = np.unique(ids[ids >= 0]).astype(np.int64)
        per_frame.append(ids)
        max_count = max(max_count, int(ids.shape[0]))
    out = np.full((frame_count, max_count), -1, dtype=np.int64)
    for frame_idx, ids in enumerate(per_frame):
        out[frame_idx, : ids.shape[0]] = ids
    return out


def _clip_completed_particles_to_ground(
    completed_shell_sequence: np.ndarray,
    completed_interior_sequence: np.ndarray,
    completed_full_sequence: np.ndarray,
    *,
    ground_height: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    shell = np.asarray(completed_shell_sequence, dtype=np.float32).copy()
    interior = np.asarray(completed_interior_sequence, dtype=np.float32).copy()
    full = np.asarray(completed_full_sequence, dtype=np.float32).copy()
    threshold = float(ground_height)

    shell_min_before = float(np.nanmin(shell[..., 2])) if shell.size else threshold
    interior_min_before = float(np.nanmin(interior[..., 2])) if interior.size else threshold
    full_min_before = float(np.nanmin(full[..., 2])) if full.size else threshold
    shell_clipped = int(np.sum(shell[..., 2] < threshold))
    interior_clipped = int(np.sum(interior[..., 2] < threshold))
    full_clipped = int(np.sum(full[..., 2] < threshold))

    shell[..., 2] = np.maximum(shell[..., 2], threshold)
    interior[..., 2] = np.maximum(interior[..., 2], threshold)
    full[..., 2] = np.maximum(full[..., 2], threshold)

    metadata = {
        "enabled": True,
        "ground_height": threshold,
        "shell_points_clipped": shell_clipped,
        "interior_points_clipped": interior_clipped,
        "full_points_clipped": full_clipped,
        "shell_min_z_before": shell_min_before,
        "interior_min_z_before": interior_min_before,
        "full_min_z_before": full_min_before,
        "full_min_z_after": float(np.nanmin(full[..., 2])) if full.size else threshold,
    }
    return shell, interior, full, metadata


def _contact_points(
    object_points: np.ndarray,
    controller_points: np.ndarray,
    num_contact_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames, num_controller, _ = controller_points.shape
    contact_controller = np.zeros((frames, num_contact_points, 3), dtype=np.float32)
    contact_object = np.zeros((frames, num_contact_points, 3), dtype=np.float32)
    contact_distances = np.zeros((frames, num_contact_points), dtype=np.float32)
    contact_object_ids = np.full((frames, num_contact_points), -1, dtype=np.int64)

    for frame_idx in range(frames):
        controller = controller_points[frame_idx].astype(np.float32)
        obj = object_points[frame_idx].astype(np.float32)
        distances = np.linalg.norm(controller[:, None, :] - obj[None, :, :], axis=-1)
        nearest_object_idx = distances.argmin(axis=1)
        nearest_distance = distances[np.arange(num_controller), nearest_object_idx]
        order = np.argsort(nearest_distance)
        selected = order[: min(num_contact_points, num_controller)]

        if selected.size < num_contact_points:
            selected = np.pad(selected, (0, num_contact_points - selected.size), mode="edge")

        contact_controller[frame_idx] = controller[selected]
        contact_object[frame_idx] = obj[nearest_object_idx[selected]]
        contact_distances[frame_idx] = nearest_distance[selected]
        contact_object_ids[frame_idx] = nearest_object_idx[selected].astype(np.int64)

    return contact_controller, contact_object, contact_distances, contact_object_ids


def _export_completed_shell_obj(
    source_root: Path,
    output_path: Path,
    fallback_vertices: np.ndarray,
    flip_z_to_z_up: bool,
) -> Dict:
    """Always emit the frame-0 shell point cloud as a vertex-only OBJ.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# frame-0 completed shell points (cotracker + depth dedup)\n")
        for point in fallback_vertices:
            f.write(f"v {point[0]:.8f} {point[1]:.8f} {point[2]:.8f}\n")
    return {
        "source": "frame_0_completed_shell_points",
        "format": "vertices",
        "vertices": int(fallback_vertices.shape[0]),
        "faces": 0,
        "flip_z_to_z_up": bool(flip_z_to_z_up),
    }


def _dedup_depth_against_cotracker(
    cotracker_frame0: np.ndarray,
    depth_points: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """Return depth points NOT within voxel_size/2 of any cotracker point.

    Cotracker is ~99% contained in the back-projected depth (verified for
    sloth: 99% within 5mm, median NN distance 1.9mm), so the survivors here
    are the depth-only coverage that the cotracker doesn't already represent.
    """
    from scipy.spatial import cKDTree

    if depth_points.shape[0] == 0 or cotracker_frame0.shape[0] == 0:
        return depth_points.astype(np.float32)
    tree = cKDTree(cotracker_frame0.astype(np.float32))
    distances, _ = tree.query(depth_points.astype(np.float32), k=1)
    keep = distances > (0.5 * float(voxel_size))
    return depth_points[keep].astype(np.float32)


def _select_valid_cotracker_indices(motion_valid_frame0: np.ndarray) -> np.ndarray:
    """Indices of cotracker points usable as warp anchors.

    """
    return np.where(np.asarray(motion_valid_frame0, dtype=bool))[0].astype(np.int64)


def _tracks_off_gripper_mask(
    tracks_world_frame0: np.ndarray,
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
    hand_mask: np.ndarray,
    dilate_pixels: int,
    max_drop_fraction: float,
) -> np.ndarray:
    """Boolean keep-mask over cotracker tracks: drop any whose frame-0 position projects ONTO the
    manipulator (gripper) silhouette in the primary view.

    The MPM particle cloud is built once from the frame-0 cotracker tracks (plus depth/DPSR seeded
    FROM those tracks), so a track that landed on the gripper -- e.g. where the object mask bleeds
    onto the gripper at the grasp contact -- seeds permanent object particles ON the gripper that
    float off the object and carry a meaningless confidence. Every confidence-mapped particle is an
    object particle, so its seed must lie in the object region; this confines the seeds there by
    removing the ones sitting on the gripper. The grasped object corner is unaffected: it lies on the
    object outside the gripper silhouette, so it does not project into the hand mask.

    Safety: if more than `max_drop_fraction` of tracks fall on the gripper the hand mask is suspect
    (over-covering / wrong view), so the filter is skipped (keep all) rather than gut the object.
    """
    n = int(tracks_world_frame0.shape[0])
    keep = np.ones(n, dtype=bool)
    if n == 0 or hand_mask is None or not np.asarray(hand_mask).any():
        return keep
    mask = np.asarray(hand_mask).astype(bool)
    if int(dilate_pixels) > 0:
        from scipy.ndimage import binary_dilation
        d = int(dilate_pixels)
        mask = binary_dilation(mask, structure=np.ones((2 * d + 1, 2 * d + 1), dtype=bool))
    h, w = mask.shape[:2]
    w2c = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64))
    hom = np.concatenate([tracks_world_frame0.astype(np.float64), np.ones((n, 1))], axis=1)
    cam = (w2c @ hom.T).T[:, :3]
    proj = (np.asarray(intrinsic, dtype=np.float64) @ cam.T).T
    z = cam[:, 2]
    denom = np.where(np.abs(proj[:, 2]) < 1e-9, 1e-9, proj[:, 2])
    u = np.round(proj[:, 0] / denom).astype(np.int64)
    v = np.round(proj[:, 1] / denom).astype(np.int64)
    in_view = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    on_gripper = np.zeros(n, dtype=bool)
    vi = np.where(in_view)[0]
    on_gripper[vi] = mask[v[vi], u[vi]]
    if on_gripper.mean() > float(max_drop_fraction):
        print(
            f"gripper-exclusion: SKIPPED -- {int(on_gripper.sum())}/{n} tracks project onto the "
            f"gripper mask (> {max_drop_fraction:.0%}); hand mask looks unreliable, keeping all",
            flush=True,
        )
        return keep
    keep[on_gripper] = False
    return keep


def _filter_depth_near_cotracker(
    depth_points: np.ndarray,
    depth_camera_centers: np.ndarray,
    cotracker_frame0: np.ndarray,
    max_distance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep depth points within `max_distance` of any cotracker anchor.

    Raw segmented depth back-projection can leak mask noise (hand mask,
    background); those leak points sit far from any motions-valid cotracker
    anchor.  Filtering depth this way removes the drift outliers in both
    the DPSR input and the depth-extra shell block.
    """
    from scipy.spatial import cKDTree

    if depth_points.shape[0] == 0 or cotracker_frame0.shape[0] == 0:
        return depth_points.astype(np.float32), depth_camera_centers.astype(np.float32)
    tree = cKDTree(cotracker_frame0.astype(np.float32))
    distances, _ = tree.query(depth_points.astype(np.float32), k=1)
    keep = distances <= float(max_distance)
    return depth_points[keep].astype(np.float32), depth_camera_centers[keep].astype(np.float32)


def _random_subsample(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    count = int(count)
    if points.shape[0] <= count:
        return points
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(points.shape[0], size=count, replace=False)
    return points[np.sort(idx)]


def _largest_spatial_cluster(points: np.ndarray, eps: float) -> np.ndarray:
    """Boolean keep-mask: keep the largest spatially-connected cluster of `points` (an eps-radius graph),
    dropping blobs disconnected from the object body by a gap > eps. A contiguous object (towel sheet,
    rope) forms a single cluster so nothing is trimmed; a segmentation false-positive that back-projects
    far away (e.g. a ~0.7 m-detached blob on the ground) is a separate small cluster and is removed.
    3-D, so it also catches blobs off the object plane."""
    n = int(points.shape[0])
    keep = np.ones(n, dtype=bool)
    if n <= 1:
        return keep
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    pairs = cKDTree(points.astype(np.float64)).query_pairs(r=float(eps), output_type="ndarray")
    if pairs.shape[0] == 0:
        labels = np.arange(n)  # every point isolated -> each its own cluster
    else:
        g = coo_matrix((np.ones(pairs.shape[0]), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        _, labels = connected_components(g, directed=False)
    main = int(np.bincount(labels).argmax())
    return labels == main


def convert(args: argparse.Namespace) -> None:
    source_root = Path(args.source_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    episode_dir = output_root / "episode_0000"
    episode_dir.mkdir(parents=True, exist_ok=True)

    final_data = _load_pickle(source_root / "sampled_tracks.pkl")
    metadata = json.loads((source_root / "metadata.json").read_text(encoding="utf-8"))
    calibrations = _load_pickle(source_root / "calibrate.pkl")

    intrinsics = np.asarray(metadata["intrinsics"], dtype=np.float32)
    extrinsics = np.asarray(calibrations, dtype=np.float32)
    flip_z_to_z_up = bool(args.flip_z_to_z_up)
    if flip_z_to_z_up:
        extrinsics = _flip_z_extrinsics(extrinsics)
    frame_count = int(metadata["frame_num"])
    views = tuple(int(v) for v in args.views)
    primary_view = int(args.primary_view)
    if primary_view not in views:
        raise ValueError(f"--primary_view {primary_view} must be included in --views {views}")

    object_points = np.asarray(final_data["object_points"], dtype=np.float32)
    object_colors = np.asarray(final_data["object_colors"], dtype=np.float32)
    controller_points = np.asarray(final_data["controller_points"], dtype=np.float32)
    object_visibilities = np.asarray(final_data["object_visibilities"], dtype=bool)
    object_motions_valid = np.asarray(final_data["object_motions_valid"], dtype=bool)
    if flip_z_to_z_up:
        object_points = _flip_z_points(object_points)
        controller_points = _flip_z_points(controller_points)

    if object_points.shape[0] != frame_count:
        frame_count = int(object_points.shape[0])
    if int(args.frame_limit) > 0:
        frame_count = min(frame_count, int(args.frame_limit))
        object_points = object_points[:frame_count]
        if object_colors.ndim == 3:
            object_colors = object_colors[:frame_count]
        controller_points = controller_points[:frame_count]
        object_visibilities = object_visibilities[:frame_count]
        object_motions_valid = object_motions_valid[:frame_count]

    tracked_count = int(object_points.shape[1])

    object_mask_ids = {
        view_idx: _mask_ids_from_info(
            source_root / "mask" / f"mask_info_{view_idx}.json",
            str(args.object_mask_label),
            exclude_labels=(str(args.controller_mask_label),),
        )
        for view_idx in views
    }
    controller_mask_ids = {
        view_idx: _mask_ids(source_root / "mask" / f"mask_info_{view_idx}.json", str(args.controller_mask_label))
        for view_idx in views
    }

    observed_points = np.zeros((len(views), frame_count, args.max_observed_points, 3), dtype=np.float32)
    observed_pixels = np.zeros((len(views), frame_count, args.max_observed_points, 2), dtype=np.int32)
    observed_valid = np.zeros((len(views), frame_count, args.max_observed_points), dtype=bool)

    primary_depth = np.zeros((frame_count, metadata["WH"][1], metadata["WH"][0]), dtype=np.float32)
    primary_segmented_depth = np.full_like(primary_depth, np.nan)
    primary_object_masks = np.zeros_like(primary_depth, dtype=np.uint8)
    primary_hand_masks = np.zeros_like(primary_depth, dtype=np.uint8)
    primary_rgb = np.zeros((frame_count, metadata["WH"][1], metadata["WH"][0], 3), dtype=np.uint8)

    mask_erode_pixels = int(args.mask_erode_pixels)
    if mask_erode_pixels > 0:
        from scipy.ndimage import binary_erosion
        _erode_struct = np.ones((2 * mask_erode_pixels + 1, 2 * mask_erode_pixels + 1), dtype=bool)
        print(f"eroding object masks by {mask_erode_pixels} pixels", flush=True)

    for local_view_idx, view_idx in enumerate(views):
        for frame_idx in range(frame_count):
            depth_mm = np.load(source_root / "depth" / str(view_idx) / f"{frame_idx}.npy")
            object_mask = _read_mask(source_root / "mask", view_idx, object_mask_ids[view_idx], frame_idx)
            if mask_erode_pixels > 0:
                object_mask = binary_erosion(object_mask, structure=_erode_struct).astype(object_mask.dtype)
            points, pixels, valid = _depth_to_world_points(
                depth_mm=depth_mm,
                object_mask=object_mask,
                intrinsic=intrinsics[view_idx],
                camera_to_world=extrinsics[view_idx],
                max_points=args.max_observed_points,
                frame_idx=frame_idx,
                view_idx=view_idx,
            )
            observed_points[local_view_idx, frame_idx] = points
            observed_pixels[local_view_idx, frame_idx] = pixels
            observed_valid[local_view_idx, frame_idx] = valid

            if view_idx == primary_view:
                hand_mask = _read_mask(source_root / "mask", view_idx, controller_mask_ids[view_idx], frame_idx)
                depth_m = depth_mm.astype(np.float32) / 1000.0
                depth_m[(depth_mm <= 0) | (depth_mm >= 60000)] = np.nan
                primary_depth[frame_idx] = depth_m
                primary_segmented_depth[frame_idx] = np.where(object_mask, depth_m, np.nan)
                primary_object_masks[frame_idx] = object_mask.astype(np.uint8)
                primary_hand_masks[frame_idx] = hand_mask.astype(np.uint8)
                primary_rgb[frame_idx] = _read_rgb(source_root / "color", view_idx, frame_idx)

        print(
            f"view {view_idx}: observed valid points/frame "
            f"{observed_valid[local_view_idx].sum(axis=1).min()}.."
            f"{observed_valid[local_view_idx].sum(axis=1).max()}",
            flush=True,
        )

    contact_controller, contact_object, contact_distances, contact_object_ids = _contact_points(
        object_points=object_points,
        controller_points=controller_points,
        num_contact_points=args.contact_points,
    )

    # ---------------- Canonical shape completion (frame 0 only) ----------------
    voxel_size = float(args.particle_voxel_size)
    target_total = int(args.target_total_particles)
    shell_depth_max_distance = float(args.shell_depth_max_distance)
    particle_mode = str(getattr(args, "particle_mode", "completed")).lower()
    use_dpsr = bool(getattr(args, "use_dpsr", True))
    fill_interior = bool(getattr(args, "fill_interior", True))
    depth_only = particle_mode in {"depth", "depth_only", "backprojected_depth"} or not use_dpsr

    # 1. Gather back-projected depth + per-point camera centers (3 views, frame 0).
    cam_centers_by_view = extrinsics[np.asarray(views, dtype=np.int64)][:, :3, 3].astype(np.float32)
    depth_pts_list = []
    depth_cams_list = []
    for local_view_idx in range(len(views)):
        valid_mask = observed_valid[local_view_idx, 0]
        if not valid_mask.any():
            continue
        view_pts = observed_points[local_view_idx, 0, valid_mask]
        view_cams = np.broadcast_to(
            cam_centers_by_view[local_view_idx][None, :], view_pts.shape
        ).copy()
        depth_pts_list.append(view_pts.astype(np.float32))
        depth_cams_list.append(view_cams.astype(np.float32))
    if not depth_pts_list:
        raise RuntimeError("Frame 0 has no valid back-projected depth points across any view.")
    depth_pts_raw = np.concatenate(depth_pts_list, axis=0).astype(np.float32)
    depth_cams_raw = np.concatenate(depth_cams_list, axis=0).astype(np.float32)

    # 2. Filter cotracker to motions_valid[0].  Drift outliers are NOT in this
    #    subset (verified by direct plot); only depth side has mask noise.
    valid_indices = _select_valid_cotracker_indices(object_motions_valid[0])
    invalid_count = int(object_points.shape[1] - valid_indices.shape[0])
    print(
        f"cotracker filter: motions_valid[0] kept "
        f"{valid_indices.shape[0]} / {object_points.shape[1]}  (dropped {invalid_count})",
        flush=True,
    )
    # 2b. Confine seeds to the object region: drop tracks that project onto the gripper silhouette.
    #     The whole MPM cloud (cotracker + depth/DPSR + interior) is seeded from these frame-0 tracks,
    #     and depth is kept only within `shell_depth_max_distance` of a track -- so removing the
    #     on-gripper tracks here cascades to drop the gripper depth + DPSR shell + interior too, i.e.
    #     no object particles get initialized on the gripper. Uses the primary view's hand mask.
    if bool(getattr(args, "exclude_gripper_particles", True)):
        # The "robot gripper" mask frequently swallows the grasped object (a thin rope can be ~98%
        # inside it), which makes "track projects onto the gripper mask" meaningless and trips the
        # >50% safety bail -> the exclusion does nothing and grasp-region particles survive. Test
        # against the gripper-ONLY region (hand minus object) so a track on the rope is never counted
        # as on the gripper; only tracks on the actual gripper body are dropped.
        hand_only = primary_hand_masks[0].astype(bool) & ~primary_object_masks[0].astype(bool)
        keep_track = _tracks_off_gripper_mask(
            object_points[0],
            intrinsics[primary_view],
            extrinsics[primary_view],
            hand_only,
            dilate_pixels=int(getattr(args, "gripper_mask_exclude_dilate_pixels", 4)),
            max_drop_fraction=float(getattr(args, "gripper_mask_exclude_max_fraction", 0.5)),
        )
        n_on_gripper = int((~keep_track[valid_indices]).sum())
        if n_on_gripper > 0:
            valid_indices = valid_indices[keep_track[valid_indices]]
            print(
                f"gripper-exclusion: dropped {n_on_gripper} cotracker tracks projecting onto the "
                f"gripper; {valid_indices.shape[0]} object-region tracks remain",
                flush=True,
            )
    # 2c. Spatial outlier rejection: keep only the main spatially-connected cluster of object tracks, so
    #     a segmentation false-positive that back-projects far from the object body (observed as a blob
    #     ~0.7 m off the towel that the descend later pulls across the silhouette into view) is dropped.
    #     A contiguous object is one cluster -> nothing trimmed. Removing the blob's tracks cascades
    #     through the depth-near-cotracker filter (its depth + DPSR + interior go too). Safety: skip if it
    #     would drop more than the max fraction (a sign the clustering, not the data, is off).
    if bool(getattr(args, "exclude_spatial_outliers", True)) and valid_indices.shape[0] > 1:
        eps = float(getattr(args, "spatial_outlier_eps", 0.05))
        max_frac = float(getattr(args, "spatial_outlier_max_drop_fraction", 0.5))
        keep_main = _largest_spatial_cluster(object_points[0][valid_indices], eps)
        n_drop = int((~keep_main).sum())
        frac = n_drop / max(int(keep_main.shape[0]), 1)
        if n_drop > 0 and frac <= max_frac:
            valid_indices = valid_indices[keep_main]
            print(
                f"spatial-outlier: dropped {n_drop} tracks in blob(s) detached (> {eps * 100:.0f}cm gap) "
                f"from the main object body; {valid_indices.shape[0]} remain",
                flush=True,
            )
        elif frac > max_frac:
            print(
                f"spatial-outlier: SKIPPED -- main cluster is only {(1 - frac) * 100:.0f}% of tracks "
                f"(> {max_frac:.0%} would be dropped); keeping all",
                flush=True,
            )
    object_points = object_points[:, valid_indices]
    object_visibilities = object_visibilities[:, valid_indices]
    object_motions_valid = object_motions_valid[:, valid_indices]
    if object_colors.ndim == 3:
        object_colors = object_colors[:, valid_indices]
    elif object_colors.ndim == 2:
        object_colors = object_colors[valid_indices]
    tracked_count = int(object_points.shape[1])
    cotracker_frame0 = object_points[0].astype(np.float32)

    # Recompute contact points against the filtered cotracker set so contact
    # indices stay consistent with particle_coords.
    contact_controller, contact_object, contact_distances, contact_object_ids = _contact_points(
        object_points=object_points,
        controller_points=controller_points,
        num_contact_points=args.contact_points,
    )

    # 3. Filter depth by proximity to cotracker -> removes hand/background mask
    #    noise from the segmentation.
    depth_pts_frame0, depth_cams_frame0 = _filter_depth_near_cotracker(
        depth_points=depth_pts_raw,
        depth_camera_centers=depth_cams_raw,
        cotracker_frame0=cotracker_frame0,
        max_distance=shell_depth_max_distance,
    )
    print(
        f"depth filter: kept {depth_pts_frame0.shape[0]} / {depth_pts_raw.shape[0]}  "
        f"(within {shell_depth_max_distance:.3f} m of cotracker)",
        flush=True,
    )

    if depth_only:
        # Depth-only particles are segmented/back-projected frame-0 depth
        # samples. They remain surface particles; no DPSR mesh or interior fill.
        depth_particle_frame0 = depth_pts_frame0.astype(np.float32)
        if voxel_size > 0.0:
            depth_particle_frame0 = _voxel_downsample(depth_particle_frame0, voxel_size)
        if target_total > 0 and depth_particle_frame0.shape[0] > target_total:
            depth_particle_frame0 = _random_subsample(
                depth_particle_frame0,
                target_total,
                seed=int(args.dpsr_seed) + 3,
            )
        if depth_particle_frame0.shape[0] == 0:
            raise RuntimeError("depth_only particle mode has no valid frame-0 depth particles")
        completed_shell_sequence = _warp_with_cotracker(
            object_points,
            depth_particle_frame0,
            knn_k=int(args.tracking_knn_k),
            knn_power=float(args.tracking_knn_power),
        )
        completed_interior_sequence = np.zeros((frame_count, 0, 3), dtype=np.float32)
        completed_full_sequence = completed_shell_sequence.astype(np.float32)
        extra_shell_frame0 = np.zeros((0, 3), dtype=np.float32)
        interior_frame0 = np.zeros((0, 3), dtype=np.float32)
        dpsr_meta = {
            "enabled": False,
            "skipped": True,
            "reason": "particle_mode=depth_only",
            "input_points": int(depth_particle_frame0.shape[0]),
        }
        voxel_meta = {"enabled": False, "pool_count": 0}
        print(
            f"depth-only particles: {depth_particle_frame0.shape[0]} "
            f"(from filtered frame-0 depth; target {target_total})",
            flush=True,
        )
    else:
        # 4. DPSR input = filtered depth only (cotracker excluded — it's a near-subset
        #    of depth and including it adds nothing).  Voxel-downsample for speed.
        dpsr_input_pts = depth_pts_frame0
        dpsr_input_cams = depth_cams_frame0
        if voxel_size > 0.0 and dpsr_input_pts.shape[0] > 0:
            order = np.floor(dpsr_input_pts / float(voxel_size)).astype(np.int64)
            _, keep_idx = np.unique(order, axis=0, return_index=True)
            keep_idx = np.sort(keep_idx)
            dpsr_input_pts = dpsr_input_pts[keep_idx]
            dpsr_input_cams = dpsr_input_cams[keep_idx]

        mesh, dpsr_meta = _dpsr_reconstruct_mesh(
            dpsr_input_pts,
            dpsr_input_cams,
            res=int(args.dpsr_grid_resolution),
            sigma=float(args.dpsr_sigma),
            normal_knn=int(args.dpsr_normal_knn),
            bbox_padding=float(args.dpsr_bbox_padding),
            level=float(args.dpsr_level),
            device=str(args.dpsr_device),
            seed=int(args.dpsr_seed),
        )
        print(
            f"DPSR mesh: {dpsr_meta['mesh_vertices']} verts, {dpsr_meta['mesh_faces']} faces "
            f"(input={dpsr_meta['input_points']} depth points)",
            flush=True,
        )

        # 5. Voxel-grid the DPSR mesh -> the uniform 3D particle pool. For thin
        #    objects every "inside the mesh" voxel also touches the surface, so
        #    we don't split surface vs interior at the voxel level; the caller
        #    derives shell-vs-interior identity from proximity to cotracker/depth.
        voxel_pool, voxel_meta = _voxelize_mesh_pool(mesh, voxel_size=voxel_size)
        print(
            f"DPSR voxel pool: {voxel_meta['pool_count']} cells  "
            f"(grid {voxel_meta.get('voxel_grid_shape')})",
            flush=True,
        )

        # 6. Shell-extra candidates: depth points NOT already covered by cotracker,
        #    then voxel-downsampled so they pack at the same area density as the
        #    interior pool.  Each unique depth-extra cell becomes one shell point.
        depth_extras_all = _dedup_depth_against_cotracker(
            cotracker_frame0=cotracker_frame0,
            depth_points=depth_pts_frame0,
            voxel_size=voxel_size,
        )
        extra_shell_frame0 = _voxel_downsample(depth_extras_all, voxel_size)
        shell_actual = tracked_count + int(extra_shell_frame0.shape[0])

        # 7. Interior candidates = voxel pool minus voxels already represented by
        #    a shell point (cotracker or depth-extra).  These are the "everything
        #    else" voxels that need filling for uniform 3D density.
        shell_anchors = np.concatenate(
            [cotracker_frame0, extra_shell_frame0], axis=0
        ).astype(np.float32)
        interior_candidates = _drop_pool_near_anchors(
            voxel_pool, shell_anchors, voxel_size=voxel_size
        )

        # 8. Safety filter: drop interior candidates whose nearest shell anchor is
        #    farther than `interior_to_shell_max_distance` (DPSR smooths concavity
        #    between limbs; this trims voxels that ended up in those voids).
        shell_clip_max_distance = float(args.interior_to_shell_max_distance)
        if shell_clip_max_distance > 0.0 and interior_candidates.shape[0] > 0 and shell_anchors.shape[0] > 0:
            from scipy.spatial import cKDTree
            dists, _ = cKDTree(shell_anchors).query(interior_candidates, k=1)
            keep = dists <= shell_clip_max_distance
            dropped = int((~keep).sum())
            if dropped > 0:
                print(
                    f"interior concavity filter: dropped {dropped} pts farther "
                    f"than {shell_clip_max_distance:.3f} m from any shell anchor",
                    flush=True,
                )
            interior_candidates = interior_candidates[keep].astype(np.float32)

        if fill_interior:
            # 9. Allocate remaining budget to interior, random-subsampled for
            #    spatially uniform thinning when the pool exceeds target.
            interior_budget = max(0, target_total - shell_actual)
            interior_budget = min(interior_budget, int(interior_candidates.shape[0]))
            interior_frame0 = _random_subsample(
                interior_candidates, interior_budget, seed=int(args.dpsr_seed) + 2
            )
        else:
            interior_frame0 = np.zeros((0, 3), dtype=np.float32)
            print("interior fill disabled by --no-fill_interior", flush=True)

        # 10. Warp non-cotracker blocks through frames using cotracker KNN.
        #     Cotracker block uses its own per-frame positions.
        extra_shell_seq = _warp_with_cotracker(
            object_points,
            extra_shell_frame0,
            knn_k=int(args.tracking_knn_k),
            knn_power=float(args.tracking_knn_power),
        )
        interior_sequence = _warp_with_cotracker(
            object_points,
            interior_frame0,
            knn_k=int(args.tracking_knn_k),
            knn_power=float(args.tracking_knn_power),
        )
        completed_shell_sequence = np.concatenate(
            [object_points.astype(np.float32), extra_shell_seq], axis=1
        ).astype(np.float32)
        completed_interior_sequence = interior_sequence.astype(np.float32)
        completed_full_sequence = np.concatenate(
            [completed_shell_sequence, completed_interior_sequence], axis=1
        ).astype(np.float32)

    shell_count = int(completed_shell_sequence.shape[1])
    interior_count = int(completed_interior_sequence.shape[1])
    print(
        f"shell={shell_count} (cotracker={tracked_count}, extra={extra_shell_frame0.shape[0]})  "
        f"interior={interior_count}  total={shell_count + interior_count} "
        f"(target {target_total})",
        flush=True,
    )

    ground_height = float(
        np.nanpercentile(
            object_points[: min(frame_count, int(args.ground_clip_reference_frames)), :, 2],
            float(args.ground_clip_quantile),
        )
        + float(args.ground_clip_offset)
    )
    ground_clip_meta = {"enabled": False, "ground_height": ground_height}
    if bool(args.clip_completed_particles_to_ground):
        (
            completed_shell_sequence,
            completed_interior_sequence,
            completed_full_sequence,
            ground_clip_meta,
        ) = _clip_completed_particles_to_ground(
            completed_shell_sequence,
            completed_interior_sequence,
            completed_full_sequence,
            ground_height=ground_height,
        )

    completed_shell_initial = completed_shell_sequence[0].astype(np.float32)
    interior_initial = completed_interior_sequence[0].astype(np.float32)
    particle_target_np = completed_full_sequence
    particle_count = int(particle_target_np.shape[1])
    particle_tracked_count = particle_count if depth_only else tracked_count

    # Observation -> particle id resolution.  In completed mode, cotracker
    # particles are the identity carriers and occupy the first block.  In
    # depth-only mode the backprojected depth particles themselves are the
    # particle set, so observations map directly to them.
    observation_particle_points = particle_target_np if depth_only else object_points
    observed_particle_ids, observed_particle_distances = _nearest_particle_ids(
        observed_points=observed_points,
        observed_valid=observed_valid,
        particle_points=observation_particle_points,
    )
    observed_valid = observed_valid & (observed_particle_distances <= float(args.max_observation_particle_distance))
    observed_particle_ids[~observed_valid] = -1
    contact_object_ids = _nearest_sequence_particle_ids(contact_object, particle_target_np)
    visible_particle_indices = _visible_indices_from_observation_ids(observed_particle_ids)

    particle_coords = torch.from_numpy(particle_target_np).float()
    particle_flows = _compute_point_flows(particle_coords)
    frame_dt = 1.0 / float(metadata.get("fps", 30))
    particle_velocities = particle_flows / max(frame_dt, 1.0e-8)
    rigid_coords = torch.from_numpy(contact_object).float()

    shell_obj_meta = _export_completed_shell_obj(
        source_root=source_root,
        output_path=episode_dir / args.shell_obj_name,
        fallback_vertices=completed_shell_initial,
        flip_z_to_z_up=flip_z_to_z_up,
    )

    if depth_only:
        particle_colors = _colors_from_nearest_cotracker(
            object_colors=object_colors,
            object_points_initial=object_points[0],
            query_points=completed_shell_initial,
            frame_count=frame_count,
        )
        particle_motion_valid = np.ones((frame_count, particle_count), dtype=bool)
    else:
        # depth-extra + interior particles inherit the nearest cotracker color.
        completion_query_points = np.concatenate(
            [extra_shell_frame0, interior_initial], axis=0
        ).astype(np.float32)
        particle_colors = _extend_particle_colors(
            object_colors=object_colors,
            object_points_initial=object_points[0],
            query_points=completion_query_points,
            full_count=particle_count,
            frame_count=frame_count,
        )

        # particle_motion_valid: cotracker block keeps the source's validity
        # mask; warped particles are valid because their motion is derived from
        # cotracker.
        extra_motion_valid = np.ones(
            (frame_count, particle_count - object_motions_valid.shape[1]),
            dtype=bool,
        )
        particle_motion_valid = np.concatenate([object_motions_valid, extra_motion_valid], axis=1)

    shape_completion_meta = {
        "method": "backprojected_depth_only" if depth_only else "cotracker_anchor_plus_dpsr_voxel_interior",
        "particle_mode": "depth_only" if depth_only else "completed",
        "use_dpsr": bool(not depth_only),
        "fill_interior": bool((not depth_only) and fill_interior),
        "particle_voxel_size": voxel_size,
        "dpsr": dpsr_meta,
        "voxel_grid": voxel_meta,
        "tracking_knn_k": int(args.tracking_knn_k),
        "tracking_knn_power": float(args.tracking_knn_power),
        "ground_clip": ground_clip_meta,
        "cotracker_anchor_count": tracked_count,
        "tracked_particle_count": particle_tracked_count,
        "shell_extra_count": int(extra_shell_frame0.shape[0]),
        "shell_count": shell_count,
        "interior_count": interior_count,
        "full_particle_count": particle_count,
    }

    data = {
        "particle_coords": particle_coords,
        "particle_flows": particle_flows,
        "particle_velocities": particle_velocities,
        # particle_deformation_gradients (identity) and particle_affine_velocity_matrices
        # (zeros) are not stored — the loader reconstructs eye/zeros from particle_coords.
        "particle_colors": torch.from_numpy(particle_colors).float(),
        "particle_motion_valid": torch.from_numpy(particle_motion_valid).bool(),
        "particle_material_params": {
            "log_E": torch.full((particle_count,), float(args.default_log_e), dtype=torch.float32),
            "nu": torch.full((particle_count,), float(args.default_nu), dtype=torch.float32),
        },
        "tracked_visible_particle_indices": torch.from_numpy(visible_particle_indices).long(),
        "tracked_particle_count": particle_tracked_count,
        "completed_shell_count": shell_count,
        "completed_interior_count": interior_count,
        "completed_full_particle_count": particle_count,
        "rigid_body_coords": rigid_coords,
        "rigid_body_flows": _compute_point_flows(rigid_coords),
        "controller_points": torch.from_numpy(controller_points).float(),
        "contact_controller_points": torch.from_numpy(contact_controller).float(),
        "contact_object_points": torch.from_numpy(contact_object).float(),
        "contact_object_particle_ids": torch.from_numpy(contact_object_ids).long(),
        "contact_distances": torch.from_numpy(contact_distances).float(),
        # episode flags (is_manipulation, manipulation_flag, include_tracked_surface_points,
        # interaction_start_frame, depth_available) live in config.yaml ("episode"); the
        # loader injects them into the sample. Not stored here.
        "manipulation_contact_points": torch.from_numpy(contact_object).float(),
        "manipulation_contact_particle_ids": torch.from_numpy(contact_object_ids).long(),
        # rigid collision config lives only in config.yaml (rigid_body.collision);
        # the dataset loader and augmentation read it from there, not from this .pt.
        # Shell/interior sequences kept; their frame-0 slices and the full-cloud
        # alias are NOT stored — they duplicate particle_coords / [0] exactly:
        #   completed_full_particle_coords  == particle_coords
        #   completed_shell_initial_points  == completed_surface_points == completed_shell_points[0]
        #   completed_interior_points       == completed_interior_particle_coords[0]
        "completed_shell_points": torch.from_numpy(completed_shell_sequence).float(),
        "completed_interior_particle_coords": torch.from_numpy(completed_interior_sequence).float(),
        "shape_completion": shape_completion_meta,
        "observation_data": {
            "frame_indices": torch.arange(frame_count, dtype=torch.long),
            "view_indices": torch.tensor(views, dtype=torch.long),
            "object_points_clean": torch.from_numpy(observed_points).float(),
            "object_points_noisy": torch.from_numpy(observed_points).float(),
            "object_valid_mask": torch.from_numpy(observed_valid).bool(),
            "object_pixels": torch.from_numpy(observed_pixels).long(),
            "object_particle_ids": torch.from_numpy(observed_particle_ids).long(),
            "object_particle_distances": torch.from_numpy(observed_particle_distances).float(),
        },
        "depth_maps": torch.from_numpy(primary_depth).float(),
        "segmented_depth_maps": torch.from_numpy(primary_segmented_depth).float(),
        "object_masks": torch.from_numpy(primary_object_masks).to(torch.uint8),
        "hand_masks": torch.from_numpy(primary_hand_masks).to(torch.uint8),
        "rgb_images": torch.from_numpy(primary_rgb).to(torch.uint8),
        "camera_intrinsics": torch.from_numpy(intrinsics).float(),
        "camera_extrinsics": torch.from_numpy(extrinsics).float(),
        "primary_view": primary_view,
        "frame_dt": frame_dt,
        "ground_height": float(ground_height),
        "is_real_world": True,
        # NOTE: provenance/debug fields (source_dir, source_sequence,
        # source_ground_height, coordinate_system, source_coordinate_transform,
        # completed_shell_obj[_meta]) live in config.yaml / source_metadata.json,
        # not in this .pt — nothing reads them from here.
    }
    torch.save(data, episode_dir / "episode_data.pt")

    _write_json(
        episode_dir / "config.yaml",
        {
            "is_real_world": True,
            "source_sequence": source_root.name,
            "episode": {
                "is_manipulation": True,
                "manipulation_flag": 1.0,
                "include_tracked_surface_points": True,
                "interaction_start_frame": 0,
                "depth_available": True,
            },
            "simulation": {
                "dt": 1.0 / float(metadata.get("fps", 30)),
                "steps_per_frame": 1,
                "ground_height": ground_height,
            },
            "rigid_body": {
                "n_surface_fps": int(args.contact_points),
                "friction": 0.5,
                "surface_proxy": "nearest object points to hand/controller per frame",
                "collision": {
                    "manipulation_contact_radius": float(args.manipulation_contact_radius),
                    "manipulation_position_blend": 1.0,
                    "manipulation_velocity_blend": 1.0,
                },
            },
            "observation": {
                "views": list(views),
                "primary_view": primary_view,
                "image_size_wh": metadata["WH"],
                "max_observed_points_per_view": int(args.max_observed_points),
                "max_observation_particle_distance": float(args.max_observation_particle_distance),
                "object_mask_label": str(args.object_mask_label),
                "controller_mask_label": str(args.controller_mask_label),
                "depth_units": "meters",
                "coordinate_system": "physcore_z_up" if flip_z_to_z_up else "phystwin_source",
                "source_coordinate_transform": {
                    "flip_z_to_z_up": flip_z_to_z_up,
                },
            },
            "shape_completion": shape_completion_meta,
        },
    )
    _write_json(
        episode_dir / "source_metadata.json",
        {
            "source_sequence": source_root.name,
            "source_dir": str(source_root),
            "num_frames": int(frame_count),
            "views": list(views),
            "primary_view": primary_view,
            "source_ground_height": 0.0,
            "converted_ground_height": float(ground_height),
            "ground_height": float(ground_height),
            "object_mask_ids": object_mask_ids,
            "controller_mask_ids": controller_mask_ids,
            "particle_shape": list(object_points.shape),
            "controller_shape": list(controller_points.shape),
            "contact_object_particle_ids_shape": list(contact_object_ids.shape),
            "observed_points_shape": list(observed_points.shape),
            "particle_target_shape": list(particle_target_np.shape),
            "cotracker_anchor_count": tracked_count,
            "tracked_particle_count": particle_tracked_count,
            "completed_shell_points": shell_count,
            "completed_interior_points": interior_count,
            "completed_full_particle_count": particle_count,
            "shape_completion": shape_completion_meta,
            "max_observation_particle_distance": float(args.max_observation_particle_distance),
            "coordinate_system": "physcore_z_up" if flip_z_to_z_up else "phystwin_source",
            "source_coordinate_transform": {
                "flip_z_to_z_up": flip_z_to_z_up,
            },
            "completed_shell_obj": str((episode_dir / args.shell_obj_name).resolve()),
            "completed_shell_obj_meta": shell_obj_meta,
        },
    )

    print(f"wrote {episode_dir / 'episode_data.pt'}")
    print(f"wrote {episode_dir / args.shell_obj_name} ({shell_obj_meta})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_dir",
        required=True,
        help="raw sequence directory.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory containing episode_0000.",
    )
    parser.add_argument("--views", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--primary_view", type=int, default=0)
    parser.add_argument(
        "--frame_limit",
        type=int,
        default=0,
        help="Optional limit on number of frames to convert; 0 converts the whole sequence.",
    )
    parser.add_argument("--mask_erode_pixels", type=int, default=0, help="Erode object masks by this many pixels before use.")
    parser.add_argument("--max_observed_points", type=int, default=4096)
    parser.add_argument("--max_observation_particle_distance", type=float, default=0.05)
    parser.add_argument("--contact_points", type=int, default=16)
    parser.add_argument("--manipulation_contact_radius", type=float, default=0.025)
    parser.add_argument("--shell_obj_name", default="completed_initial_exterior_shell.obj")
    parser.add_argument(
        "--object_mask_label",
        default="auto",
        help="Mask label for the deformable object. 'auto' uses every non-hand mask ID.",
    )
    parser.add_argument("--controller_mask_label", default="hand", help="Mask label for the controller (e.g. 'hand' or 'robot gripper').")
    parser.add_argument("--default_log_e", type=float, default=10.0)
    parser.add_argument("--default_nu", type=float, default=0.35)

    parser.add_argument(
        "--particle_voxel_size",
        type=float,
        default=0.008,
        help="Spatial spacing (m) for shell dedup and interior voxel grid.",
    )
    parser.add_argument(
        "--target_total_particles",
        type=int,
        default=8000,
        help="Total particle budget. Shell vs interior split follows the DPSR "
             "voxel-grid surface:interior ratio for uniform 3D density. "
             "Cotracker is kept entirely; depth extras fill shell up to the "
             "ideal count; interior fills the remainder.",
    )
    parser.add_argument(
        "--particle_mode",
        choices=("completed", "depth_only"),
        default="completed",
        help="completed uses cotracker + depth shell + optional DPSR interior; "
             "depth_only uses only frame-0 segmented backprojected depth particles.",
    )
    parser.add_argument(
        "--use_dpsr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run DPSR shape completion. --no-use_dpsr implies depth_only particles.",
    )
    parser.add_argument(
        "--fill_interior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add DPSR voxel interior particles in completed mode.",
    )
    parser.add_argument(
        "--exclude_gripper_particles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop cotracker tracks projecting onto the controller/gripper mask. "
             "--no-exclude_gripper_particles disables it.",
    )
    parser.add_argument(
        "--shell_depth_max_distance",
        type=float,
        default=0.024,
        help="Drop back-projected depth points farther than this from any "
             "motions-valid cotracker anchor.  Rejects hand/background mask "
             "noise that would otherwise inflate the DPSR mesh and shell.",
    )
    parser.add_argument(
        "--interior_to_shell_max_distance",
        type=float,
        default=0.020,
        help="Drop interior voxel points farther than this from any shell "
             "anchor (cotracker + depth extras).  DPSR's smooth field can "
             "fill concavities (e.g., the void between limbs); this filter "
             "trims interior voxels that ended up there. Set to 0 to disable.",
    )

    parser.add_argument("--dpsr_grid_resolution", type=int, default=64)
    parser.add_argument("--dpsr_sigma", type=float, default=0.5)
    parser.add_argument("--dpsr_normal_knn", type=int, default=24)
    parser.add_argument("--dpsr_bbox_padding", type=float, default=0.12)
    parser.add_argument(
        "--dpsr_level",
        type=float,
        default=0.0,
        help="Marching-cubes iso-level for the DPSR field. 0.0 = raw zero "
             "level set. Small positive values can tighten the mesh, but go "
             "too high and the mesh collapses on thin features; the "
             "`--interior_to_shell_max_distance` filter is a safer way to "
             "trim concavity-fill.",
    )
    parser.add_argument("--dpsr_device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--dpsr_seed", type=int, default=12345)

    parser.add_argument(
        "--tracking_knn_k",
        type=int,
        default=8,
        help="Number of cotracker neighbors used to warp non-cotracker particles.",
    )
    parser.add_argument(
        "--tracking_knn_power",
        type=float,
        default=2.0,
        help="Inverse-distance power for cotracker KNN warp weights.",
    )

    parser.add_argument(
        "--clip_completed_particles_to_ground",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clamp completed particles so reconstruction artifacts don't go below ground.",
    )
    parser.add_argument("--ground_clip_quantile", type=float, default=0.1)
    parser.add_argument("--ground_clip_reference_frames", type=int, default=24)
    parser.add_argument("--ground_clip_offset", type=float, default=0.0)

    parser.add_argument(
        "--no_flip_z_to_z_up",
        dest="flip_z_to_z_up",
        action="store_false",
        help="Keep original vertical convention instead of converting to PhysCoRe z-up.",
    )
    parser.set_defaults(flip_z_to_z_up=True)
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
