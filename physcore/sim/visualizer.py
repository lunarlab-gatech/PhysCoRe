"""
Low-resolution video visualization of MPM simulation.
"""

import numpy as np
from typing import Optional, Tuple


def project_points(points: np.ndarray,
                   azimuth: float = 45.0,
                   elevation: float = 25.0,
                   distance: float = 2.0,
                   center: np.ndarray = None,
                   width: int = 320,
                   height: int = 240,
                   fov: float = 40.0) -> np.ndarray:
    """Simple perspective projection from a viewpoint.

    Args:
        points: (N, 3) in world space [0,1]^3
        Returns: (N, 2) pixel coordinates, (N,) depth
    """
    if center is None:
        center = np.array([0.5, 0.5, 0.3])

    az = np.radians(azimuth)
    el = np.radians(elevation)

    # Camera position
    cam_x = center[0] + distance * np.cos(el) * np.cos(az)
    cam_y = center[1] + distance * np.cos(el) * np.sin(az)
    cam_z = center[2] + distance * np.sin(el)
    cam_pos = np.array([cam_x, cam_y, cam_z])

    # Camera basis
    forward = center - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)

    # Transform to camera space
    rel = points - cam_pos  # (N, 3)
    cam_z_vals = rel @ forward  # depth
    cam_x_vals = rel @ right
    cam_y_vals = rel @ up

    # Perspective divide
    focal = 0.5 * width / np.tan(np.radians(fov / 2))
    valid = cam_z_vals > 0.01
    px = np.zeros(len(points))
    py = np.zeros(len(points))
    px[valid] = focal * cam_x_vals[valid] / cam_z_vals[valid] + width / 2
    py[valid] = height / 2 - focal * cam_y_vals[valid] / cam_z_vals[valid]

    coords = np.stack([px, py], axis=-1)
    return coords, cam_z_vals


def render_frame(
    particles: np.ndarray,
    rigid_points: Optional[np.ndarray] = None,
    particle_colors: Optional[np.ndarray] = None,
    particle_alphas: Optional[np.ndarray] = None,
    width: int = 320,
    height: int = 240,
    azimuth: float = 45.0,
    elevation: float = 25.0,
    bg_color: Tuple[int, int, int] = (245, 245, 245),
    point_radius: int = 1,
    ground_height: float = 0.02,
) -> np.ndarray:
    """Render a single frame as a numpy image (H, W, 3) uint8.

    Uses z-buffered point splatting with depth shading, a ground-plane grid,
    and projected drop shadows for spatial context.

    Args:
        particle_alphas: Optional (N,) float in [0, 1]. Per-particle opacity;
            1.0 = fully opaque, 0.0 = fully transparent (shows background).
    """
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)

    # ── Draw ground plane as a perspective-projected grid ──────────────
    grid_lines = 11
    grid_lo, grid_hi = 0.0, 1.0
    grid_vals = np.linspace(grid_lo, grid_hi, grid_lines)
    grid_color = np.array([195, 195, 195], dtype=np.uint8)
    grid_edge_color = np.array([175, 175, 175], dtype=np.uint8)
    # Lines parallel to X
    for i, v in enumerate(grid_vals):
        pts = np.array([[x, v, ground_height] for x in np.linspace(grid_lo, grid_hi, 80)])
        px, _ = project_points(pts, azimuth, elevation, width=width, height=height)
        pxi = np.round(px[:, 0]).astype(int)
        pyi = np.round(px[:, 1]).astype(int)
        valid = (pxi >= 0) & (pxi < width) & (pyi >= 0) & (pyi < height)
        c = grid_edge_color if (i == 0 or i == grid_lines - 1) else grid_color
        img[pyi[valid], pxi[valid]] = c
    # Lines parallel to Y
    for i, v in enumerate(grid_vals):
        pts = np.array([[v, y, ground_height] for y in np.linspace(grid_lo, grid_hi, 80)])
        px, _ = project_points(pts, azimuth, elevation, width=width, height=height)
        pxi = np.round(px[:, 0]).astype(int)
        pyi = np.round(px[:, 1]).astype(int)
        valid = (pxi >= 0) & (pxi < width) & (pyi >= 0) & (pyi < height)
        c = grid_edge_color if (i == 0 or i == grid_lines - 1) else grid_color
        img[pyi[valid], pxi[valid]] = c

    # ── Project drop shadows onto ground plane ─────────────────────────
    shadow_color = np.array([210, 210, 215], dtype=np.uint8)
    for pts_3d in [particles, rigid_points]:
        if pts_3d is not None and len(pts_3d) > 0:
            shadow_pts = pts_3d.copy()
            shadow_pts[:, 2] = ground_height
            scoords, _ = project_points(shadow_pts, azimuth, elevation, width=width, height=height)
            sx = np.round(scoords[:, 0]).astype(int)
            sy = np.round(scoords[:, 1]).astype(int)
            valid = (sx >= 0) & (sx < width) & (sy >= 0) & (sy < height)
            img[sy[valid], sx[valid]] = shadow_color

    # ── Collect all 3D points for splatting ────────────────────────────
    all_points = []
    all_colors = []
    all_depths = []
    all_alphas = []

    # MPM particles
    if particles is not None and len(particles) > 0:
        if particle_colors is None:
            particle_colors = np.full((len(particles), 3), [70, 130, 230], dtype=np.uint8)

        coords, depths = project_points(particles, azimuth, elevation, width=width, height=height)
        all_points.append(coords)
        all_colors.append(particle_colors.copy())
        all_depths.append(depths)
        if particle_alphas is not None:
            all_alphas.append(np.clip(particle_alphas, 0.0, 1.0).astype(np.float32))
        else:
            all_alphas.append(np.ones(len(particles), dtype=np.float32))

    # Rigid body points
    if rigid_points is not None and len(rigid_points) > 0:
        rcoords, rdepths = project_points(rigid_points, azimuth, elevation, width=width, height=height)
        rcolors = np.full((len(rigid_points), 3), [220, 80, 60], dtype=np.uint8)
        all_points.append(rcoords)
        all_colors.append(rcolors)
        all_depths.append(rdepths)
        all_alphas.append(np.ones(len(rigid_points), dtype=np.float32))

    if not all_points:
        return img

    coords = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    depths = np.concatenate(all_depths, axis=0)
    alphas = np.concatenate(all_alphas, axis=0)

    # ── Depth-based shading: nearer points brighter, farther darker ───
    valid_depth = depths > 0.01
    if valid_depth.any():
        d_min = depths[valid_depth].min()
        d_max = depths[valid_depth].max()
        d_range = max(d_max - d_min, 1e-6)
        # shade factor: 1.0 (nearest) → 0.55 (farthest)
        shade = np.where(valid_depth, 1.0 - 0.45 * (depths - d_min) / d_range, 0.7)
        colors = (colors.astype(np.float32) * shade[:, None]).clip(0, 255).astype(np.uint8)

    # Z-sort (painter's algorithm: draw far points first)
    order = np.argsort(-depths)
    coords = coords[order]
    colors = colors[order]
    alphas = alphas[order]

    # Rasterize
    px = np.round(coords[:, 0]).astype(int)
    py = np.round(coords[:, 1]).astype(int)

    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px = px[valid]
    py = py[valid]
    colors = colors[valid]
    alphas = alphas[valid]

    has_alpha = not np.all(alphas >= 0.99)

    if point_radius <= 1:
        if has_alpha:
            # Alpha-blend each point onto the image
            bg = img[py, px].astype(np.float32)
            fg = colors.astype(np.float32)
            a = alphas[:, None]
            img[py, px] = (a * fg + (1.0 - a) * bg).clip(0, 255).astype(np.uint8)
        else:
            img[py, px] = colors
    else:
        for dx in range(-point_radius, point_radius + 1):
            for dy in range(-point_radius, point_radius + 1):
                if dx * dx + dy * dy <= point_radius * point_radius:
                    cx = np.clip(px + dx, 0, width - 1)
                    cy = np.clip(py + dy, 0, height - 1)
                    if has_alpha:
                        bg = img[cy, cx].astype(np.float32)
                        fg = colors.astype(np.float32)
                        a = alphas[:, None]
                        img[cy, cx] = (a * fg + (1.0 - a) * bg).clip(0, 255).astype(np.uint8)
                    else:
                        img[cy, cx] = colors

    return img


def _deformation_magnitude(F: np.ndarray) -> np.ndarray:
    """Compute per-particle deformation magnitude from deformation gradient.

    Args:
        F: (N, 3, 3) deformation gradient per particle.
    Returns:
        (N,) scalar >= 0 measuring deviation from identity (undeformed).
    """
    # Green-Lagrange strain: E = 0.5 * (F^T F - I)
    # Use Frobenius norm of E as the scalar magnitude.
    FtF = np.einsum('...ji,...jk->...ik', F, F)  # (N, 3, 3)
    I = np.eye(3, dtype=F.dtype)
    E = 0.5 * (FtF - I)
    return np.sqrt((E * E).sum(axis=(-2, -1)))  # Frobenius norm of E


def _deformation_colormap(magnitudes: np.ndarray, vmin: float = 0.0, vmax: float = 0.15) -> np.ndarray:
    """Map deformation magnitudes to a blue → cyan → yellow → red colormap.

    Args:
        magnitudes: (N,) non-negative scalars.
        vmin, vmax: clamp range for the colormap.
    Returns:
        (N, 3) uint8 RGB colors.
    """
    t = np.clip((magnitudes - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)
    # Piecewise linear: blue(0) → cyan(0.33) → yellow(0.66) → red(1.0)
    r = np.where(t < 0.33, 30,
        np.where(t < 0.66, 30 + (t - 0.33) / 0.33 * 225,
                 255))
    g = np.where(t < 0.33, 80 + t / 0.33 * 170,
        np.where(t < 0.66, 250,
                 250 - (t - 0.66) / 0.34 * 210))
    b = np.where(t < 0.33, 230,
        np.where(t < 0.66, 230 - (t - 0.33) / 0.33 * 200,
                 30))
    return np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)


def _particle_colors_from_material_ids(
    particle_trajectory: np.ndarray,
    particle_material_ids: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    if particle_material_ids is None:
        return None

    unique_ids = np.unique(particle_material_ids)
    color_palette = np.array([
        [70, 130, 230],
        [230, 160, 50],
        [80, 200, 120],
        [200, 80, 200],
        [200, 200, 80],
        [80, 200, 200],
        [230, 100, 100],
        [150, 100, 200],
        [100, 180, 100],
    ], dtype=np.uint8)
    n_particles = particle_trajectory.shape[1]
    particle_colors = np.zeros((n_particles, 3), dtype=np.uint8)
    for i, uid in enumerate(unique_ids):
        mask = particle_material_ids == uid
        particle_colors[mask] = color_palette[i % len(color_palette)]
    return particle_colors


def scalar_to_heatmap_colors(
    values: np.ndarray,
    vmin: float = 0.0,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Map per-particle scalar values to a blue→cyan→green→yellow→red heatmap.

    Args:
        values: (N,) non-negative floats (e.g. per-particle loss).
        vmin: Value mapped to the cool end (blue).
        vmax: Value mapped to the hot end (red). Defaults to 95th percentile.
    Returns:
        (N, 3) uint8 RGB colors.
    """
    if vmax is None:
        vmax = float(np.percentile(values, 95)) if len(values) > 0 else 1.0
    vmax = max(vmax, vmin + 1e-8)
    t = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)

    # 5-stop gradient: blue(0) → cyan(0.25) → green(0.5) → yellow(0.75) → red(1.0)
    anchors = np.array([
        [30, 80, 220],    # blue
        [30, 200, 220],   # cyan
        [50, 200, 50],    # green
        [230, 220, 30],   # yellow
        [220, 50, 30],    # red
    ], dtype=np.float32)
    n_seg = len(anchors) - 1
    seg = np.clip((t * n_seg).astype(int), 0, n_seg - 1)
    frac = (t * n_seg - seg).astype(np.float32)
    colors = (anchors[seg] * (1 - frac[:, None]) + anchors[seg + 1] * frac[:, None])
    return np.clip(colors, 0, 255).astype(np.uint8)


def _write_video_frames(frames: np.ndarray, output_path: str, fps: int) -> None:
    macro_block_size = 16
    height, width = frames.shape[1], frames.shape[2]
    pad_height = (macro_block_size - (height % macro_block_size)) % macro_block_size
    pad_width = (macro_block_size - (width % macro_block_size)) % macro_block_size
    if pad_height or pad_width:
        frames = np.pad(
            frames,
            ((0, 0), (0, pad_height), (0, pad_width), (0, 0)),
            mode='constant',
            constant_values=240,
        )

    try:
        import imageio
    except ImportError:
        print("Warning: imageio not available. Skipping video export.")
        return

    try:
        imageio.mimwrite(output_path, list(frames), fps=fps)
    except Exception:
        try:
            writer = imageio.get_writer(output_path, fps=fps)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
        except Exception:
            gif_path = output_path.rsplit('.', 1)[0] + '.gif'
            imageio.mimwrite(gif_path, list(frames), duration=1000 // fps, loop=0)
            output_path = gif_path
    print(f"Video saved to {output_path}")


def render_video(
    particle_trajectory: np.ndarray,
    rigid_trajectory: Optional[np.ndarray] = None,
    particle_material_ids: Optional[np.ndarray] = None,
    output_path: str = "simulation.mp4",
    fps: int = 30,
    width: int = 320,
    height: int = 240,
    azimuth: float = 45.0,
    elevation: float = 25.0,
):
    """Render full simulation video with constant material-id colors.

    Args:
        particle_trajectory: (T, N, 3) particle positions
        rigid_trajectory: (T, M, 3) rigid body surface points or None
        particle_material_ids: (N,) int for coloring by material
        output_path: path to save video
    """
    T = particle_trajectory.shape[0]
    particle_colors = _particle_colors_from_material_ids(particle_trajectory, particle_material_ids)

    frames = []
    for t in range(T):
        rigid_pts = rigid_trajectory[t] if rigid_trajectory is not None else None
        frame = render_frame(
            particle_trajectory[t],
            rigid_pts,
            particle_colors=particle_colors,
            width=width, height=height,
            azimuth=azimuth, elevation=elevation,
        )
        frames.append(frame)

    frames = np.stack(frames)
    _write_video_frames(frames, output_path=output_path, fps=fps)


def render_deformation_video(
    particle_trajectory: np.ndarray,
    deformation_gradients: np.ndarray,
    rigid_trajectory: Optional[np.ndarray] = None,
    output_path: str = "deformation.gif",
    fps: int = 30,
    width: int = 320,
    height: int = 240,
    azimuth: float = 45.0,
    elevation: float = 25.0,
    alpha_min: float = 0.05,
    alpha_max: float = 1.0,
):
    """Render a video where particle opacity reflects deformation magnitude.

    Higher deformation gradient norm → more opaque.  Rigid body points are
    always fully opaque.

    Args:
        particle_trajectory: (T, N, 3) particle positions.
        deformation_gradients: (T, N, 3, 3) deformation gradient per particle.
        rigid_trajectory: (T, M, 3) rigid body surface points or None.
        alpha_min: minimum opacity for undeformed particles.
        alpha_max: maximum opacity for highly deformed particles.
    """
    T = particle_trajectory.shape[0]
    # Compute global normalization for consistent opacity across frames
    all_mag = _deformation_magnitude(deformation_gradients.reshape(-1, 3, 3))
    vmax = float(np.percentile(all_mag, 99))
    vmax = max(vmax, 0.005)

    frames = []
    for t in range(T):
        rigid_pts = rigid_trajectory[t] if rigid_trajectory is not None else None
        mag = _deformation_magnitude(deformation_gradients[t])
        norm_mag = np.clip(mag / vmax, 0.0, 1.0)
        alphas = alpha_min + (alpha_max - alpha_min) * norm_mag
        frame = render_frame(
            particle_trajectory[t],
            rigid_pts,
            particle_alphas=alphas,
            width=width, height=height,
            azimuth=azimuth, elevation=elevation,
        )
        frames.append(frame)

    frames = np.stack(frames)
    _write_video_frames(frames, output_path=output_path, fps=fps)


def render_comparison_video(
    predicted_particle_trajectory: np.ndarray,
    gt_particle_trajectory: np.ndarray,
    predicted_rigid_trajectory: Optional[np.ndarray] = None,
    gt_rigid_trajectory: Optional[np.ndarray] = None,
    particle_material_ids: Optional[np.ndarray] = None,
    particle_colors: Optional[np.ndarray] = None,
    output_path: str = "comparison.mp4",
    fps: int = 30,
    width: int = 320,
    height: int = 240,
    azimuth: float = 45.0,
    elevation: float = 25.0,
    separator_width: int = 4,
) -> None:
    """Render predicted and ground-truth rollouts side by side in one video.

    ``particle_colors`` (N, 3) uint8 takes precedence over ``particle_material_ids``.
    """
    t_frames = min(predicted_particle_trajectory.shape[0], gt_particle_trajectory.shape[0])
    if particle_colors is None:
        particle_colors = _particle_colors_from_material_ids(predicted_particle_trajectory, particle_material_ids)
    separator = np.full((height, separator_width, 3), 210, dtype=np.uint8)

    frames = []
    for t in range(t_frames):
        pred_rigid = predicted_rigid_trajectory[t] if predicted_rigid_trajectory is not None else None
        gt_rigid = gt_rigid_trajectory[t] if gt_rigid_trajectory is not None else None
        predicted_frame = render_frame(
            predicted_particle_trajectory[t],
            pred_rigid,
            particle_colors=particle_colors,
            width=width,
            height=height,
            azimuth=azimuth,
            elevation=elevation,
        )
        gt_frame = render_frame(
            gt_particle_trajectory[t],
            gt_rigid,
            particle_colors=particle_colors,
            width=width,
            height=height,
            azimuth=azimuth,
            elevation=elevation,
        )
        frames.append(np.concatenate([predicted_frame, separator, gt_frame], axis=1))

    _write_video_frames(np.stack(frames), output_path=output_path, fps=fps)
