"""
Render full particle-cloud videos for MPM augmented episodes, colored by log_E.
"""

import argparse
import os
import time

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def _sample_track_indices(num_tracks: int, max_index: int, seed: int) -> np.ndarray:
    if num_tracks <= 0 or max_index <= 0:
        return np.empty(0, dtype=np.int64)
    if num_tracks >= max_index:
        return np.arange(max_index, dtype=np.int64)
    return np.linspace(0, max_index - 1, num=num_tracks).round().astype(np.int64)


def _trail_segments(pts: np.ndarray, indices: np.ndarray, frame: int, trail_length: int, dims):
    if indices.size == 0 or trail_length <= 0:
        return []
    start = max(0, frame - trail_length)
    window = pts[start : frame + 1, indices][..., list(dims)]  # (L, K, D)
    return [window[:, i] for i in range(indices.size)]


def _comet_window(ctrl: np.ndarray, frame: int, trail_length: int) -> np.ndarray:
    """Return (trail_length, ctrl_n, 3) of swarm positions ending at `frame - 1`,
    oldest first. Pre-history slots are clamped to frame 0 so the ghost alpha
    gradient stays consistent across the run."""
    n_ctrl = ctrl.shape[1]
    out = np.empty((trail_length, n_ctrl, 3), dtype=ctrl.dtype)
    for k in range(trail_length):
        src = max(0, frame - trail_length + k)
        out[k] = ctrl[src]
    return out


def render_episode(
    ep_dir: str,
    out_path: str,
    fps: int = 30,
    dpi: int = 110,
    point_size: int = 6,
    num_tracks: int = 64,
    trail_length: int = 30,
    track_seed: int = 0,
    max_controls: int = 16,
    disp_scale: float = 10.0,
):
    t = torch.load(os.path.join(ep_dir, "episode_data.pt"), map_location="cpu", weights_only=False)
    pts = t["particle_coords"].numpy()                        # (T, N, 3)
    log_E = t["particle_material_params"]["log_E"].numpy()    # (N,)
    norm = Normalize(vmin=5.0, vmax=11.0)
    colors = cm.viridis(norm(log_E))

    tracked_count = int(t.get("tracked_particle_count", 0) or 0)
    tracked_count = min(tracked_count, pts.shape[1])
    track_idx = _sample_track_indices(num_tracks, tracked_count, track_seed)

    # particle_motion_valid is loaded for the displacement-quiver panel only.
    # It is a cotracker-validity flag carried through augmentation, so
    # it should NOT be applied as a position mask to augmented episodes (whose
    # positions come from MPM simulation and are always physically valid).
    motion_valid_t = t.get("particle_motion_valid")
    if torch.is_tensor(motion_valid_t):
        motion_valid = motion_valid_t.numpy().astype(bool)  # (T, N)
    else:
        motion_valid = np.ones((pts.shape[0], pts.shape[1]), dtype=bool)

    # Control points fed into the model: ep["r_coords"] = rigid_body_coords, truncated to max_controls.
    ctrl_all = t.get("rigid_body_coords")
    if torch.is_tensor(ctrl_all):
        ctrl = ctrl_all[:, :max_controls].numpy() if ctrl_all.shape[1] > 0 else np.empty((pts.shape[0], 0, 3))
    else:
        ctrl = np.empty((pts.shape[0], 0, 3))
    ctrl_n = int(ctrl.shape[1])

    # Bounds from the full per-frame trajectory so the camera doesn't jitter.
    combined = pts.reshape(-1, 3)
    if ctrl_n > 0:
        combined = np.concatenate([combined, ctrl.reshape(-1, 3)], axis=0)
    mn = combined.min(0)
    mx = combined.max(0)
    pad = 0.02
    xlim = (mn[0] - pad, mx[0] + pad)
    ylim = (mn[1] - pad, mx[1] + pad)
    zlim = (mn[2] - pad, mx[2] + pad)

    fig = plt.figure(figsize=(24, 6), dpi=dpi)
    # Top-down: regular 2D axes with equal aspect; correctly fills the panel.
    ax_top = fig.add_subplot(1, 4, 1)
    sc_top = ax_top.scatter(
        pts[0, :, 0], pts[0, :, 1], s=point_size, c=colors
    )
    trails_top = LineCollection([], colors="#ff5252", linewidths=0.9, alpha=0.85)
    ax_top.add_collection(trails_top)
    sc_top_track = ax_top.scatter(
        pts[0, track_idx, 0] if track_idx.size else [],
        pts[0, track_idx, 1] if track_idx.size else [],
        s=point_size * 3, facecolors="none", edgecolors="#ff5252", linewidths=1.0,
    )
    ax_top.set_xlim(*xlim); ax_top.set_ylim(*ylim)
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.set_xlabel("x"); ax_top.set_ylabel("y")
    ax_top.set_title("top-down (xy)")

    # Iso 3D view with explicit box aspect matching the data extent.
    ax_iso = fig.add_subplot(1, 4, 2, projection="3d")
    sc_iso = ax_iso.scatter(
        pts[0, :, 0], pts[0, :, 1], pts[0, :, 2],
        s=point_size, c=colors, depthshade=False,
    )
    # Line3DCollection requires at least one segment so add_collection3d can autoscale.
    _placeholder_seg = np.zeros((1, 2, 3))
    trails_iso = Line3DCollection(_placeholder_seg, colors="#ff5252", linewidths=0.9, alpha=0.85)
    ax_iso.add_collection3d(trails_iso)
    trails_iso.set_segments([])
    sc_iso_track = ax_iso.scatter(
        pts[0, track_idx, 0] if track_idx.size else [],
        pts[0, track_idx, 1] if track_idx.size else [],
        pts[0, track_idx, 2] if track_idx.size else [],
        s=point_size * 3, facecolors="none", edgecolors="#ff5252", linewidths=1.0,
        depthshade=False,
    )
    ax_iso.set_xlim(*xlim); ax_iso.set_ylim(*ylim); ax_iso.set_zlim(*zlim)
    ax_iso.set_box_aspect(
        (xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
    )
    ax_iso.view_init(elev=25, azim=-60)
    ax_iso.set_xlabel("x"); ax_iso.set_ylabel("y"); ax_iso.set_zlabel("z")
    ax_iso.set_title("iso (xyz)")

    # Controls panel: 3D iso of object cloud + the exact 16 control points fed to the model,
    # with a comet trail (recent frames of the swarm, alpha-faded oldest-first).
    ax_ctrl = fig.add_subplot(1, 4, 3, projection="3d")
    sc_ctrl_obj = ax_ctrl.scatter(
        pts[0, :, 0], pts[0, :, 1], pts[0, :, 2],
        s=max(point_size // 2, 2), c=colors, alpha=0.35, depthshade=False,
    )
    # Comet: trail_length×ctrl_n ghost points with a fixed alpha gradient (oldest → newest).
    n_ghost = max(trail_length, 1) * max(ctrl_n, 1)
    if ctrl_n > 0 and trail_length > 0:
        ghost_alpha = np.linspace(0.04, 0.45, num=trail_length, endpoint=True)
        ghost_alpha_per_point = np.repeat(ghost_alpha, ctrl_n)
        ghost_rgba = np.tile(np.array([1.0, 0.596, 0.0, 1.0]), (n_ghost, 1))
        ghost_rgba[:, 3] = ghost_alpha_per_point
        sc_ctrl_ghost = ax_ctrl.scatter(
            np.zeros(n_ghost), np.zeros(n_ghost), np.zeros(n_ghost),
            s=max(point_size * 2, 4), c=ghost_rgba, depthshade=False,
        )
    else:
        sc_ctrl_ghost = None
    sc_ctrl_pts = ax_ctrl.scatter(
        ctrl[0, :, 0] if ctrl_n else [],
        ctrl[0, :, 1] if ctrl_n else [],
        ctrl[0, :, 2] if ctrl_n else [],
        s=point_size * 6, c="#ff9800", edgecolors="black", linewidths=0.6, depthshade=False,
    )
    ax_ctrl.set_xlim(*xlim); ax_ctrl.set_ylim(*ylim); ax_ctrl.set_zlim(*zlim)
    ax_ctrl.set_box_aspect(
        (xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
    )
    ax_ctrl.view_init(elev=25, azim=-60)
    ax_ctrl.set_xlabel("x"); ax_ctrl.set_ylabel("y"); ax_ctrl.set_zlabel("z")
    ax_ctrl.set_title(f"controls (max_controls={max_controls})")

    # Displacement panel: per-tracked-particle incremental displacement (model's tracked_disp input).
    # Matches `observed_features` in mfm_training.py with tracked_disp_mode='incremental' +
    # persistent_track_use_motion_valid=True: tracked_disp[i] = coords[f,i] - coords[f-1,i] gated by motion_valid.
    ax_disp = fig.add_subplot(1, 4, 4, projection="3d")
    sc_disp_obj = ax_disp.scatter(
        pts[0, :, 0], pts[0, :, 1], pts[0, :, 2],
        s=max(point_size // 2, 2), c=colors, alpha=0.25, depthshade=False,
    )
    quiver_holder = {"q": None}  # mutable holder so we can swap inside the loop
    ax_disp.set_xlim(*xlim); ax_disp.set_ylim(*ylim); ax_disp.set_zlim(*zlim)
    ax_disp.set_box_aspect(
        (xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
    )
    ax_disp.view_init(elev=25, azim=-60)
    ax_disp.set_xlabel("x"); ax_disp.set_ylabel("y"); ax_disp.set_zlabel("z")
    ax_disp.set_title(f"tracked_disp ×{disp_scale:g} (incremental, motion_valid)")

    sm = cm.ScalarMappable(norm=norm, cmap="viridis"); sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_top, ax_iso, ax_ctrl, ax_disp], shrink=0.7, pad=0.08, location="right")
    cbar.set_label("log_E (material stiffness)")

    suptitle = fig.suptitle("", y=0.98)

    T = pts.shape[0]
    ep_name = os.path.basename(ep_dir.rstrip("/"))
    writer = imageio.get_writer(
        out_path,
        format="FFMPEG",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    t0 = time.time()
    for f in range(T):
        # 2D scatter uses .set_offsets; 3D uses ._offsets3d.
        sc_top.set_offsets(np.stack([pts[f, :, 0], pts[f, :, 1]], axis=-1))
        sc_iso._offsets3d = (pts[f, :, 0], pts[f, :, 1], pts[f, :, 2])
        sc_ctrl_obj._offsets3d = (pts[f, :, 0], pts[f, :, 1], pts[f, :, 2])
        if track_idx.size:
            sc_top_track.set_offsets(np.stack([pts[f, track_idx, 0], pts[f, track_idx, 1]], axis=-1))
            sc_iso_track._offsets3d = (pts[f, track_idx, 0], pts[f, track_idx, 1], pts[f, track_idx, 2])
            trails_top.set_segments(_trail_segments(pts, track_idx, f, trail_length, (0, 1)))
            trails_iso.set_segments(_trail_segments(pts, track_idx, f, trail_length, (0, 1, 2)))
        if ctrl_n:
            sc_ctrl_pts._offsets3d = (ctrl[f, :, 0], ctrl[f, :, 1], ctrl[f, :, 2])
            if sc_ctrl_ghost is not None:
                ghost = _comet_window(ctrl, f, trail_length).reshape(-1, 3)
                sc_ctrl_ghost._offsets3d = (ghost[:, 0], ghost[:, 1], ghost[:, 2])
        # Displacement panel: refresh background cloud + per-tracked-particle quiver.
        sc_disp_obj._offsets3d = (pts[f, :, 0], pts[f, :, 1], pts[f, :, 2])
        if quiver_holder["q"] is not None:
            quiver_holder["q"].remove()
            quiver_holder["q"] = None
        if track_idx.size and f > 0:
            ref_frame = f - 1
            origin = pts[f, track_idx]                    # (K, 3)
            disp = (pts[f, track_idx] - pts[ref_frame, track_idx]) * float(disp_scale)
            valid = motion_valid[f, track_idx] & motion_valid[ref_frame, track_idx]
            if valid.any():
                origin_v = origin[valid]; disp_v = disp[valid]
                quiver_holder["q"] = ax_disp.quiver(
                    origin_v[:, 0], origin_v[:, 1], origin_v[:, 2],
                    disp_v[:, 0], disp_v[:, 1], disp_v[:, 2],
                    color="#00bcd4", arrow_length_ratio=0.3, linewidths=1.2,
                )
        suptitle.set_text(
            f"{ep_name}  all particles (N={pts.shape[1]})"
            f"  tracked={track_idx.size}/{tracked_count}"
            f"  controls={ctrl_n}  frame {f + 1}/{T}"
        )
        fig.canvas.draw()
        img = np.asarray(fig.canvas.renderer.buffer_rgba())[..., :3]
        writer.append_data(img)
    writer.close()
    plt.close(fig)
    print(f"wrote {out_path}  ({T} frames, {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="data_episodes_augmented/episodes/single_lift_rope_elastic",
    )
    ap.add_argument("--episodes", nargs="+", default=["episode_0000"])
    ap.add_argument("--out_dir", default="data_episodes_augmented/video")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--num_tracks", type=int, default=64,
                    help="Number of tracked points to overlay (0 disables trails).")
    ap.add_argument("--trail_length", type=int, default=30,
                    help="Frames of history to draw behind each tracked point.")
    ap.add_argument("--max_controls", type=int, default=16,
                    help="Number of leading rigid_body_coords rows to show; must match model.max_controls.")
    ap.add_argument("--disp_scale", type=float, default=10.0,
                    help="Scale factor applied to per-frame tracked_disp arrows for visibility.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for ep in args.episodes:
        ep_dir = os.path.join(args.root, ep)
        out = os.path.join(args.out_dir, f"{ep}.mp4")
        render_episode(
            ep_dir, out, fps=args.fps,
            num_tracks=args.num_tracks, trail_length=args.trail_length,
            max_controls=args.max_controls, disp_scale=args.disp_scale,
        )
