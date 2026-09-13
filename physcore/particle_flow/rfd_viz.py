"""
Videos and loss curves for the RfD entrypoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import imageio.v2 as imageio
import numpy as np
import torch

from physcore.particle_flow import episode_runtime as ert
from physcore.particle_flow.mfm_training import material_guess
from physcore.particle_flow.rfd_runtime import (
    _default_material_confidence,
    first_half_windows,
)


def _log_E_row(material: torch.Tensor) -> torch.Tensor:
    """(…, N, 2) material → the (N,) log_E channel, for per-frame viz colors."""
    m = material
    while m.dim() > 2:
        m = m[0]
    return m[:, 0]


@torch.no_grad()
def _full_episode_eval_rollout(
    *,
    engine,
    refiner,
    corrector,
    ep: Dict,
    cfg,
    kinematic_ids: Optional[torch.Tensor],
    n_full_windows: int,
    tail_frames: int,
    k: int,
    rollout_steps: int,
    total_outer_steps: int,
    pid_chunk_fn=None,
    first_half_only: bool = False,
) -> "np.ndarray":
    """Run the same per-window MPM rollout used in training but no-grad and
    collect predicted positions per camera frame.

    Two mirroring modes:
      * ``first_half_only`` (training videos) — the Refiner observes the first
        half with no MPM, material is frozen there, and only frames
        ``0..mid_frame`` are rolled with the GVC on. Matches what train() optimizes.
      * ``pid_chunk_fn`` given (validation videos, from validate_RfD.py) — the
        first half is rolled GVC-off with per-frame PID correction and a
        per-window Refiner refresh, the second half GVC-on with the material
        frozen at the midpoint. Matches what _run_validation scores (except that
        the trailing partial window is rendered here but no longer scored).

    Returns ``(pred_pts, log_E_seq, mid_frame)``; ``log_E_seq`` is (T, N), the
    per-frame log_E actually used by the MPM. ``mid_frame`` is 0 when the clip
    has a single phase, so the renderer draws no phase banner.
    """
    coords = ep["coords"]
    T, N, _ = coords.shape
    device = coords.device

    half_window = first_half_windows(cfg, n_full_windows)
    # Windows rolled with PID (validation clips only); 0 elsewhere.
    mid_window = half_window if pid_chunk_fn is not None else 0
    mid_frame = mid_window * k

    was_active = bool(getattr(engine, "_gvc_active", False))
    engine.enable_corrector(True)
    try:
        refiner.reset(coords[0:1])
        material = material_guess(ep, cfg, 0, seed=int(cfg.train.get("seed", 0)))
        confidence = _default_material_confidence(material)  # placeholder; refreshed by predict_window
        if first_half_only:
            # Mirror train(): Refiner observes windows 0..half-1 with no MPM,
            # then the material stays frozen for the whole adaptation rollout.
            for w in range(half_window):
                material, confidence = refiner.predict_window(
                    ep, start_frame=w * k, end_frame=(w + 1) * k, material=material,
                )

        positions = coords[0:1].to(device)
        velocities = ep["particle_v"][0:1].to(device)
        F_state = ep["particle_F"][0:1].to(device)
        C_state = (
            ep["particle_C"][0:1].to(device)
            if ep.get("particle_C", None) is not None else None
        )

        pred_pts = coords.new_empty((T, N, 3))
        pred_pts[0] = coords[0]
        # Per-frame log_E so the clip is colored by the material actually in
        # effect when each frame was rolled (phase 1 refreshes; phase 2 frozen).
        log_E_seq = coords.new_empty((T, N))

        chunk_dt = ert._real_world_chunk_dt(ep, engine, rollout_steps)
        ground_height = float(ep.get("ground_height", engine.ground_height))

        def _run_one(f_start: int, f_end: int, outer_steps: int, mat, conf):
            nonlocal positions, velocities, F_state, C_state
            log_E = mat[..., 0]
            nu = mat[..., 1]
            rigid_window = ert._piecewise_catmull_rom_window(
                ep["r_coords"], f_start, f_end, steps_per_frame=rollout_steps,
            )
            controller_window = ert._piecewise_catmull_rom_window(
                ep.get("controller_grid_points", None), f_start, f_end, steps_per_frame=rollout_steps,
            )
            contact_window = (
                ert._hold_index_window(kinematic_ids, f_start, outer_steps)
                if torch.is_tensor(kinematic_ids) else None
            )
            rigid_in = rigid_window.unsqueeze(0) if rigid_window is not None else None
            controller_in = controller_window.unsqueeze(0) if controller_window is not None else None
            B = positions.shape[0]
            delta_v = torch.zeros(outer_steps, B, N, 3, device=device, dtype=positions.dtype)
            with ert._temporary_rollout_timestep(
                engine, dt=chunk_dt, ground_height=ground_height,
            ):
                out = engine(
                    positions, velocities, F_state, delta_v, log_E, nu,
                    C=C_state,
                    material_model_info=ep.get("particle_material_models", None),
                    rigid_points=rigid_in,
                    rigid_collision_cfg=ep.get("rigid_collision_cfg", {}),
                    rigid_body_primitives=ep.get("rigid_body_primitives", []),
                    manipulation_indicator=ep.get("manipulation_flag", None),
                    manipulation_contact_particle_ids=contact_window,
                    controller_grid_points=controller_in,
                    material_confidence=conf,
                )
            pred_pos_all = out["predicted_positions"][:, 0]
            n_predicted = (f_end - f_start)
            for t in range(n_predicted):
                gt_frame = f_start + 1 + t
                if gt_frame >= T:
                    break
                cam_step_idx = (t + 1) * rollout_steps - 1
                pred_pts[gt_frame] = pred_pos_all[cam_step_idx]
                log_E_seq[gt_frame] = _log_E_row(mat)
            positions = out["predicted_positions"][-1].detach()
            velocities = out["final_velocity"].detach()
            F_state = out["final_deformation_gradient"].detach()
            C_state = out["final_C"].detach() if out.get("final_C", None) is not None else None

        correction_steps_count = int(
            cfg.model.get("correction_rollout_steps", cfg.model.get("correction_steps", 4))
        )
        n_windows = half_window if first_half_only else n_full_windows
        for window_idx in range(n_windows):
            f_start = window_idx * k
            f_end = f_start + k
            # Training freezes phi at the midpoint estimate, so no per-window
            # refresh in first_half_only mode. In two-phase (validation) mode the
            # Refiner refreshes over the PID half ONLY and freezes at the midpoint,
            # matching _run_validation; single-phase adapt clips refresh throughout.
            refresh = (
                (not first_half_only)
                and (window_idx < mid_window if pid_chunk_fn is not None else True)
            )
            if refresh:
                material, confidence = refiner.predict_window(
                    ep, start_frame=f_start, end_frame=f_end, material=material,
                )
            if window_idx < mid_window:
                # Phase 1 — material identification: GVC off, per-frame PID correction.
                engine.enable_corrector(False)
                for i in range(k):
                    if f_start + i + 1 >= T:
                        break
                    positions, velocities, F_state, C_state = pid_chunk_fn(
                        engine=engine,
                        ctx={"ep": ep, "coords": coords, "kinematic_ids": kinematic_ids, "N": N},
                        f_start=f_start + i, chunk_steps=rollout_steps,
                        correction_steps_count=correction_steps_count,
                        run_correction=True,
                        positions=positions, velocities=velocities,
                        F_state=F_state, C_state=C_state,
                        material=material,
                        chunk_dt=chunk_dt, ground_height=ground_height,
                    )
                    pred_pts[f_start + i + 1] = positions[0]
                    log_E_seq[f_start + i + 1] = _log_E_row(material)
                continue
            engine.enable_corrector(True)  # Phase 2 — RfD validation.
            _run_one(f_start, f_end, total_outer_steps, material, confidence)

        if tail_frames > 0 and not first_half_only:
            f_start = n_full_windows * k
            f_end = f_start + tail_frames
            _run_one(f_start, f_end, tail_frames * rollout_steps, material, confidence)
    finally:
        engine.enable_corrector(was_active)

    # Frame 0 is the un-rolled initial state; color it like the first rolled
    # frame rather than flashing the pre-Refiner material_guess.
    if T > 1:
        log_E_seq[0] = log_E_seq[1]

    # Training clips stop at the adaptation boundary; nothing past it was rolled.
    if first_half_only:
        pred_pts = pred_pts[: half_window * k + 1]
        log_E_seq = log_E_seq[: half_window * k + 1]
    return (
        pred_pts.detach().cpu().numpy(),
        log_E_seq.detach().cpu().numpy(),
        mid_frame,
    )


def _render_gvc_epoch_video(
    *,
    engine, refiner, corrector,
    ep: Dict,
    cfg,
    kinematic_ids: Optional[torch.Tensor],
    n_full_windows: int,
    tail_frames: int,
    k: int,
    rollout_steps: int,
    total_outer_steps: int,
    viz_dir: Path,
    epoch_num: int,
    cfg_train_loss: float,
    log_prefix: str = "",
    ep_label: Optional[str] = None,
    kind: str = "train",
    pid_chunk_fn=None,
    first_half_only: bool = False,
) -> None:
    """No-grad full-episode rollout + render to mp4 under viz_dir.

    ``viz_dir`` is the *target directory* (e.g. ``videos/train/epoch_0020``);
    the file name inside is just ``<ep_label>.mp4`` since the dir already
    encodes the epoch and split.
    """
    import time as _time
    vt0 = _time.perf_counter()
    pred_pts, log_E_seq, mid_frame = _full_episode_eval_rollout(
        engine=engine, refiner=refiner, corrector=corrector, ep=ep, cfg=cfg,
        kinematic_ids=kinematic_ids,
        n_full_windows=n_full_windows, tail_frames=tail_frames,
        k=k, rollout_steps=rollout_steps, total_outer_steps=total_outer_steps,
        pid_chunk_fn=pid_chunk_fn,
        first_half_only=first_half_only,
    )
    if pred_pts.shape[0] < 2:
        print(f"{log_prefix}epoch {epoch_num:03d} {ep_label}: fewer than 2 frames "
              f"to render — skipping", flush=True)
        return
    gt_pts = ep["coords"].detach().cpu().numpy()[: pred_pts.shape[0]]

    max_ctrl = int(cfg.train.get("viz_max_controls", 16))
    ctrl_t = ep.get("r_coords")
    ctrl = (
        ctrl_t[: pred_pts.shape[0], :max_ctrl].detach().cpu().numpy()
        if torch.is_tensor(ctrl_t) and ctrl_t.shape[1] > 0
        else np.empty((pred_pts.shape[0], 0, 3), dtype=pred_pts.dtype)
    )
    viz_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{ep_label}.mp4" if ep_label else f"gvc_epoch_{epoch_num:04d}.mp4"
    out_path = viz_dir / filename
    fps = int(cfg.train.get("viz_fps", 30))
    title_ep = f"  ep={ep_label}" if ep_label else ""
    _render_gvc_video(
        out_path=out_path,
        pred_pts=pred_pts,
        gt_pts=gt_pts,
        log_E=log_E_seq,
        ctrl=ctrl,
        title=f"RfD {kind}  epoch {epoch_num:03d}{title_ep}  loss={cfg_train_loss:.4f}",
        fps=fps,
        mid_frame=mid_frame,
    )
    print(
        f"{log_prefix}epoch {epoch_num:03d}{title_ep} wrote {out_path} "
        f"({pred_pts.shape[0]} frames, {_time.perf_counter() - vt0:.1f}s)",
        flush=True,
    )


def _render_gvc_video(
    out_path: Path,
    pred_pts: np.ndarray,        # (T, N, 3)
    gt_pts: np.ndarray,          # (T, N, 3)
    log_E: np.ndarray,           # (T, N) per-frame estimate, or (N,) constant
    ctrl: np.ndarray,            # (T, K, 3) — controller points (already truncated to max_ctrl)
    title: str,
    fps: int = 30,
    dpi: int = 110,
    point_size: int = 6,
    mid_frame: int = 0,
) -> None:
    """Side-by-side top-down + iso video. Predicted points colored by current
    per-particle log_E; GT in light gray; controllers in orange."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=5.0, vmax=11.0)
    # Fixed color scale, so a per-frame recolor tracks the estimate and not the range.
    log_E_seq = log_E if log_E.ndim == 2 else np.broadcast_to(log_E, (pred_pts.shape[0], log_E.shape[0]))
    colors = cm.viridis(norm(log_E_seq[0]))

    combined = np.concatenate([pred_pts.reshape(-1, 3), gt_pts.reshape(-1, 3)], axis=0)
    if ctrl.size:
        combined = np.concatenate([combined, ctrl.reshape(-1, 3)], axis=0)
    mn, mx = combined.min(0), combined.max(0)
    pad = 0.02
    xlim = (mn[0] - pad, mx[0] + pad)
    ylim = (mn[1] - pad, mx[1] + pad)
    zlim = (mn[2] - pad, mx[2] + pad)

    fig = plt.figure(figsize=(13, 6), dpi=dpi)
    ax_top = fig.add_subplot(1, 2, 1)
    sc_top_gt = ax_top.scatter(gt_pts[0, :, 0], gt_pts[0, :, 1], s=point_size, c="lightgray", alpha=0.45)
    sc_top = ax_top.scatter(pred_pts[0, :, 0], pred_pts[0, :, 1], s=point_size, c=colors)
    ax_top.set_xlim(*xlim); ax_top.set_ylim(*ylim)
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.set_xlabel("x"); ax_top.set_ylabel("y")
    ax_top.set_title("top-down (xy) — pred (color) + GT (gray)")

    ax_iso = fig.add_subplot(1, 2, 2, projection="3d")
    sc_iso_gt = ax_iso.scatter(
        gt_pts[0, :, 0], gt_pts[0, :, 1], gt_pts[0, :, 2],
        s=max(point_size // 2, 2), c="lightgray", alpha=0.35, depthshade=False,
    )
    sc_iso = ax_iso.scatter(
        pred_pts[0, :, 0], pred_pts[0, :, 1], pred_pts[0, :, 2],
        s=point_size, c=colors, depthshade=False,
    )
    ctrl_n = int(ctrl.shape[1]) if ctrl.ndim == 3 else 0
    sc_iso_ctrl = ax_iso.scatter(
        ctrl[0, :, 0] if ctrl_n else [],
        ctrl[0, :, 1] if ctrl_n else [],
        ctrl[0, :, 2] if ctrl_n else [],
        s=point_size * 6, c="#ff9800", edgecolors="black", linewidths=0.6, depthshade=False,
    )
    ax_iso.set_xlim(*xlim); ax_iso.set_ylim(*ylim); ax_iso.set_zlim(*zlim)
    ax_iso.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    ax_iso.view_init(elev=25, azim=-60)
    ax_iso.set_xlabel("x"); ax_iso.set_ylabel("y"); ax_iso.set_zlabel("z")
    ax_iso.set_title("iso (xyz) — pred + GT (gray) + controls (orange)")

    sm = cm.ScalarMappable(norm=norm, cmap="viridis"); sm.set_array([])
    fig.colorbar(sm, ax=[ax_top, ax_iso], shrink=0.7, pad=0.08, location="right").set_label(
        "log_E (current per-particle estimate)"
    )
    suptitle = fig.suptitle("", y=0.99)
    # Red phase banner: frames before mid_frame are the PID / material-ID half.
    phase_text = fig.text(0.5, 0.945, "", color="red", fontsize=15,
                          fontweight="bold", ha="center", va="top")

    T = int(pred_pts.shape[0])
    writer = imageio.get_writer(
        str(out_path),
        format="FFMPEG",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    try:
        for f in range(T):
            sc_top.set_offsets(np.stack([pred_pts[f, :, 0], pred_pts[f, :, 1]], axis=-1))
            sc_top_gt.set_offsets(np.stack([gt_pts[f, :, 0], gt_pts[f, :, 1]], axis=-1))
            sc_iso._offsets3d = (pred_pts[f, :, 0], pred_pts[f, :, 1], pred_pts[f, :, 2])
            sc_iso_gt._offsets3d = (gt_pts[f, :, 0], gt_pts[f, :, 1], gt_pts[f, :, 2])
            frame_colors = cm.viridis(norm(log_E_seq[f]))
            sc_top.set_color(frame_colors)
            sc_iso.set_color(frame_colors)
            if ctrl_n:
                sc_iso_ctrl._offsets3d = (ctrl[f, :, 0], ctrl[f, :, 1], ctrl[f, :, 2])
            suptitle.set_text(f"{title}  frame {f + 1}/{T}")
            if int(mid_frame) > 0:  # only the two-phase (validation) rollout has halves
                # Frame 0 is the initial state and phase 1 advances the state
                # through frame mid_frame, so the PID half is frames [0, mid_frame];
                # phase 2's first predicted frame is mid_frame + 1.
                phase_text.set_text("Material Identification (no RfD)"
                                    if f <= int(mid_frame) else "RfD Validation")
            fig.canvas.draw()
            img = np.asarray(fig.canvas.renderer.buffer_rgba())[..., :3]
            writer.append_data(img)
    finally:
        writer.close()
        plt.close(fig)


# --- Loss-curve plotting ---


def _save_loss_plots(history: list, plots_dir: Path) -> None:
    """Write loss + F-stability PNGs and ``history.json`` under ``plots_dir``.

    Train curves are solid no-marker; val entries (``val_loss/chamfer/l2``,
    when present) overlay as dashed + dot markers in the matching color.
    """
    if not history:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    # Rows written by a standalone validate_RfD.py sweep carry only `epoch` +
    # `val_*`; the train series must skip them.
    train_rows = [h for h in history if "loss" in h]
    epochs = [h["epoch"] for h in train_rows]
    metrics = {
        "total_loss": ("loss",    "val_loss",    [h["loss"]    for h in train_rows]),
        "chamfer":    ("chamfer", "val_chamfer", [h["chamfer"] for h in train_rows]),
        "tracked_l2": ("l2",      "val_l2",      [h["l2"]      for h in train_rows]),
    }

    def _val_series(val_key):
        pts = [(h["epoch"], h[val_key]) for h in history if val_key in h]
        if not pts:
            return None, None
        xs, ys = zip(*pts)
        return list(xs), list(ys)

    # Combined plot
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    for name, (_, val_key, values) in metrics.items():
        line, = ax.plot(epochs, values, linewidth=1.5, label=f"train {name}")
        xs, ys = _val_series(val_key)
        if xs is not None:
            ax.plot(
                xs, ys,
                linestyle="--", marker="o", markersize=4, linewidth=1.5,
                color=line.get_color(), label=f"val {name}",
            )
    ax.set(xlabel="epoch", ylabel="loss", title="RfD training loss")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.4); ax.legend()
    fig.tight_layout(); fig.savefig(plots_dir / "loss_curves.png"); plt.close(fig)

    # Per-metric plots
    for name, (_, val_key, values) in metrics.items():
        fig, ax = plt.subplots(figsize=(7, 4.0), dpi=120)
        ax.plot(epochs, values, linewidth=1.5, color="C0", label=f"train {name}")
        xs, ys = _val_series(val_key)
        if xs is not None:
            ax.plot(
                xs, ys,
                linestyle="--", marker="o", markersize=4, linewidth=1.5,
                color="C3", label=f"val {name}",
            )
            ax.legend()
        ax.set(xlabel="epoch", ylabel=name, title=f"RfD training — {name}")
        ax.grid(True, linestyle=":", alpha=0.4)
        fig.tight_layout(); fig.savefig(plots_dir / f"loss_{name}.png"); plt.close(fig)

    if train_rows and all("stability" in h for h in train_rows):
        S = [h["stability"] for h in train_rows]
        fig, axes = plt.subplots(3, 1, figsize=(8, 9), dpi=120, sharex=True)
        for k_, lbl in (("det_F_min", "det(F) min"), ("det_F_max", "det(F) max")):
            axes[0].plot(epochs, [s[k_] for s in S], linewidth=1.2, label=lbl)
        axes[0].axhline(0.0, linestyle="--", color="red", alpha=0.4, label="0 (singular)")
        axes[0].set_ylabel("det(F)"); axes[0].grid(True, linestyle=":", alpha=0.4); axes[0].legend(fontsize=8)
        for k_, lbl in (("sigma_min_F", "σ_min(F)"), ("sigma_max_F", "σ_max(F)"), ("cond_F_max", "cond(F) max")):
            axes[1].plot(epochs, [s[k_] for s in S], linewidth=1.2, label=lbl)
        axes[1].set_ylabel("σ / cond"); axes[1].set_yscale("log")
        axes[1].grid(True, which="both", linestyle=":", alpha=0.4); axes[1].legend(fontsize=8)
        for k_, lbl in (("grad_norm_max", "grad_norm max"), ("grad_norm_mean", "grad_norm mean")):
            axes[2].plot(epochs, [s[k_] for s in S], linewidth=1.2, label=lbl)
        axes[2].set(xlabel="epoch", ylabel="grad norm (pre-clip)"); axes[2].set_yscale("log")
        axes[2].grid(True, which="both", linestyle=":", alpha=0.4); axes[2].legend(fontsize=8)
        fig.suptitle("RfD training — F-stability monitor")
        fig.tight_layout(); fig.savefig(plots_dir / "stability_curves.png"); plt.close(fig)

    (plots_dir / "history.json").write_text(json.dumps(history, indent=2))
