"""
Render a rollout saved by optimize_fixed_material.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from physcore.particle_flow import episode_runtime as ert
from physcore.fixed_material.optimize_fixed_material import _load_first_episode  # noqa: E402
from physcore.sim.visualizer import (  # noqa: E402
    _write_video_frames,
    render_frame,
    scalar_to_heatmap_colors,
)
from physcore.particle_flow.rollout import (  # noqa: E402
    BatchedDifferentiableRolloutEngine,
    DifferentiableRolloutEngine,
)
from physcore.particle_flow.rollout_runtime import (  # noqa: E402
    real_world_chunk_dt as _real_world_chunk_dt,
    temporary_rollout_timestep as _temporary_rollout_timestep,
)


def _tracking_l2_at_frame(predicted_pos: torch.Tensor, episode: Dict, frame_idx: int) -> Optional[float]:
    """Mean Euclidean distance over cotracker-observed correspondences only.

    For each valid (view, obs-point) at `frame_idx`, gather the predicted
    particle at `object_particle_ids[v, frame_idx, p]` and take Euclidean
    distance to the observed point `object_points_clean[v, frame_idx, p]`.
    Average over all valid points. Returns None when no observation is available
    (synthetic episodes without observation_data, or no valid points at frame).
    """
    od = episode.get("observation_data")
    if not od:
        return None
    pts = od.get("object_points_clean")
    vmask = od.get("object_valid_mask")
    pids = od.get("object_particle_ids")
    if pts is None or vmask is None or pids is None:
        return None
    N = int(predicted_pos.shape[0])
    diffs = []
    for v in range(int(pts.shape[0])):
        ids = pids[v, frame_idx].long()
        valid = vmask[v, frame_idx].bool() & (ids >= 0) & (ids < N)
        if not bool(valid.any()):
            continue
        gathered = predicted_pos.index_select(0, ids[valid].to(predicted_pos.device))
        obs_pts = pts[v, frame_idx][valid].to(predicted_pos.device)
        diffs.append((gathered - obs_pts).norm(dim=-1))
    if not diffs:
        return None
    return float(torch.cat(diffs, dim=0).mean().item())


def _overlay_text(frame: np.ndarray, text: str, *, x: int = 8, y: int = 8) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    pad = 5
    bbox = draw.textbbox((x, y), text, font=font)
    rect = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    draw.rectangle(rect, fill=(245, 245, 245))
    draw.rectangle(rect, outline=(35, 35, 35), width=1)
    draw.text((x, y), text, fill=(20, 20, 20), font=font)
    return np.asarray(image)


def _load_material(path: Path, device: torch.device, num_particles: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "decoded_material" not in payload:
        raise KeyError(f"{path} does not contain decoded_material")
    material = payload["decoded_material"].float()
    if material.ndim == 1:
        material = material.view(1, 2).expand(num_particles, 2).clone()
    if material.shape != (num_particles, 2):
        raise ValueError(f"Material shape {tuple(material.shape)} does not match {(num_particles, 2)}")
    return material.to(device)


def _rollout_fixed_material(
    *,
    cfg,
    episode: Dict[str, object],
    rollout_engine,
    material: torch.Tensor,
    start_frame: int,
    end_frame: int,
    rollout_steps: int,
    correction_steps_count: int,
    correction_stop_frame: int,
) -> Dict[str, object]:
    coords = episode["coords"]
    flows = episode["flows"]
    particle_v = episode["particle_v"]
    particle_F = episode["particle_F"]
    particle_C = episode["particle_C"]
    r_coords = episode["r_coords"]
    vis_idx = episode["vis_idx"]
    rigid_collision_cfg = episode["rigid_collision_cfg"]
    rigid_body_primitives = episode["rigid_body_primitives"]
    particle_material_models = episode.get("particle_material_models", None)
    manipulation_flag = episode.get("manipulation_flag", coords.new_zeros(()))
    manipulation_contact_particle_ids = episode.get("manipulation_contact_particle_ids", None)
    controller_grid_points = episode.get("controller_grid_points", None)
    is_real_world_episode = bool(episode.get("is_real_world", False))

    use_contact_kinematic_targets = ert._should_use_contact_kinematic_targets(
        is_real_world_episode=is_real_world_episode,
        manipulation_contact_particle_ids=manipulation_contact_particle_ids,
        controller_grid_points=controller_grid_points,
    )
    kinematic_contact_particle_ids = (
        manipulation_contact_particle_ids if use_contact_kinematic_targets else None
    )

    total_frames, num_particles, _ = coords.shape
    predicted = coords.detach().cpu().numpy().copy()
    corrected_flags = np.zeros(total_frames, dtype=np.bool_)
    tracking_l2 = []
    obs_to_pred = []

    current_frame = int(start_frame)
    current_positions = coords[current_frame].contiguous()
    current_velocities = particle_v[current_frame].contiguous()
    current_F = particle_F[current_frame].contiguous()
    current_C = None if particle_C is None else particle_C[current_frame].contiguous()
    material_state = material.view(1, num_particles, 2)

    with torch.no_grad():
        _, _, initial_real_terms = ert._compute_loss(
            current_positions.view(1, num_particles, 3),
            torch.zeros_like(current_positions).view(1, num_particles, 3),
            coords[current_frame : current_frame + 1],
            flows[current_frame : current_frame + 1],
            float(cfg.loss.get("position_weight", 1.0)),
            float(cfg.loss.get("flow_weight", 0.5)),
            episode=episode,
            start_frame=current_frame,
            target_frame=current_frame,
            loss_config=cfg.loss,
            return_terms=True,
        )
        print(
            f"  frame {current_frame:04d}/{end_frame} mode=init "
            f"obs_to_pred={float(initial_real_terms['real_obs_to_pred_loss'].detach().item()):.6g}",
            flush=True,
        )
        while current_frame < end_frame:
            chunk_steps = int(rollout_steps)
            next_chunk_frame = current_frame + 1
            run_correction = current_frame < int(correction_stop_frame)
            next_chunk_observed, next_chunk_vis = ert._get_correction_observation(
                episode,
                coords,
                vis_idx,
                next_chunk_frame,
                num_particles,
            )
            rigid_points_window = ert._catmull_rom_window(
                r_coords,
                current_frame,
                next_chunk_frame,
                chunk_steps,
            )
            controller_grid_points_window = ert._catmull_rom_window(
                controller_grid_points,
                current_frame,
                next_chunk_frame,
                chunk_steps,
            )
            manipulation_contact_particle_ids_window = ert._hold_index_window(
                kinematic_contact_particle_ids,
                current_frame,
                chunk_steps,
            )
            real_chunk_dt = _real_world_chunk_dt(episode, rollout_engine, chunk_steps)
            real_chunk_ground_height = float(episode.get("ground_height", rollout_engine.ground_height))
            with _temporary_rollout_timestep(
                rollout_engine,
                dt=real_chunk_dt,
                ground_height=real_chunk_ground_height,
            ):
                rollout_outputs = ert._run_chunk_rollout_and_correction(
                    rollout_engine,
                    current_positions,
                    current_velocities,
                    current_F,
                    current_C,
                    material_state,
                    chunk_steps,
                    current_frame,
                    next_chunk_frame,
                    next_chunk_observed,
                    next_chunk_vis,
                    coords,
                    r_coords,
                    particle_material_models,
                    rigid_collision_cfg,
                    rigid_body_primitives,
                    correction_steps_count,
                    run_correction,
                    manipulation_flag=manipulation_flag,
                    manipulation_contact_particle_ids=kinematic_contact_particle_ids,
                    controller_grid_points=controller_grid_points,
                    rigid_points_window=rigid_points_window,
                    manipulation_contact_particle_ids_window=manipulation_contact_particle_ids_window,
                    controller_grid_points_window=controller_grid_points_window,
                )

            terminal = rollout_outputs["correction_result"]["predicted_positions"][-1].detach()
            predicted[next_chunk_frame] = terminal.cpu().numpy()
            corrected_flags[next_chunk_frame] = bool(run_correction)

            gt_pos = coords[next_chunk_frame : next_chunk_frame + 1]
            gt_flow = flows[next_chunk_frame : next_chunk_frame + 1]
            _, _, real_terms = ert._compute_loss(
                terminal.view(1, num_particles, 3),
                rollout_outputs["chunk_flows"],
                gt_pos,
                gt_flow,
                float(cfg.loss.get("position_weight", 1.0)),
                float(cfg.loss.get("flow_weight", 0.5)),
                episode=episode,
                start_frame=current_frame,
                target_frame=next_chunk_frame,
                loss_config=cfg.loss,
                return_terms=True,
            )
            tl = _tracking_l2_at_frame(terminal, episode, next_chunk_frame)
            if tl is not None:
                tracking_l2.append(tl)
            obs_to_pred.append(float(real_terms["real_obs_to_pred_loss"].detach().item()))

            pure_mpm_result = rollout_outputs["pure_mpm_result"]
            current_positions = terminal.contiguous()
            current_velocities = pure_mpm_result["final_velocity"].detach().contiguous()
            current_F = rollout_outputs["correction_result"]["final_deformation_gradient"].detach().contiguous()
            final_C = rollout_outputs["correction_result"].get("final_C", pure_mpm_result.get("final_C", None))
            current_C = None if final_C is None else final_C.detach().contiguous()
            current_frame = next_chunk_frame

            if current_frame % 25 == 0 or current_frame >= end_frame:
                print(
                    f"  frame {current_frame:04d}/{end_frame} "
                    f"mode={'corr' if run_correction else 'pure'} "
                    f"obs_to_pred={np.mean(obs_to_pred):.6g}",
                    flush=True,
                )

    return {
        "predicted": predicted[: end_frame + 1],
        "ground_truth": coords[: end_frame + 1].detach().cpu().numpy(),
        "rigid": r_coords[: end_frame + 1].detach().cpu().numpy(),
        "corrected_flags": corrected_flags[: end_frame + 1],
        "tracking_l2": float(np.mean(tracking_l2)) if tracking_l2 else float("nan"),
        "obs_to_pred": float(np.mean(obs_to_pred)) if obs_to_pred else float("nan"),
        "final_velocity": current_velocities.detach().cpu(),
        "final_deformation_gradient": current_F.detach().cpu(),
        "final_C": None if current_C is None else current_C.detach().cpu(),
    }


def _render_comparison(
    *,
    rollout: Dict[str, object],
    material_log_e: np.ndarray,
    output_path: Path,
    fps: int,
    width: int,
    height: int,
    frame_stride: int,
    azimuth: float,
    elevation: float,
    color_min: float,
    color_max: float,
    title: str,
) -> None:
    predicted = rollout["predicted"]
    gt = rollout["ground_truth"]
    rigid = rollout["rigid"]
    colors = scalar_to_heatmap_colors(material_log_e, vmin=color_min, vmax=color_max)
    separator = np.full((height, 4, 3), 210, dtype=np.uint8)
    frames = []
    frame_indices = np.arange(0, predicted.shape[0], max(int(frame_stride), 1), dtype=np.int64)
    if frame_indices[-1] != predicted.shape[0] - 1:
        frame_indices = np.concatenate([frame_indices, np.array([predicted.shape[0] - 1], dtype=np.int64)])
    for out_idx, frame_idx in enumerate(frame_indices):
        pred_frame = render_frame(
            predicted[frame_idx],
            rigid[frame_idx],
            particle_colors=colors,
            width=width,
            height=height,
            azimuth=azimuth,
            elevation=elevation,
        )
        gt_frame = render_frame(
            gt[frame_idx],
            rigid[frame_idx],
            particle_colors=colors,
            width=width,
            height=height,
            azimuth=azimuth,
            elevation=elevation,
        )
        mode = "corr" if rollout["corrected_flags"][frame_idx] else "pure"
        pred_frame = _overlay_text(pred_frame, f"{title} pred frame={frame_idx} {mode}")
        gt_frame = _overlay_text(gt_frame, "observed")
        frames.append(np.concatenate([pred_frame, separator, gt_frame], axis=1))
        if (out_idx + 1) % 50 == 0 or out_idx + 1 == len(frame_indices):
            print(f"  rendered {out_idx + 1}/{len(frame_indices)} frames for {output_path.name}", flush=True)
    _write_video_frames(np.stack(frames), str(output_path), fps=fps)


def _save_material_histogram(material: torch.Tensor, output_path: Path, log_e_min: float, log_e_max: float) -> None:
    log_e = material[:, 0].detach().cpu().numpy()
    nu = material[:, 1].detach().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].hist(log_e, bins=64, color="tab:blue", alpha=0.85)
    axes[0].axvline(log_e_min, color="black", linestyle="--", linewidth=1)
    axes[0].axvline(log_e_max, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("optimized log_E")
    axes[0].set_xlabel("log_E")
    axes[0].set_ylabel("particles")
    axes[1].hist(nu, bins=64, color="tab:green", alpha=0.85)
    axes[1].set_title("optimized nu")
    axes[1].set_xlabel("nu")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--material-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=360)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--azimuth", type=float, default=45.0)
    parser.add_argument("--elevation", type=float, default=25.0)
    parser.add_argument("--pure-tail-frac", type=float, default=0.5)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode = _load_first_episode(cfg, device)
    coords = episode["coords"]
    total_frames, num_particles, _ = coords.shape
    history_steps = int(cfg.dataset.get("history_steps", 1))
    start_frame = max(history_steps - 1, 0) if args.start_frame is None else int(args.start_frame)
    end_frame = total_frames - 1 if args.end_frame is None else min(int(args.end_frame), total_frames - 1)
    material = _load_material(Path(args.material_checkpoint), device, num_particles)
    rollout_engine_cls = (
        BatchedDifferentiableRolloutEngine
        if bool(cfg.rollout.get("use_batched_mpm", True))
        else DifferentiableRolloutEngine
    )
    rollout_engine = rollout_engine_cls(cfg.rollout, device=str(device))
    rollout_steps = max(int(cfg.dataset.get("rollout_steps", 25)), 1)
    correction_steps_count = int(cfg.model.get("correction_rollout_steps", cfg.model.get("correction_steps", 4)))

    log_e_min = float(cfg.model.get("material_log_E_min", 8.0))
    log_e_max = float(cfg.model.get("material_log_E_max", 12.0))
    material_log_e = material[:, 0].detach().cpu().numpy()
    print(
        "Material stats: "
        f"log_E min={material_log_e.min():.6f} max={material_log_e.max():.6f} "
        f"mean={material_log_e.mean():.6f} std={material_log_e.std():.6f}",
        flush=True,
    )
    _save_material_histogram(material, output_dir / "material_histogram.png", log_e_min, log_e_max)

    print("Running correction-assisted rollout for visualization", flush=True)
    corrected_rollout = _rollout_fixed_material(
        cfg=cfg,
        episode=episode,
        rollout_engine=rollout_engine,
        material=material,
        start_frame=start_frame,
        end_frame=end_frame,
        rollout_steps=rollout_steps,
        correction_steps_count=correction_steps_count,
        correction_stop_frame=end_frame,
    )
    correction_stop_frame = start_frame + int(round((end_frame - start_frame) * (1.0 - float(args.pure_tail_frac))))
    print(
        f"Running validation-style rollout: correction until frame {correction_stop_frame}, then pure MPM",
        flush=True,
    )
    validation_rollout = _rollout_fixed_material(
        cfg=cfg,
        episode=episode,
        rollout_engine=rollout_engine,
        material=material,
        start_frame=start_frame,
        end_frame=end_frame,
        rollout_steps=rollout_steps,
        correction_steps_count=correction_steps_count,
        correction_stop_frame=correction_stop_frame,
    )

    torch.save(
        {
            "corrected": corrected_rollout,
            "validation": validation_rollout,
            "material": material.detach().cpu(),
            "material_checkpoint": str(args.material_checkpoint),
        },
        output_dir / "fixed_material_rollout_payload.pt",
    )
    metrics = {
        "corrected_tracking_l2": corrected_rollout["tracking_l2"],
        "corrected_obs_to_pred": corrected_rollout["obs_to_pred"],
        "validation_tracking_l2": validation_rollout["tracking_l2"],
        "validation_obs_to_pred": validation_rollout["obs_to_pred"],
        "validation_correction_stop_frame": int(correction_stop_frame),
        "min_log_E": float(material_log_e.min()),
        "max_log_E": float(material_log_e.max()),
        "mean_log_E": float(material_log_e.mean()),
        "std_log_E": float(material_log_e.std()),
    }
    (output_dir / "render_metrics.json").write_text(
        __import__("json").dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    _render_comparison(
        rollout=corrected_rollout,
        material_log_e=material_log_e,
        output_path=output_dir / "fixed_material_corrected_vs_observed.mp4",
        fps=int(args.fps),
        width=int(args.width),
        height=int(args.height),
        frame_stride=int(args.frame_stride),
        azimuth=float(args.azimuth),
        elevation=float(args.elevation),
        color_min=log_e_min,
        color_max=log_e_max,
        title="corrected",
    )
    _render_comparison(
        rollout=validation_rollout,
        material_log_e=material_log_e,
        output_path=output_dir / "fixed_material_halfcorr_puretail_vs_observed.mp4",
        fps=int(args.fps),
        width=int(args.width),
        height=int(args.height),
        frame_stride=int(args.frame_stride),
        azimuth=float(args.azimuth),
        elevation=float(args.elevation),
        color_min=log_e_min,
        color_max=log_e_max,
        title="half-corr/pure-tail",
    )
    print(f"Wrote visualization outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
