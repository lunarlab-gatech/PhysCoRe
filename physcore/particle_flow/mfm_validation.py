"""
Validation helpers shared by the MfM entrypoints.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from physcore.fixed_material.render_fixed_material_rollout import _rollout_fixed_material
from physcore.model_MfM import Refiner
from physcore.particle_flow import episode_runtime as ert
from physcore.particle_flow.mfm_training import material_guess, observed_features


def correction_stop_frame_for_mode(
    end_frame: int,
    correction_mode: str,
    correction_stop_frame: Optional[int] = None,
) -> int:
    if correction_stop_frame is not None:
        return int(correction_stop_frame)
    if correction_mode == "none":
        return 0
    if correction_mode == "half":
        return int(end_frame) // 2
    if correction_mode == "throughout":
        return int(end_frame)
    raise ValueError(f"unknown correction_mode: {correction_mode}")


def correction_steps_for_mode(cfg, correction_mode: str) -> int:
    if correction_mode == "none":
        return 0
    return int(cfg.model.get("correction_rollout_steps", cfg.model.get("correction_steps", 4)))


def midpoint_frame(n_frames: int, update_every: int) -> int:
    """The episode midpoint, floored to a Refiner update boundary.

    Same formula the `pure_tail` branch of the validation rollout uses, so a material read out here
    is the one validate_MfM.py freezes at the midpoint.
    """
    K = max(int(update_every), 1)
    end_frame = ((int(n_frames) - 1) // K) * K
    return (end_frame // 2 // K) * K


def material_from_final_chunk(
    model: Refiner,
    ep: Dict[str, object],
    cfg,
    seed: int,
    gate: str = "none",
    gate_temperature: float = 1.0,
    gate_min: float = 0.0,
    gate_max: float = 1.0,
    end_frame: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Per-particle material read out after streaming the episode to `end_frame`.

    `end_frame=None` (the default) runs to the last frame. Pass `midpoint_frame(...)` to stop at the
    midpoint instead -- the pure-tail estimate -- without touching the episode tensors: frames past
    `end_frame` are simply never indexed.
    """
    coords, vel, Fm = ep["coords"], ep["particle_v"], ep["particle_F"]
    start = 0
    end = max(int(coords.shape[0]) - 1, 1) if end_frame is None else max(int(end_frame), 1)
    canonical = coords[start : start + 1]
    cache = model.cache(canonical)
    state: Optional[object] = None
    material = material_guess(ep, cfg, start, seed)
    correction = torch.zeros_like(canonical)
    mask = torch.zeros(canonical.shape[:2], device=coords.device, dtype=torch.bool)
    last = material
    last_conf = material.new_ones(*material.shape[:2], 1)
    alpha_sum = 0.0
    alpha_sq_sum = 0.0
    alpha_count = 0
    initialized = False
    feature_window = []
    for frame in range(start + 1, end + 1):
        cur, prev = coords[frame : frame + 1], coords[frame - 1 : frame]
        obs = observed_features(
            ep,
            frame,
            canonical[0],
            cur[0],
            int(cfg.model.get("max_controls", 0)),
            bool(cfg.model.get("fixed_tracked_mask", False)),
            str(cfg.model.get("tracked_disp_mode", "incremental")),
            str(cfg.model.get("control_disp_mode", "absolute")),
            bool(cfg.model.get("use_persistent_tracks", True)),
            bool(cfg.model.get("persistent_track_use_motion_valid", False)),
            aggregate_controls=True,
        )
        feature_window.append(
            model.build_features(
                cur,
                prev,
                canonical,
                material,
                vel[frame : frame + 1],
                Fm[frame : frame + 1],
                correction,
                mask,
                obs,
            )
        )
        if (frame - start) % int(cfg.train.update_every):
            continue
        x_window = torch.stack(feature_window, dim=2)
        pred = model.forward_features(x_window, cur, prev, cache, state)
        feature_window.clear()
        state = pred["state"]
        predicted_material = pred["material"].detach()
        last_conf = pred.get("material_confidence", last_conf).detach()
        if gate == "none" or not initialized:
            material = predicted_material
            alpha = material.new_ones(*material.shape[:2], 1)
        elif gate in {"confidence", "bounded_confidence"}:
            raw_conf = (last_conf - 1.0).clamp_min(1.0e-8).log()
            alpha = torch.sigmoid(raw_conf / max(float(gate_temperature), 1.0e-8))
            if gate == "bounded_confidence":
                alpha = float(gate_min) + (float(gate_max) - float(gate_min)) * alpha
            else:
                alpha = alpha.clamp(float(gate_min), float(gate_max))
            material = material + alpha * (predicted_material - material)
        else:
            raise ValueError(f"unknown material gate: {gate}")
        alpha_sum += float(alpha.mean().item())
        alpha_sq_sum += float(alpha.square().mean().item())
        alpha_count += 1
        initialized = True
        last = material
    gate_stats = {
        "gate_alpha_mean": alpha_sum / max(alpha_count, 1),
        "gate_alpha_rms": (alpha_sq_sum / max(alpha_count, 1)) ** 0.5,
        "gate_updates": float(alpha_count),
    }
    return last.squeeze(0).detach(), last_conf.squeeze(0).detach(), gate_stats


def contact_ids_for_rollout(ep: Dict[str, object], controller_grid_points: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    ids = ep.get("manipulation_contact_particle_ids", None)
    if getattr(ert, "_should_use_contact_kinematic_targets")(
        is_real_world_episode=bool(ep.get("is_real_world", False)),
        manipulation_contact_particle_ids=ids,
        controller_grid_points=controller_grid_points,
    ):
        return ids
    return None


def rollout_windows(ep: Dict[str, object], frame: int, next_frame: int, chunk_steps: int):
    r = ep["r_coords"]
    controller = ep.get("controller_grid_points", None)
    contact_ids = contact_ids_for_rollout(ep, controller)
    if bool(ep.get("is_real_world", False)):
        rigid = ert._catmull_rom_window(r, frame, next_frame, chunk_steps)
        control = ert._catmull_rom_window(controller, frame, next_frame, chunk_steps)
        contact = ert._hold_index_window(contact_ids, frame, chunk_steps)
    else:
        rigid = r[frame : next_frame + 1]
        control = None if controller is None else controller[frame : next_frame + 1]
        contact = None if contact_ids is None else contact_ids[frame : next_frame + 1]
    return rigid, contact, control, contact_ids


def validation_rollout_loss(
    engine,
    ep: Dict[str, object],
    material: torch.Tensor,
    cfg,
    save_rollout: bool = False,
    correction_mode: str = "none",
    correction_stop_frame: Optional[int] = None,
) -> Dict[str, object]:
    end_frame = int(ep["coords"].shape[0] - 1)
    correction_stop_frame = correction_stop_frame_for_mode(end_frame, correction_mode, correction_stop_frame)
    correction_steps_count = correction_steps_for_mode(cfg, correction_mode)
    rollout = _rollout_fixed_material(
        cfg=cfg,
        episode=ep,
        rollout_engine=engine,
        material=material,
        start_frame=0,
        end_frame=end_frame,
        rollout_steps=max(int(cfg.dataset.get("rollout_steps", 25)), 1),
        correction_steps_count=correction_steps_count,
        correction_stop_frame=int(correction_stop_frame),
    )
    metrics: Dict[str, object] = {
        "avg_recon_loss": float(rollout.get("obs_to_pred", float("nan"))),
        "tracking_l2": float(rollout.get("tracking_l2", float("nan"))),
        "chunks": float(max(end_frame, 0)),
        "last_frame": float(max(end_frame, 0)),
        "correction_mode": str(correction_mode),
        "correction_stop_frame": float(int(correction_stop_frame)),
        "correction_steps_count": float(int(correction_steps_count)),
    }
    if save_rollout:
        metrics["rollout"] = {
            "predicted": torch.as_tensor(rollout["predicted"]).detach().cpu(),
            "ground_truth": torch.as_tensor(rollout["ground_truth"]).detach().cpu(),
            "rigid": torch.as_tensor(rollout["rigid"]).detach().cpu(),
            "corrected_flags": torch.as_tensor(rollout["corrected_flags"]).detach().cpu(),
            "tracking_l2": metrics["tracking_l2"],
            "obs_to_pred": metrics["avg_recon_loss"],
            "final_velocity": torch.as_tensor(rollout["final_velocity"]).detach().cpu(),
            "final_deformation_gradient": torch.as_tensor(rollout["final_deformation_gradient"]).detach().cpu(),
            "final_C": None if rollout.get("final_C") is None else torch.as_tensor(rollout["final_C"]).detach().cpu(),
        }
    return metrics


def _run_chunk_with_correction(
    engine,
    ep,
    coords,
    r,
    cur_pos,
    cur_vel,
    cur_F,
    cur_C,
    material,
    from_frame,
    to_frame,
    chunk_steps,
    correction_steps,
    run_correction,
):
    mat_for_mpm = material
    while mat_for_mpm.dim() > 3:
        mat_for_mpm = mat_for_mpm.squeeze(0)
    manipulation_contact_ids = ep.get("manipulation_contact_particle_ids", None)
    controller_pts = ep.get("controller_grid_points", None)
    use_contact = ert._should_use_contact_kinematic_targets(
        is_real_world_episode=bool(ep.get("is_real_world", False)),
        manipulation_contact_particle_ids=manipulation_contact_ids,
        controller_grid_points=controller_pts,
    )
    kinematic_contact_ids = manipulation_contact_ids if use_contact else None
    rigid_window = ert._catmull_rom_window(r, from_frame, to_frame, chunk_steps)
    controller_window = ert._catmull_rom_window(controller_pts, from_frame, to_frame, chunk_steps)
    contact_window = ert._hold_index_window(kinematic_contact_ids, from_frame, chunk_steps)
    obs_corr, vis = ert._get_correction_observation(
        ep,
        coords,
        ep["vis_idx"],
        to_frame,
        coords.shape[1],
    )
    with ert._temporary_rollout_timestep(
        engine,
        dt=ert._real_world_chunk_dt(ep, engine, chunk_steps),
        ground_height=float(ep.get("ground_height", engine.ground_height)),
    ):
        out = ert._run_chunk_rollout_and_correction(
            engine,
            cur_pos,
            cur_vel,
            cur_F,
            cur_C,
            mat_for_mpm,
            chunk_steps,
            from_frame,
            to_frame,
            obs_corr,
            vis,
            coords,
            r,
            ep.get("particle_material_models"),
            ep.get("rigid_collision_cfg"),
            ep.get("rigid_body_primitives"),
            correction_steps,
            run_correction,
            ep.get("manipulation_flag"),
            kinematic_contact_ids,
            controller_pts,
            rigid_points_window=rigid_window,
            manipulation_contact_particle_ids_window=contact_window,
            controller_grid_points_window=controller_window,
        )
    return out


def _chamfer_obs_to_pred(pred_pos: torch.Tensor, episode: Dict, frame_idx: int, chunk: int = 4096) -> torch.Tensor:
    od = episode.get("observation_data")
    if not od:
        return pred_pos.new_zeros(())
    pts = od.get("object_points_clean")
    vmask = od.get("object_valid_mask")
    if pts is None or vmask is None or frame_idx >= int(pts.shape[1]):
        return pred_pos.new_zeros(())
    flat = pts[:, frame_idx].reshape(-1, 3).to(pred_pos.device)
    mask = vmask[:, frame_idx].reshape(-1).to(pred_pos.device)
    valid = flat[mask & torch.isfinite(flat).all(dim=-1)]
    if valid.shape[0] == 0:
        return pred_pos.new_zeros(())
    total = pred_pos.new_zeros(())
    count = 0
    for i in range(0, valid.shape[0], int(chunk)):
        block = valid[i : i + int(chunk)]
        nearest = torch.cdist(block, pred_pos).min(dim=1).values
        total = total + nearest.sum()
        count += int(block.shape[0])
    return total / max(count, 1)


def _tracked_l2_at_obs(pred_pos: torch.Tensor, episode: Dict, frame_idx: int) -> torch.Tensor:
    od = episode.get("observation_data")
    if not od:
        return pred_pos.new_zeros(())
    pts = od.get("object_points_clean")
    vmask = od.get("object_valid_mask")
    pids = od.get("object_particle_ids")
    if pts is None or vmask is None or pids is None or frame_idx >= int(pts.shape[1]):
        return pred_pos.new_zeros(())
    n_particles = int(pred_pos.shape[0])
    diffs = []
    for view in range(int(pts.shape[0])):
        ids = pids[view, frame_idx].long().to(pred_pos.device)
        valid = vmask[view, frame_idx].bool().to(pred_pos.device) & (ids >= 0) & (ids < n_particles)
        if not bool(valid.any()):
            continue
        gathered = pred_pos.index_select(0, ids[valid])
        obs_pts = pts[view, frame_idx].to(pred_pos.device)[valid]
        diffs.append((gathered - obs_pts).norm(dim=-1))
    if not diffs:
        return pred_pos.new_zeros(())
    return torch.cat(diffs, dim=0).mean()


def interleaved_rollout_validation(
    model: Refiner,
    engine,
    ep: Dict[str, object],
    cfg,
    seed: int,
    pure_tail: bool = False,
    save_rollout: bool = False,
    refresh_tail: bool = False,
    correction_mode: str = "none",
    correction_stop_frame: Optional[int] = None,
) -> Dict[str, object]:
    coords, vel, Fm = ep["coords"], ep["particle_v"], ep["particle_F"]
    C = ep.get("particle_C", None)
    K = max(int(cfg.train.update_every), 1)
    chunk_steps = max(int(cfg.dataset.get("rollout_steps", 25)), 1)
    start = 0
    end_frame = int(coords.shape[0]) - 1
    end_frame = start + ((end_frame - start) // K) * K
    if end_frame <= start:
        return {"avg_recon_loss": float("nan"), "tracking_l2": float("nan")}
    mid_frame = start + ((end_frame - start) // 2 // K) * K if pure_tail else end_frame
    correction_stop_frame = correction_stop_frame_for_mode(end_frame, correction_mode, correction_stop_frame)
    if pure_tail:
        # Correction must never reach the scored half. `correction_stop_frame` is a
        # raw frame (end_frame//2 for "half") while `mid_frame` is floored to a K
        # multiple, so for odd window counts they disagree by K/2 and PID would snap
        # the first K/2 scored frames to observations. Clamp only downward, so an
        # explicit --correction-stop-frame can still stop earlier.
        correction_stop_frame = min(int(correction_stop_frame), int(mid_frame))
    correction_steps_count = correction_steps_for_mode(cfg, correction_mode)

    def should_correct(from_frame: int, to_frame: int) -> bool:
        return correction_steps_count > 0 and int(from_frame) < int(correction_stop_frame) and int(to_frame) <= int(correction_stop_frame)

    canonical = coords[start : start + 1]
    cache = model.cache(canonical)
    state: Optional[object] = None
    material = material_guess(ep, cfg, start, seed)
    correction = torch.zeros_like(canonical)
    mask = torch.zeros(canonical.shape[:2], device=coords.device, dtype=torch.bool)
    run_pos = coords[start].contiguous()
    run_vel = vel[start].contiguous()
    run_F = Fm[start].contiguous()
    run_C = None if C is None else C[start].contiguous()
    cur_frame = start
    feature_window = []
    obs_vals, trk_vals = [], []
    logE_means = []
    confidence_mean_mid = float("nan")
    pred_traj = [run_pos.detach().cpu().clone()] if save_rollout else None
    pred_frames = [start] if save_rollout else None

    for frame in range(start + 1, mid_frame + 1):
        cur_obs = coords[frame : frame + 1]
        prev_obs = coords[frame - 1 : frame]
        obs = observed_features(
            ep,
            frame,
            canonical[0],
            cur_obs[0],
            int(cfg.model.get("max_controls", 0)),
            bool(cfg.model.get("fixed_tracked_mask", False)),
            str(cfg.model.get("tracked_disp_mode", "incremental")),
            str(cfg.model.get("control_disp_mode", "absolute")),
            bool(cfg.model.get("use_persistent_tracks", True)),
            bool(cfg.model.get("persistent_track_use_motion_valid", False)),
            aggregate_controls=True,
        )
        feature_window.append(
            model.build_features(
                cur_obs,
                prev_obs,
                canonical,
                material,
                vel[frame : frame + 1],
                Fm[frame : frame + 1],
                correction,
                mask,
                obs,
            )
        )
        if (frame - start) % K:
            continue
        x_window = torch.stack(feature_window, dim=2)
        pred = model.forward_features(x_window, cur_obs, prev_obs, cache, state)
        feature_window.clear()
        state = pred["state"]
        material = pred["material"].detach()
        # Global (all-particle) mean confidence for the material standing at this
        # window boundary. In pure_tail the last one written is the midpoint value
        # that gets frozen for the scored half.
        conf = pred.get("material_confidence", None)
        if conf is not None:
            confidence_mean_mid = float(conf.detach().mean().item())
        logE_means.append(float(material[..., 0].mean().item()))
        for i in range(K):
            from_f = cur_frame + i
            to_f = from_f + 1
            out = _run_chunk_with_correction(
                engine,
                ep,
                coords,
                ep["r_coords"],
                run_pos,
                run_vel,
                run_F,
                run_C,
                material,
                from_f,
                to_f,
                chunk_steps,
                correction_steps_count,
                should_correct(from_f, to_f),
            )
            corr = out["correction_result"]
            pure = out["pure_mpm_result"]
            if should_correct(from_f, to_f):
                run_pos = corr["predicted_positions"][-1].detach()
                run_vel = corr["final_velocity"].detach()
                run_F = corr["final_deformation_gradient"].detach()
                run_C = None if corr.get("final_C") is None else corr["final_C"].detach()
                pred_pos = corr["predicted_positions"][-1]
            else:
                run_pos = pure["predicted_positions"][-1].detach()
                run_vel = pure["final_velocity"].detach()
                run_F = pure["final_deformation_gradient"].detach()
                run_C = None if pure.get("final_C") is None else pure["final_C"].detach()
                pred_pos = out["chunk_positions"][-1]
            if not pure_tail:
                obs_vals.append(float(_chamfer_obs_to_pred(pred_pos, ep, to_f).item()))
                trk_vals.append(float(_tracked_l2_at_obs(pred_pos, ep, to_f).item()))
            if save_rollout:
                pred_traj.append(pred_pos.detach().cpu().clone())
                pred_frames.append(to_f)
        cur_frame += K

    if pure_tail and cur_frame < end_frame and refresh_tail:
        for w_start in range(cur_frame, end_frame, K):
            w_end = w_start + K
            for frame in range(w_start + 1, w_end + 1):
                cur_obs = coords[frame : frame + 1]
                prev_obs = coords[frame - 1 : frame]
                obs = observed_features(
                    ep,
                    frame,
                    canonical[0],
                    cur_obs[0],
                    int(cfg.model.get("max_controls", 0)),
                    bool(cfg.model.get("fixed_tracked_mask", False)),
                    str(cfg.model.get("tracked_disp_mode", "incremental")),
                    str(cfg.model.get("control_disp_mode", "absolute")),
                    bool(cfg.model.get("use_persistent_tracks", True)),
                    bool(cfg.model.get("persistent_track_use_motion_valid", False)),
                    aggregate_controls=True,
                )
                feature_window.append(
                    model.build_features(
                        cur_obs,
                        prev_obs,
                        canonical,
                        material,
                        vel[frame : frame + 1],
                        Fm[frame : frame + 1],
                        correction,
                        mask,
                        obs,
                    )
                )
            x_window = torch.stack(feature_window, dim=2)
            pred = model.forward_features(x_window, coords[w_end : w_end + 1], coords[w_end - 1 : w_end], cache, state)
            feature_window.clear()
            state = pred["state"]
            material = pred["material"].detach()
            logE_means.append(float(material[..., 0].mean().item()))
            for frame in range(w_start + 1, w_end + 1):
                from_f = frame - 1
                to_f = frame
                out = _run_chunk_with_correction(
                    engine,
                    ep,
                    coords,
                    ep["r_coords"],
                    run_pos,
                    run_vel,
                    run_F,
                    run_C,
                    material,
                    from_f,
                    to_f,
                    chunk_steps,
                    correction_steps_count,
                    should_correct(from_f, to_f),
                )
                corr = out["correction_result"]
                pure = out["pure_mpm_result"]
                if should_correct(from_f, to_f):
                    run_pos = corr["predicted_positions"][-1].detach()
                    run_vel = corr["final_velocity"].detach()
                    run_F = corr["final_deformation_gradient"].detach()
                    run_C = None if corr.get("final_C") is None else corr["final_C"].detach()
                else:
                    run_pos = pure["predicted_positions"][-1].detach()
                    run_vel = pure["final_velocity"].detach()
                    run_F = pure["final_deformation_gradient"].detach()
                    run_C = None if pure.get("final_C") is None else pure["final_C"].detach()
                obs_vals.append(float(_chamfer_obs_to_pred(run_pos, ep, to_f).item()))
                trk_vals.append(float(_tracked_l2_at_obs(run_pos, ep, to_f).item()))
                if save_rollout:
                    pred_traj.append(run_pos.detach().cpu().clone())
                    pred_frames.append(to_f)
    elif pure_tail and cur_frame < end_frame:
        for frame in range(cur_frame + 1, end_frame + 1):
            from_f = frame - 1
            to_f = frame
            out = _run_chunk_with_correction(
                engine,
                ep,
                coords,
                ep["r_coords"],
                run_pos,
                run_vel,
                run_F,
                run_C,
                material,
                from_f,
                to_f,
                chunk_steps,
                correction_steps_count,
                should_correct(from_f, to_f),
            )
            corr = out["correction_result"]
            pure = out["pure_mpm_result"]
            if should_correct(from_f, to_f):
                run_pos = corr["predicted_positions"][-1].detach()
                run_vel = corr["final_velocity"].detach()
                run_F = corr["final_deformation_gradient"].detach()
                run_C = None if corr.get("final_C") is None else corr["final_C"].detach()
            else:
                run_pos = pure["predicted_positions"][-1].detach()
                run_vel = pure["final_velocity"].detach()
                run_F = pure["final_deformation_gradient"].detach()
                run_C = None if pure.get("final_C") is None else pure["final_C"].detach()
            obs_vals.append(float(_chamfer_obs_to_pred(run_pos, ep, to_f).item()))
            trk_vals.append(float(_tracked_l2_at_obs(run_pos, ep, to_f).item()))
            if save_rollout:
                pred_traj.append(run_pos.detach().cpu().clone())
                pred_frames.append(to_f)

    avg_obs = sum(obs_vals) / max(len(obs_vals), 1)
    avg_trk = sum(trk_vals) / max(len(trk_vals), 1)
    result = {
        "avg_recon_loss": avg_obs,
        "tracking_l2": avg_trk,
        "recon_loss": avg_obs + avg_trk,
        "chunks": float(len(obs_vals)),
        "logE_mean": float(sum(logE_means) / max(len(logE_means), 1)),
        "confidence_mean_mid": confidence_mean_mid,
        "correction_mode": str(correction_mode),
        "correction_stop_frame": float(int(correction_stop_frame)),
        "correction_steps_count": float(int(correction_steps_count)),
        **({"mid_frame": float(mid_frame)} if pure_tail else {}),
    }
    if save_rollout:
        result["rollout"] = {
            "predicted": torch.stack(pred_traj, dim=0),
            "predicted_frames": torch.tensor(pred_frames, dtype=torch.long),
            "ground_truth": coords.detach().cpu().clone(),
            "mid_frame": int(mid_frame),
        }
    return result


def episode_roots(cfg) -> list[str]:
    roots = [str(r) for r in cfg.dataset.get("validation_roots", [])]
    return ert._expand_simulation_batch_roots(roots)
