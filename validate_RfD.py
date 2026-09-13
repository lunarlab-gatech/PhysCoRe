"""
Sweep validation over the RfD checkpoints written by train_RfD.py.
`gvc`/`GVC` in identifiers and on-disk names all mean RfD.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from physcore.particle_flow import episode_runtime as ert
from physcore.particle_flow.dataset import ParticleFlowEpisodeDataset, _resolve_episode_roots
from physcore.particle_flow.mfm_validation import _tracked_l2_at_obs

from physcore.particle_flow.mfm_training import defaults, material_guess

from physcore.model_RfD import GridVelocityCorrector
from physcore.particle_flow.rfd_runtime import (
    FrozenRefiner,
    build_engine,
    chamfer_gt_to_pred,
    first_half_windows,
    load_refiner,
    _default_material_confidence,
)
from physcore.particle_flow.rfd_viz import _render_gvc_epoch_video, _save_loss_plots


# --- Config + material loaders ---


def _load_cfg(args: argparse.Namespace) -> "OmegaConf":
    cfg = defaults(OmegaConf.load(args.config))
    cfg.model.input_mode = "observed_control"
    overrides = list(getattr(args, "cli_overrides", []) or [])
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


# --- Per-camera-frame PID correction (first-half anchor, mirrors validate_MfM.py) ---


@torch.no_grad()
def _val_chunk_with_correction(
    *,
    engine,
    ctx: Dict,
    f_start: int,
    chunk_steps: int,
    correction_steps_count: int,
    run_correction: bool,
    positions, velocities, F_state, C_state,
    material,
    chunk_dt: float,
    ground_height: float,
):
    """One camera-frame rollout with optional PID-style correction substeps
    (wraps ``ert._run_chunk_rollout_and_correction``). Snaps tracked/surface
    particles toward the observed positions at frame ``f_start + 1``; RfD stays
    off during the correction sub-rollout. ert helpers expect unbatched (N, 3),
    so we squeeze on the way in and restore the batch dim on the way out.
    """
    ep = ctx["ep"]
    coords = ctx["coords"]
    kinematic_ids = ctx["kinematic_ids"]
    f_end = f_start + 1

    mat_for_mpm = material
    while mat_for_mpm.dim() > 3:
        mat_for_mpm = mat_for_mpm.squeeze(0)

    pos_u = positions.squeeze(0) if positions.dim() == 3 else positions
    vel_u = velocities.squeeze(0) if velocities.dim() == 3 else velocities
    F_u   = F_state.squeeze(0)   if F_state.dim()   == 4 else F_state
    C_u   = (None if C_state is None
             else (C_state.squeeze(0) if C_state.dim() == 4 else C_state))

    obs_corr, vis = ert._get_correction_observation(
        ep, coords, ep["vis_idx"], f_end, ctx["N"],
    )
    rigid_window      = ert._catmull_rom_window(ep["r_coords"], f_start, f_end, chunk_steps)
    controller_window = ert._catmull_rom_window(
        ep.get("controller_grid_points", None), f_start, f_end, chunk_steps,
    )
    contact_window = (ert._hold_index_window(kinematic_ids, f_start, chunk_steps)
                      if torch.is_tensor(kinematic_ids) else None)

    with ert._temporary_rollout_timestep(engine, dt=chunk_dt, ground_height=ground_height):
        out = ert._run_chunk_rollout_and_correction(
            engine, pos_u, vel_u, F_u, C_u, mat_for_mpm,
            chunk_steps, f_start, f_end,
            obs_corr, vis, coords, ep["r_coords"],
            ep.get("particle_material_models", None),
            ep.get("rigid_collision_cfg", {}),
            ep.get("rigid_body_primitives", []),
            int(correction_steps_count), bool(run_correction),
            ep.get("manipulation_flag", None),
            kinematic_ids if torch.is_tensor(kinematic_ids) else None,
            # Positional `controller_grid_points`: the correction sub-rollout reads
            # ONLY this (the _window arg feeds the main chunk rollout). Passing None
            # drops the gripper for every correction substep — with kinematic contact
            # ids gated off, grid-velocity coupling is the sole grip during phase 1.
            ep.get("controller_grid_points", None),
            rigid_points_window=rigid_window,
            manipulation_contact_particle_ids_window=contact_window,
            controller_grid_points_window=controller_window,
        )
    src = out["correction_result"] if run_correction else out["pure_mpm_result"]
    new_pos = src["predicted_positions"][-1].detach().unsqueeze(0)
    new_vel = src["final_velocity"].detach().unsqueeze(0)
    new_F   = src["final_deformation_gradient"].detach().unsqueeze(0)
    new_C   = (None if src.get("final_C") is None
               else src["final_C"].detach().unsqueeze(0))
    return new_pos, new_vel, new_F, new_C


# --- Per-window no-grad rollout (mirrors train_RfD.py's training rollout) ---


@torch.no_grad()
def _val_rollout_window(
    *,
    engine,
    ctx: Dict,
    f_start: int,
    f_end: int,
    n_predicted: int,
    outer_steps: int,
    positions, velocities, F_state, C_state,
    material, confidence,
    w_chamfer: float, w_l2: float,
    chunk_dt: float, ground_height: float,
    rollout_steps: int,
    device: torch.device,
):
    """One-window no-grad analog of train_RfD.py's _rollout_and_train (no backprop)."""
    ep = ctx["ep"]
    N = ctx["N"]
    T = ctx["T"]
    kinematic_ids = ctx["kinematic_ids"]
    tracked_visible_indices = ctx["tracked_visible_indices"]

    log_E = material[..., 0]
    nu    = material[..., 1]

    # Piecewise per-frame Catmull-Rom so every intermediate observed pose is
    # respected — matches _rollout_and_train. A single end-to-end segment would
    # skip the k-1 intermediate controller poses inside the window.
    rigid_window      = ert._piecewise_catmull_rom_window(ep["r_coords"], f_start, f_end, steps_per_frame=rollout_steps)
    controller_window = ert._piecewise_catmull_rom_window(ep.get("controller_grid_points", None), f_start, f_end, steps_per_frame=rollout_steps)
    contact_window    = (ert._hold_index_window(kinematic_ids, f_start, outer_steps)
                         if torch.is_tensor(kinematic_ids) else None)
    rigid_in      = rigid_window.unsqueeze(0)      if rigid_window      is not None else None
    controller_in = controller_window.unsqueeze(0) if controller_window is not None else None

    B = positions.shape[0]
    delta_v = torch.zeros(outer_steps, B, N, 3, device=device, dtype=positions.dtype)

    with ert._temporary_rollout_timestep(engine, dt=chunk_dt, ground_height=ground_height):
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
            material_confidence=confidence,
        )
    pred_pos_all = out["predicted_positions"][:, 0]

    chamfer_acc = pred_pos_all.new_zeros(())
    l2_acc      = pred_pos_all.new_zeros(())
    obs = ep.get("observation_data", None)
    frames_used = 0
    for t in range(n_predicted):
        gt_frame = f_start + 1 + t
        if gt_frame >= T:
            break
        cam_step_idx = (t + 1) * rollout_steps - 1
        p = pred_pos_all[cam_step_idx]
        if obs is not None:
            chamfer_acc = chamfer_acc + chamfer_gt_to_pred(
                obs["object_points_clean"][:, gt_frame].to(device),
                obs["object_valid_mask"][:, gt_frame].to(device),
                p,
            )
        if tracked_visible_indices is not None:
            # Obs-space tracking metric (validate_MfM.py convention), not train_RfD's
            # GT-space tracked_l2.
            l2_acc = l2_acc + _tracked_l2_at_obs(p, ep, gt_frame)
        frames_used += 1
    chamfer_acc = chamfer_acc / max(frames_used, 1)
    l2_acc      = l2_acc     / max(frames_used, 1)
    loss = w_chamfer * chamfer_acc + w_l2 * l2_acc

    new_positions  = out["predicted_positions"][-1].detach()
    new_velocities = out["final_velocity"].detach()
    new_F          = out["final_deformation_gradient"].detach()
    new_C          = out["final_C"].detach() if out.get("final_C", None) is not None else None
    return (
        new_positions, new_velocities, new_F, new_C,
        float(loss.item()), float(chamfer_acc.item()), float(l2_acc.item()),
    )


# --- Full-sweep validation pass for one loaded corrector ---


@torch.no_grad()
def _run_validation(
    *,
    refiner: FrozenRefiner,
    val_episodes: List[Dict],
    cfg,
    device: torch.device,
    k: int,
    rollout_steps: int,
    w_chamfer: float,
    w_l2: float,
    use_gvc: bool = True,
    correction_mode: str = "half",
) -> Optional[Dict[str, float]]:
    """Two-phase held-out validation (mirrors old validate_RfD.py's --correction-mode):

      * Phase 1 — first-half windows: MfM refreshes material per window; RfD
        OFF; state advances but its loss is NOT counted. When
        ``correction_mode == "half"`` a validate_MfM.py-style PID target-velocity
        correction (snap to observation) is applied at every camera frame, so the
        tracked/surface particles are anchored to the observed first half.
      * Phase 2 — second half: material frozen at the midpoint; RfD ON (unless ``use_gvc``
        is False, an ablation); rollout loss IS counted. These are the reported
        numbers.
    """
    if not val_episodes:
        return None
    # PID correction-substep count comes from the same cfg fields validate_MfM.py uses.
    correction_steps_count = (
        int(cfg.model.get("correction_rollout_steps", cfg.model.get("correction_steps", 4)))
        if str(correction_mode) != "none" else 0
    )
    val_loss_sum = val_chamfer_sum = val_l2_sum = 0.0
    val_windows = 0
    for ctx in val_episodes:
        ep_loss = ep_chamfer = ep_l2 = 0.0
        ep_windows = 0
        engine = ctx["engine"]
        ep = ctx["ep"]
        coords = ctx["coords"]
        n_full_windows    = ctx["n_full_windows"]
        total_outer_steps = ctx["total_outer_steps"]
        refiner.reset(coords[0:1].to(device))
        material   = material_guess(ep, cfg, 0, seed=int(cfg.train.get("seed", 0)))
        confidence = _default_material_confidence(material)
        positions  = coords[0:1].to(device)
        velocities = ep["particle_v"][0:1].to(device)
        F_state    = ep["particle_F"][0:1].to(device)
        C_state    = ep["particle_C"][0:1].to(device) if ep.get("particle_C", None) is not None else None

        chunk_dt      = ert._real_world_chunk_dt(ep, engine, rollout_steps)
        ground_height = float(ep.get("ground_height", engine.ground_height))

        # K-window-aligned midpoint = first-half / second-half boundary.
        mid_window_idx = first_half_windows(cfg, n_full_windows)
        if str(correction_mode) == "none":
            corr_stop_window = 0
        elif str(correction_mode) == "half":
            corr_stop_window = mid_window_idx
        else:
            raise ValueError(f"unknown correction_mode: {correction_mode}")

        warmup = min(int(cfg.gvc.get("warmup_windows", 0)), mid_window_idx)
        for w in range(warmup):
            material, confidence = refiner.predict_window(ep, start_frame=w*k, end_frame=(w+1)*k, material=material)

        # Phase 1: material refresh, RfD OFF, per-frame PID correction, loss ignored.
        engine.enable_corrector(False)
        for window_idx in range(mid_window_idx):
            f_start = window_idx * k
            f_end   = f_start + k
            if window_idx >= warmup:
                material, confidence = refiner.predict_window(ep, start_frame=f_start, end_frame=f_end, material=material)
            if window_idx < corr_stop_window:
                for i in range(k):
                    positions, velocities, F_state, C_state = _val_chunk_with_correction(
                        engine=engine, ctx=ctx, f_start=f_start + i,
                        chunk_steps=rollout_steps,
                        correction_steps_count=correction_steps_count,
                        run_correction=True,
                        positions=positions, velocities=velocities,
                        F_state=F_state, C_state=C_state,
                        material=material,
                        chunk_dt=chunk_dt, ground_height=ground_height,
                    )
            else:
                positions, velocities, F_state, C_state, _, _, _ = _val_rollout_window(
                    engine=engine, ctx=ctx, f_start=f_start, f_end=f_end,
                    n_predicted=k, outer_steps=total_outer_steps,
                    positions=positions, velocities=velocities,
                    F_state=F_state, C_state=C_state,
                    material=material, confidence=confidence,
                    w_chamfer=w_chamfer, w_l2=w_l2,
                    chunk_dt=chunk_dt, ground_height=ground_height,
                    rollout_steps=rollout_steps, device=device,
                )

        # Phase 2: material frozen, RfD ON (caller-controlled), loss counted.
        engine.enable_corrector(use_gvc)
        for window_idx in range(mid_window_idx, n_full_windows):
            f_start = window_idx * k
            f_end   = f_start + k
            positions, velocities, F_state, C_state, l, c, l2 = _val_rollout_window(
                engine=engine, ctx=ctx, f_start=f_start, f_end=f_end,
                n_predicted=k, outer_steps=total_outer_steps,
                positions=positions, velocities=velocities,
                F_state=F_state, C_state=C_state,
                material=material, confidence=confidence,
                w_chamfer=w_chamfer, w_l2=w_l2,
                chunk_dt=chunk_dt, ground_height=ground_height,
                rollout_steps=rollout_steps, device=device,
            )
            ep_loss += l; ep_chamfer += c; ep_l2 += l2; ep_windows += 1
        # Trailing partial window (tail_frames) is neither rolled nor scored:
        # metric range is mid_frame+1 .. n_full_windows*k, matching validate_MfM.py.
        # Each episode contributes ONE unit, so the mean is unweighted over
        # episodes rather than over windows (long episodes don't count more).
        if ep_windows:
            val_loss_sum    += ep_loss    / ep_windows
            val_chamfer_sum += ep_chamfer / ep_windows
            val_l2_sum      += ep_l2      / ep_windows
            val_windows += 1
            print(f"[val] ep {ctx['label']}: chamfer={ep_chamfer / ep_windows:.5f} "
                  f"l2={ep_l2 / ep_windows:.5f} frames={ep_windows * k}", flush=True)
    if val_windows == 0:
        return None
    return {
        "loss":    val_loss_sum    / val_windows,
        "chamfer": val_chamfer_sum / val_windows,
        "l2":      val_l2_sum      / val_windows,
        "windows": val_windows,
    }


# --- Build a single val episode context ---


def _build_val_episode_context(
    ep_idx: int, root: str, raw_entry,
    k: int, rollout_steps: int, device: torch.device,
) -> Dict:
    ep_tensors = ert._load_episode_tensors(raw_entry, device)
    tracked_visible_indices = ert._pull_tracked_visible_indices(root, device)
    coords_e = ep_tensors["coords"]
    T_e, N_e, _ = coords_e.shape
    kinematic_ids_e = ert._kinematic_contact_ids(ep_tensors)
    n_full_windows_e = (T_e - 1) // k
    tail_frames_e    = (T_e - 1) - n_full_windows_e * k
    label = f"val{ep_idx:02d}_{Path(root).parent.name}_{Path(root).name}"
    return {
        "idx": ep_idx,
        "root": root,
        "label": label,
        "ep": ep_tensors,
        "coords": coords_e,
        "T": T_e, "N": N_e,
        "n_full_windows": n_full_windows_e,
        "tail_frames": tail_frames_e,
        "total_outer_steps": k * rollout_steps,
        "tracked_visible_indices": tracked_visible_indices,
        "kinematic_ids": kinematic_ids_e,
    }


# --- Main sweep ---


def validate(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args)
    device = torch.device(cfg.train.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg.train.get("seed", 0)))

    output_dir = Path(cfg.train.output_dir)
    ckpt_dir   = output_dir / "checkpoints"
    plots_dir  = output_dir / "plots"
    viz_dir    = output_dir / "videos"
    if not ckpt_dir.exists():
        raise RuntimeError(f"checkpoints dir not found: {ckpt_dir}")

    # Enumerate gvc_epoch_*.pt
    ckpt_re = re.compile(r"^gvc_epoch_(\d+)\.pt$")
    ckpt_files = sorted(
        ((int(ckpt_re.match(p.name).group(1)), p) for p in ckpt_dir.glob("gvc_epoch_*.pt")
         if ckpt_re.match(p.name)),
        key=lambda x: x[0],
    )
    if args.epoch is not None:
        target = int(args.epoch)
        ckpt_files = [(e, p) for (e, p) in ckpt_files if e == target]
    if not ckpt_files:
        raise RuntimeError(f"no gvc_epoch_*.pt found in {ckpt_dir}")
    print(f"[val] {len(ckpt_files)} checkpoint(s) to evaluate: "
          f"epochs={[e for e, _ in ckpt_files]}", flush=True)

    gvc_cfg = cfg.get("gvc", {})
    # MfM (built once for the whole sweep).
    refiner_ckpt_path = args.refiner_checkpoint or gvc_cfg.get("refiner_checkpoint", None)
    if not refiner_ckpt_path:
        raise RuntimeError("RfD validation requires gvc.refiner_checkpoint set in the config "
                           "or --refiner-checkpoint on the CLI.")
    refiner_model = load_refiner(str(refiner_ckpt_path), cfg, device)
    refiner: FrozenRefiner = FrozenRefiner(refiner_model, cfg, device)
    print(f"[val] loaded MfM: {refiner_ckpt_path}", flush=True)

    # Corrector weights are shared by all per-episode engines.
    corrector = GridVelocityCorrector(
        in_channels=19, cond_dim=16,
        base_channels=int(gvc_cfg.get("base_channels", 32)),
        levels=int(gvc_cfg.get("levels", 2)),
        max_delta_v=float(gvc_cfg.get("max_delta_v", 0.05)),
    ).to(device)
    use_gvc = not bool(args.deactivate_gvc)
    if not use_gvc:
        print("[val] --deactivate-gvc — pure-MPM ablation (no RfD residual)", flush=True)

    # Validation dataset (loaded once, GPU-resident through the sweep).
    val_roots = _resolve_episode_roots(
        [str(r) for r in cfg.dataset.get("validation_roots", []) or []]
    )
    if not val_roots:
        raise RuntimeError("cfg.dataset.validation_roots is empty — nothing to validate")
    val_dataset = ParticleFlowEpisodeDataset(
        val_roots,
        cache_size=int(cfg.dataset.get("cache_size", 2)),
        real_world_domain_center=cfg.dataset.get("real_world_domain_center", None),
        observation_views=cfg.dataset.get("observation_views", None),
        observation_view_index=int(cfg.dataset.get("observation_view_index", 0)),
    )
    k = int(cfg.train.update_every)
    rollout_steps = int(cfg.dataset.get("rollout_steps", 25))
    w_chamfer = float(cfg.loss.get("chamfer_weight", 1.0))
    w_l2      = float(cfg.loss.get("tracked_l2_weight", 1.0))

    val_episodes: list = []
    for ep_idx, root in enumerate(val_roots):
        ep_engine = build_engine(cfg, device, root)
        ep_engine.attach_corrector(corrector, h=int(gvc_cfg.get("h", 10)))
        ep_engine.enable_corrector(use_gvc)
        ctx = _build_val_episode_context(
            ep_idx, root, val_dataset[ep_idx],
            k=k, rollout_steps=rollout_steps, device=device,
        )
        ctx["engine"] = ep_engine
        val_episodes.append(ctx)
        print(f"[val] episode {ep_idx} ({ctx['label']}): T={ctx['T']} N={ctx['N']} "
              f"n_full_windows={ctx['n_full_windows']} tail_frames={ctx['tail_frames']}",
              flush=True)

    # Load history.json from training (preserve training rows; overlay val).
    history_path = plots_dir / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        print(f"[val] loaded existing history.json with {len(history)} rows", flush=True)
    else:
        history = []
        print("[val] no existing history.json — creating one", flush=True)
    by_epoch = {int(r["epoch"]): r for r in history if "epoch" in r}

    viz_every   = int(cfg.train.get("viz_every", 0) or 0)
    final_epoch = max(e for e, _ in ckpt_files)
    skip_render = bool(args.skip_render)
    best_val_loss, best_val_epoch = float("inf"), 0

    for epoch_num, ckpt_path in ckpt_files:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        corrector.load_state_dict(ckpt["model"], strict=True)
        corrector.eval()
        for ctx in val_episodes:
            ctx["engine"].enable_corrector(use_gvc)

        t0 = time.perf_counter()
        v = _run_validation(
            refiner=refiner, val_episodes=val_episodes, cfg=cfg, device=device,
            k=k, rollout_steps=rollout_steps,
            w_chamfer=w_chamfer, w_l2=w_l2,
            use_gvc=use_gvc, correction_mode=args.correction_mode,
        )
        elapsed = time.perf_counter() - t0
        if v is None:
            print(f"[val] epoch {epoch_num:03d} produced no windows — skipping", flush=True)
            continue

        # Update / insert the history row for this epoch.
        row = by_epoch.get(epoch_num)
        if row is None:
            row = {"epoch": epoch_num}
            history.append(row)
            by_epoch[epoch_num] = row
        row["val_loss"]    = v["loss"]
        row["val_chamfer"] = v["chamfer"]
        row["val_l2"]      = v["l2"]
        row["val_windows"] = v["windows"]
        row["val_time_s"]  = float(elapsed)

        if v["loss"] < best_val_loss:
            best_val_loss  = v["loss"]
            best_val_epoch = epoch_num

        print(f"[val] epoch {epoch_num:03d} val loss={v['loss']:.6g} chamfer={v['chamfer']:.6g} "
              f"l2={v['l2']:.6g} windows={v['windows']} episodes={len(val_episodes)} "
              f"val_time={elapsed:.1f}s", flush=True)

        # Persist incrementally (crash-safe + live plot refresh).
        history.sort(key=lambda r: int(r.get("epoch", 0)))
        plots_dir.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(history, indent=2))
        _save_loss_plots(history, plots_dir)

        # Val viz at viz_every cadence (and the final ckpt).
        if not skip_render and viz_every > 0 and (
            epoch_num % viz_every == 0 or epoch_num == final_epoch
        ):
            sub_dir = viz_dir / "val" / f"epoch_{epoch_num:04d}"
            for viz_ctx in val_episodes:
                _render_gvc_epoch_video(
                    engine=viz_ctx["engine"], refiner=refiner, corrector=corrector,
                    ep=viz_ctx["ep"], cfg=cfg,
                    kinematic_ids=viz_ctx["kinematic_ids"],
                    n_full_windows=viz_ctx["n_full_windows"],
                    tail_frames=viz_ctx["tail_frames"],
                    k=k, rollout_steps=rollout_steps,
                    total_outer_steps=viz_ctx["total_outer_steps"],
                    viz_dir=sub_dir,
                    epoch_num=epoch_num,
                    cfg_train_loss=v["loss"],
                    log_prefix="[val] ",
                    ep_label=viz_ctx["label"],
                    kind="val",
                    pid_chunk_fn=(_val_chunk_with_correction
                                  if args.correction_mode != "none" else None),
                )

    # Copy the best-val checkpoint.
    if best_val_epoch > 0:
        best_src = ckpt_dir / f"gvc_epoch_{best_val_epoch:04d}.pt"
        best_dst = ckpt_dir / "gvc_best_val.pt"
        shutil.copy2(best_src, best_dst)
        print(f"[val] best val epoch {best_val_epoch:03d} (val_loss={best_val_loss:.6g}) "
              f"→ copied {best_src.name} to {best_dst}", flush=True)

    print(f"[val] sweep complete; history + plots refreshed in {plots_dir}", flush=True)


# --- CLI ---


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validation sweep over RfD checkpoints.")
    p.add_argument("--config", default="configs/validate_RfD.yaml",
                   help="RfD (GVC) validation config YAML")
    p.add_argument("--refiner-checkpoint", default=None, help="override cfg.gvc.refiner_checkpoint")
    p.add_argument("--epoch", default=None, help="only validate this single epoch; default: all checkpoints")
    p.add_argument("--skip-render", action="store_true", help="skip val mp4 rendering (val loss only)")
    p.add_argument("--deactivate-gvc", action="store_true",
                   help="pure MPM (no RfD residual) — ablation")
    p.add_argument("--correction-mode", choices=["none", "half"], default="half",
                   help="first-half PID correction: 'half' (default) snaps tracked "
                        "particles to observations through the first half before the "
                        "RfD-scored second half; 'none' disables it (ablation)")
    args, overrides = p.parse_known_args()
    bad = [o for o in overrides if "=" not in o or o.startswith("-")]
    if bad:
        p.error(f"unrecognized argument(s): {bad}")
    args.cli_overrides = overrides
    return args


if __name__ == "__main__":
    validate(parse_args())
