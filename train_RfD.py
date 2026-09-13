"""
Train RfD (Residual from Dynamics) on real-world converted episodes.
`gvc`/`GVC` in identifiers, checkpoint names and log tags all mean RfD.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from physcore.particle_flow import episode_runtime as ert
from physcore.particle_flow.dataset import ParticleFlowEpisodeDataset, _resolve_episode_roots
from physcore.particle_flow.runtime import (
    setup_distributed,
    is_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    allreduce_gradients,
    broadcast_parameters,
    cleanup_distributed,
)

from physcore.particle_flow.mfm_training import defaults, material_guess

from physcore.model_RfD import GridVelocityCorrector
from physcore.particle_flow.rfd_runtime import (
    FrozenRefiner,
    build_engine,
    chamfer_gt_to_pred,
    first_half_windows,
    load_refiner,
    tracked_l2,
    _default_material_confidence,
)
from physcore.particle_flow.rfd_viz import _render_gvc_epoch_video, _save_loss_plots


# --- Augmentation: z-rotation + 50% x-flip, one R per (episode, epoch) ---
# MPM physics is exactly equivariant under R (z-axis = gravity is preserved).


def _random_z_rotation_and_flip(seed: int, device: torch.device) -> torch.Tensor:
    """Return a 3x3 rotation matrix around world z, with 50% chance of mirroring x."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    angle = torch.rand(1, generator=g).item() * 2.0 * math.pi
    c, s = math.cos(angle), math.sin(angle)
    R = torch.tensor(
        [[c, -s, 0.0],
         [s,  c, 0.0],
         [0.0, 0.0, 1.0]],
        device=device, dtype=torch.float32,
    )
    if torch.rand(1, generator=g).item() < 0.5:
        R[0] = -R[0]
    return R


def _rotate_episode(ep: Dict, coords: torch.Tensor, R: torch.Tensor) -> tuple[Dict, torch.Tensor]:
    """Return ``(aug_ep, aug_coords)`` (shallow copy of ``ep``) under a
    rotate-then-translate augmentation that keeps the object on the MPM grid.

    Implementation = mfm_training.py-style rotation (``x @ R.T``) for every world-frame
    tensor, followed by a translation that puts the rotated frame-0 AABB center
    back at its original location ``C0``. Mathematically identical to pivot
    rotation around ``C0``; written this way to match mfm_training.py's style and
    make the re-centering step explicit.

    Transformed:
      - coords, r_coords, controller_grid_points,
        observation_data.object_points_clean: positions → ``x @ R.T + delta``
        where ``delta = C0 - R @ C0`` and ``C0`` = frame-0 AABB center of the
        un-rotated ``coords``.
      - particle_v: free vector → ``v @ R.T`` (no translation).
      - particle_F, particle_C: similarity transform ``R @ M @ R.T`` (no
        translation). Both are identity / zero at frame 0 in the loaded
        episode, so this is also a no-op for MPM init in practice.
    Untouched: indices, scalars, masks (tracked_visible_indices,
    manipulation_contact_particle_ids, ground_height, particle_motion_valid,
    object_valid_mask, ...).
    """
    aug_ep = dict(ep)
    Rt = R.transpose(-1, -2)

    # Frame-0 AABB center of the un-rotated object, and the shift that puts
    # the rotated frame-0 center back to the same world location.
    c0 = 0.5 * (coords[0].amin(dim=0) + coords[0].amax(dim=0))   # (3,)
    delta = c0 - c0 @ Rt                                          # (3,)

    def _xform_positions(x: torch.Tensor) -> torch.Tensor:
        # ``delta`` broadcasts over any leading dims (T, N, K, ...).
        return x @ Rt + delta.view(*([1] * (x.ndim - 1)), 3)

    aug_coords = _xform_positions(coords)
    aug_ep["coords"] = aug_coords
    if isinstance(ep.get("r_coords"), torch.Tensor):
        aug_ep["r_coords"] = _xform_positions(ep["r_coords"])
    if isinstance(ep.get("controller_grid_points"), torch.Tensor):
        aug_ep["controller_grid_points"] = _xform_positions(ep["controller_grid_points"])
    obs = ep.get("observation_data")
    if isinstance(obs, dict) and isinstance(obs.get("object_points_clean"), torch.Tensor):
        new_obs = dict(obs)
        new_obs["object_points_clean"] = _xform_positions(obs["object_points_clean"])
        aug_ep["observation_data"] = new_obs

    # Free vectors / tensors: rotation only, no translation.
    if isinstance(ep.get("particle_v"), torch.Tensor):
        aug_ep["particle_v"] = ep["particle_v"] @ Rt
    if isinstance(ep.get("particle_F"), torch.Tensor):
        # F: (T, N, 3, 3) — broadcast ``R @ F @ R.T`` over leading dims.
        aug_ep["particle_F"] = (R @ ep["particle_F"]) @ Rt
    if isinstance(ep.get("particle_C"), torch.Tensor):
        aug_ep["particle_C"] = (R @ ep["particle_C"]) @ Rt

    return aug_ep, aug_coords


def _epoch_lr(cfg, epoch: int) -> float:
    """Linear LR decay between ``lr_decay_start`` and ``lr_decay_end``.
    Returns the configured ``train.lr`` if any of those knobs are unset.
    """
    base = float(cfg.train.lr)
    final = cfg.train.get("final_lr", None)
    start = cfg.train.get("lr_decay_start", None)
    end = cfg.train.get("lr_decay_end", None)
    if final is None or start is None or end is None:
        return base
    if epoch <= int(start):
        return base
    if epoch >= int(end):
        return float(final)
    span = max(int(end) - int(start), 1)
    alpha = (epoch - int(start)) / span
    return base + alpha * (float(final) - base)


# --- Loading utilities ---


def load_cfg(args: argparse.Namespace) -> "OmegaConf":
    cfg = defaults(OmegaConf.load(args.config))
    cfg.model.input_mode = "observed_control"
    if args.episode_root:
        cfg.dataset.train_roots = [str(args.episode_root)]
    # Any unparsed `key=value` args are merged as OmegaConf dotlist overrides
    # so YAML knobs (e.g. train.output_dir) can be set from the CLI
    # without editing the config file.
    overrides = list(getattr(args, "cli_overrides", []) or [])
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


# --- Training loop ---


def train(args: argparse.Namespace) -> None:
    rank, local_rank, world_size = setup_distributed()
    cfg = load_cfg(args)
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(cfg.train.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg.train.get("seed", 0)) + rank)

    gvc_cfg_early = cfg.get("gvc", {})
    # CLI overrides YAML when explicitly passed (`--refiner-checkpoint`); otherwise the knob comes
    # from `gvc:` in the YAML.
    refiner_ckpt_path = args.refiner_checkpoint or gvc_cfg_early.get("refiner_checkpoint", None)
    if not refiner_ckpt_path:
        raise RuntimeError(
            "RfD training requires `gvc.refiner_checkpoint` set in the config "
            "(path to a frozen MfM/Refiner .pt). Alternatively pass --refiner-checkpoint on the CLI."
        )
    refiner_model = load_refiner(str(refiner_ckpt_path), cfg, device)
    refiner: FrozenRefiner = FrozenRefiner(refiner_model, cfg, device)

    gvc_cfg = cfg.get("gvc", {})
    corrector = GridVelocityCorrector(
        in_channels=19,  # was 17; +2 channels for per-particle material_confidence
        cond_dim=16,
        base_channels=int(gvc_cfg.get("base_channels", 32)),
        levels=int(gvc_cfg.get("levels", 2)),
        max_delta_v=float(gvc_cfg.get("max_delta_v", 0.05)),
    ).to(device)
    if is_distributed():
        broadcast_parameters(corrector, src=0)

    opt = torch.optim.AdamW(
        corrector.parameters(),
        lr=float(cfg.train.lr),
        betas=(float(cfg.train.adam_beta1), float(cfg.train.adam_beta2)),
        weight_decay=float(cfg.train.weight_decay),
    )

    train_roots = _resolve_episode_roots([str(r) for r in cfg.dataset.train_roots])
    dataset = ParticleFlowEpisodeDataset(
        train_roots,
        cache_size=int(cfg.dataset.get("cache_size", 2)),
        real_world_domain_center=cfg.dataset.get("real_world_domain_center", None),
        observation_views=cfg.dataset.get("observation_views", None),
        observation_view_index=int(cfg.dataset.get("observation_view_index", 0)),
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty — check cfg.dataset.train_roots")

    # Validation (loading val episodes, running val rollouts, writing
    # val curves/videos/best-val ckpt) is handled separately by
    # validate_RfD.py — run it post-training over the saved checkpoints.

    k = int(cfg.train.update_every)
    rollout_steps = int(cfg.dataset.get("rollout_steps", 25))
    w_chamfer = float(cfg.loss.get("chamfer_weight", 1.0))
    w_l2 = float(cfg.loss.get("tracked_l2_weight", 1.0))

    def _build_episode_context(ep_idx: int, root: str, raw_entry, tag: str) -> Dict:
        ep_tensors = ert._load_episode_tensors(raw_entry, device)
        ep_engine = build_engine(cfg, device, root)
        ep_engine.attach_corrector(corrector, h=int(gvc_cfg.get("h", 10)))
        tracked_visible_indices = ert._pull_tracked_visible_indices(root, device)
        coords_e = ep_tensors["coords"]
        T_e, N_e, _ = coords_e.shape
        kinematic_ids_e = ert._kinematic_contact_ids(ep_tensors)
        n_full_windows_e = (T_e - 1) // k
        tail_frames_e = (T_e - 1) - n_full_windows_e * k
        total_outer_steps_e = k * rollout_steps
        label = f"{tag}{ep_idx:02d}_{Path(root).parent.name}_{Path(root).name}"
        return {
            "idx": ep_idx,
            "root": root,
            "label": label,
            "engine": ep_engine,
            "ep": ep_tensors,
            "coords": coords_e,
            "T": T_e,
            "N": N_e,
            "n_full_windows": n_full_windows_e,
            "tail_frames": tail_frames_e,
            "total_outer_steps": total_outer_steps_e,
            "tracked_visible_indices": tracked_visible_indices,
            "kinematic_ids": kinematic_ids_e,
        }

    episodes: list = []
    for ep_idx, root in enumerate(train_roots):
        ctx = _build_episode_context(ep_idx, root, dataset[ep_idx], tag="ep")
        episodes.append(ctx)
        print(
            f"[gvc] episode {ep_idx} ({ctx['label']}): T={ctx['T']} N={ctx['N']} "
            f"n_full_windows={ctx['n_full_windows']} tail_frames={ctx['tail_frames']} "
            f"kinematic_contact_ids={'active' if ctx['kinematic_ids'] is not None else 'gated_off'} "
            f"tracked_visible={'yes' if ctx['tracked_visible_indices'] is not None else 'NO (L2 will be 0)'}",
            flush=True,
        )


    # First-half adaptation: a single shared RfD is adapted across all episodes,
    # each contributing only its first `warmup_fraction` of windows, in truncated-
    # BPTT segments of `bptt_frames` frames (memory-safe for the large episodes).
    warmup_fraction = float(gvc_cfg.get("warmup_fraction", 0.5))
    bptt_frames = max(1, int(gvc_cfg.get("bptt_frames", 3)))
    engine = episodes[0]["engine"]
    print(f"[gvc] warmup_fraction={warmup_fraction} bptt_frames={bptt_frames} k={k} "
          f"steps_per_frame={engine.steps_per_frame} dt={engine.dt} "
          f"episodes={len(episodes)}", flush=True)

    output_dir = Path(cfg.train.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    if is_distributed():
        import torch.distributed as _dist
        _dist.barrier()

    history = []
    # On-disk names keep the legacy `gvc_` prefix (gvc_epoch_NNNN.pt, gvc_latest.pt,
    # gvc_best_train.pt) and logs keep the `[gvc]` tag — both mean RfD. They are held
    # stable so existing runs and the reference outputs stay directly comparable.
    plots_dir = output_dir / "plots"
    ckpt_dir = output_dir / "checkpoints"
    viz_dir = output_dir / "videos"
    if is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        viz_dir.mkdir(parents=True, exist_ok=True)
    plot_every = int(cfg.train.get("plot_every", 1))

    def _rollout_and_train(
        ctx: Dict,
        f_start: int,
        f_end: int,
        n_predicted: int,
        outer_steps: int,
        positions, velocities, F_state, C_state,
        material,
        confidence,
    ):
        """One window's MPM rollout + loss + backprop for episode ``ctx``.
        Returns updated MPM state plus (loss, chamfer, l2) floats. Uses
        *current* `material` and `confidence` as phi for the engine — caller
        is responsible for refreshing them first."""
        ep = ctx["ep"]
        engine = ctx["engine"]
        coords = ctx["coords"]
        N = ctx["N"]
        T = ctx["T"]
        kinematic_ids = ctx["kinematic_ids"]
        tracked_visible_indices = ctx["tracked_visible_indices"]

        log_E = material[..., 0]
        nu = material[..., 1]

        # Piecewise per-frame Catmull-Rom so every intermediate frame is respected.
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

        # Per-episode dt override so chunk sim time = ep["frame_dt"].
        with ert._temporary_rollout_timestep(
            engine,
            dt=ert._real_world_chunk_dt(ep, engine, rollout_steps),
            ground_height=float(ep.get("ground_height", engine.ground_height)),
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
                material_confidence=confidence,
            )
        pred_pos_all = out["predicted_positions"][:, 0]

        chamfer_acc = pred_pos_all.new_zeros(())
        l2_acc = pred_pos_all.new_zeros(())
        obs = ep.get("observation_data", None)
        frames_used = 0
        for t in range(n_predicted):
            gt_frame = f_start + 1 + t
            if gt_frame >= T:
                break
            cam_step_idx = (t + 1) * rollout_steps - 1
            p = pred_pos_all[cam_step_idx]
            gt = coords[gt_frame].to(device)
            if obs is not None:
                chamfer_acc = chamfer_acc + chamfer_gt_to_pred(
                    obs["object_points_clean"][:, gt_frame].to(device),
                    obs["object_valid_mask"][:, gt_frame].to(device),
                    p,
                )
            if tracked_visible_indices is not None:
                chosen = tracked_visible_indices[gt_frame]
                l2_acc = l2_acc + tracked_l2(gt, p, chosen)
            frames_used += 1
        chamfer_acc = chamfer_acc / max(frames_used, 1)
        l2_acc = l2_acc / max(frames_used, 1)
        loss = w_chamfer * chamfer_acc + w_l2 * l2_acc

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm_pre_clip = float(torch.nn.utils.clip_grad_norm_(
            corrector.parameters(), float(cfg.train.grad_clip),
        ).item())
        opt.step()

        new_positions = out["predicted_positions"][-1].detach()
        new_velocities = out["final_velocity"].detach()
        new_F = out["final_deformation_gradient"].detach()
        new_C = out["final_C"].detach() if out.get("final_C", None) is not None else None

        # ---- F-stability stats (final F of this window across all particles) ----
        # Shape is (B, N, 3, 3). Reshape to (P, 3, 3) where P = B*N for batched linalg.
        F_flat = new_F.reshape(-1, 3, 3)
        with torch.no_grad():
            det_F = torch.linalg.det(F_flat)              # (P,)
            sv = torch.linalg.svdvals(F_flat)             # (P, 3), sorted descending
            sigma_max = sv[:, 0]
            sigma_min = sv[:, -1]
            cond = sigma_max / sigma_min.clamp_min(1e-12)
            fro = F_flat.reshape(-1, 9).norm(dim=-1)      # Frobenius norm per particle
            abs_max = F_flat.abs().reshape(-1).max()      # element-wise max |F|

        stats = {
            "det_F_min": float(det_F.min().item()),
            "det_F_max": float(det_F.max().item()),
            "sigma_min_F": float(sigma_min.min().item()),
            "sigma_max_F": float(sigma_max.max().item()),
            "cond_F_max": float(cond.max().item()),
            "F_fro_max": float(fro.max().item()),
            "F_abs_max": float(abs_max.item()),
            "log_E_min": float(log_E.detach().min().item()),
            "log_E_max": float(log_E.detach().max().item()),
            "nu_min": float(nu.detach().min().item()),
            "nu_max": float(nu.detach().max().item()),
            "grad_norm": grad_norm_pre_clip,
        }
        return (
            new_positions, new_velocities, new_F, new_C,
            float(loss.item()), float(chamfer_acc.item()), float(l2_acc.item()),
            stats,
        )

    import random as _random

    best_loss = float("inf")
    best_epoch_num = 0
    for epoch in range(int(cfg.train.epochs)):
        epoch_num = epoch + 1  # 1-indexed for logs/checkpoints
        epoch_t0 = time.perf_counter()    # wall-time start for this epoch
        lr_now = _epoch_lr(cfg, epoch_num)
        for group in opt.param_groups:
            group["lr"] = float(lr_now)

        # Shuffle the episode order for this epoch (deterministic by seed + epoch).
        shuffled_order = list(range(len(episodes)))
        _random.Random(int(cfg.train.get("seed", 0)) + epoch).shuffle(shuffled_order)

        epoch_loss = epoch_chamfer = epoch_l2 = 0.0
        windows_done = 0
        # NaN-aware aggregators: a value v is NaN iff v != v.
        stab = {
            "det_F_min":  float("inf"),  "det_F_max":  float("-inf"),
            "sigma_min":  float("inf"),  "sigma_max":  float("-inf"),
            "cond_max":   float("-inf"),
            "fro_max":    float("-inf"),
            "abs_max":    float("-inf"),
            "logE_min":   float("inf"),  "logE_max":   float("-inf"),
            "nu_min":     float("inf"),  "nu_max":     float("-inf"),
            "grad_max":   float("-inf"), "grad_sum":   0.0,
            "grad_count": 0,             "nan_grad_windows": 0,
        }

        def _accumulate_stats(s: dict, window_idx: int, label: str) -> None:
            def _smin(k, v): stab[k] = stab[k] if v != v else min(stab[k], v)
            def _smax(k, v): stab[k] = stab[k] if v != v else max(stab[k], v)
            _smin("det_F_min", s["det_F_min"]); _smax("det_F_max", s["det_F_max"])
            _smin("sigma_min", s["sigma_min_F"]); _smax("sigma_max", s["sigma_max_F"])
            _smax("cond_max",  s["cond_F_max"])
            _smax("fro_max",   s["F_fro_max"]); _smax("abs_max", s["F_abs_max"])
            _smin("logE_min",  s["log_E_min"]); _smax("logE_max", s["log_E_max"])
            _smin("nu_min",    s["nu_min"]);    _smax("nu_max",   s["nu_max"])
            g = s["grad_norm"]
            if g != g or g in (float("inf"), float("-inf")):
                stab["nan_grad_windows"] += 1
                print(
                    f"  [warn] {label} window {window_idx:02d} non-finite grad_norm ({g}) — "
                    f"sigma_min={s['sigma_min_F']:.3g} det_F_min={s['det_F_min']:.3g}",
                    flush=True,
                )
            else:
                _smax("grad_max", g)
                stab["grad_sum"] += g
                stab["grad_count"] += 1

        # Per epoch: shuffled episodes, each re-inits MPM from its own frame-0
        # GT then rolls n_full_windows + tail with per-window loss + backprop.
        # In DDP mode, episodes are sharded across ranks for parallel compute.
        augment = bool(cfg.train.get("augment_z_rotation", False))
        if world_size > 1:
            rank_order = [ep_idx for i, ep_idx in enumerate(shuffled_order) if i % world_size == rank]
        else:
            rank_order = shuffled_order
        for ep_idx in rank_order:
            ctx = episodes[ep_idx]
            engine = ctx["engine"]
            engine.enable_corrector(True)
            ep = ctx["ep"]
            coords = ctx["coords"]
            n_full_windows = ctx["n_full_windows"]
            label = ctx["label"]

            # Augment: fresh R for this (epoch, episode), shared across all
            # frames. Both MfM predictions and MPM see the rotated tensors;
            # val and viz remain un-augmented (they read the original ctx).
            if augment:
                R_aug = _random_z_rotation_and_flip(
                    seed=int(cfg.train.get("seed", 0)) * 1009 + epoch * 17 + ep_idx,
                    device=device,
                )
                ep, coords = _rotate_episode(ep, coords, R_aug)
                ctx = {**ctx, "ep": ep, "coords": coords}

            # Material + confidence init for this episode.
            refiner.reset(coords[0:1].to(device))
            material = material_guess(ep, cfg, 0, seed=int(cfg.train.seed) + epoch + ep_idx)
            confidence = _default_material_confidence(material)

            # MPM initial state at this episode's frame 0.
            positions = coords[0:1].to(device)
            velocities = ep["particle_v"][0:1].to(device)
            F_state = ep["particle_F"][0:1].to(device)
            C_state = (
                ep["particle_C"][0:1].to(device)
                if ep.get("particle_C", None) is not None else None
            )

            # Proportional warmup = first-half window count (warmup_fraction of the
            # episode; 0.5 = midpoint). The frozen MfM observes the first half
            # (no MPM) → material is FROZEN at the midpoint estimate; the shared RfD
            # then adapts over that first half only.
            half = first_half_windows(cfg, n_full_windows)
            warmup = half
            for w in range(warmup):
                material, confidence = refiner.predict_window(ep, start_frame=w*k, end_frame=(w+1)*k, material=material)
            if warmup > 0:
                print(f"  [{label}] MfM observed windows 0..{warmup-1} "
                      f"(frames 1..{warmup*k}); material frozen — adapting RfD over "
                      f"the first {half} window(s).", flush=True)

            # Train the FIRST HALF ONLY (frames 0..mid_frame) with the FIXED material,
            # free-running MPM + RfD. Backprop in truncated-BPTT segments of
            # `bptt_frames` frames so the per-window MPM backward graph fits on the
            # card for the large (12k-14k particle) episodes; state is detached
            # between segments (returned detached by _rollout_and_train). No tail /
            # second half — adaptation trains the first half only.
            mid_frame = half * k
            seg_start = 0
            seg_idx = 0
            while seg_start < mid_frame:
                seg = min(bptt_frames, mid_frame - seg_start)
                f_start = seg_start
                f_end = seg_start + seg
                positions, velocities, F_state, C_state, l, c, l2, stats = _rollout_and_train(
                    ctx, f_start, f_end, n_predicted=seg, outer_steps=seg * rollout_steps,
                    positions=positions, velocities=velocities, F_state=F_state, C_state=C_state,
                    material=material,
                    confidence=confidence,
                )
                epoch_loss += l; epoch_chamfer += c; epoch_l2 += l2
                windows_done += 1
                _accumulate_stats(stats, seg_idx, label)
                seg_start = f_end
                seg_idx += 1

        # DDP: average corrector weights across ranks at epoch boundary.
        if world_size > 1:
            import torch.distributed as _dist
            for p in corrector.parameters():
                _dist.all_reduce(p.data, op=_dist.ReduceOp.SUM)
                p.data.div_(world_size)
            _dist.barrier()

        train_elapsed = time.perf_counter() - epoch_t0  # training-phase wall time
        avg_loss = epoch_loss / max(windows_done, 1)
        avg_chamfer = epoch_chamfer / max(windows_done, 1)
        avg_l2 = epoch_l2 / max(windows_done, 1)
        grad_norm_mean = (
            stab["grad_sum"] / stab["grad_count"]
            if stab["grad_count"] > 0 else float("nan")
        )
        if not is_main_process():
            continue
        print(
            f"[gvc] epoch {epoch_num:03d} lr={lr_now:.3g} loss={avg_loss:.6g} "
            f"chamfer={avg_chamfer:.6g} l2={avg_l2:.6g} windows={windows_done} "
            f"train_time={train_elapsed:.1f}s "
            f"order={shuffled_order} "
            f"| log_E=[{stab['logE_min']:.3g},{stab['logE_max']:.3g}] "
            f"nu=[{stab['nu_min']:.3g},{stab['nu_max']:.3g}] "
            f"det_F=[{stab['det_F_min']:.3g},{stab['det_F_max']:.3g}] "
            f"sigma_min={stab['sigma_min']:.3g} cond_max={stab['cond_max']:.3g} "
            f"F_abs_max={stab['abs_max']:.3g} grad_max={stab['grad_max']:.3g} "
            f"nan_grad_windows={stab['nan_grad_windows']}/{windows_done}",
            flush=True,
        )
        history.append({
            "epoch": epoch_num,
            "lr": float(lr_now),
            "loss": avg_loss,
            "chamfer": avg_chamfer,
            "l2": avg_l2,
            "windows": windows_done,
            "episode_order": shuffled_order,
            "train_time_s": float(train_elapsed),
            "stability": {
                "det_F_min": stab["det_F_min"], "det_F_max": stab["det_F_max"],
                "sigma_min_F": stab["sigma_min"], "sigma_max_F": stab["sigma_max"],
                "cond_F_max": stab["cond_max"],
                "F_fro_max": stab["fro_max"], "F_abs_max": stab["abs_max"],
                "log_E_min": stab["logE_min"], "log_E_max": stab["logE_max"],
                "nu_min": stab["nu_min"], "nu_max": stab["nu_max"],
                "grad_norm_max": stab["grad_max"],
                "grad_norm_mean": grad_norm_mean,
                "nan_grad_windows": stab["nan_grad_windows"],
            },
        })

        if plot_every > 0 and epoch_num % plot_every == 0:
            _save_loss_plots(history, plots_dir)

        # `gvc_latest.pt` — refreshed every epoch (crash-safe resumption point).
        latest_path = ckpt_dir / "gvc_latest.pt"
        torch.save(
            {
                "epoch": epoch_num,
                "model": corrector.state_dict(),
                "optimizer": opt.state_dict(),
                "cfg": OmegaConf.to_container(cfg, resolve=True),
                "history": history,
                "best_epoch": best_epoch_num,
                "best_loss": best_loss,
            },
            latest_path,
        )

        # Periodic checkpoint: save when either checkpoint_every or
        # validation_every fires (dedupe naturally since both write the same
        # filename), and always on the final epoch.  validate_RfD.py iterates
        # these and computes val curves post-hoc.
        ckpt_every = int(cfg.train.get("checkpoint_every", 0) or 0)
        val_every  = int(cfg.train.get("validation_every", 0) or 0)
        is_final   = epoch_num == int(cfg.train.epochs)
        save_due   = (ckpt_every > 0 and epoch_num % ckpt_every == 0) \
                  or (val_every  > 0 and epoch_num % val_every  == 0) \
                  or is_final
        if save_due:
            ckpt_path = ckpt_dir / f"gvc_epoch_{epoch_num:04d}.pt"
            # Weights only; the sweep takes the epoch from the filename.
            torch.save({"model": corrector.state_dict()}, ckpt_path)
            print(f"[gvc] wrote {ckpt_path}", flush=True)

        # Best-by-train checkpoint — overwrite whenever avg_loss improves.
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch_num = epoch_num
            best_path = ckpt_dir / "gvc_best_train.pt"
            torch.save({"model": corrector.state_dict()}, best_path)
            print(f"[gvc] new best train at epoch {epoch_num:03d} (loss={avg_loss:.6g}) → {best_path}", flush=True)

        # Viz: every viz_every (and final) epoch, render one mp4 per train/val
        # Train-only viz: every viz_every (and final) epoch, render one mp4
        # per train episode → videos/train/epoch_NNNN/<ep_label>.mp4 (no-grad).
        # Validation videos are produced by validate_RfD.py over the saved
        # gvc_epoch_NNNN.pt checkpoints.
        viz_every = int(cfg.train.get("viz_every", 0) or 0)
        if viz_dir is not None and viz_every > 0 and (
            epoch_num % viz_every == 0 or epoch_num == int(cfg.train.epochs)
        ):
            sub_dir = viz_dir / "train" / f"epoch_{epoch_num:04d}"
            for viz_ctx in episodes:
                _render_gvc_epoch_video(
                    engine=viz_ctx["engine"],
                    refiner=refiner,
                    corrector=corrector,
                    ep=viz_ctx["ep"],
                    cfg=cfg,
                    kinematic_ids=viz_ctx["kinematic_ids"],
                    n_full_windows=viz_ctx["n_full_windows"],
                    tail_frames=viz_ctx["tail_frames"],
                    k=k,
                    rollout_steps=rollout_steps,
                    total_outer_steps=viz_ctx["total_outer_steps"],
                    viz_dir=sub_dir,
                    epoch_num=epoch_num,
                    cfg_train_loss=avg_loss,
                    log_prefix="[gvc] train ",
                    ep_label=viz_ctx["label"],
                    kind="train",
                    first_half_only=True,  # render exactly what train() optimizes
                )

        # Per-epoch total wall time (training + ckpt I/O + viz).
        epoch_elapsed = time.perf_counter() - epoch_t0
        history[-1]["epoch_time_s"] = float(epoch_elapsed)
        print(f"[gvc] epoch {epoch_num:03d} done in {epoch_elapsed:.1f}s "
              f"(train {train_elapsed:.1f}s)", flush=True)

    if is_main_process():
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))
        _save_loss_plots(history, plots_dir)
        print(
            f"[gvc] training complete; checkpoints in {ckpt_dir} "
            f"(gvc_latest.pt, gvc_best_train.pt, gvc_epoch_NNNN.pt) "
            f"+ plots at {plots_dir} "
            f"(best_train epoch {best_epoch_num:03d}, loss={best_loss:.6g}). "
            f"Run validate_RfD.py to compute val curves and add val videos.",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GridVelocityCorrector.")
    p.add_argument("--config", default="configs/train_RfD.yaml", help="RfD (GVC) training config YAML")
    p.add_argument("--refiner-checkpoint", default=None, help="override cfg.gvc.refiner_checkpoint")
    p.add_argument("--episode-root", default=None, help="override cfg.dataset.train_roots (single)")
    args, overrides = p.parse_known_args()
    bad = [o for o in overrides if "=" not in o or o.startswith("-")]
    if bad:
        p.error(f"unrecognized argument(s): {bad}")
    args.cli_overrides = overrides  # OmegaConf dotlist (e.g. train.output_dir=foo)
    return args


if __name__ == "__main__":
    try:
        train(parse_args())
    finally:
        cleanup_distributed()
