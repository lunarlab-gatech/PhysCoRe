"""
Training helpers shared by the MfM entrypoints.
"""

from __future__ import annotations

import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler


from physcore.particle_flow import episode_runtime as ert


from physcore.model_MfM import Refiner
from physcore.particle_flow.rollout import RandomMaterialGuessInitializer
from physcore.particle_flow.runtime import (
    allreduce_gradients,
    get_rank,
    get_world_size,
    is_distributed,
)


def defaults(cfg):
    cfg.setdefault("dataset", {})
    d = cfg.dataset
    d.setdefault("validation_roots", [])
    d.setdefault("history_steps", 1)
    d.setdefault("sample_stride", 8)
    d.setdefault("observation_views", "all")
    d.setdefault("use_noisy_observation", True)
    d.setdefault("require_observation", True)
    d.setdefault("interaction_only", True)
    d.setdefault("min_interaction_offset", 40)
    d.setdefault("min_future_flow_l2_mean", 0.0)
    d.setdefault("min_future_flow_l2_max", 0.0)
    d.setdefault("sort_by_motion", "descending")
    d.setdefault("cache_size", 2)
    d.setdefault("default_ground_height", 0.02)
    d.setdefault("real_world_domain_center", [0.5, 0.5, 0.2])
    d.setdefault("max_train_windows", None)
    d.setdefault("max_val_windows", None)
    cfg.setdefault("train", {})
    t = cfg.train
    t.setdefault("batch_size", 1)
    t.setdefault("num_workers", 0)
    t.setdefault("shuffle", False)
    t.setdefault("pin_memory", False)
    t.setdefault("weight_decay", 0.0)
    t.setdefault("adam_beta1", 0.9)
    t.setdefault("adam_beta2", 0.999)
    t.setdefault("grad_clip", 1.0)
    t.setdefault("precision", "fp32")
    t.setdefault("checkpoint_every", 0)
    t.setdefault("best_checkpoint_metric", "logE_shape")
    t.setdefault("log_every_epochs", 1)
    t.setdefault("save_latest_every_epoch", True)
    t.setdefault("tbptt_chunks", 60)
    t.setdefault("update_every", 10)
    t.setdefault("supervision_fraction", 0.5)
    t.setdefault("lr_decay_start", None)
    t.setdefault("lr_decay_end", None)
    t.setdefault("final_lr", None)
    t.setdefault("lr_warmup_epochs", 0)
    t.setdefault("warmup_start_lr", None)
    t.setdefault("validation_every", 1)
    cfg.setdefault("loss", {})
    cfg.loss.setdefault("material_weight", 1.0)
    cfg.loss.setdefault("logE_shape_weight", 0.0)
    cfg.loss.setdefault("logE_mean_weight", 1.0)
    cfg.loss.setdefault("nu_weight", 1.0)
    cfg.loss.setdefault("logE_std_weight", 0.0)
    cfg.loss.setdefault("latent_logE_contrast_weight", 0.0)
    cfg.loss.setdefault("latent_logE_contrast_max_particles", 2048)
    cfg.loss.setdefault("material_confidence_log_weight", 1.0)
    cfg.loss.setdefault("plasticity_weight", 0.0)
    cfg.setdefault("material_guess", {
        "mode": "uniform",
        "log_E_range": [8.735597610473633, 8.735597610473633],
        "nu_range": [0.3600001335144043, 0.3600001335144043],
        "log_E_clamp_range": [5.0, 11.0],
        "nu_clamp_range": [0.05, 0.45],
    })
    return cfg




class _ForwardFeaturesAdapter(nn.Module):
    """Routes Refiner.forward_features through nn.Module.__call__ so DDP can
    install gradient-sync hooks on it."""

    def __init__(self, refiner: Refiner) -> None:
        super().__init__()
        self.refiner = refiner

    def forward(self, x, cur, prev, cache, state):
        return self.refiner.forward_features(x, cur, prev, cache, state)


_ddp_wrapper: Optional[nn.Module] = None


def _fwd_features(model: Refiner, *args):
    if _ddp_wrapper is not None:
        return _ddp_wrapper(*args)
    return model.forward_features(*args)


def sync_and_step(model, opt, cfg) -> None:
    if _ddp_wrapper is None:
        allreduce_gradients(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
    opt.step()


_canonical_cache_dict: Dict[str, Dict[str, object]] = {}


def _model_cache_shared(model, canonical: torch.Tensor, source_name: str = ""):
    if source_name and source_name in _canonical_cache_dict:
        return _canonical_cache_dict[source_name]
    c0 = canonical[:1] if canonical.ndim == 3 else canonical.unsqueeze(0)
    cache = model.cache(c0)
    if source_name:
        _canonical_cache_dict[source_name] = cache
    return cache


def supervision_end_frame(num_frames: int, cfg) -> int:
    frac = float(cfg.train.get("supervision_fraction", 0.5))
    return max(1, min(int(num_frames) - 1, int(round((int(num_frames) - 1) * frac))))


def epoch_lr(cfg, epoch: int) -> float:
    """LR schedule with three phases (warmup -> constant -> linear decay).

    1. epoch in [1, lr_warmup_epochs]:  linear ramp from warmup_start_lr -> lr
    2. epoch in (lr_warmup_epochs, lr_decay_start]:  constant at lr
    3. epoch in (lr_decay_start, lr_decay_end]:  linear decay to final_lr
    4. epoch > lr_decay_end:  constant at final_lr

    Setting `lr_warmup_epochs <= 0` disables warmup; setting any of
    `final_lr / lr_decay_start / lr_decay_end` to None disables decay.
    """
    base = float(cfg.train.lr)
    warmup = int(cfg.train.get("lr_warmup_epochs", 0) or 0)
    if warmup > 0 and epoch <= warmup:
        start_lr = float(cfg.train.get("warmup_start_lr", base * 0.01))
        a = float(epoch) / float(warmup)
        return start_lr + a * (base - start_lr)
    final = cfg.train.get("final_lr", None)
    decay_start = cfg.train.get("lr_decay_start", None)
    decay_end = cfg.train.get("lr_decay_end", None)
    if final is None or decay_start is None or decay_end is None:
        return base
    decay_start, decay_end, final = int(decay_start), int(decay_end), float(final)
    if epoch <= decay_start:
        return base
    if epoch >= decay_end:
        return final
    a = (float(epoch) - float(decay_start)) / max(float(decay_end - decay_start), 1.0)
    if bool(cfg.train.get("cosine_decay", False)):
        return final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * a))
    return base + a * (final - base)


def set_optimizer_lr(opt, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = float(lr)


def _source_group_size(dataset, cfg) -> int:
    """Return the augmentation-group size if 'same_source' batching is enabled
    in the config, else 0. Episodes within a source must be consecutive in the
    dataset's index ordering (which is the order of `train_roots` in the YAML)."""
    if str(cfg.train.get("batch_grouping", "")) != "same_source":
        return 0
    g = int(cfg.train.get("source_group_size", 0))
    if g <= 0:
        return 0
    n = len(dataset)
    if n % g != 0:
        raise ValueError(
            f"dataset length {n} is not divisible by source_group_size {g}; "
            f"either fix train_roots ordering or set batch_grouping to '' (default)."
        )
    return g


def local_epoch_indices(dataset, cfg, epoch: int) -> list[int]:
    n = len(dataset)
    grp = _source_group_size(dataset, cfg)
    shuffle = bool(cfg.train.get("shuffle", False))

    if grp > 0:
        # Group dataset into (n_groups, grp) source-groups. At each "step", all
        # ranks process the SAME source â€” rank r gets augmentation r % grp.
        # This guarantees identical N per rank per step (no DDP workload
        # imbalance / NCCL timeout when source sizes vary heavily across
        # objects). Effective global batch = grp (8 ranks Ã— 1 aug each).
        # Requires train.batch_size = 1 in the config.
        n_groups = n // grp
        group_order = list(range(n_groups))
        if shuffle:
            g = torch.Generator()
            g.manual_seed(int(cfg.train.seed) + int(epoch))
            group_order = torch.randperm(n_groups, generator=g).tolist()
        world = get_world_size() if is_distributed() else 1
        rank = get_rank() if is_distributed() else 0
        # Each epoch must visit ALL grp augmentations per source group. With
        # `world` ranks each picking one aug per step, we need ceil(grp/world)
        # passes per group. When grp == world this reduces to one pass with
        # aug = rank (original behavior).
        n_passes = (grp + world - 1) // world
        # Per-epoch random pass-offset so different ranks see different augs
        # across epochs (still cover all grp every epoch).
        if shuffle:
            g2 = torch.Generator()
            g2.manual_seed(int(cfg.train.seed) + int(epoch) * 1009 + 17)
            offset = int(torch.randint(grp, (1,), generator=g2).item())
        else:
            offset = 0
        return [
            gi * grp + (rank + s * world + offset) % grp
            for gi in group_order
            for s in range(n_passes)
        ]

    if not is_distributed():
        indices = list(range(n))
        if shuffle:
            g = torch.Generator()
            g.manual_seed(int(cfg.train.seed) + int(epoch))
            indices = torch.randperm(n, generator=g).tolist()
        return indices
    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
        drop_last=False,
    )
    sampler.set_epoch(epoch)
    return [int(i) for i in sampler]


def gather_rows(rows: list[Dict[str, object]]) -> list[Dict[str, object]]:
    if not is_distributed():
        return rows
    gathered = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, rows)
    return [row for rank_rows in gathered for row in rank_rows]


def average_rows(rows: list[Dict[str, object]]) -> Dict[str, float]:
    rows = gather_rows(rows)
    if not rows:
        return {}
    # Some rows may be missing keys, compute weighted mean only
    # over the rows that actually contain each key.
    all_keys: set = set()
    for r in rows:
        for k, v in r.items():
            if not k.startswith("_") and not isinstance(v, list):
                all_keys.add(k)
    avg: Dict[str, float] = {}
    for k in all_keys:
        tw = 0.0
        tv = 0.0
        for r in rows:
            if k in r:
                try:
                    val = float(r[k])
                except (TypeError, ValueError):
                    continue
                w = float(r.get("_samples", 1.0))
                tw += w
                tv += val * w
        if tw > 0.0:
            avg[k] = tv / tw
    avg["episodes"] = sum(float(r.get("_samples", 1.0)) for r in rows)
    return avg


def average_chunk_traces(rows: list[Dict[str, object]]) -> Dict[str, list[float]]:
    rows = [r for r in gather_rows(rows) if r.get("chunk_logE_mean")]
    if not rows:
        return {}
    n = min(len(r["chunk_logE_mean"]) for r in rows)
    out: Dict[str, list[float]] = {}
    series_names = ["chunk_logE_mean", "chunk_logE_mae"]
    if rows[0].get("chunk_obs_to_pred"):
        series_names += ["chunk_obs_to_pred", "chunk_tracking_l2"]
    for name in series_names:
        vals = []
        for i in range(n):
            denom = sum(float(r.get("_samples", 1.0)) for r in rows if len(r.get(name, [])) > i)
            vals.append(sum(float(r[name][i]) * float(r.get("_samples", 1.0)) for r in rows if len(r.get(name, [])) > i) / max(denom, 1.0))
        out[name] = vals
    out["chunk_frames"] = [float(v) for v in rows[0].get("chunk_frames", [])[:n]]
    return out


def fmt_series(xs: list[float]) -> str:
    return "[" + ", ".join(f"{float(x):.4f}" for x in xs) + "]"


def plot_loss_curves(history: list[Dict[str, object]], out_path: Path) -> None:
    if not history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"skipping loss plot ({e})", flush=True)
        return
    def series(key):
        return [(int(row.get("epoch", i + 1)), float(row[key])) for i, row in enumerate(history) if key in row]
    has_plasticity = any("plasticity_acc_elastic" in row for row in history)
    logE_curves = [
        ("train logE MAE", series("logE"), "C0", "-", None),
        ("val logE MAE", series("val_logE"), "C0", "--", "o"),
    ]
    nu_curves = [
        ("train nu MAE", series("nu"), "C1", "-", None),
        ("val nu MAE", series("val_nu"), "C1", "--", "o"),
    ]
    plast_curves = []
    if has_plasticity:
        train_bal = [(int(row.get("epoch", i + 1)), (float(row["plasticity_acc_elastic"]) + float(row["plasticity_acc_plastic"])) / 2) for i, row in enumerate(history) if "plasticity_acc_elastic" in row and "plasticity_acc_plastic" in row]
        val_bal = [(int(row.get("epoch", i + 1)), float(row["val_plasticity_bal"])) for i, row in enumerate(history) if "val_plasticity_bal" in row]
        plast_curves = [
            ("train balanced acc", train_bal, "C0", "-", None),
            ("val balanced acc", val_bal, "C0", "--", "o"),
        ]
    n_plots = 1 + (1 if nu_curves else 0) + (1 if plast_curves else 0)
    if not any(c[1] for c in logE_curves):
        return
    fig, axes = plt.subplots(n_plots, 1, figsize=(9, 4 * n_plots), squeeze=False)
    ax_idx = 0
    ax = axes[ax_idx, 0]
    for label, data, color, ls, marker in logE_curves:
        if not data:
            continue
        xs, ys = zip(*data)
        ax.plot(xs, ys, label=label, color=color, linewidth=1.5, linestyle=ls, marker=marker, markersize=4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("logE MAE")
    ax.set_title("logE loss")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    if nu_curves:
        ax_idx += 1
        ax = axes[ax_idx, 0]
        for label, data, color, ls, marker in nu_curves:
            if not data:
                continue
            xs, ys = zip(*data)
            ax.plot(xs, ys, label=label, color=color, linewidth=1.5, linestyle=ls, marker=marker, markersize=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("nu MAE")
        ax.set_title("nu loss")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    if plast_curves:
        ax_idx += 1
        ax = axes[ax_idx, 0]
        for label, data, color, ls, marker in plast_curves:
            if not data:
                continue
            xs, ys = zip(*data)
            ax.plot(xs, ys, label=label, color=color, linewidth=1.5, linestyle=ls, marker=marker, markersize=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("balanced accuracy")
        ax.set_ylim(0, 1)
        ax.set_title("plasticity classification")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)


def val_indices(dataset) -> list[int]:
    if not is_distributed():
        return list(range(len(dataset)))
    sampler = DistributedSampler(dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=False, drop_last=False)
    return [int(i) for i in sampler]


def evaluate_episode(model: Refiner, ep: Dict[str, object], cfg, seed: int) -> Dict[str, float]:
    """Mirror the training-time forward pass exactly.

    The training loop accumulates `update_every` frames of per-frame features
    into a window and feeds the 4D `(B, N, K, F)` window to
    `model.forward_features`.  The model's temporal aggregator (configured
    with `temporal_conv: 9`, `attention_window: 9`) expects K>=window-size to
    be meaningful; collapsing to K=1 at eval time produces a different
    inference regime than training and makes the val metric un-improvable.

    Build the same feature window here so train and val see the same input
    shape.
    """
    coords, vel, Fm = ep["coords"], ep["particle_v"], ep["particle_F"]
    start, end = 0, supervision_end_frame(coords.shape[0], cfg)
    canonical = coords[start:start + 1]
    cache = _model_cache_shared(model, canonical, episode_source_name(ep))
    state: Optional[object] = None
    material = material_guess(ep, cfg, start, seed)
    correction = torch.zeros_like(canonical)
    mask = torch.zeros(canonical.shape[:2], device=coords.device, dtype=torch.bool)
    last, conf, latent = material, None, None
    feature_window: list[torch.Tensor] = []
    for frame in range(start + 1, end + 1):
        cur = coords[frame:frame + 1]
        prev = coords[frame - 1:frame]
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
                vel[frame:frame + 1],
                Fm[frame:frame + 1],
                correction,
                mask,
                obs,
            )
        )
        if (frame - start) % int(cfg.train.update_every):
            continue
        x_window = torch.stack(feature_window, dim=2)  # B,N,K,F  (K = update_every)
        with amp_context(cfg, coords.device):
            pred = model.forward_features(x_window, cur, prev, cache, state)
        feature_window.clear()
        state, last = pred["state"], pred["material"]
        conf, latent = pred.get("material_confidence"), pred.get("latent")
        plast = pred.get("plasticity")
    m = update_loss(last, ep, model, cfg, conf, latent, plast)
    return {k: float(v.detach().item()) for k, v in m.items()}


def evaluate(model: Refiner, val_loader, cfg, device, rank: int) -> Optional[Dict[str, float]]:
    if val_loader is None:
        return None
    was_training = model.training
    model.eval()
    rows: list[Dict[str, object]] = []
    indices = val_indices(val_loader.dataset)
    seed_base = int(cfg.train.seed) + rank * 100000
    with torch.no_grad():
        for offset, j in enumerate(indices):
            ep = load_episode(val_loader.dataset, j, device)
            row = evaluate_episode(model, ep, cfg, seed_base + offset + int(j))
            row["_samples"] = 1.0
            rows.append(row)
    if was_training:
        model.train()
    return average_rows(rows) or None

def amp_context(cfg, device):
    if str(cfg.train.get("precision", "fp32")).lower() == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def norm_material(m: torch.Tensor, model: Refiner) -> torch.Tensor:
    return model.material_norm(m.float())


def load_episode(dataset, i: int, device: torch.device) -> Dict[str, object]:
    ep = ert._load_episode_tensors(dataset[i], device)
    ep["root_dir"] = dataset.episodes[i].root_dir
    return ep


def episode_is_elastic(ep: Dict[str, object]) -> bool:
    root_dir = ep.get("root_dir", "")
    parent_name = Path(root_dir).parent.name
    return parent_name.endswith("_elastic") or "_elastic_" in parent_name


def episode_source_name(ep: Dict[str, object]) -> str:
    root_dir = ep.get("root_dir", "")
    if not root_dir:
        return ""
    return Path(root_dir).parent.name


def material_guess(ep: Dict[str, object], cfg, frame: int, seed: int) -> torch.Tensor:
    return ert._sample_material_guess(
        RandomMaterialGuessInitializer(cfg.material_guess),
        ep["coords"][frame],
        ep["gt_log_E"],
        ep["gt_nu"],
        seed=seed,
    ).unsqueeze(0)


def actual_control_points(ep: Dict[str, object], frame: int, max_controls: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    pts = ep.get("controller_grid_points", ep.get("r_coords", None))
    if pts is None or max_controls <= 0:
        return torch.zeros(max_controls, 3, device=device, dtype=dtype), torch.zeros(max_controls, device=device, dtype=dtype)
    x = pts[frame].to(device=device, dtype=dtype)
    out = torch.zeros(max_controls, 3, device=device, dtype=dtype)
    mask = torch.zeros(max_controls, device=device, dtype=dtype)
    m = min(max_controls, int(x.shape[0]))
    if m > 0:
        out[:m] = x[:m]
        mask[:m] = 1.0
    return out, mask


def observed_features(ep: Dict[str, object], frame: int, canonical: torch.Tensor, cur: torch.Tensor,
                      max_controls: int, fixed_tracked: bool = False, tracked_disp_mode: str = "incremental",
                      control_disp_mode: str = "absolute", use_persistent_tracks: bool = True,
                      persistent_track_use_motion_valid: bool = False, aggregate_controls: bool = False) -> Dict[str, torch.Tensor]:
    n, device, dtype = cur.shape[0], cur.device, cur.dtype
    tracked_disp = torch.zeros(n, 3, device=device, dtype=dtype)
    tracked_count = torch.zeros(n, 1, device=device, dtype=dtype)
    tracked_n = min(int(ep.get("tracked_particle_count", 0) or 0), n)
    if bool(use_persistent_tracks) and bool(ep.get("include_tracked_surface_points", False)) and tracked_n > 0:
        ids_k = torch.arange(tracked_n, device=device, dtype=torch.long)
        pts_k = ep["coords"][int(frame), :tracked_n].to(device=device, dtype=dtype)
        if str(tracked_disp_mode).lower() in {"canonical", "absolute", "from_canonical"}:
            ref_frame = 0
            ref = canonical[:tracked_n]
        else:
            ref_frame = max(int(frame) - 1, 0)
            ref = ep["coords"][ref_frame, :tracked_n].to(device=device, dtype=dtype)
        valid_k = torch.isfinite(pts_k).all(dim=-1) & torch.isfinite(ref).all(dim=-1)
        if bool(persistent_track_use_motion_valid) and "particle_motion_valid" in ep:
            motion_valid = ep["particle_motion_valid"][:, :tracked_n].to(device=device).bool()
            valid_k = valid_k & motion_valid[int(frame)] & motion_valid[int(ref_frame)]
        if valid_k.any():
            ids_k = ids_k[valid_k]
            tracked_disp[ids_k] = pts_k[valid_k] - ref[valid_k]
            tracked_count[ids_k, 0] = 1.0
    else:
        obs_data = ep.get("observation_data", None)
        if obs_data:
            ids = obs_data["object_particle_ids"]
            pts = obs_data["object_points_clean"]
            valid = obs_data["object_valid_mask"]
            track_frame = 0 if fixed_tracked else frame
            ids_f, valid_f = ids[:, track_frame].reshape(-1), valid[:, track_frame].reshape(-1)
            pts_f = pts[:, frame].reshape(-1, 3)
            keep = valid_f & (ids_f >= 0) & (ids_f < n)
            if keep.any():
                ids_k, pts_k = ids_f[keep].long(), pts_f[keep].to(device=device, dtype=dtype)
                if str(tracked_disp_mode).lower() in {"canonical", "absolute", "from_canonical"}:
                    ref = canonical
                else:
                    ref = ep["coords"][max(int(frame) - 1, 0)].to(device=device, dtype=dtype)
                tracked_disp.index_add_(0, ids_k, pts_k - ref[ids_k])
                tracked_count.index_add_(0, ids_k, torch.ones(ids_k.numel(), 1, device=device, dtype=dtype))
    tracked_mask = tracked_count[:, 0] > 0
    tracked_disp = tracked_disp / tracked_count.clamp_min(1.0)
    ctrlt, cmask = actual_control_points(ep, frame, max_controls, device, dtype)
    if str(control_disp_mode).lower() in {"incremental", "previous", "delta"}:
        ctrl_ref, cmask_ref = actual_control_points(ep, max(int(frame) - 1, 0), max_controls, device, dtype)
    else:
        ctrl_ref, cmask_ref = actual_control_points(ep, 0, max_controls, device, dtype)
    if aggregate_controls:
        all_pts = ep.get("controller_grid_points", ep.get("r_coords", None))
        if all_pts is not None and all_pts.shape[1] > 0:
            pts_now = all_pts[frame].to(device=device, dtype=dtype)
            if str(control_disp_mode).lower() in {"incremental", "previous", "delta"}:
                pts_ref = all_pts[max(int(frame) - 1, 0)].to(device=device, dtype=dtype)
            else:
                pts_ref = all_pts[0].to(device=device, dtype=dtype)
            dist = (pts_now[None] - cur[:, None]).norm(dim=-1).clamp_min(1e-6)
            w = torch.exp(-dist / 0.04)
            w_sum = w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            control_vecs = (w[..., None] * (pts_now[None] - cur[:, None])).sum(dim=1) / w_sum
            control_disp = (w[..., None] * (pts_now - pts_ref)[None]).sum(dim=1) / w_sum
            control_mask = w_sum.squeeze(-1) / w.shape[-1]
        else:
            control_vecs = torch.zeros(n, 3, device=device, dtype=dtype)
            control_disp = torch.zeros(n, 3, device=device, dtype=dtype)
            control_mask = torch.zeros(n, 1, device=device, dtype=dtype)
        return {
            "tracked_disp": tracked_disp[None],
            "tracked_mask": tracked_mask[None],
            "control_vecs": control_vecs[None],
            "control_disp": control_disp[None],
            "control_mask": control_mask[..., None][None] if control_mask.ndim == 1 else control_mask[None],
        }
    control_vecs = (ctrlt[None] - cur[:, None]).flatten(1)
    control_disp = (ctrlt - ctrl_ref).flatten()[None].expand(n, -1)
    control_mask = (cmask * cmask_ref)[None].expand(n, -1)
    return {
        "tracked_disp": tracked_disp[None],
        "tracked_mask": tracked_mask[None],
        "control_vecs": control_vecs[None],
        "control_disp": control_disp[None],
        "control_mask": control_mask[None],
    }


def point_material_loss(pred: torch.Tensor, gt: torch.Tensor, model: Refiner, cfg=None) -> torch.Tensor:
    pred_n, gt_n = norm_material(pred, model), norm_material(gt, model)
    if cfg is None or float(cfg.loss.get("logE_shape_weight", 0.0)) <= 0.0:
        return torch.nn.functional.smooth_l1_loss(pred_n, gt_n, beta=0.02, reduction="none").mean(dim=-1)
    shape_w = float(cfg.loss.get("logE_shape_weight", 1.0))
    mean_w = float(cfg.loss.get("logE_mean_weight", 0.05))
    nu_w = float(cfg.loss.get("nu_weight", 1.0))
    pred_logE, gt_logE = pred_n[..., 0], gt_n[..., 0]
    pred_mean, gt_mean = pred_logE.mean(dim=1, keepdim=True), gt_logE.mean(dim=1, keepdim=True)
    shape_loss = torch.nn.functional.smooth_l1_loss(pred_logE - pred_mean, gt_logE - gt_mean, beta=0.02, reduction="none")
    mean_loss = torch.nn.functional.smooth_l1_loss(pred_mean, gt_mean, beta=0.02, reduction="none").expand_as(shape_loss)
    nu_loss = torch.nn.functional.smooth_l1_loss(pred_n[..., 1], gt_n[..., 1], beta=0.02, reduction="none")
    return shape_w * shape_loss + mean_w * mean_loss + nu_w * nu_loss


def logE_std_loss(pred: torch.Tensor, gt: torch.Tensor, cfg=None) -> torch.Tensor:
    if cfg is None or float(cfg.loss.get("logE_std_weight", 0.0)) <= 0.0:
        return pred[..., 0].new_zeros(())
    pred_std = pred[..., 0].float().flatten(1).std(dim=1, unbiased=False)
    gt_std = gt[..., 0].float().flatten(1).std(dim=1, unbiased=False)
    return torch.nn.functional.smooth_l1_loss(pred_std, gt_std, beta=0.02)



def latent_logE_contrast_loss(latent: torch.Tensor, gt_logE: torch.Tensor, model: Refiner, cfg=None) -> torch.Tensor:
    if cfg is None or float(cfg.loss.get("latent_logE_contrast_weight", 0.0)) <= 0.0:
        return latent.new_zeros(())
    if latent.ndim != 3 or latent.shape[0] < 2:
        return latent.new_zeros(())
    b = latent.shape[0]
    z = torch.nn.functional.normalize(latent.float().mean(dim=1), dim=-1, eps=1.0e-6)
    gt = gt_logE.to(z.device).float()
    if gt.ndim == 2:
        gt = gt.mean(dim=1)
    y = ((gt - model.log_E_min) / (model.log_E_max - model.log_E_min)).clamp(0, 1)
    iu, ju = torch.triu_indices(b, b, offset=1, device=z.device)
    cos = (z[iu] * z[ju]).sum(dim=-1).clamp(-1.0, 1.0)
    target_cos = 1.0 - 2.0 * (y[iu] - y[ju]).abs()
    return torch.nn.functional.smooth_l1_loss(cos, target_cos, beta=0.1)

def material_loss(pred: torch.Tensor, gt: torch.Tensor, model: Refiner, cfg=None, confidence: Optional[torch.Tensor] = None, latent: Optional[torch.Tensor] = None) -> torch.Tensor:
    point_loss = point_material_loss(pred, gt, model, cfg)
    std_loss = logE_std_loss(pred, gt, cfg)
    contrast_loss = latent_logE_contrast_loss(latent, gt[..., 0], model, cfg) if latent is not None else pred[..., 0].new_zeros(())
    std_w = 0.0 if cfg is None else float(cfg.loss.get("logE_std_weight", 0.0))
    contrast_w = 0.0 if cfg is None else float(cfg.loss.get("latent_logE_contrast_weight", 0.0))
    if confidence is None:
        return point_loss.mean() + std_w * std_loss + contrast_w * contrast_loss
    reg = float(cfg.loss.get("material_confidence_log_weight", 1.0))
    if confidence.shape[-1] == 2:
        pred_n, gt_n = norm_material(pred, model), norm_material(gt, model)
        logE_loss = torch.nn.functional.smooth_l1_loss(pred_n[..., 0], gt_n[..., 0], beta=0.02, reduction="none")
        nu_loss = torch.nn.functional.smooth_l1_loss(pred_n[..., 1], gt_n[..., 1], beta=0.02, reduction="none")
        conf_E = confidence[..., 0]
        conf_nu = confidence[..., 1]
        weighted = (logE_loss * conf_E - reg * conf_E.log() + nu_loss * conf_nu - reg * conf_nu.log()).mean()
        return weighted + std_w * std_loss + contrast_w * contrast_loss
    conf = confidence.squeeze(-1).to(point_loss.dtype).clamp_min(1.0e-6)
    return (point_loss * conf - reg * torch.log(conf)).mean() + std_w * std_loss + contrast_w * contrast_loss


def _per_class_mae(
    pred_logE: torch.Tensor,
    gt_logE: torch.Tensor,
    pred_nu: torch.Tensor,
    gt_nu: torch.Tensor,
    shell_count: int,
    out: Dict[str, torch.Tensor],
) -> None:
    """Split per-particle MAE into shell [0, shell_count) vs interior [shell_count, N).

    Shell = cotracker + depth-extras (particles that receive direct displacement
    signal in observed_features). Interior = the rest (rely on aggregated
    signal from shell neighbors via GraphUNet).
    """
    n = int(pred_logE.shape[-1])
    shell_count = max(0, min(int(shell_count), n))
    if shell_count > 0:
        out["logE_shell"] = (pred_logE[..., :shell_count] - gt_logE[..., :shell_count]).abs().mean()
        out["nu_shell"] = (pred_nu[..., :shell_count] - gt_nu[..., :shell_count]).abs().mean()
    if shell_count < n:
        out["logE_interior"] = (pred_logE[..., shell_count:] - gt_logE[..., shell_count:]).abs().mean()
        out["nu_interior"] = (pred_nu[..., shell_count:] - gt_nu[..., shell_count:]).abs().mean()


def _plasticity_class_weight(gt: torch.Tensor, elastic_ratio: float = 7.0) -> torch.Tensor:
    return torch.where(gt > 0.5, torch.ones_like(gt), torch.full_like(gt, elastic_ratio))


def plasticity_loss(plasticity_pred: Optional[torch.Tensor], ep: Dict[str, object], cfg=None) -> torch.Tensor:
    if plasticity_pred is None or cfg is None:
        return plasticity_pred.new_zeros(()) if plasticity_pred is not None else torch.zeros(())
    w = float(cfg.loss.get("plasticity_weight", 0.0))
    if w <= 0.0:
        return plasticity_pred.new_zeros(())
    gt = 1.0 if episode_is_elastic(ep) else 0.0
    target = plasticity_pred.new_full(plasticity_pred.shape, gt)
    sample_w = _plasticity_class_weight(target)
    bce = torch.nn.functional.binary_cross_entropy(plasticity_pred, target, reduction="none")
    return w * (bce * sample_w).mean()


def plasticity_loss_batch(plasticity_pred: Optional[torch.Tensor], eps: list, cfg=None) -> torch.Tensor:
    if plasticity_pred is None or cfg is None:
        return plasticity_pred.new_zeros(()) if plasticity_pred is not None else torch.zeros(())
    w = float(cfg.loss.get("plasticity_weight", 0.0))
    if w <= 0.0:
        return plasticity_pred.new_zeros(())
    gt = torch.tensor([1.0 if episode_is_elastic(ep) else 0.0 for ep in eps],
                       device=plasticity_pred.device, dtype=plasticity_pred.dtype).unsqueeze(-1)
    sample_w = _plasticity_class_weight(gt)
    bce = torch.nn.functional.binary_cross_entropy(plasticity_pred, gt, reduction="none")
    return w * (bce * sample_w).mean()


def update_loss(pred: torch.Tensor, ep: Dict[str, object], model: Refiner, cfg=None, confidence: Optional[torch.Tensor] = None, latent: Optional[torch.Tensor] = None, plasticity_pred: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    gt = torch.stack([ep["gt_log_E"], ep["gt_nu"]], dim=-1).unsqueeze(0)
    pred_logE, gt_logE = pred[..., 0].float(), gt[..., 0].to(pred.device).float()
    pred_nu, gt_nu = pred[..., 1].float(), gt[..., 1].to(pred.device).float()
    mat_loss = material_loss(pred, gt.to(pred.device), model, cfg, confidence, latent)
    plast_loss = plasticity_loss(plasticity_pred, ep, cfg)
    out = {
        "material": mat_loss + plast_loss,
        "material_raw": point_material_loss(pred, gt.to(pred.device), model, cfg).mean(),
        "logE_std_loss": logE_std_loss(pred, gt.to(pred.device), cfg),
        "latent_logE_contrast": latent_logE_contrast_loss(latent, gt[..., 0].to(pred.device), model, cfg) if latent is not None else pred[..., 0].new_zeros(()),
        "logE": (pred_logE - gt_logE).abs().mean(),
        "logE_shape": ((pred_logE - pred_logE.mean(dim=1, keepdim=True)) - (gt_logE - gt_logE.mean(dim=1, keepdim=True))).abs().mean(),
        "logE_mean": (pred_logE.mean(dim=1) - gt_logE.mean(dim=1)).abs().mean(),
        "nu": (pred_nu - gt_nu).abs().mean(),
        "std": pred_logE.std(unbiased=False),
        "plasticity_loss": plast_loss.detach(),
    }
    if plasticity_pred is not None:
        gt_label = 1.0 if episode_is_elastic(ep) else 0.0
        correct = 1.0 if (plasticity_pred.detach().mean().item() > 0.5) == (gt_label > 0.5) else 0.0
        out["plasticity_pred"] = plasticity_pred.detach().mean()
        out["plasticity_gt"] = plasticity_pred.new_tensor(gt_label)
        out["plasticity_acc"] = plasticity_pred.new_tensor(correct)
        if gt_label > 0.5:
            out["plasticity_acc_elastic"] = plasticity_pred.new_tensor(correct)
        else:
            out["plasticity_acc_plastic"] = plasticity_pred.new_tensor(correct)
    shell_count = int(ep.get("completed_shell_count", pred_logE.shape[-1]))
    _per_class_mae(pred_logE, gt_logE, pred_nu, gt_nu, shell_count, out)
    if confidence is not None:
        conf = confidence.detach().float().clamp_min(1.0e-6)
        out["conf"] = conf.mean()
        out["conf_log"] = conf.log().mean()
    return out


def material_guess_batch(eps: list[Dict[str, object]], cfg, frame: int, seeds: list[int]) -> torch.Tensor:
    return torch.cat([material_guess(ep, cfg, frame, seed) for ep, seed in zip(eps, seeds)], dim=0)


def observed_features_batch(eps: list[Dict[str, object]], frame: int, canonical: torch.Tensor, cur: torch.Tensor, cfg) -> Dict[str, torch.Tensor]:
    rows = [observed_features(
        ep, frame, canonical[i], cur[i], int(cfg.model.get("max_controls", 0)), bool(cfg.model.get("fixed_tracked_mask", False)), str(cfg.model.get("tracked_disp_mode", "incremental")), str(cfg.model.get("control_disp_mode", "absolute")), bool(cfg.model.get("use_persistent_tracks", True)), bool(cfg.model.get("persistent_track_use_motion_valid", False)), aggregate_controls=True
    ) for i, ep in enumerate(eps)]
    return {k: torch.cat([r[k] for r in rows], dim=0) for k in rows[0]}


def update_loss_batch(pred: torch.Tensor, eps: list[Dict[str, object]], model: Refiner, cfg=None, confidence: Optional[torch.Tensor] = None, latent: Optional[torch.Tensor] = None, plasticity_pred: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    gt = torch.stack([torch.stack([ep["gt_log_E"], ep["gt_nu"]], dim=-1) for ep in eps], dim=0).to(pred.device)
    pred_logE, gt_logE = pred[..., 0].float(), gt[..., 0].float()
    pred_nu, gt_nu = pred[..., 1].float(), gt[..., 1].float()
    mat_loss = material_loss(pred, gt, model, cfg, confidence, latent)
    plast_loss = plasticity_loss_batch(plasticity_pred, eps, cfg)
    out = {
        "material": mat_loss + plast_loss,
        "material_raw": point_material_loss(pred, gt, model, cfg).mean(),
        "logE_std_loss": logE_std_loss(pred, gt, cfg),
        "latent_logE_contrast": latent_logE_contrast_loss(latent, gt[..., 0], model, cfg) if latent is not None else pred[..., 0].new_zeros(()),
        "logE": (pred_logE - gt_logE).abs().mean(),
        "logE_shape": ((pred_logE - pred_logE.mean(dim=1, keepdim=True)) - (gt_logE - gt_logE.mean(dim=1, keepdim=True))).abs().mean(),
        "logE_mean": (pred_logE.mean(dim=1) - gt_logE.mean(dim=1)).abs().mean(),
        "nu": (pred_nu - gt_nu).abs().mean(),
        "std": pred_logE.flatten(1).std(dim=1, unbiased=False).mean(),
        "plasticity_loss": plast_loss.detach(),
    }
    if plasticity_pred is not None:
        gt_labels = torch.tensor([1.0 if episode_is_elastic(ep) else 0.0 for ep in eps],
                                  device=plasticity_pred.device)
        correct = ((plasticity_pred.detach().squeeze(-1) > 0.5) == (gt_labels > 0.5)).float()
        out["plasticity_pred"] = plasticity_pred.detach().mean()
        out["plasticity_gt"] = gt_labels.mean()
        out["plasticity_acc"] = correct.mean()
        elastic_mask = gt_labels > 0.5
        if elastic_mask.any():
            out["plasticity_acc_elastic"] = correct[elastic_mask].mean()
        if (~elastic_mask).any():
            out["plasticity_acc_plastic"] = correct[~elastic_mask].mean()
    shell_count = int(eps[0].get("completed_shell_count", pred_logE.shape[-1]))
    _per_class_mae(pred_logE, gt_logE, pred_nu, gt_nu, shell_count, out)
    if confidence is not None:
        conf = confidence.detach().float().clamp_min(1.0e-6)
        out["conf"] = conf.mean()
        out["conf_log"] = conf.log().mean()
    return out


def material_only_batch(model: Refiner, opt, eps: list[Dict[str, object]], cfg, seeds: list[int]) -> Dict[str, float]:
    aug_eps = []
    for i, ep in enumerate(eps):
        new_ep = {**ep}
        if bool(cfg.train.get("use_source_motion_valid", False)):
            src_mv = _load_source_motion_valid(ep, ep["coords"].device)
            if src_mv is not None:
                n_frames = min(src_mv.shape[0], ep["coords"].shape[0])
                mv = ep.get("particle_motion_valid", torch.ones_like(ep["coords"][:, :, 0], dtype=torch.bool)).clone()
                mv[:n_frames] = mv[:n_frames] & src_mv[:n_frames, :mv.shape[1]].to(mv.device)
                new_ep["particle_motion_valid"] = mv
        if bool(cfg.train.get("augment_z_rotation", False)):
            R = _random_z_rotation_and_flip(seeds[i] * 7 + 31, ep["coords"].device)
            new_ep["coords"] = ep["coords"] @ R.T
            cg = ep.get("controller_grid_points")
            if cg is not None:
                new_ep["controller_grid_points"] = cg @ R.T
        aug_eps.append(new_ep)
    eps = aug_eps
    coords = torch.stack([ep["coords"] for ep in eps], dim=0)
    vel = torch.stack([ep["particle_v"] for ep in eps], dim=0)
    Fm = torch.stack([ep["particle_F"] for ep in eps], dim=0)
    end = min(supervision_end_frame(ep["coords"].shape[0], cfg) for ep in eps)
    max_start = max(end - int(cfg.train.update_every) * 2, 0)
    if max_start > 0 and model.training and float(cfg.train.get("temporal_crop_fraction", 0.0)) > 0:
        grp_size = max(int(cfg.train.get("source_group_size", 1)), 1)
        group_idx = int(eps[0].get("episode_index", 0)) // grp_size
        start = int((int(cfg.train.seed) + group_idx * 997) % (max_start + 1))
    else:
        start = 0
    canonical = coords[:, start]
    cache = _model_cache_shared(model, coords[:, 0], episode_source_name(eps[0]))
    state: Optional[object] = None
    material = material_guess_batch(eps, cfg, start, seeds)
    correction = torch.zeros_like(canonical)
    mask = torch.zeros(canonical.shape[:2], device=coords.device, dtype=torch.bool)
    losses, last, chunks = [], material, 0
    chunk_frames, chunk_logE_mean, chunk_logE_mae = [], [], []
    feature_window = []
    track_drop_p = float(cfg.train.get("tracking_dropout", 0.0))
    for frame in range(start + 1, end + 1):
        cur, prev = coords[:, frame], coords[:, frame - 1]
        obs = observed_features_batch(eps, frame, canonical, cur, cfg)
        if obs is not None and track_drop_p > 0 and model.training:
            drop = torch.rand_like(obs["tracked_mask"].float()) < track_drop_p
            obs["tracked_mask"] = obs["tracked_mask"] & ~drop
            obs["tracked_disp"] = obs["tracked_disp"] * obs["tracked_mask"][..., None].float()
        track_jitter = float(cfg.train.get("tracking_jitter_std", 0.0))
        if obs is not None and track_jitter > 0 and model.training:
            noise = torch.randn_like(obs["tracked_disp"]) * track_jitter
            obs["tracked_disp"] = obs["tracked_disp"] + noise * obs["tracked_mask"][..., None].float()
            obs["control_vecs"] = obs["control_vecs"] + torch.randn_like(obs["control_vecs"]) * track_jitter
            obs["control_disp"] = obs["control_disp"] + torch.randn_like(obs["control_disp"]) * track_jitter
        feature_window.append(
            model.build_features(
                cur,
                prev,
                canonical,
                material,
                vel[:, frame],
                Fm[:, frame],
                correction,
                mask,
                obs,
            )
        )
        if (frame - start) % int(cfg.train.update_every):
            continue
        x_window = torch.stack(feature_window, dim=2)  # B,N,K,F
        with amp_context(cfg, coords.device):
            pred = _fwd_features(model, x_window, cur, prev, cache, state)
        feature_window.clear()

        state, last = pred["state"], pred["material"]
        gt_logE = torch.stack([ep["gt_log_E"] for ep in eps], dim=0).to(last.device)
        chunk_frames.append(float(frame))
        chunk_logE_mean.append(float(last[..., 0].float().mean().detach().item()))
        chunk_logE_mae.append(float((last[..., 0].float() - gt_logE).abs().mean().detach().item()))
        losses.append(update_loss_batch(last, eps, model, cfg, pred.get("material_confidence"), pred.get("latent"), pred.get("plasticity"))["material"])
        chunks += 1
        if chunks % int(cfg.train.tbptt_chunks) == 0:
            opt.zero_grad(set_to_none=True)
            torch.stack(losses).mean().backward()
            sync_and_step(model, opt, cfg)
            losses.clear()
            state = model.detach(state)
    if losses:
        opt.zero_grad(set_to_none=True)
        torch.stack(losses).mean().backward()
        sync_and_step(model, opt, cfg)
    with torch.no_grad():
        m = update_loss_batch(last, eps, model, cfg, pred.get("material_confidence") if "pred" in locals() else None, pred.get("latent") if "pred" in locals() else None, pred.get("plasticity") if "pred" in locals() else None)
    return {k: float(v.detach().item()) for k, v in m.items()} | {
        "chunks": float(chunks),
        "chunk_frames": chunk_frames,
        "chunk_logE_mean": chunk_logE_mean,
        "chunk_logE_mae": chunk_logE_mae,
    }


def _random_z_rotation_and_flip(seed: int, device) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    angle = torch.rand(1, generator=g).item() * 2 * 3.141592653589793
    c, s = math.cos(angle), math.sin(angle)
    R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], device=device, dtype=torch.float32)
    if torch.rand(1, generator=g).item() < 0.5:
        R[0] = -R[0]
    return R


_source_motion_valid_cache: Dict[str, torch.Tensor] = {}


def _load_source_motion_valid(ep: Dict[str, object], device) -> Optional[torch.Tensor]:
    src_dir = ep.get("source_episode_dir", "")
    if not src_dir:
        return None
    if src_dir in _source_motion_valid_cache:
        return _source_motion_valid_cache[src_dir].to(device)
    traj_path = os.path.join(src_dir, "episode_data.pt")
    if not os.path.exists(traj_path):
        _source_motion_valid_cache[src_dir] = None
        return None
    src = torch.load(traj_path, map_location="cpu", weights_only=False)
    mv = src.get("particle_motion_valid")
    _source_motion_valid_cache[src_dir] = mv
    return mv.to(device) if mv is not None else None


def material_only_episode(model: Refiner, opt, ep: Dict[str, object], cfg, seed: int) -> Dict[str, float]:
    coords, vel, Fm = ep["coords"], ep["particle_v"], ep["particle_F"]
    if bool(cfg.train.get("use_source_motion_valid", False)):
        src_mv = _load_source_motion_valid(ep, coords.device)
        if src_mv is not None:
            n_frames = min(src_mv.shape[0], coords.shape[0])
            mv = ep.get("particle_motion_valid", torch.ones_like(coords[:, :, 0], dtype=torch.bool))
            mv = mv.clone()
            mv[:n_frames] = mv[:n_frames] & src_mv[:n_frames, :mv.shape[1]].to(mv.device)
            ep = {**ep, "particle_motion_valid": mv}
    if bool(cfg.train.get("augment_z_rotation", False)):
        R = _random_z_rotation_and_flip(seed * 7 + 31, coords.device)
        coords = coords @ R.T
        cg = ep.get("controller_grid_points")
        if cg is not None:
            ep = {**ep, "controller_grid_points": cg @ R.T, "coords": coords}
        else:
            ep = {**ep, "coords": coords}
    end = supervision_end_frame(coords.shape[0], cfg)
    max_start = max(end - int(cfg.train.update_every) * 2, 0)
    if max_start > 0 and model.training and float(cfg.train.get("temporal_crop_fraction", 0.0)) > 0:
        grp_size = max(int(cfg.train.get("source_group_size", 1)), 1)
        group_idx = int(ep.get("episode_index", 0)) // grp_size
        start = int((int(cfg.train.seed) + group_idx * 997) % (max_start + 1))
    else:
        start = 0
    canonical = coords[start:start + 1]
    cache = _model_cache_shared(model, coords[0:1], episode_source_name(ep))
    state: Optional[object] = None
    material = material_guess(ep, cfg, start, seed)
    correction = torch.zeros_like(canonical)
    mask = torch.zeros(canonical.shape[:2], device=coords.device, dtype=torch.bool)
    losses, last, chunks = [], material, 0
    chunk_frames, chunk_logE_mean, chunk_logE_mae = [], [], []
    feature_window = []
    track_drop_p = float(cfg.train.get("tracking_dropout", 0.0))
    for frame in range(start + 1, end + 1):
        cur = coords[frame:frame + 1]
        prev = coords[frame - 1:frame]
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
        if obs is not None and track_drop_p > 0 and model.training:
            drop = torch.rand_like(obs["tracked_mask"].float()) < track_drop_p
            obs["tracked_mask"] = obs["tracked_mask"] & ~drop
            obs["tracked_disp"] = obs["tracked_disp"] * obs["tracked_mask"][..., None].float()
        track_jitter = float(cfg.train.get("tracking_jitter_std", 0.0))
        if obs is not None and track_jitter > 0 and model.training:
            noise = torch.randn_like(obs["tracked_disp"]) * track_jitter
            obs["tracked_disp"] = obs["tracked_disp"] + noise * obs["tracked_mask"][..., None].float()
            obs["control_vecs"] = obs["control_vecs"] + torch.randn_like(obs["control_vecs"]) * track_jitter
            obs["control_disp"] = obs["control_disp"] + torch.randn_like(obs["control_disp"]) * track_jitter
        feature_window.append(
            model.build_features(
                cur,
                prev,
                canonical,
                material,
                vel[frame:frame + 1],
                Fm[frame:frame + 1],
                correction,
                mask,
                obs,
            )
        )

        if (frame - start) % int(cfg.train.update_every):
            continue

        x_window = torch.stack(feature_window, dim=2)  # B,N,K,F
        with amp_context(cfg, coords.device):
            pred = _fwd_features(model, x_window, cur, prev, cache, state)
        feature_window.clear()

        state, last = pred["state"], pred["material"]
        chunk_frames.append(float(frame))
        chunk_logE_mean.append(float(last[..., 0].float().mean().detach().item()))
        chunk_logE_mae.append(float((last[..., 0].float() - ep["gt_log_E"].to(last.device)).abs().mean().detach().item()))
        losses.append(update_loss(last, ep, model, cfg, pred.get("material_confidence"), pred.get("latent"), pred.get("plasticity"))["material"])
        chunks += 1
        if chunks % int(cfg.train.tbptt_chunks) == 0:
            opt.zero_grad(set_to_none=True)
            torch.stack(losses).mean().backward()
            sync_and_step(model, opt, cfg)
            losses.clear()
            state = model.detach(state)
    if losses:
        opt.zero_grad(set_to_none=True)
        torch.stack(losses).mean().backward()
        sync_and_step(model, opt, cfg)
    with torch.no_grad():
        m = update_loss(last, ep, model, cfg, pred.get("material_confidence") if "pred" in locals() else None, pred.get("latent") if "pred" in locals() else None, pred.get("plasticity") if "pred" in locals() else None)
    return {k: float(v.detach().item()) for k, v in m.items()} | {
        "chunks": float(chunks),
        "chunk_frames": chunk_frames,
        "chunk_logE_mean": chunk_logE_mean,
        "chunk_logE_mae": chunk_logE_mae,
    }


