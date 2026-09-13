"""
Train MfM (Material from Motion) on the augmented episodes.
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from physcore.model_MfM import Refiner
from physcore.particle_flow import episode_runtime as ert
from physcore.particle_flow import mfm_training
from physcore.particle_flow.dataset import build_dataloaders
from physcore.particle_flow.mfm_artifacts import (
    MfMRunArtifacts,
    load_checkpoint,
    load_model_weights,
    model_state_from_checkpoint,
    save_model_checkpoint,
    save_training_state,
)
from physcore.particle_flow.mfm_training import (
    average_chunk_traces,
    average_rows,
    defaults,
    epoch_lr,
    evaluate,
    gather_rows,
    load_episode,
    local_epoch_indices,
    material_only_batch,
    material_only_episode,
    plot_loss_curves,
    set_optimizer_lr,
)
from physcore.particle_flow.runtime import (
    broadcast_parameters,
    cleanup_distributed,
    is_distributed,
    is_main_process,
    setup_distributed,
)


class _Tee:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _load_result_messages(result) -> tuple[list[str], list[str]]:
    return list(getattr(result, "missing_keys", [])), list(getattr(result, "unexpected_keys", []))


def _format_epoch_summary(row: Dict[str, object], lr: float) -> str:
    epoch = int(row["epoch"])
    extra = ""
    if "logE_shape" in row and "logE_mean" in row:
        extra = f" shape={float(row['logE_shape']):.4f} mean={float(row['logE_mean']):.4f}"
    if "logE_shell" in row or "logE_interior" in row:
        extra += (
            f" shell={float(row.get('logE_shell', float('nan'))):.4f}"
            f" interior={float(row.get('logE_interior', float('nan'))):.4f}"
        )
    plast_str = ""
    if "plasticity_acc_elastic" in row and "plasticity_acc_plastic" in row:
        bal = (float(row["plasticity_acc_elastic"]) + float(row["plasticity_acc_plastic"])) / 2
        plast_str = f" plast_bal={bal:.3f}"
    conf_str = f" conf={float(row['conf']):.3f} raw={float(row['material_raw']):.4f}" if "conf" in row else ""
    return (
        f"epoch {epoch:03d} avg logE={float(row['logE']):.4f}{extra} "
        f"nu={float(row['nu']):.5f} std={float(row['std']):.4f}"
        f"{conf_str}{plast_str} lr={lr:.3g} time={float(row['time']):.1f}s"
    )


def _restore_old_history_if_needed(artifacts: Optional[MfMRunArtifacts], history: list[Dict[str, object]]) -> None:
    if artifacts is None or not history or artifacts.history_jsonl_path.exists():
        return
    for row in history:
        artifacts.append_epoch_metrics(row)


def _run_training(cfg, rank: int, local_rank: int, world_size: int, artifacts: Optional[MfMRunArtifacts]) -> None:
    if artifacts is not None:
        artifacts.write_config(cfg)
    torch.manual_seed(int(cfg.train.seed) + rank)
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
        cfg.train.device = str(device)
    else:
        device = torch.device(str(cfg.train.device if torch.cuda.is_available() else "cpu"))

    if is_distributed():
        dist.barrier()

    cfg.dataset.train_roots = ert._expand_simulation_batch_roots(list(cfg.dataset.train_roots))
    val_roots_raw = list(cfg.dataset.get("validation_roots", []) or [])
    if val_roots_raw:
        cfg.dataset.val_roots = ert._expand_simulation_batch_roots(val_roots_raw)
    loader, val_loader = build_dataloaders(cfg)
    model = Refiner(cfg.model).to(device)
    if is_distributed():
        broadcast_parameters(model, src=0)
        mfm_training._ddp_wrapper = nn.parallel.DistributedDataParallel(
            mfm_training._ForwardFeaturesAdapter(model),
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        betas=(float(cfg.train.adam_beta1), float(cfg.train.adam_beta2)),
    )

    history: list[Dict[str, object]] = []
    best_metric = float("inf")
    best_epoch = 0
    best_checkpoint: Optional[str] = None
    start_epoch = 1
    metric_name = str(cfg.train.best_checkpoint_metric)
    resume_from = cfg.train.get("resume_from", None)
    init_from = cfg.train.get("init_from", None)

    if init_from and not resume_from:
        result, _ = load_model_weights(model, init_from, strict=False, map_location=device)
        if is_distributed():
            broadcast_parameters(model, src=0)
        if is_main_process():
            missing, unexpected = _load_result_messages(result)
            print(f"initialized model from {init_from}", flush=True)
            if missing:
                print(f"  missing keys (randomly initialized): {missing}", flush=True)
            if unexpected:
                print(f"  unexpected keys (ignored): {unexpected}", flush=True)

    if resume_from:
        ckpt = load_checkpoint(resume_from, map_location=device)
        result = model.load_state_dict(model_state_from_checkpoint(ckpt), strict=False)
        if is_main_process():
            missing, unexpected = _load_result_messages(result)
            if missing:
                print(f"  missing keys (randomly initialized): {missing}", flush=True)
            if unexpected:
                print(f"  unexpected keys (ignored): {unexpected}", flush=True)
        if "optimizer" not in ckpt:
            raise KeyError(
                f"resume checkpoint {resume_from} has no optimizer state; "
                "use train.init_from for model-only checkpoints"
            )
        opt.load_state_dict(ckpt["optimizer"])
        if ckpt.get("format") == "physcore.mfm.state.v2":
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_metric = float(ckpt.get("best_metric", float("inf")))
            best_epoch = int(ckpt.get("best_epoch", 0))
            best_checkpoint = ckpt.get("best_checkpoint", None)
            history = artifacts.load_history() if artifacts is not None else []
        else:
            history = list(ckpt.get("history", []))
            start_epoch = len(history) + 1
            vals = [float(row[metric_name]) for row in history if metric_name in row]
            if vals:
                best_metric = min(vals)
            best_epoch = next(
                (int(row.get("epoch", 0)) for row in history if float(row.get(metric_name, float("inf"))) == best_metric),
                0,
            )
            _restore_old_history_if_needed(artifacts, history)
        if is_distributed():
            broadcast_parameters(model, src=0)
        if is_main_process():
            print(f"resumed {resume_from} at epoch {start_epoch - 1}", flush=True)

    if is_main_process() and world_size > 1:
        print(f"distributed training: world_size={world_size} batch_size_per_gpu={int(cfg.train.batch_size)}", flush=True)

    total_epochs = int(cfg.train.epochs)
    epoch_range = range(start_epoch, total_epochs + 1)
    epoch_bar = (
        tqdm(epoch_range, total=total_epochs, initial=start_epoch - 1, desc="epochs", unit="ep", dynamic_ncols=True)
        if is_main_process()
        else None
    )

    def emit(msg: str) -> None:
        if epoch_bar is not None:
            epoch_bar.write(msg)
        else:
            print(msg, flush=True)

    for epoch in (epoch_bar if epoch_bar is not None else epoch_range):
        lr_now = epoch_lr(cfg, epoch)
        set_optimizer_lr(opt, lr_now)
        t0 = time.perf_counter()
        rows = []
        indices = local_epoch_indices(loader.dataset, cfg, epoch)
        batch_size = int(cfg.train.batch_size)
        for offset in range(0, len(indices), batch_size):
            batch_indices = indices[offset:offset + batch_size]
            eps = [load_episode(loader.dataset, j, device) for j in batch_indices]
            seed_base = int(cfg.train.seed) + rank * 100000 + offset
            if len(eps) > 1:
                row = material_only_batch(model, opt, eps, cfg, [seed_base + int(j) for j in batch_indices])
                label = "ep " + ",".join(f"{int(j):03d}" for j in batch_indices)
            else:
                row = material_only_episode(model, opt, eps[0], cfg, seed_base + int(batch_indices[0]))
                label = f"ep {batch_indices[0]:03d}"
            row["_samples"] = float(len(eps))
            row["_rank"] = int(rank)
            row["_epoch"] = int(epoch)
            row["_label"] = label
            rows.append(row)
        avg = average_rows(rows)
        chunk_avg = average_chunk_traces(rows)
        detail_rows = gather_rows(rows)
        val_every = int(cfg.train.get("validation_every", 1) or 0)
        val_avg = None
        if val_every > 0 and epoch % val_every == 0:
            vt0 = time.perf_counter()
            if val_loader is not None:
                val_avg = evaluate(model, val_loader, cfg, device, rank)
            vt_mat = time.perf_counter() - vt0
            if is_main_process() and val_avg is not None:
                val_extra = ""
                if "logE_shell" in val_avg or "logE_interior" in val_avg:
                    ls = val_avg.get("logE_shell", float("nan"))
                    li = val_avg.get("logE_interior", float("nan"))
                    val_extra = f" shell={ls:.4f} interior={li:.4f}"
                if "plasticity_acc" in val_avg:
                    val_bal = (val_avg.get("plasticity_acc_elastic", 0.5) + val_avg.get("plasticity_acc_plastic", 0.5)) / 2
                    val_extra += f" plast_bal={val_bal:.3f}"
                emit(f"epoch {epoch:03d} val logE={val_avg['logE']:.4f}{val_extra} nu={val_avg['nu']:.5f} episodes={val_avg.get('episodes', 0):.0f} time={vt_mat:.1f}s")
        if is_main_process():
            avg["epoch"], avg["time"] = epoch, time.perf_counter() - t0
            avg.update(chunk_avg)
            if val_avg is not None:
                avg["val_logE"] = float(val_avg["logE"])
                avg["val_nu"] = float(val_avg["nu"])
                if "logE_shell" in val_avg:
                    avg["val_logE_shell"] = float(val_avg["logE_shell"])
                if "logE_interior" in val_avg:
                    avg["val_logE_interior"] = float(val_avg["logE_interior"])
                if "plasticity_acc" in val_avg:
                    avg["val_plasticity_acc"] = float(val_avg["plasticity_acc"])
                if "plasticity_acc_elastic" in val_avg and "plasticity_acc_plastic" in val_avg:
                    avg["val_plasticity_bal"] = (float(val_avg["plasticity_acc_elastic"]) + float(val_avg["plasticity_acc_plastic"])) / 2
                if "plasticity_loss" in val_avg:
                    avg["val_plasticity_loss"] = float(val_avg["plasticity_loss"])
            if artifacts is not None:
                artifacts.append_detail_rows(detail_rows)
                history = artifacts.append_epoch_metrics(avg)
            else:
                history.append(avg)
            if int(cfg.train.get("log_every_epochs", 1) or 0) > 0 and epoch % int(cfg.train.get("log_every_epochs", 1)) == 0:
                emit(_format_epoch_summary(avg, lr_now))
            if epoch_bar is not None:
                postfix = {"logE": f"{float(avg['logE']):.3f}", "s/ep": f"{float(avg['time']):.1f}"}
                if "val_logE" in avg:
                    postfix["vlogE"] = f"{float(avg['val_logE']):.3f}"
                epoch_bar.set_postfix(postfix)
            metric = float(avg.get(metric_name, avg["logE"]))
            if metric < best_metric:
                best_metric = metric
                best_epoch = int(epoch)
                if artifacts is not None:
                    best_checkpoint = str(artifacts.best_model_path(metric_name))
                    save_model_checkpoint(best_checkpoint, model, epoch)
            if artifacts is not None:
                if bool(cfg.train.get("save_latest_every_epoch", True)):
                    save_model_checkpoint(artifacts.latest_model_path, model, epoch)
                if int(cfg.train.checkpoint_every) > 0 and epoch % int(cfg.train.checkpoint_every) == 0:
                    save_model_checkpoint(artifacts.epoch_model_path(epoch), model, epoch)
                save_training_state(
                    artifacts.latest_state_path,
                    epoch=epoch,
                    model=model,
                    optimizer=opt,
                    best_metric=best_metric,
                    best_epoch=best_epoch,
                    best_checkpoint=best_checkpoint,
                )
                artifacts.write_summary({
                    "latest_epoch": int(epoch),
                    "best_metric_name": metric_name,
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                    "best_checkpoint": best_checkpoint,
                    "latest_checkpoint": str(artifacts.latest_model_path),
                    "latest_state": str(artifacts.latest_state_path),
                })
                plot_loss_curves(history, artifacts.plots_dir / "loss_curves.png")
        if is_distributed():
            dist.barrier()

    if epoch_bar is not None:
        epoch_bar.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_MfM.yaml")
    args = p.parse_args()
    rank, local_rank, world_size = setup_distributed()
    try:
        cfg = defaults(OmegaConf.load(args.config))
        cfg.train.setdefault("log_every_epochs", 1)
        cfg.train.setdefault("save_latest_every_epoch", True)
        artifacts = MfMRunArtifacts.create(cfg.train.output_dir) if is_main_process() else None
        if artifacts is not None:
            with artifacts.train_log_path.open("a", encoding="utf-8") as log_fh:
                tee_out = _Tee(sys.stdout, log_fh)
                tee_err = _Tee(sys.stderr, log_fh)
                with redirect_stdout(tee_out), redirect_stderr(tee_err):
                    _run_training(cfg, rank, local_rank, world_size, artifacts)
        else:
            _run_training(cfg, rank, local_rank, world_size, artifacts)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
