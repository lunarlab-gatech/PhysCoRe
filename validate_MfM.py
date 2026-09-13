"""
Validate MfM on real-world episodes and write a JSON report.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from physcore.model_MfM import Refiner
from physcore.particle_flow import episode_runtime as ert
from physcore.particle_flow.dataset import ParticleFlowEpisodeDataset, _resolve_episode_roots
from physcore.particle_flow.mfm_artifacts import load_checkpoint, model_state_from_checkpoint
from physcore.particle_flow.mfm_training import defaults
from physcore.particle_flow.mfm_validation import (
    episode_roots,
    interleaved_rollout_validation,
    material_from_final_chunk,
    validation_rollout_loss,
)
from physcore.particle_flow.rollout import BatchedDifferentiableRolloutEngine, DifferentiableRolloutEngine


def load_cfg(args: argparse.Namespace, ckpt: Dict[str, object], user_cfg):
    cfg = OmegaConf.create(ckpt.get("cfg", {}))
    cfg = OmegaConf.merge(cfg, user_cfg)
    cfg = defaults(cfg)
    if args.root:
        cfg.dataset.validation_roots = list(args.root)
    if not cfg.dataset.get("validation_roots", []):
        raise ValueError("validate_MfM.py requires cfg.dataset.validation_roots or at least one --root")
    if args.rollout_steps is not None:
        cfg.dataset.rollout_steps = int(args.rollout_steps)
    return cfg


def resolve_validation_mode(args: argparse.Namespace, vcfg) -> str:
    if bool(getattr(args, "legacy_refresh_tail", False)):
        return "refresh-tail"
    if bool(getattr(args, "legacy_pure_tail", False)):
        return "pure-tail"
    if args.validation_mode is not None:
        return str(args.validation_mode)
    return str(vcfg.get("validation_mode", "pure-tail"))


def resolve_correction_mode(args: argparse.Namespace, validation_mode: str) -> str:
    if args.correction_mode is not None:
        return str(args.correction_mode)
    if validation_mode in {"pure-tail", "refresh-tail"}:
        return "half"
    return "none"


def episode_label(root: str) -> str:
    path = Path(root)
    return path.parent.name if path.name == "episode_0000" else path.name


def cfg_for_episode(cfg, root: str):
    overrides = cfg.get("per_sample_rollout_config", {}) or {}
    label = episode_label(root)
    override = overrides.get(label, None)
    if override is None:
        return cfg
    ep_cfg = copy.deepcopy(cfg)
    ep_cfg.rollout = OmegaConf.merge(ep_cfg.rollout, override)
    return ep_cfg


def save_tail_rollout(args, rollout, row, episode_idx: int, controller_radius: float, plasticity_model: str) -> None:
    if rollout is None or args.out is None:
        return
    traj_path = Path(args.out).with_suffix("").as_posix() + f"_episode_{episode_idx:04d}_trajectory.pt"
    traj_path = Path(traj_path)
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "predicted": rollout["predicted"],
            "predicted_frames": rollout["predicted_frames"],
            "ground_truth": rollout["ground_truth"],
            "mid_frame": rollout["mid_frame"],
            "controller_radius": controller_radius,
            "plasticity_model": plasticity_model,
        },
        traj_path,
    )
    row["trajectory_path"] = str(traj_path)
    print(f"wrote {traj_path}", flush=True)


def save_full_rollout(args, rollout, row, material: torch.Tensor, confidence: torch.Tensor, episode_idx: int) -> None:
    if rollout is None:
        return
    payload_path = (
        Path(args.checkpoint).resolve().parent
        / "visualization"
        / f"validation_{args.resolved_correction_mode}_episode_{episode_idx:04d}_rollout_payload.pt"
    )
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "validation": rollout,
            "material": material.detach().cpu(),
            "confidence": confidence.detach().cpu(),
        },
        payload_path,
    )
    row["rollout_payload"] = str(payload_path)
    print(f"wrote {payload_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate MfM (Material from Motion, the Refiner) on real-world "
                    "episodes with MPM rollout. Everything needed comes from the "
                    "config; the flags below only override it for ad-hoc runs.",
    )
    p.add_argument("--config", default="configs/validate_MfM.yaml")
    # Default None → fall back to the config's `validation:` block.
    p.add_argument("--checkpoint", default=None, help="override validation.checkpoint")
    p.add_argument("--root", action="append", default=[])
    p.add_argument("--device", default=None, help="override validation.device")
    p.add_argument("--out", default=None, help="override validation.out")
    p.add_argument("--rollout-steps", type=int, default=None)
    p.add_argument("--save-rollout", action=argparse.BooleanOptionalAction, default=None,
                   help="override validation.save_rollout")
    p.add_argument("--validation-mode", choices=["full", "pure-tail", "refresh-tail"], default=None,
                   help="override validation.validation_mode")
    p.add_argument("--material-gate", choices=["none", "confidence", "bounded_confidence"], default="none")
    p.add_argument("--correction-mode", choices=["none", "throughout", "half"], default=None)
    p.add_argument("--correction-stop-frame", type=int, default=None)
    p.add_argument("--gate-temperature", type=float, default=1.0)
    p.add_argument("--gate-min", type=float, default=0.0)
    p.add_argument("--gate-max", type=float, default=1.0)
    p.add_argument("--pure-tail", dest="legacy_pure_tail", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--refresh-tail", dest="legacy_refresh_tail", action="store_true", help=argparse.SUPPRESS)
    return p


def main() -> None:
    p = build_parser()
    args, overrides = p.parse_known_args()
    bad = [o for o in overrides if "=" not in o or o.startswith("-")]
    if bad:
        p.error(f"unrecognized argument(s): {bad}")

    # The config is the source of truth; a CLI flag only wins when passed.
    user_cfg = OmegaConf.load(args.config) if args.config else OmegaConf.create({})
    if overrides:
        user_cfg = OmegaConf.merge(user_cfg, OmegaConf.from_dotlist(overrides))
    vcfg = user_cfg.get("validation", {}) or {}

    args.checkpoint = args.checkpoint or vcfg.get("checkpoint", None)
    if not args.checkpoint:
        p.error("no checkpoint: set `validation.checkpoint` in the config or pass --checkpoint")
    args.device = args.device or vcfg.get("device", "cuda")
    args.out = args.out or vcfg.get("out", None)
    args.save_rollout = (
        bool(vcfg.get("save_rollout", True)) if args.save_rollout is None else bool(args.save_rollout)
    )

    validation_mode = resolve_validation_mode(args, vcfg)
    correction_mode = resolve_correction_mode(args, validation_mode)
    args.resolved_correction_mode = correction_mode

    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = load_cfg(args, ckpt, user_cfg)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = Refiner(cfg.model).to(device)
    missing, unexpected = model.load_state_dict(model_state_from_checkpoint(ckpt), strict=False)
    if missing or unexpected:
        print(f"  missing keys: {missing}", flush=True)
        print(f"  unexpected keys: {unexpected}", flush=True)
    model.eval()

    engines = {}
    base_controller_radius = float(cfg.rollout.get("manipulation_controller_grid_contact_radius", float("nan")))
    base_plasticity_model = str(cfg.rollout.get("plasticity_model", "unknown"))
    print(
        "[config] "
        f"validation_mode={validation_mode} "
        f"correction_mode={correction_mode} "
        f"controller_radius={base_controller_radius} "
        f"plasticity_model={base_plasticity_model}",
        flush=True,
    )

    # episode_roots() passes glob patterns through unexpanded; the dataset takes
    # roots literally, so resolve them here (same as validate_RfD.py).
    roots = _resolve_episode_roots(episode_roots(cfg))
    dataset = ParticleFlowEpisodeDataset(
        roots,
        cache_size=int(cfg.dataset.get("cache_size", 2)),
        real_world_domain_center=cfg.dataset.get("real_world_domain_center", None),
        observation_views=cfg.dataset.get("observation_views", None),
        observation_view_index=int(cfg.dataset.get("observation_view_index", 0)),
    )

    rows = []
    with torch.no_grad():
        for i in range(len(dataset)):
            root = dataset.episodes[i].root_dir
            ep_cfg = cfg_for_episode(cfg, root)
            engine_cls = BatchedDifferentiableRolloutEngine if bool(ep_cfg.rollout.get("use_batched_mpm", True)) else DifferentiableRolloutEngine
            controller_radius = float(ep_cfg.rollout.get("manipulation_controller_grid_contact_radius", float("nan")))
            plasticity_model = str(ep_cfg.rollout.get("plasticity_model", "unknown"))
            engine_key = f"{engine_cls.__name__}\n{OmegaConf.to_yaml(ep_cfg.rollout, resolve=True)}"
            engine = engines.get(engine_key)
            if engine is None:
                engine = engine_cls(ep_cfg.rollout, device=device)
                engines[engine_key] = engine
            label = episode_label(root)
            print(
                f"[episode {i:03d}] {label} controller_radius={controller_radius} plasticity_model={plasticity_model}",
                flush=True,
            )
            ep = ert._load_episode_tensors(dataset[i], device)
            seed = int(ep_cfg.train.get("seed", 0)) + i

            if validation_mode in {"pure-tail", "refresh-tail"}:
                row = interleaved_rollout_validation(
                    model,
                    engine,
                    ep,
                    ep_cfg,
                    seed,
                    pure_tail=True,
                    save_rollout=bool(args.save_rollout),
                    refresh_tail=(validation_mode == "refresh-tail"),
                    correction_mode=correction_mode,
                    correction_stop_frame=args.correction_stop_frame,
                )
                rollout = row.pop("rollout", None)
                save_tail_rollout(args, rollout, row, i, controller_radius, plasticity_model)
                row.update(
                    {
                        "episode": float(i),
                        "episode_label": label,
                        "episode_root": root,
                        "validation_mode": validation_mode,
                        # Global all-particle mean of the confidence frozen at the midpoint.
                        "confidence_mean": row.get("confidence_mean_mid", 0.0),
                        "confidence_std": 0.0,
                        "nu_mean": 0.0,
                        "nu_std": 0.0,
                        "logE_std": 0.0,
                        "gate_alpha_mean": 0.0,
                        "gate_alpha_rms": 0.0,
                        "gate_updates": 0.0,
                    }
                )
                rows.append(row)
                print(
                    f"episode {i:03d} avg_recon={row['avg_recon_loss']:.6g} "
                    f"tracking_l2={row['tracking_l2']:.6g} logE_mean={row['logE_mean']:.4f}",
                    flush=True,
                )
                continue

            material, confidence, gate_stats = material_from_final_chunk(
                model,
                ep,
                ep_cfg,
                seed,
                gate=str(args.material_gate),
                gate_temperature=float(args.gate_temperature),
                gate_min=float(args.gate_min),
                gate_max=float(args.gate_max),
            )
            row = validation_rollout_loss(
                engine,
                ep,
                material,
                ep_cfg,
                save_rollout=bool(args.save_rollout),
                correction_mode=correction_mode,
                correction_stop_frame=args.correction_stop_frame,
            )
            rollout = row.pop("rollout", None)
            save_full_rollout(args, rollout, row, material, confidence, i)
            row.update(
                {
                    "episode": float(i),
                    "episode_label": label,
                    "episode_root": root,
                    "validation_mode": validation_mode,
                    "logE_mean": float(material[:, 0].mean().item()),
                    "logE_std": float(material[:, 0].std(unbiased=False).item()),
                    "nu_mean": float(material[:, 1].mean().item()),
                    "nu_std": float(material[:, 1].std(unbiased=False).item()),
                    "confidence_mean": float(confidence.mean().item()),
                    "confidence_std": float(confidence.std(unbiased=False).item()),
                    **gate_stats,
                }
            )
            rows.append(row)
            print(
                f"episode {i:03d} avg_recon={row['avg_recon_loss']:.6g} "
                f"chunks={int(row['chunks'])} logE_mean={row['logE_mean']:.4f} logE_std={row['logE_std']:.4f}",
                flush=True,
            )

    avg = (
        {k: sum(r[k] for r in rows) / max(len(rows), 1) for k in rows[0] if isinstance(rows[0][k], (int, float))}
        if rows
        else {"avg_recon_loss": 0.0}
    )
    print("average " + json.dumps(avg, sort_keys=True), flush=True)
    output = Path(args.out) if args.out else Path(args.checkpoint).resolve().parent / f"real_validation_{correction_mode}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"episodes": rows, "average": avg}, indent=2))
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
