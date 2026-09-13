"""
Optimize a fixed material field on one real-world episode, as an MfM baseline.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from physcore.particle_flow import episode_runtime as ert
from physcore.particle_flow.dataset import ParticleFlowEpisodeDataset  # noqa: E402
from physcore.particle_flow.rollout import (  # noqa: E402
    BatchedDifferentiableRolloutEngine,
    DifferentiableRolloutEngine,
)
from physcore.particle_flow.rollout_runtime import (  # noqa: E402
    real_world_chunk_dt as _real_world_chunk_dt,
    temporary_rollout_timestep as _temporary_rollout_timestep,
)


def _logit_from_value(value: float, lo: float, hi: float) -> float:
    t = (float(value) - float(lo)) / max(float(hi) - float(lo), 1.0e-8)
    t = min(max(t, 1.0e-6), 1.0 - 1.0e-6)
    return math.log(t / (1.0 - t))


def _direct_e_bounds(cfg, direct_e_min: Optional[float], direct_e_max: Optional[float]) -> tuple[float, float]:
    log_e_min = float(cfg.model.get("material_log_E_min", 8.0))
    log_e_max = float(cfg.model.get("material_log_E_max", 12.0))
    e_min = math.exp(log_e_min) if direct_e_min is None else float(direct_e_min)
    e_max = math.exp(log_e_max) if direct_e_max is None else float(direct_e_max)
    if e_max <= e_min:
        raise ValueError(f"Invalid direct E bounds: min={e_min:g}, max={e_max:g}")
    return e_min, e_max


def _decode_material(
    raw: torch.Tensor,
    cfg,
    *,
    e_parameterization: str = "log_e",
    direct_e_min: Optional[float] = None,
    direct_e_max: Optional[float] = None,
) -> torch.Tensor:
    log_e_min = float(cfg.model.get("material_log_E_min", 8.0))
    log_e_max = float(cfg.model.get("material_log_E_max", 12.0))
    nu_min = float(cfg.model.get("material_nu_min", 0.3))
    nu_max = float(cfg.model.get("material_nu_max", 0.49))
    normalized = torch.sigmoid(raw)

    if e_parameterization == "log_e":
        log_e = log_e_min + normalized[..., 0] * (log_e_max - log_e_min)
    elif e_parameterization == "raw_log_e":
        log_e = raw[..., 0]
    elif e_parameterization == "direct_e":
        e_min, e_max = _direct_e_bounds(cfg, direct_e_min, direct_e_max)
        youngs_modulus = e_min + normalized[..., 0] * (e_max - e_min)
        log_e = torch.log(torch.clamp(youngs_modulus, min=1.0e-12))
    elif e_parameterization == "direct_e_value":
        e_min, e_max = _direct_e_bounds(cfg, direct_e_min, direct_e_max)
        normalized_e = raw[..., 0].clamp(0.0, 1.0)
        youngs_modulus = e_min + normalized_e * (e_max - e_min)
        log_e = torch.log(torch.clamp(youngs_modulus, min=1.0e-12))
    else:
        raise ValueError(f"Unknown E parameterization: {e_parameterization}")

    return torch.stack(
        [
            log_e,
            nu_min + normalized[..., 1] * (nu_max - nu_min),
        ],
        dim=-1,
    )


def _override_material_and_rollout_cfg(cfg, args: argparse.Namespace) -> None:
    if args.material_log_e_min is not None:
        cfg.model.material_log_E_min = float(args.material_log_e_min)
    if args.material_log_e_max is not None:
        cfg.model.material_log_E_max = float(args.material_log_e_max)
    if args.material_nu_min is not None:
        cfg.model.material_nu_min = float(args.material_nu_min)
    if args.material_nu_max is not None:
        cfg.model.material_nu_max = float(args.material_nu_max)
    if args.ground_friction is not None:
        cfg.rollout.ground_friction = float(args.ground_friction)
    if args.ground_elasticity is not None:
        cfg.rollout.ground_elasticity = float(args.ground_elasticity)
    if args.rigid_friction is not None:
        cfg.rollout.rigid_friction = float(args.rigid_friction)


def _init_material_logits(
    *,
    num_particles: int,
    cfg,
    device: torch.device,
    init_log_e: float,
    init_nu: float,
    global_material: bool,
    e_parameterization: str,
    direct_e_min: Optional[float],
    direct_e_max: Optional[float],
) -> torch.nn.Parameter:
    log_e_min = float(cfg.model.get("material_log_E_min", 8.0))
    log_e_max = float(cfg.model.get("material_log_E_max", 12.0))
    nu_min = float(cfg.model.get("material_nu_min", 0.3))
    nu_max = float(cfg.model.get("material_nu_max", 0.49))
    shape = (2,) if bool(global_material) else (int(num_particles), 2)
    logits = torch.empty(shape, device=device, dtype=torch.float32)
    if e_parameterization == "log_e":
        logits[..., 0] = _logit_from_value(init_log_e, log_e_min, log_e_max)
    elif e_parameterization == "raw_log_e":
        logits[..., 0] = float(init_log_e)
    elif e_parameterization == "direct_e":
        e_min, e_max = _direct_e_bounds(cfg, direct_e_min, direct_e_max)
        logits[..., 0] = _logit_from_value(math.exp(init_log_e), e_min, e_max)
    elif e_parameterization == "direct_e_value":
        e_min, e_max = _direct_e_bounds(cfg, direct_e_min, direct_e_max)
        normalized_e = (math.exp(init_log_e) - e_min) / max(e_max - e_min, 1.0e-8)
        logits[..., 0] = min(max(normalized_e, 0.0), 1.0)
    else:
        raise ValueError(f"Unknown E parameterization: {e_parameterization}")
    logits[..., 1] = _logit_from_value(init_nu, nu_min, nu_max)
    return torch.nn.Parameter(logits)


def _encode_material_as_raw(
    material: torch.Tensor,
    cfg,
    *,
    e_parameterization: str,
    direct_e_min: Optional[float],
    direct_e_max: Optional[float],
) -> torch.Tensor:
    """Encode physical (log_E, nu) values into the current raw parameter space."""
    log_e_min = float(cfg.model.get("material_log_E_min", 8.0))
    log_e_max = float(cfg.model.get("material_log_E_max", 12.0))
    nu_min = float(cfg.model.get("material_nu_min", 0.3))
    nu_max = float(cfg.model.get("material_nu_max", 0.49))
    raw = torch.empty_like(material)
    log_e = material[..., 0]
    nu = material[..., 1]
    if e_parameterization == "log_e":
        normalized_log_e = (log_e - log_e_min) / max(log_e_max - log_e_min, 1.0e-8)
        raw[..., 0] = torch.logit(normalized_log_e.clamp(1.0e-6, 1.0 - 1.0e-6))
    elif e_parameterization == "raw_log_e":
        raw[..., 0] = log_e
    elif e_parameterization == "direct_e":
        e_min, e_max = _direct_e_bounds(cfg, direct_e_min, direct_e_max)
        normalized_e = (log_e.exp() - e_min) / max(e_max - e_min, 1.0e-8)
        raw[..., 0] = torch.logit(normalized_e.clamp(1.0e-6, 1.0 - 1.0e-6))
    elif e_parameterization == "direct_e_value":
        e_min, e_max = _direct_e_bounds(cfg, direct_e_min, direct_e_max)
        raw[..., 0] = ((log_e.exp() - e_min) / max(e_max - e_min, 1.0e-8)).clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unknown E parameterization: {e_parameterization}")
    normalized_nu = (nu - nu_min) / max(nu_max - nu_min, 1.0e-8)
    raw[..., 1] = torch.logit(normalized_nu.clamp(1.0e-6, 1.0 - 1.0e-6))
    return raw


def _load_first_episode(cfg, device: torch.device) -> Dict[str, object]:
    roots = ert._expand_simulation_batch_roots(list(cfg.dataset.get("train_roots", [])))
    if not roots:
        raise ValueError("dataset.train_roots is empty")
    dataset = ParticleFlowEpisodeDataset(
        roots,
        cache_size=int(cfg.dataset.get("cache_size", 2)),
        real_world_domain_center=cfg.dataset.get("real_world_domain_center", None),
        observation_views=cfg.dataset.get("observation_views", None),
        observation_view_index=int(cfg.dataset.get("observation_view_index", 0)),
    )
    raw_episode = dataset[0]
    episode = ert._load_episode_tensors(raw_episode, device)
    episode["episode_root"] = str(raw_episode.get("episode_root", dataset.episodes[0].root_dir))
    return episode


def _project_material_parameter(raw_material: torch.Tensor, e_parameterization: str) -> None:
    if e_parameterization == "direct_e_value":
        with torch.no_grad():
            raw_material[..., 0].clamp_(0.0, 1.0)


def _build_knn_edges(points: torch.Tensor, k: int, chunk_size: int = 1024) -> Optional[torch.Tensor]:
    if k <= 0:
        return None
    points_cpu = points.detach().float().cpu()
    num_points = int(points_cpu.shape[0])
    src_chunks = []
    dst_chunks = []
    for start in range(0, num_points, int(chunk_size)):
        end = min(start + int(chunk_size), num_points)
        distances = torch.cdist(points_cpu[start:end], points_cpu)
        _, neighbors = torch.topk(distances, k=min(int(k) + 1, num_points), largest=False, dim=1)
        neighbors = neighbors[:, 1:]
        src = torch.arange(start, end, dtype=torch.long).view(-1, 1).expand_as(neighbors)
        src_chunks.append(src.reshape(-1))
        dst_chunks.append(neighbors.reshape(-1))
    return torch.stack([torch.cat(src_chunks), torch.cat(dst_chunks)], dim=0)


def _material_smoothness_loss(material: torch.Tensor, edges: Optional[torch.Tensor], nu_weight: float) -> torch.Tensor:
    if edges is None or material.ndim == 1:
        return material.new_zeros(())
    edge_index = edges.to(device=material.device, non_blocking=True)
    diff = material[edge_index[0]] - material[edge_index[1]]
    log_e_loss = diff[:, 0].square().mean()
    nu_loss = diff[:, 1].square().mean()
    return log_e_loss + float(nu_weight) * nu_loss


def _material_checkpoint_payload(
    *,
    raw_material: torch.Tensor,
    cfg,
    metrics: Dict[str, float],
    args: argparse.Namespace,
    num_particles: int,
) -> Dict[str, object]:
    decoded_material = _decode_material(
        raw_material.detach(),
        cfg,
        e_parameterization=str(args.e_parameterization),
        direct_e_min=args.direct_e_min,
        direct_e_max=args.direct_e_max,
    ).detach().cpu()
    if decoded_material.ndim == 1:
        decoded_material = decoded_material.view(1, 2).expand(int(num_particles), 2).clone()
    return {
        "raw_material": raw_material.detach().cpu(),
        "decoded_material": decoded_material,
        "metrics": dict(metrics),
        "e_parameterization": str(args.e_parameterization),
        "direct_e_min": args.direct_e_min,
        "direct_e_max": args.direct_e_max,
        "global_material": bool(args.global_material),
        "config_path": str(args.config),
    }


def _run_episode_objective(
    *,
    cfg,
    episode: Dict[str, object],
    rollout_engine,
    raw_material: torch.Tensor,
    correction_steps_count: int,
    run_correction: bool,
    start_frame: int,
    end_frame: int,
    rollout_steps: int,
    e_parameterization: str,
    direct_e_min: Optional[float],
    direct_e_max: Optional[float],
    correction_stop_frame: Optional[int],
    pure_tail_loss_only: bool,
) -> Dict[str, float]:
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
    if not is_real_world_episode:
        raise ValueError("This sanity check is intended for a real-world episode.")

    use_contact_kinematic_targets = ert._should_use_contact_kinematic_targets(
        is_real_world_episode=is_real_world_episode,
        manipulation_contact_particle_ids=manipulation_contact_particle_ids,
        controller_grid_points=controller_grid_points,
    )
    kinematic_contact_particle_ids = (
        manipulation_contact_particle_ids if use_contact_kinematic_targets else None
    )

    _, num_particles, _ = coords.shape
    current_frame = int(start_frame)
    current_positions = coords[current_frame].contiguous()
    current_velocities = particle_v[current_frame].contiguous()
    current_F = particle_F[current_frame].contiguous()
    current_C = None if particle_C is None else particle_C[current_frame].contiguous()

    denom_chunks = max(int(end_frame) - int(start_frame), 1)
    tail_denom_chunks = (
        max(int(end_frame) - int(correction_stop_frame), 1)
        if correction_stop_frame is not None
        else denom_chunks
    )
    total_loss_value = 0.0
    total_raw_recon = 0.0
    total_obs_to_pred = 0.0
    total_corr = 0.0
    total_mpm_endpoint = 0.0
    total_correction_endpoint = 0.0
    tail_loss_value = 0.0
    tail_raw_recon = 0.0
    tail_obs_to_pred = 0.0
    tail_corr = 0.0
    tail_chunks = 0
    corrected_chunks = 0
    chunks = 0

    while current_frame < end_frame:
        material_value = _decode_material(
            raw_material,
            cfg,
            e_parameterization=e_parameterization,
            direct_e_min=direct_e_min,
            direct_e_max=direct_e_max,
        )
        if material_value.ndim == 1:
            material_state = material_value.view(1, 1, 2).expand(1, num_particles, 2)
        else:
            material_state = material_value.view(1, num_particles, 2)
        chunk_steps = int(rollout_steps)
        next_chunk_frame = current_frame + 1
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
        chunk_run_correction = bool(run_correction)
        if correction_stop_frame is not None:
            chunk_run_correction = chunk_run_correction and current_frame < int(correction_stop_frame)
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
                chunk_run_correction,
                manipulation_flag=manipulation_flag,
                manipulation_contact_particle_ids=kinematic_contact_particle_ids,
                controller_grid_points=controller_grid_points,
                rigid_points_window=rigid_points_window,
                manipulation_contact_particle_ids_window=manipulation_contact_particle_ids_window,
                controller_grid_points_window=controller_grid_points_window,
            )

        chunk_positions = rollout_outputs["chunk_positions"]
        chunk_flows = rollout_outputs["chunk_flows"]
        gt_pos = coords[next_chunk_frame : next_chunk_frame + 1]
        gt_flow = flows[next_chunk_frame : next_chunk_frame + 1]
        reconstruction_loss, raw_reconstruction_loss, real_terms = ert._compute_loss(
            chunk_positions,
            chunk_flows,
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
        if not torch.isfinite(reconstruction_loss):
            raise FloatingPointError(f"non-finite reconstruction loss at frame {current_frame}")

        chunk_is_tail = correction_stop_frame is not None and current_frame >= int(correction_stop_frame)
        if bool(pure_tail_loss_only) and correction_stop_frame is not None:
            if chunk_is_tail:
                (reconstruction_loss / float(tail_denom_chunks)).backward()
        else:
            (reconstruction_loss / float(denom_chunks)).backward()

        total_loss_value += float(reconstruction_loss.detach().item())
        total_raw_recon += float(raw_reconstruction_loss.detach().item())
        total_obs_to_pred += float(real_terms["real_obs_to_pred_loss"].detach().item())
        total_corr += float(real_terms["real_correspondence_loss"].detach().item())
        total_mpm_endpoint += float(rollout_outputs["mpm_endpoint_loss"].detach().item())
        total_correction_endpoint += float(rollout_outputs["correction_loss"].detach().item())
        chunks += 1
        if chunk_run_correction:
            corrected_chunks += 1
        else:
            tail_loss_value += float(reconstruction_loss.detach().item())
            tail_raw_recon += float(raw_reconstruction_loss.detach().item())
            tail_obs_to_pred += float(real_terms["real_obs_to_pred_loss"].detach().item())
            tail_corr += float(real_terms["real_correspondence_loss"].detach().item())
            tail_chunks += 1

        corrected_terminal_position = rollout_outputs["correction_result"]["predicted_positions"][-1].detach()
        pure_mpm_result = rollout_outputs["pure_mpm_result"]
        current_positions = corrected_terminal_position.contiguous()
        current_velocities = pure_mpm_result["final_velocity"].detach().contiguous()
        current_F = rollout_outputs["correction_result"]["final_deformation_gradient"].detach().contiguous()
        final_C = pure_mpm_result.get("final_C", None)
        current_C = None if final_C is None else final_C.detach().contiguous()
        current_frame = next_chunk_frame

    denom = max(chunks, 1)
    tail_denom = max(tail_chunks, 1)
    material_detached = _decode_material(
        raw_material.detach(),
        cfg,
        e_parameterization=e_parameterization,
        direct_e_min=direct_e_min,
        direct_e_max=direct_e_max,
    )
    if material_detached.ndim == 1:
        mean_material = material_detached
        std_material = material_detached.new_zeros(2)
        mean_E = material_detached[0].exp()
        std_E = material_detached.new_zeros(())
        min_material = material_detached
        max_material = material_detached
        min_E = mean_E
        max_E = mean_E
    else:
        mean_material = material_detached.mean(dim=0)
        std_material = material_detached.std(dim=0)
        e_values = material_detached[..., 0].exp()
        mean_E = e_values.mean()
        std_E = e_values.std()
        min_material = material_detached.amin(dim=0)
        max_material = material_detached.amax(dim=0)
        min_E = e_values.min()
        max_E = e_values.max()
    return {
        "loss": total_loss_value / denom,
        "recon": total_raw_recon / denom,
        "obs_to_pred": total_obs_to_pred / denom,
        "corr": total_corr / denom,
        "mpm_endpoint": total_mpm_endpoint / denom,
        "correction_endpoint": total_correction_endpoint / denom,
        "tail_loss": tail_loss_value / tail_denom,
        "tail_recon": tail_raw_recon / tail_denom,
        "tail_obs_to_pred": tail_obs_to_pred / tail_denom,
        "tail_corr": tail_corr / tail_denom,
        "tail_chunks": float(tail_chunks),
        "corrected_chunks": float(corrected_chunks),
        "mean_log_E": float(mean_material[0].item()),
        "mean_E": float(mean_E.item()),
        "mean_nu": float(mean_material[1].item()),
        "std_log_E": float(std_material[0].item()),
        "std_E": float(std_E.item()),
        "std_nu": float(std_material[1].item()),
        "min_log_E": float(min_material[0].item()),
        "max_log_E": float(max_material[0].item()),
        "min_E": float(min_E.item()),
        "max_E": float(max_E.item()),
        "min_nu": float(min_material[1].item()),
        "max_nu": float(max_material[1].item()),
        "chunks": float(chunks),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument("--init-log-e", type=float, default=10.0)
    parser.add_argument("--init-nu", type=float, default=0.4)
    parser.add_argument("--output-dir", default="outputs/fixed_material/case_name")
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--no-correction", action="store_true")
    parser.add_argument(
        "--pure-tail-frac",
        type=float,
        default=0.0,
        help="If >0, use correction for the prefix and pure MPM for this fraction of the episode tail.",
    )
    parser.add_argument(
        "--pure-tail-loss-only",
        action="store_true",
        help="When using --pure-tail-frac, backpropagate only pure-tail chunk losses.",
    )
    parser.add_argument("--global-material", action="store_true")
    parser.add_argument(
        "--e-parameterization",
        choices=("log_e", "raw_log_e", "direct_e", "direct_e_value"),
        default="log_e",
        help=(
            "log_e: sigmoid-bounded log_E; raw_log_e: optimize unbounded log_E directly; "
            "direct_e: sigmoid-bounded direct E; direct_e_value: optimize projected normalized E directly."
        ),
    )
    parser.add_argument(
        "--direct-e-min",
        type=float,
        default=None,
        help="Lower Young's modulus bound for --e-parameterization direct_e; defaults to exp(material_log_E_min).",
    )
    parser.add_argument(
        "--direct-e-max",
        type=float,
        default=None,
        help="Upper Young's modulus bound for --e-parameterization direct_e; defaults to exp(material_log_E_max).",
    )
    parser.add_argument("--material-log-e-min", type=float, default=None)
    parser.add_argument("--material-log-e-max", type=float, default=None)
    parser.add_argument("--material-nu-min", type=float, default=None)
    parser.add_argument("--material-nu-max", type=float, default=None)
    parser.add_argument("--ground-friction", type=float, default=None)
    parser.add_argument("--ground-elasticity", type=float, default=None)
    parser.add_argument("--rigid-friction", type=float, default=None)
    parser.add_argument(
        "--init-material-checkpoint",
        type=str,
        default=None,
        help="Optional fixed-material checkpoint whose raw_material tensor initializes optimization.",
    )
    parser.add_argument("--material-smoothness-weight", type=float, default=0.0)
    parser.add_argument("--material-smoothness-k", type=int, default=8)
    parser.add_argument("--material-smoothness-nu-weight", type=float, default=0.0)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, False)
    _override_material_and_rollout_cfg(cfg, args)
    device = torch.device(str(cfg.train.get("device", "cuda")))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode = _load_first_episode(cfg, device)
    coords = episode["coords"]
    num_particles = int(coords.shape[1])
    smooth_edges = _build_knn_edges(coords[0], int(args.material_smoothness_k)) if float(args.material_smoothness_weight) > 0.0 else None
    raw_material = _init_material_logits(
        num_particles=num_particles,
        cfg=cfg,
        device=device,
        init_log_e=args.init_log_e,
        init_nu=args.init_nu,
        global_material=bool(args.global_material),
        e_parameterization=str(args.e_parameterization),
        direct_e_min=args.direct_e_min,
        direct_e_max=args.direct_e_max,
    )
    if args.init_material_checkpoint:
        checkpoint = torch.load(Path(args.init_material_checkpoint).expanduser(), map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict) or "raw_material" not in checkpoint:
            raise ValueError(f"Material checkpoint missing raw_material: {args.init_material_checkpoint}")
        if "decoded_material" in checkpoint:
            loaded_material = checkpoint["decoded_material"].to(device=device, dtype=raw_material.dtype)
            if tuple(loaded_material.shape) != tuple(raw_material.shape):
                if raw_material.ndim == 1 and loaded_material.ndim == 2 and loaded_material.shape[-1] == 2:
                    loaded_material = loaded_material.mean(dim=0)
                else:
                    raise ValueError(
                        f"Checkpoint decoded_material shape {tuple(loaded_material.shape)} does not match "
                        f"expected {tuple(raw_material.shape)}"
                    )
            loaded_raw = _encode_material_as_raw(
                loaded_material,
                cfg,
                e_parameterization=str(args.e_parameterization),
                direct_e_min=args.direct_e_min,
                direct_e_max=args.direct_e_max,
            )
        else:
            loaded_raw = checkpoint["raw_material"].to(device=device, dtype=raw_material.dtype)
        if tuple(loaded_raw.shape) != tuple(raw_material.shape):
            raise ValueError(
                f"Checkpoint raw_material shape {tuple(loaded_raw.shape)} does not match "
                f"expected {tuple(raw_material.shape)}"
            )
        with torch.no_grad():
            raw_material.copy_(loaded_raw)
            _project_material_parameter(raw_material, str(args.e_parameterization))
    optimizer = torch.optim.Adam([raw_material], lr=float(args.lr))
    total_frames = int(coords.shape[0])
    history_steps = int(cfg.dataset.get("history_steps", 1))
    start_frame = max(history_steps - 1, 0) if args.start_frame is None else int(args.start_frame)
    end_frame = total_frames - 1 if args.end_frame is None else min(int(args.end_frame), total_frames - 1)
    if start_frame >= end_frame:
        raise ValueError(f"Invalid frame range {start_frame}->{end_frame}")
    run_correction = not bool(args.no_correction)
    pure_tail_frac = max(0.0, min(float(args.pure_tail_frac), 1.0))
    correction_stop_frame = None
    if pure_tail_frac > 0.0 and run_correction:
        correction_stop_frame = start_frame + int(round((end_frame - start_frame) * (1.0 - pure_tail_frac)))

    rollout_engine_cls = (
        BatchedDifferentiableRolloutEngine
        if bool(cfg.rollout.get("use_batched_mpm", True))
        else DifferentiableRolloutEngine
    )
    rollout_engine = rollout_engine_cls(cfg.rollout, device=str(device))
    rollout_steps = max(int(cfg.dataset.get("rollout_steps", 25)), 1)
    correction_steps_count = int(cfg.model.get("correction_rollout_steps", cfg.model.get("correction_steps", 4)))

    print(
        "Fixed material sanity: "
        f"mode={'global' if args.global_material else 'per_particle'} "
        f"e_parameterization={args.e_parameterization} "
        f"episode={episode.get('episode_root', 'unknown')} frames={start_frame}->{end_frame} "
        f"chunks={end_frame - start_frame} rollout_steps={rollout_steps} "
        f"correction={run_correction} pure_tail_frac={pure_tail_frac:g} "
        f"pure_tail_loss_only={bool(args.pure_tail_loss_only)} "
        f"correction_stop_frame={correction_stop_frame} lr={args.lr:g} iters={args.iters} "
        f"smooth_w={float(args.material_smoothness_weight):g} smooth_k={int(args.material_smoothness_k)}",
        flush=True,
    )
    print(
        "Overrides: "
        f"log_E=[{float(cfg.model.get('material_log_E_min', 8.0)):g},"
        f"{float(cfg.model.get('material_log_E_max', 12.0)):g}] "
        f"nu=[{float(cfg.model.get('material_nu_min', 0.3)):g},"
        f"{float(cfg.model.get('material_nu_max', 0.49)):g}] "
        f"ground_friction={float(cfg.rollout.get('ground_friction', 0.3)):g} "
        f"rigid_friction={float(cfg.rollout.get('rigid_friction', 0.5)):g}",
        flush=True,
    )
    if args.e_parameterization in {"direct_e", "direct_e_value"}:
        e_min, e_max = _direct_e_bounds(cfg, args.direct_e_min, args.direct_e_max)
        print(f"Direct E bounds: min={e_min:g} max={e_max:g}", flush=True)

    history = []
    best: Optional[Dict[str, float]] = None
    best_metric_key = (
        "tail_obs_to_pred"
        if pure_tail_frac > 0.0 and bool(args.pure_tail_loss_only)
        else "obs_to_pred"
    )
    for iteration in range(1, int(args.iters) + 1):
        tic = time.time()
        optimizer.zero_grad(set_to_none=True)
        metrics = _run_episode_objective(
            cfg=cfg,
            episode=episode,
            rollout_engine=rollout_engine,
            raw_material=raw_material,
            correction_steps_count=correction_steps_count,
            run_correction=run_correction,
            start_frame=start_frame,
            end_frame=end_frame,
            rollout_steps=rollout_steps,
            e_parameterization=str(args.e_parameterization),
            direct_e_min=args.direct_e_min,
            direct_e_max=args.direct_e_max,
            correction_stop_frame=correction_stop_frame,
            pure_tail_loss_only=bool(args.pure_tail_loss_only),
        )
        decoded_for_smoothness = _decode_material(
            raw_material,
            cfg,
            e_parameterization=str(args.e_parameterization),
            direct_e_min=args.direct_e_min,
            direct_e_max=args.direct_e_max,
        )
        smoothness_loss = _material_smoothness_loss(
            decoded_for_smoothness,
            smooth_edges,
            float(args.material_smoothness_nu_weight),
        )
        if float(args.material_smoothness_weight) > 0.0:
            (float(args.material_smoothness_weight) * smoothness_loss).backward()
        metrics["material_smoothness_loss"] = float(smoothness_loss.detach().item())
        metrics["material_smoothness_weight"] = float(args.material_smoothness_weight)
        metrics["smooth_objective"] = float(metrics[best_metric_key] + float(args.material_smoothness_weight) * metrics["material_smoothness_loss"])
        grad_norm = float(raw_material.grad.detach().norm().item()) if raw_material.grad is not None else 0.0
        evaluated_raw_material = raw_material.detach().clone()
        torch.nn.utils.clip_grad_norm_([raw_material], max_norm=10.0)
        optimizer.step()
        _project_material_parameter(raw_material, str(args.e_parameterization))
        metrics["iter"] = float(iteration)
        metrics["grad_norm"] = grad_norm
        metrics["time_sec"] = time.time() - tic
        metrics["best_metric_key"] = best_metric_key
        history.append(metrics)
        best_value_key = "smooth_objective" if float(args.material_smoothness_weight) > 0.0 else best_metric_key
        metrics["best_value_key"] = best_value_key
        if best is None or metrics[best_value_key] < best[best_value_key]:
            best = dict(metrics)
            torch.save(
                _material_checkpoint_payload(
                    raw_material=evaluated_raw_material,
                    cfg=cfg,
                    metrics=metrics,
                    args=args,
                    num_particles=num_particles,
                ),
                output_dir / "best_material.pt",
            )
        torch.save(
            _material_checkpoint_payload(
                raw_material=evaluated_raw_material,
                cfg=cfg,
                metrics=metrics,
                args=args,
                num_particles=num_particles,
            ),
            output_dir / "latest_evaluated_material.pt",
        )
        torch.save(
            _material_checkpoint_payload(
                raw_material=raw_material,
                cfg=cfg,
                metrics=metrics,
                args=args,
                num_particles=num_particles,
            ),
            output_dir / "latest_material.pt",
        )
        print(
            f"iter {iteration:04d}/{args.iters} "
            f"loss={metrics['loss']:.6g} recon={metrics['recon']:.6g} "
            f"obs_to_pred={metrics['obs_to_pred']:.6g} corr={metrics['corr']:.6g} "
            f"tail_obs_to_pred={metrics['tail_obs_to_pred']:.6g} "
            f"mean_log_E={metrics['mean_log_E']:.6f} mean_E={metrics['mean_E']:.3g} "
            f"mean_nu={metrics['mean_nu']:.6f} "
            f"std_log_E={metrics['std_log_E']:.6f} std_E={metrics['std_E']:.3g} "
            f"std_nu={metrics['std_nu']:.6f} "
            f"log_E_range=[{metrics['min_log_E']:.3f},{metrics['max_log_E']:.3f}] "
            f"smooth={metrics['material_smoothness_loss']:.6g} "
            f"objective={metrics['smooth_objective']:.6g} "
            f"grad={grad_norm:.3g} time={metrics['time_sec']:.1f}s",
            flush=True,
        )

        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(metrics, sort_keys=True) + "\n")
        if best is not None:
            (output_dir / "best.json").write_text(json.dumps(best, indent=2, sort_keys=True), encoding="utf-8")

    if history:
        xs = [int(row["iter"]) for row in history]
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(xs, [row["obs_to_pred"] for row in history], label="obs_to_pred")
        axes[0].plot(xs, [row["recon"] for row in history], label="recon")
        axes[0].legend()
        axes[0].grid(True, alpha=0.25)
        axes[1].plot(xs, [row["mean_log_E"] for row in history], label="mean_log_E")
        axes[1].plot(xs, [row["mean_nu"] for row in history], label="mean_nu")
        axes[1].legend()
        axes[1].grid(True, alpha=0.25)
        axes[1].set_xlabel("iteration")
        fig.tight_layout()
        fig.savefig(output_dir / "fixed_material_optimization.png", dpi=160)
        plt.close(fig)

    if best is not None:
        print("best " + json.dumps(best, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
