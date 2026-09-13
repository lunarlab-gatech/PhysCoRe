"""
Runtime helpers shared by the RfD entrypoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from omegaconf import OmegaConf

from physcore.model_MfM import Refiner
from physcore.model_RfD import GridVelocityCorrector  # noqa: F401  (re-exported)
from physcore.particle_flow.rfd_engine import GVCRolloutEngine
from physcore.particle_flow.mfm_artifacts import model_state_from_checkpoint
from physcore.particle_flow.mfm_training import observed_features


# --- First-half / second-half boundary ---


def first_half_windows(cfg, n_full_windows: int) -> int:
    """First-half / second-half window boundary. Single definition shared by
    training (adaptation half) and validate_RfD.py (PID half) so they can't drift."""
    return int(float((cfg.get("gvc", {}) or {}).get("warmup_fraction", 0.5)) * int(n_full_windows))


DEACTIVATE_REFINER_CONFIDENCE: float = 5.0


def _default_material_confidence(material: torch.Tensor, value: float = DEACTIVATE_REFINER_CONFIDENCE) -> torch.Tensor:
    """Constant (B, N, 2) confidence tensor matching `separate_confidence: true` layout."""
    return material.new_full((material.shape[0], material.shape[1], 2), float(value))


# --- MfM (Refiner) inference wrapper ---


class FrozenRefiner:
    """Holds a frozen MfM (Refiner) + its persistent recurrent state across windows.

    Architecture-dependent knobs (max_controls, tracked_disp_mode, …) are read
    from the Refiner's own cfg (`self.model.cfg`, which was loaded from the
    checkpoint) so the observed-features tensor matches what the network was
    trained on. The outer RfD cfg is only used for ancillary fields.
    """

    def __init__(self, model: Refiner, cfg, device: torch.device):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.state = None
        self.canonical: Optional[torch.Tensor] = None
        self.cache = None

    def reset(self, canonical: torch.Tensor) -> None:
        self.canonical = canonical
        self.cache = self.model.cache(canonical)
        self.state = None

    @torch.no_grad()
    def predict_window(
        self,
        ep: Dict[str, object],
        start_frame: int,
        end_frame: int,
        material: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Refiner on frames (start_frame+1 .. end_frame) (=k frames).

        Returns ``(material, confidence)`` both detached:
          - material:   (1, N, 2) = [log_E, nu] per particle.
          - confidence: (1, N, 2) when `separate_confidence: true`, else (1, N, 1).
            Raw Refiner output, range ``[1, +inf)`` per channel.
        The Refiner's `plasticity` output is also computed but discarded here.
        ``material`` is the prior estimate (used as the running guess feature,
        which `observed_control` mode ignores but the API still requires).
        """
        coords = ep["coords"]
        mcfg = self.model.cfg  # the Refiner's own architecture cfg (from ckpt)
        max_controls = int(mcfg.get("max_controls", 0))
        feature_window = []
        cur = prev = None
        for frame in range(start_frame + 1, end_frame + 1):
            cur = coords[frame:frame + 1].to(self.device)
            prev = coords[max(frame - 1, 0):frame].to(self.device)
            obs = observed_features(
                ep,
                frame,
                self.canonical[0],
                cur[0],
                max_controls,
                bool(mcfg.get("fixed_tracked_mask", False)),
                str(mcfg.get("tracked_disp_mode", "canonical")),
                str(mcfg.get("control_disp_mode", "absolute")),
                bool(mcfg.get("use_persistent_tracks", True)),
                bool(mcfg.get("persistent_track_use_motion_valid", False)),
                aggregate_controls=bool(mcfg.get("aggregate_controls", False)),
            )
            zero_correction = torch.zeros_like(cur)
            zero_mask = torch.zeros(cur.shape[:2], device=self.device, dtype=torch.bool)
            feat = self.model.build_features(
                cur, prev, self.canonical, material,
                ep["particle_v"][frame:frame + 1].to(self.device),
                ep["particle_F"][frame:frame + 1].to(self.device),
                zero_correction, zero_mask,
                obs,
            )
            feature_window.append(feat)
        x_window = torch.stack(feature_window, dim=2)  # (1, N, k, F)
        pred = self.model.forward_features(x_window, cur, prev, self.cache, self.state)
        self.state = pred["state"]
        return pred["material"].detach(), pred["material_confidence"].detach()


# --- Losses ---


def chamfer_gt_to_pred(
    gt_points: torch.Tensor,
    gt_valid: torch.Tensor,
    pred_points: torch.Tensor,
    chunk: int = 4096,
) -> torch.Tensor:
    """One-directional Chamfer: mean L2 (Euclidean) nearest-neighbor distance
    from each valid GT depth point to the nearest predicted MPM particle.

    gt_points:    (V, K, 3) backprojected world points
    gt_valid:     (V, K) bool
    pred_points:  (N, 3) MPM particles for the current frame
    Returns scalar.
    """
    flat = gt_points.reshape(-1, 3)
    mask = gt_valid.reshape(-1)
    valid = flat[mask & torch.isfinite(flat).all(dim=-1)]
    if valid.shape[0] == 0:
        return pred_points.new_zeros(())
    total = pred_points.new_zeros(())
    count = 0
    for i in range(0, valid.shape[0], int(chunk)):
        block = valid[i:i + int(chunk)]
        d = torch.cdist(block, pred_points)          # L2 distance (M_block, N)
        nearest = d.min(dim=1).values
        total = total + nearest.sum()
        count += int(block.shape[0])
    return total / max(count, 1)


def tracked_l2(
    gt_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    tracked_idx: torch.Tensor,
) -> torch.Tensor:
    """Mean L2 (Euclidean) distance between predicted and GT positions at
    tracked indices.

    gt_coords:   (N, 3) GT particle positions at this frame
    pred_coords: (N, 3)
    tracked_idx: (K,) int64 — indices into particle array (use only >= 0)
    """
    idx = tracked_idx[tracked_idx >= 0].long()
    if idx.numel() == 0:
        return pred_coords.new_zeros(())
    return (pred_coords[idx] - gt_coords[idx]).norm(dim=-1).mean()


# --- Loading utilities ---


def load_refiner(checkpoint_path: str, cfg, device: torch.device) -> Refiner:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = cfg.model
    if "cfg" in ckpt and "model" in ckpt["cfg"]:
        ckpt_model_cfg = OmegaConf.create(ckpt["cfg"]["model"])
        ckpt_model_cfg.input_mode = "observed_control"
        model_cfg = ckpt_model_cfg
    model = Refiner(model_cfg).to(device)
    model.load_state_dict(model_state_from_checkpoint(ckpt), strict=True)
    model.requires_grad_(False)
    model.eval()
    return model


def build_engine(cfg, device: torch.device, root: Optional[str] = None) -> GVCRolloutEngine:
    rollout_cfg = cfg.rollout
    if root is not None:
        path = Path(root)
        case = path.parent.name if path.name.startswith("episode_") else path.name
        override = (cfg.get("per_sample_rollout_config", {}) or {}).get(case, {})
        rollout_cfg = OmegaConf.merge(rollout_cfg, override)
    return GVCRolloutEngine(rollout_cfg, device=device)
