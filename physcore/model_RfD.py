"""
The RfD network: a sparse 3D U-Net predicting per-cell velocity residuals.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

# Reuse MPM's stabilized corotated stress (Warp-backed SVD with stable backward,
# F clamp, nan_to_num, surrogate J backward, final stress clamp) instead of a
# raw torch.linalg.svd reimplementation. This is the same stress code path the
# differentiable rollout backprops through via the MPM constitutive model.
from physcore.constitutive.physical_constitutive_models.elasticity import (
    CorotatedElasticity,
)
from torch import nn, Tensor
from torch.nn import functional as F

import spconv.pytorch as spconv
from spconv.pytorch import (
    SparseConvTensor,
    SparseConv3d,
    SparseInverseConv3d,
    SubMConv3d,
)


# ---------------------------------------------------------------------------
# MPM B-spline P2G kernel + per-particle feature projection to grid
# (mirrors physcore/mpm/mpm_model.py's p2g2p B-spline so weights are bit-identical)
# ---------------------------------------------------------------------------

_OFFSET_CACHE: dict[Tuple[str, torch.dtype], Tensor] = {}


def _bspline_offsets(device: torch.device, dtype: torch.dtype) -> Tensor:
    key = (str(device), dtype)
    cached = _OFFSET_CACHE.get(key)
    if cached is None:
        offsets = torch.stack(
            torch.meshgrid(
                torch.arange(3, device=device, dtype=dtype),
                torch.arange(3, device=device, dtype=dtype),
                torch.arange(3, device=device, dtype=dtype),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(27, 3)
        _OFFSET_CACHE[key] = offsets
        cached = offsets
    return cached


def _bspline_weights_and_indices(x: Tensor, num_grids: int) -> Tuple[Tensor, Tensor]:
    """MPM quadratic B-spline weights + flat grid indices for one batched frame.

    x: (B, N, 3) particle positions in normalized world coords (0..1).
    Returns:
        weight:     (B, N, 27)
        index_flat: (B, N, 27) cell index in [0, G^3) per batch element.
    """
    inv_dx = float(num_grids)
    px = x * inv_dx
    base = (px - 0.5).long()
    fx = px - base.float()

    w0 = 0.5 * (1.5 - fx) ** 2
    w1 = 0.75 - (fx - 1.0) ** 2
    w2 = 0.5 * (fx - 0.5) ** 2
    w = torch.stack([w0, w1, w2], dim=-1)  # (B, N, 3, 3)
    w_e = torch.einsum("bni,bnj,bnk->bnijk", w[..., 0], w[..., 1], w[..., 2])
    weight = w_e.reshape(*x.shape[:2], 27)

    offsets = _bspline_offsets(x.device, torch.long)
    index_xyz = base.unsqueeze(2) + offsets.view(1, 1, 27, 3)
    g = num_grids
    index_flat = (
        index_xyz[..., 0] * g * g
        + index_xyz[..., 1] * g
        + index_xyz[..., 2]
    ).clamp(0, g ** 3 - 1)
    return weight, index_flat


_COROTATED_STRESS_CACHE: Dict[Tuple[torch.device, torch.dtype], CorotatedElasticity] = {}


def _get_corotated_module(device: torch.device, dtype: torch.dtype) -> CorotatedElasticity:
    """Return a process-wide `CorotatedElasticity` instance for (device, dtype).

    The module has no trainable state we care about (only a dummy ``useless``
    parameter from its base class). Caching by (device, dtype) avoids re-creating
    the Warp-backed SVD on every hook firing.
    """
    key = (device, dtype)
    module = _COROTATED_STRESS_CACHE.get(key)
    if module is None:
        module = CorotatedElasticity().to(device=device, dtype=dtype)
        _COROTATED_STRESS_CACHE[key] = module
    return module


def _corotated_stress_inline(F_p: Tensor, log_E: Tensor, nu: Tensor) -> Tensor:
    """Corotated stress (sigma) per particle, delegating to MPM's stabilized
    ``CorotatedElasticity``. The MPM module uses a Warp-backed SVD with a
    stable backward, clamps F to [-2, 2], `nan_to_num`s intermediates, and
    uses a surrogate backward for J — all the safeguards a raw
    ``torch.linalg.svd`` adjoint lacks.

    F_p:    (B, N, 3, 3)
    log_E:  (B, N)
    nu:     (B, N)
    Returns: (B, N, 3, 3)
    """
    B, N = F_p.shape[:2]
    F_flat = F_p.reshape(B * N, 3, 3)
    log_E_flat = log_E.reshape(B * N)
    nu_flat = nu.reshape(B * N)
    module = _get_corotated_module(F_p.device, F_p.dtype)
    stress_flat = module(F_flat, log_E_flat, nu_flat)  # (B*N, 3, 3)
    return stress_flat.reshape(B, N, 3, 3)


def _symmetric_stress_to_6(stress: Tensor) -> Tensor:
    """Pack a (near-)symmetric 3x3 tensor into 6 channels: xx, yy, zz, xy, xz, yz."""
    return torch.stack(
        [
            stress[..., 0, 0],
            stress[..., 1, 1],
            stress[..., 2, 2],
            0.5 * (stress[..., 0, 1] + stress[..., 1, 0]),
            0.5 * (stress[..., 0, 2] + stress[..., 2, 0]),
            0.5 * (stress[..., 1, 2] + stress[..., 2, 1]),
        ],
        dim=-1,
    )


# Per-active-cell channel layout:
#   [v_grid(3), log1p(m)(1), stress(6), log_E(1), nu(1), conf_logE(1), conf_nu(1),
#    v_p(3), contact(1), dist_ground(1)] = 19
NUM_CELL_FEATURES = 19


def build_active_cell_features(
    x: Tensor,
    v_p: Tensor,
    F_p: Tensor,
    log_E: Tensor,
    nu: Tensor,
    contact_flag: Tensor,
    grid_mv: Tensor,
    grid_m: Tensor,
    num_grids: int,
    ground_height: float,
    active_thresh: float = 1.0e-15,
    stress_log_normalize: bool = True,
    detach_particle_features: bool = False,
    material_confidence: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Project per-particle features to active grid cells via mass-weighted B-spline P2G.

    ``material_confidence`` is the per-particle Refiner confidence:
      - (B, N, 2)  — separate confidence for log_E and nu (current default Refiner)
      - (B, N, 1)  — shared confidence
      - None       — default to ``torch.ones(B, N, 2)`` (a caller with no MfM estimate)
    Confidence is passed raw (range ``[1, inf)``); it lives in 2 channels here.

    Returns:
        features:    (num_active, NUM_CELL_FEATURES) per-active-cell channels.
        indices:     (num_active, 4) int32 [batch, x, y, z] for spconv.
        active_flat: (B, G^3) bool — which flat cells are active.
    """
    B, N, _ = x.shape
    g = int(num_grids)

    weight, index = _bspline_weights_and_indices(x, g)  # (B, N, 27)

    # Confidence normalization: always emit 2 channels (conf_logE, conf_nu).
    if material_confidence is None:
        conf = log_E.new_ones(B, N, 2)
    elif material_confidence.shape[-1] == 1:
        conf = material_confidence.expand(B, N, 2).to(log_E.dtype)
    else:
        conf = material_confidence[..., :2].to(log_E.dtype)

    if detach_particle_features:
        x_d, v_d, F_d, log_E_d, nu_d, contact_d = (
            x.detach(), v_p.detach(), F_p.detach(),
            log_E.detach(), nu.detach(), contact_flag.detach(),
        )
        conf_d = conf.detach()
    else:
        x_d, v_d, F_d, log_E_d, nu_d, contact_d = x, v_p, F_p, log_E, nu, contact_flag
        conf_d = conf

    stress = _corotated_stress_inline(F_d, log_E_d, nu_d)  # (B, N, 3, 3)
    stress6 = _symmetric_stress_to_6(stress)
    if stress_log_normalize:
        stress6 = torch.sign(stress6) * torch.log1p(stress6.abs())

    # Per-particle bundle: stress6(6) + log_E(1) + nu(1) + conf(2) + v_p(3) + contact(1) = 14
    per_particle = torch.cat(
        [
            stress6,
            log_E_d.unsqueeze(-1),
            nu_d.unsqueeze(-1),
            conf_d,
            v_d,
            contact_d.unsqueeze(-1).to(v_d.dtype),
        ],
        dim=-1,
    )  # (B, N, 14)
    n_aux = per_particle.shape[-1]

    # Mass-weighted P2G: scatter (w_ip * f_p) and (w_ip) separately, then divide.
    tap_feat = (weight.unsqueeze(-1) * per_particle.unsqueeze(2)).reshape(B, N * 27, n_aux)
    tap_w = weight.reshape(B, N * 27)
    idx_flat_part = index.reshape(B, N * 27)

    aux_grid = torch.zeros(B, g ** 3, n_aux, device=x.device, dtype=tap_feat.dtype)
    aux_grid.scatter_add_(1, idx_flat_part.unsqueeze(-1).expand(-1, -1, n_aux), tap_feat)
    weight_grid = torch.zeros(B, g ** 3, device=x.device, dtype=tap_w.dtype)
    weight_grid.scatter_add_(1, idx_flat_part, tap_w)
    aux_grid = aux_grid / weight_grid.clamp_min(1.0e-8).unsqueeze(-1)

    # BatchedMPMModel stores grid_mv as (B*G^3, 3) and grid_m as (B*G^3,) — reshape per-batch.
    grid_mv_b = grid_mv.view(B, g ** 3, 3)
    grid_m_b = grid_m.view(B, g ** 3)
    active_flat = grid_m_b > active_thresh

    # Per-cell distance to ground (cell-center z minus ground_height)
    cell_idx = torch.arange(g ** 3, device=x.device)
    cz = (cell_idx % g).float()
    inv_dx = float(g)
    cell_center_z = (cz + 0.5) / inv_dx
    dist_ground_full = (cell_center_z - float(ground_height)).unsqueeze(0).expand(B, -1)

    mass_feat = torch.log1p(grid_m_b)

    feats_full = torch.cat(
        [
            grid_mv_b,                       # 3
            mass_feat.unsqueeze(-1),         # 1
            aux_grid,                        # 14 (stress6 + log_E + nu + conf2 + v_p + contact)
            dist_ground_full.unsqueeze(-1),  # 1
        ],
        dim=-1,
    )  # (B, G^3, 19)

    # Active-cell selection → spconv-compatible (indices, features) packed across batch.
    b_idx, flat_idx = torch.where(active_flat)
    features = feats_full[b_idx, flat_idx]  # (num_active, 19)

    # Grid order in MPM: flat = x*G^2 + y*G + z (cf. physcore/mpm/mpm_model.py p2g2p).
    cx = (flat_idx // (g * g)).to(torch.int32)
    cy = ((flat_idx // g) % g).to(torch.int32)
    cz_i = (flat_idx % g).to(torch.int32)
    indices = torch.stack([b_idx.to(torch.int32), cx, cy, cz_i], dim=-1)

    return features, indices, active_flat


# ---------------------------------------------------------------------------
# Per-substep controller velocity (Catmull-Rom window + per-substep linear interp)
# ---------------------------------------------------------------------------


def controller_v_at_substep(
    controller_window: Optional[Tensor],
    step: int,
    substep_idx: int,
    steps_per_frame: int,
    frame_dt: float,
) -> Optional[Tensor]:
    """Return controller velocity at (step, substep_idx) inside a Catmull-Rom-smoothed window.

    The window is the output of `episode_runtime._catmull_rom_window(...)` and
    has shape (B, T+1, K, 3) (or (T+1, K, 3) for the unbatched flavor), where
    consecutive entries are at consecutive camera-frame times. The engine
    internally linearly interpolates between consecutive entries per substep;
    we replicate that here so the corrector sees the same action the engine
    will apply.

    Returns velocity (m/s) at the (step, substep) timepoint, shape (B, K, 3)
    or (K, 3), or None if no controller is present.
    """
    if controller_window is None:
        return None
    if controller_window.ndim == 4:  # (B, T+1, K, 3)
        start = controller_window[:, step]
        end = controller_window[:, step + 1]
    elif controller_window.ndim == 3:  # (T+1, K, 3)
        start = controller_window[step]
        end = controller_window[step + 1]
    else:
        return None
    # Per-substep linear interp — but we want *velocity*, which is (end - start) / frame_dt
    # (constant across substeps in the engine's linear-interp scheme). This matches
    # the rate at which the controller hook drives grid velocity inside p2g2p.
    if frame_dt <= 0.0:
        return None
    return (end - start) / float(frame_dt)


# ---------------------------------------------------------------------------
# Sparse 3D U-Net (spconv) with FiLM
# ---------------------------------------------------------------------------


class _FiLMSubM(nn.Module):
    """SubMConv3d (preserves sparsity) -> GroupNorm -> FiLM -> SiLU."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, indice_key: str, groups: int = 8):
        super().__init__()
        self.conv = SubMConv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        self.norm = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.film = nn.Linear(cond_dim, 2 * out_ch)
        self.out_ch = out_ch

    def forward(self, x: SparseConvTensor, cond: Tensor) -> SparseConvTensor:
        x = self.conv(x)
        feats = x.features
        # GroupNorm over channel axis — treat active-cell axis as (N,) and add a
        # dummy spatial dim so GroupNorm normalizes per channel only.
        feats = self.norm(feats.unsqueeze(-1)).squeeze(-1)
        batch_idx = x.indices[:, 0].long()
        gb = self.film(cond)  # (B, 2*out_ch)
        gamma, beta = gb.chunk(2, dim=-1)
        feats = feats * (1.0 + gamma[batch_idx]) + beta[batch_idx]
        return x.replace_feature(F.silu(feats))


class _FiLMSubMBlock(nn.Module):
    """Two stacked _FiLMSubM at the same sparsity level (shared indice_key)."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, indice_key: str):
        super().__init__()
        self.c1 = _FiLMSubM(in_ch, out_ch, cond_dim, indice_key)
        self.c2 = _FiLMSubM(out_ch, out_ch, cond_dim, indice_key)

    def forward(self, x: SparseConvTensor, cond: Tensor) -> SparseConvTensor:
        return self.c2(self.c1(x, cond), cond)


class GridVelocityCorrector(nn.Module):
    """Sparse 3D U-Net predicting per-grid velocity residuals on active cells.

    Input:
        sp_in:  spconv.SparseConvTensor of (num_active, in_channels) features
                with spatial_shape [G, G, G] and explicit batch_size.
        cond:   (B, cond_dim) FiLM conditioning vector.

    Output:
        delta_v_features: (num_active, 3) per-active-cell residual bounded by
                          max_delta_v * tanh(.). Zero-initialized head so the
                          corrector starts as the identity (Delta v = 0).
        out_indices:      (num_active, 4) int32 [batch, x, y, z] — identical to
                          sp_in.indices (SubM blocks preserve sparsity).
    """

    def __init__(
        self,
        in_channels: int = NUM_CELL_FEATURES,
        cond_dim: int = 16,
        base_channels: int = 32,
        levels: int = 2,
        max_delta_v: float = 0.05,
    ):
        super().__init__()
        self.max_delta_v = float(max_delta_v)
        self.levels = int(levels)

        c = [base_channels * (2 ** i) for i in range(levels + 1)]

        self.stem = _FiLMSubMBlock(in_channels, c[0], cond_dim, indice_key="sub0")

        self.downs = nn.ModuleList()
        self.enc_blocks = nn.ModuleList()
        for i in range(levels):
            self.downs.append(
                SparseConv3d(c[i], c[i + 1], kernel_size=2, stride=2, bias=False, indice_key=f"down{i}")
            )
            self.enc_blocks.append(
                _FiLMSubMBlock(c[i + 1], c[i + 1], cond_dim, indice_key=f"sub{i + 1}")
            )

        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(levels)):
            self.ups.append(
                SparseInverseConv3d(c[i + 1], c[i], kernel_size=2, bias=False, indice_key=f"down{i}")
            )
            self.dec_blocks.append(
                _FiLMSubMBlock(2 * c[i], c[i], cond_dim, indice_key=f"sub{i}")
            )

        self.head = SubMConv3d(c[0], 3, kernel_size=1, bias=True, indice_key="sub0_head")
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, sp_in: SparseConvTensor, cond: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.stem(sp_in, cond)
        skips = [x]
        for dn, blk in zip(self.downs, self.enc_blocks):
            x = dn(x)
            x = blk(x, cond)
            skips.append(x)

        x = skips[-1]
        for level, (up, dec) in enumerate(zip(self.ups, self.dec_blocks)):
            x = up(x)
            skip = skips[self.levels - 1 - level]
            fused = torch.cat([x.features, skip.features], dim=-1)
            x = x.replace_feature(fused)
            x = dec(x, cond)

        out = self.head(x)
        delta_v = self.max_delta_v * torch.tanh(out.features)  # (num_active, 3)
        return delta_v, out.indices


# ---------------------------------------------------------------------------
# Conditioning vector builder
# ---------------------------------------------------------------------------


def build_film_cond(
    substep_idx: int,
    steps_per_frame: int,
    step: int,
    total_steps: int,
    controller_v: Optional[Tensor],
    log_E: Tensor,
    nu: Tensor,
    ground_friction: float,
    ground_height: float,
) -> Tensor:
    """Assemble the (B, 16) FiLM conditioning vector.

    Channels:
        0-3:   sin/cos of substep phase and step phase
        4-6:   controller mean velocity (3-vec, zero if absent)
        7:     ||controller v|| mean
        8-11:  mean/std of log_E and nu across particles
        12-13: ground_friction, ground_height
        14-15: substep / steps_per_frame, step / total_steps  (normalized)
    """
    device = log_E.device
    dtype = log_E.dtype
    B = log_E.shape[0]
    out = torch.zeros(B, 16, device=device, dtype=dtype)

    twopi = 6.283185307179586
    sub_phase = float(substep_idx) / max(1, steps_per_frame) * twopi
    step_phase = float(step) / max(1, total_steps) * twopi
    out[:, 0] = float(torch.sin(torch.tensor(sub_phase)))
    out[:, 1] = float(torch.cos(torch.tensor(sub_phase)))
    out[:, 2] = float(torch.sin(torch.tensor(step_phase)))
    out[:, 3] = float(torch.cos(torch.tensor(step_phase)))

    if controller_v is not None and controller_v.numel() > 0:
        cv = controller_v
        if cv.ndim == 3:  # (B, K, 3)
            cv_mean = cv.mean(dim=1)
            cv_norm = cv.norm(dim=-1).mean(dim=-1)
        elif cv.ndim == 2:  # (K, 3)
            cv_mean = cv.mean(dim=0, keepdim=True).expand(B, -1)
            cv_norm = cv.norm(dim=-1).mean().expand(B)
        else:
            cv_mean = torch.zeros(B, 3, device=device, dtype=dtype)
            cv_norm = torch.zeros(B, device=device, dtype=dtype)
        out[:, 4:7] = cv_mean.to(dtype)
        out[:, 7] = cv_norm.to(dtype)

    out[:, 8] = log_E.mean(dim=-1)
    out[:, 9] = log_E.std(dim=-1, unbiased=False)
    out[:, 10] = nu.mean(dim=-1)
    out[:, 11] = nu.std(dim=-1, unbiased=False)
    out[:, 12] = float(ground_friction)
    out[:, 13] = float(ground_height)
    out[:, 14] = float(substep_idx) / max(1, steps_per_frame)
    out[:, 15] = float(step) / max(1, total_steps)
    return out


# ---------------------------------------------------------------------------
# Sparse tensor builders + delta-v scatter back to MPM's flat grid_mv
# ---------------------------------------------------------------------------


def make_sparse_tensor(
    features: Tensor,
    indices: Tensor,
    num_grids: int,
    batch_size: int,
) -> SparseConvTensor:
    return SparseConvTensor(
        features=features,
        indices=indices.to(torch.int32),
        spatial_shape=[num_grids, num_grids, num_grids],
        batch_size=int(batch_size),
    )


def scatter_delta_to_grid_mv(
    delta_v: Tensor,
    indices: Tensor,
    grid_mv: Tensor,
    num_grids: int,
) -> Tensor:
    """Add per-active-cell delta_v back into MPM's flat (B*G^3, 3) grid_mv tensor.

    delta_v: (num_active, 3)
    indices: (num_active, 4) int32 [batch, x, y, z]
    grid_mv: (B*G^3, 3)

    Functional (autograd-safe) — returns a new tensor.
    """
    g = int(num_grids)
    g3 = g * g * g
    b_idx = indices[:, 0].long()
    flat_local = (
        indices[:, 1].long() * g * g
        + indices[:, 2].long() * g
        + indices[:, 3].long()
    )
    global_flat = b_idx * g3 + flat_local
    scattered = torch.zeros_like(grid_mv)
    scattered.index_add_(0, global_flat, delta_v.to(grid_mv.dtype))
    return grid_mv + scattered
