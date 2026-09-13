"""
Rollout engine that applies the RfD residual on top of the MPM solve.
The class name `GVCRolloutEngine` and the `_gvc_*` state both mean RfD.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from .rollout import BatchedDifferentiableRolloutEngine

# Callers (e.g. train_RfD.py) insert the repo root onto sys.path before
# importing this module.
from physcore.model_RfD import (  # noqa: E402
    GridVelocityCorrector,
    NUM_CELL_FEATURES,
    build_active_cell_features,
    build_film_cond,
    controller_v_at_substep,
    make_sparse_tensor,
    scatter_delta_to_grid_mv,
)


class GVCRolloutEngine(BatchedDifferentiableRolloutEngine):
    """BatchedDifferentiableRolloutEngine + GridVelocityCorrector hook."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gvc_corrector: Optional[GridVelocityCorrector] = None
        self._gvc_h: int = 10
        self._gvc_active: bool = False
        self._gvc_counter: int = 0
        self._gvc_total_substeps: int = 0
        # Stashed per-rollout context (set by forward, read by hook)
        self._gvc_log_E: Optional[Tensor] = None
        self._gvc_nu: Optional[Tensor] = None
        self._gvc_material_confidence: Optional[Tensor] = None
        self._gvc_contact_ids: Optional[Tensor] = None
        self._gvc_controller_window: Optional[Tensor] = None
        self._gvc_total_steps: int = 0
        self._gvc_frame_dt: float = 0.0

    # ------------------------------------------------------------------ API

    def attach_corrector(self, corrector: GridVelocityCorrector, h: int = 10) -> None:
        """Bind the corrector and the cadence (in MPM substeps)."""
        self._gvc_corrector = corrector
        self._gvc_h = max(int(h), 1)

    def enable_corrector(self, enable: bool = True) -> None:
        self._gvc_active = bool(enable)

    # ------------------------------------------------------------ solver hook

    def _make_batched_solver(self, init_pos: Tensor):
        solver, state = super()._make_batched_solver(init_pos)
        self._install_gvc_hooks(solver.model)
        return solver, state

    def _install_gvc_hooks(self, mpm_model) -> None:
        """Monkey-patch p2g2p to stash particle state + append the corrector hook."""
        original_p2g2p = mpm_model.p2g2p

        def patched_p2g2p(
            x: Tensor,
            v: Tensor,
            C: Tensor,
            F: Tensor,
            stress: Tensor,
            *args,
            **kwargs,
        ):
            mpm_model._gvc_x = x
            mpm_model._gvc_v = v
            mpm_model._gvc_F = F
            return original_p2g2p(x, v, C, F, stress, *args, **kwargs)

        mpm_model.p2g2p = patched_p2g2p
        mpm_model.post_grid_process.append(self._gvc_grid_hook)

    # The hook is a bound method (engine self pre-bound); when invoked from
    # `for op in post_grid_process: op(mpm_model)` it gets `mpm_model` as arg.
    def _gvc_grid_hook(self, mpm_model) -> None:
        if not self._gvc_active or self._gvc_corrector is None:
            return
        counter = self._gvc_counter
        self._gvc_counter = counter + 1
        # Fire on the LAST substep of each h-substep window.
        # With h=10 and 500 substeps per training window, this triggers the
        # corrector at counter = 9, 19, 29, ..., 499 — i.e. immediately
        # before G2P of the last substep in each h-step block, so the
        # residual is applied at the boundary that will then drive the
        # *next* h-step block's dynamics.
        if (counter + 1) % self._gvc_h != 0:
            return

        log_E = self._gvc_log_E
        nu = self._gvc_nu
        if log_E is None or nu is None:
            return

        x = mpm_model._gvc_x
        v = mpm_model._gvc_v
        F = mpm_model._gvc_F
        B, N, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Build per-particle contact flag from contact-particle IDs at the current step.
        step = counter // max(int(self.steps_per_frame), 1)
        substep_idx = counter % max(int(self.steps_per_frame), 1)
        contact_flag = self._contact_flag_at_step(step, B, N, device, dtype)

        # Per-active-cell features
        features, indices, _active = build_active_cell_features(
            x=x,
            v_p=v,
            F_p=F,
            log_E=log_E,
            nu=nu,
            contact_flag=contact_flag,
            grid_mv=mpm_model.grid_mv,
            grid_m=mpm_model.grid_m,
            num_grids=mpm_model.num_grids,
            ground_height=self.ground_height,
            stress_log_normalize=True,
            detach_particle_features=False,
            material_confidence=self._gvc_material_confidence,
        )

        if features.shape[0] == 0:
            return

        # Per-substep controller velocity (Catmull-Rom-smoothed upstream + per-substep linear)
        ctrl_v = controller_v_at_substep(
            self._gvc_controller_window,
            step=step,
            substep_idx=substep_idx,
            steps_per_frame=int(self.steps_per_frame),
            frame_dt=float(self._gvc_frame_dt or (self.dt * self.steps_per_frame)),
        )

        cond = build_film_cond(
            substep_idx=substep_idx,
            steps_per_frame=int(self.steps_per_frame),
            step=step,
            total_steps=int(self._gvc_total_steps),
            controller_v=ctrl_v,
            log_E=log_E,
            nu=nu,
            ground_friction=float(self.ground_friction),
            ground_height=float(self.ground_height),
        )

        sp = make_sparse_tensor(features, indices, mpm_model.num_grids, batch_size=B)
        delta_v, out_indices = self._gvc_corrector(sp, cond)

        delta_v = self._mask_controller_cells(
            delta_v, out_indices, step, substep_idx, mpm_model,
        )

        mpm_model.grid_mv = scatter_delta_to_grid_mv(
            delta_v=delta_v,
            indices=out_indices,
            grid_mv=mpm_model.grid_mv,
            num_grids=mpm_model.num_grids,
        )

    def _contact_flag_at_step(
        self,
        step: int,
        batch_size: int,
        num_particles: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Per-particle binary mask: 1 where the particle is a manipulation contact."""
        flag = torch.zeros(batch_size, num_particles, device=device, dtype=dtype)
        ids = self._gvc_contact_ids
        if ids is None:
            return flag
        if torch.is_tensor(ids):
            # Expected layouts:
            #   - (T, K) (per-frame per-contact index) -> use ids[step] for all batches
            #   - (B, T, K)
            #   - (B, K) (static)
            if ids.ndim == 2 and ids.shape[0] > step:
                row = ids[step].to(device=device, dtype=torch.long)
                row = row[row >= 0]
                if row.numel() > 0:
                    flag[:, row] = 1.0
            elif ids.ndim == 3 and ids.shape[1] > step:
                for b in range(min(batch_size, ids.shape[0])):
                    row = ids[b, step].to(device=device, dtype=torch.long)
                    row = row[row >= 0]
                    if row.numel() > 0:
                        flag[b, row] = 1.0
        return flag

    def _mask_controller_cells(
        self,
        delta_v: Tensor,
        out_indices: Tensor,
        step: int,
        substep_idx: int,
        mpm_model,
    ) -> Tensor:
        """Zero out corrector deltas at grid cells within the controller Dirichlet radius."""
        ctrl_window = self._gvc_controller_window
        if ctrl_window is None:
            return delta_v
        radius = self.manipulation_controller_grid_contact_radius
        if radius <= 0.0:
            return delta_v
        if delta_v.shape[0] == 0:
            return delta_v

        device = delta_v.device
        dtype = delta_v.dtype
        dx = float(mpm_model.dx)

        cw = ctrl_window.to(device=device, dtype=dtype)
        if cw.ndim == 3:
            cw = cw.unsqueeze(0)
        if cw.ndim != 4 or cw.shape[1] <= step + 1 or cw.shape[2] == 0:
            return delta_v

        alpha = float(substep_idx + 1) / float(max(int(self.steps_per_frame), 1))
        ctrl_pos = cw[:, step] + alpha * (cw[:, step + 1] - cw[:, step])  # (B, K, 3)
        B_ctrl = ctrl_pos.shape[0]

        cell_world = out_indices[:, 1:4].to(device=device, dtype=dtype) * dx
        batch_idx = out_indices[:, 0].long().to(device=device)

        mask = torch.ones(delta_v.shape[0], device=device, dtype=dtype)
        for b in range(B_ctrl):
            in_batch = batch_idx == b
            if not in_batch.any():
                continue
            pts = ctrl_pos[b]
            cells_b = cell_world[in_batch]
            dists = torch.cdist(cells_b.unsqueeze(0), pts.unsqueeze(0)).squeeze(0)
            near = dists.min(dim=-1).values <= radius
            mask[in_batch] = mask[in_batch] * (~near).to(dtype)

        return delta_v * mask.unsqueeze(-1)

    # ----------------------------------------------------------- forward wrap

    def forward(
        self,
        positions: Tensor,
        velocities: Tensor,
        F: Tensor,
        delta_velocities: Tensor,
        log_E: Tensor,
        nu: Tensor,
        C: Optional[Tensor] = None,
        material_model_info: Optional[Dict] = None,
        rigid_points: Optional[Tensor] = None,
        rigid_collision_cfg: Optional[Dict] = None,
        rigid_body_primitives: Optional[list] = None,
        direct_velocity: bool = False,
        correction_targets: Optional[Tensor] = None,
        correction_target_mask: Optional[Tensor] = None,
        manipulation_indicator: Optional[Tensor] = None,
        manipulation_contact_particle_ids: Optional[Tensor] = None,
        controller_grid_points: Optional[Tensor] = None,
        material_confidence: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        # Stash hook context for the duration of this rollout
        self._gvc_log_E = log_E
        self._gvc_nu = nu
        self._gvc_material_confidence = material_confidence
        self._gvc_contact_ids = manipulation_contact_particle_ids
        self._gvc_controller_window = controller_grid_points
        if positions.ndim == 3:
            self._gvc_total_steps = int(delta_velocities.shape[0])
        else:
            self._gvc_total_steps = int(delta_velocities.shape[0])
        self._gvc_total_substeps = self._gvc_total_steps * max(int(self.steps_per_frame), 1)
        self._gvc_frame_dt = float(self.dt * self.steps_per_frame)
        self._gvc_counter = 0
        try:
            return super().forward(
                positions,
                velocities,
                F,
                delta_velocities,
                log_E,
                nu,
                C=C,
                material_model_info=material_model_info,
                rigid_points=rigid_points,
                rigid_collision_cfg=rigid_collision_cfg,
                rigid_body_primitives=rigid_body_primitives,
                direct_velocity=direct_velocity,
                correction_targets=correction_targets,
                correction_target_mask=correction_target_mask,
                manipulation_indicator=manipulation_indicator,
                manipulation_contact_particle_ids=manipulation_contact_particle_ids,
                controller_grid_points=controller_grid_points,
            )
        finally:
            self._gvc_log_E = None
            self._gvc_nu = None
            self._gvc_material_confidence = None
            self._gvc_contact_ids = None
            self._gvc_controller_window = None
