"""
Loss terms for a particle rollout, gathered by compute_rollout_losses.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from omegaconf import DictConfig


def _time_weighted_mean(values: torch.Tensor, time_weights: torch.Tensor | None) -> torch.Tensor:
    if time_weights is None:
        return values.mean()
    weights = time_weights
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).mean() / time_weights.mean().clamp(min=1e-8)


def _reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
    smooth_l1_beta: float,
    time_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if loss_type == 'l1':
        return _time_weighted_mean((prediction - target).abs(), time_weights)
    if loss_type == 'mse':
        return _time_weighted_mean((prediction - target).pow(2), time_weights)
    if loss_type == 'smooth_l1':
        return _time_weighted_mean(
            F.smooth_l1_loss(prediction, target, beta=smooth_l1_beta, reduction='none'),
            time_weights,
        )
    raise ValueError(f'Unsupported loss type: {loss_type}')


def _zero_from(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _masked_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    loss_type: str,
    smooth_l1_beta: float,
) -> torch.Tensor:
    if valid_mask.numel() == 0 or not bool(valid_mask.any()):
        return _zero_from(prediction)
    valid = valid_mask.bool()
    while valid.ndim < prediction.ndim:
        valid = valid.unsqueeze(-1)
    if loss_type == 'l1':
        values = (prediction - target).abs()
    elif loss_type == 'mse':
        values = (prediction - target).pow(2)
    elif loss_type == 'smooth_l1':
        values = F.smooth_l1_loss(prediction, target, beta=smooth_l1_beta, reduction='none')
    else:
        raise ValueError(f'Unsupported loss type: {loss_type}')
    denom = valid.sum().to(values.dtype).clamp(min=1.0) * prediction.shape[-1]
    return (values * valid.to(values.dtype)).sum() / denom


def _distance_loss(distances: torch.Tensor, loss_type: str, smooth_l1_beta: float) -> torch.Tensor:
    if distances.numel() == 0:
        return distances.sum() * 0.0
    zeros = torch.zeros_like(distances)
    if loss_type == 'l1':
        return distances.abs().mean()
    if loss_type == 'mse':
        return distances.pow(2).mean()
    if loss_type == 'smooth_l1':
        return F.smooth_l1_loss(distances, zeros, beta=smooth_l1_beta, reduction='mean')
    raise ValueError(f'Unsupported loss type: {loss_type}')


def _select_evenly(points: torch.Tensor, max_points: int) -> torch.Tensor:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    indices = torch.linspace(
        0,
        points.shape[0] - 1,
        steps=max_points,
        device=points.device,
    ).round().long()
    return points.index_select(0, indices)


def _real_world_correspondence_loss(
    predicted_positions: torch.Tensor,
    observation_points: torch.Tensor,
    observation_ids: torch.Tensor,
    observation_valid: torch.Tensor,
    loss_type: str,
    smooth_l1_beta: float,
) -> torch.Tensor:
    losses = []
    num_steps = min(predicted_positions.shape[0], observation_points.shape[0])
    num_particles = predicted_positions.shape[1]
    for step_idx in range(num_steps):
        ids = observation_ids[step_idx].long()
        valid = observation_valid[step_idx].bool() & (ids >= 0) & (ids < num_particles)
        if not bool(valid.any()):
            continue
        gathered = predicted_positions[step_idx].index_select(0, ids[valid])
        losses.append(
            _masked_reconstruction_loss(
                gathered,
                observation_points[step_idx, valid],
                torch.ones(gathered.shape[0], dtype=torch.bool, device=gathered.device),
                loss_type,
                smooth_l1_beta,
            )
        )
    if not losses:
        return _zero_from(predicted_positions)
    return torch.stack(losses).mean()


def _real_world_tracking_l2_squared_loss(
    predicted_positions: torch.Tensor,
    observation_points: torch.Tensor,
    observation_ids: torch.Tensor,
    observation_valid: torch.Tensor,
) -> torch.Tensor:
    losses = []
    num_steps = min(predicted_positions.shape[0], observation_points.shape[0])
    num_particles = predicted_positions.shape[1]
    for step_idx in range(num_steps):
        ids = observation_ids[step_idx].long()
        valid = observation_valid[step_idx].bool() & (ids >= 0) & (ids < num_particles)
        if not bool(valid.any()):
            continue
        gathered = predicted_positions[step_idx].index_select(0, ids[valid])
        displacement = gathered - observation_points[step_idx, valid]
        losses.append(displacement.pow(2).sum(dim=-1).mean())
    if not losses:
        return _zero_from(predicted_positions)
    return torch.stack(losses).mean()


def _one_sided_observation_to_prediction_loss(
    predicted_positions: torch.Tensor,
    observation_points: torch.Tensor,
    observation_valid: torch.Tensor,
    loss_type: str,
    smooth_l1_beta: float,
    max_observed_points: int,
    max_predicted_points: int,
) -> torch.Tensor:
    losses = []
    num_steps = min(predicted_positions.shape[0], observation_points.shape[0])
    for step_idx in range(num_steps):
        observed = observation_points[step_idx, observation_valid[step_idx].bool()]
        if observed.numel() == 0:
            continue
        observed = _select_evenly(observed, max_observed_points)
        predicted = _select_evenly(predicted_positions[step_idx], max_predicted_points)
        nearest = torch.cdist(observed, predicted).amin(dim=-1)
        losses.append(_distance_loss(nearest, loss_type, smooth_l1_beta))
    if not losses:
        return _zero_from(predicted_positions)
    return torch.stack(losses).mean()


def _visible_prediction_to_observation_loss(
    predicted_positions: torch.Tensor,
    observation_points: torch.Tensor,
    observation_ids: torch.Tensor,
    observation_valid: torch.Tensor,
    loss_type: str,
    smooth_l1_beta: float,
    max_observed_points: int,
    max_visible_predicted_points: int,
) -> torch.Tensor:
    losses = []
    num_steps = min(predicted_positions.shape[0], observation_points.shape[0])
    num_particles = predicted_positions.shape[1]
    for step_idx in range(num_steps):
        ids = observation_ids[step_idx].long()
        valid = observation_valid[step_idx].bool() & (ids >= 0) & (ids < num_particles)
        if not bool(valid.any()):
            continue
        observed = _select_evenly(observation_points[step_idx, valid], max_observed_points)
        visible_ids = torch.unique(ids[valid])
        predicted_visible = predicted_positions[step_idx].index_select(0, visible_ids)
        predicted_visible = _select_evenly(predicted_visible, max_visible_predicted_points)
        nearest = torch.cdist(predicted_visible, observed).amin(dim=-1)
        losses.append(_distance_loss(nearest, loss_type, smooth_l1_beta))
    if not losses:
        return _zero_from(predicted_positions)
    return torch.stack(losses).mean()


def _temporal_smoothness_loss(predicted_flows: torch.Tensor) -> torch.Tensor:
    if predicted_flows.shape[0] <= 1:
        return _zero_from(predicted_flows)
    return (predicted_flows[1:] - predicted_flows[:-1]).pow(2).mean()


def _compute_real_world_reconstruction_loss(
    sample: Dict,
    predicted_positions: torch.Tensor,
    predicted_flows: torch.Tensor,
    loss_cfg: DictConfig,
    loss_type: str,
    smooth_l1_beta: float,
) -> Dict[str, torch.Tensor]:
    real_cfg = loss_cfg.get('real_world_reconstruction', {})
    observation_points = sample.get('future_observation_points_clean', None)
    observation_valid = sample.get('future_observation_valid_mask', None)
    observation_ids = sample.get('future_observation_particle_ids', None)
    if observation_points is None or observation_valid is None or observation_ids is None:
        zero = _zero_from(predicted_positions)
        return {
            'real_recon_loss': zero,
            'real_correspondence_loss': zero,
            'real_obs_to_pred_loss': zero,
            'real_pred_visible_to_obs_loss': zero,
            'real_temporal_smoothness_loss': _temporal_smoothness_loss(predicted_flows),
        }

    observation_points = observation_points[:predicted_positions.shape[0]]
    observation_valid = observation_valid[:predicted_positions.shape[0]]
    observation_ids = observation_ids[:predicted_positions.shape[0]]
    # Use all available points by default. Subsampling biases obs_to_pred upward
    # on dense real-world depth observations and should not be used for metrics.
    max_observed_points = int(real_cfg.get('chamfer_max_observed_points', 0))
    max_predicted_points = int(real_cfg.get('chamfer_max_predicted_points', 0))
    max_visible_predicted_points = int(real_cfg.get('chamfer_max_visible_predicted_points', 0))

    if bool(real_cfg.get('use_empm_loss', False)):
        tracking_loss = _real_world_tracking_l2_squared_loss(
            predicted_positions,
            observation_points,
            observation_ids,
            observation_valid,
        )
        dist_loss = _one_sided_observation_to_prediction_loss(
            predicted_positions,
            observation_points,
            observation_valid,
            str(real_cfg.get('dist_loss_type', loss_type)).lower(),
            smooth_l1_beta,
            max_observed_points=max_observed_points,
            max_predicted_points=max_predicted_points,
        )
        zero = _zero_from(predicted_positions)
        real_recon_loss = (
            float(real_cfg.get('dist_weight', real_cfg.get('obs_to_pred_weight', 1.0))) * dist_loss
            + float(real_cfg.get('track_weight', real_cfg.get('correspondence_weight', 1.0))) * tracking_loss
        )
        return {
            'real_recon_loss': real_recon_loss,
            'real_correspondence_loss': tracking_loss,
            'real_obs_to_pred_loss': dist_loss,
            'real_pred_visible_to_obs_loss': zero,
            'real_temporal_smoothness_loss': zero,
        }

    correspondence_loss = _real_world_correspondence_loss(
        predicted_positions,
        observation_points,
        observation_ids,
        observation_valid,
        loss_type,
        smooth_l1_beta,
    )
    obs_to_pred_loss = _one_sided_observation_to_prediction_loss(
        predicted_positions,
        observation_points,
        observation_valid,
        loss_type,
        smooth_l1_beta,
        max_observed_points=max_observed_points,
        max_predicted_points=max_predicted_points,
    )
    pred_visible_to_obs_loss = _visible_prediction_to_observation_loss(
        predicted_positions,
        observation_points,
        observation_ids,
        observation_valid,
        loss_type,
        smooth_l1_beta,
        max_observed_points=max_observed_points,
        max_visible_predicted_points=max_visible_predicted_points,
    )
    temporal_smoothness = _temporal_smoothness_loss(predicted_flows)
    real_recon_loss = (
        float(real_cfg.get('correspondence_weight', 1.0)) * correspondence_loss
        + float(real_cfg.get('obs_to_pred_weight', 0.25)) * obs_to_pred_loss
        + float(real_cfg.get('pred_visible_to_obs_weight', 0.1)) * pred_visible_to_obs_loss
        + float(real_cfg.get('temporal_smoothness_weight', 0.01)) * temporal_smoothness
    )

    return {
        'real_recon_loss': real_recon_loss,
        'real_correspondence_loss': correspondence_loss,
        'real_obs_to_pred_loss': obs_to_pred_loss,
        'real_pred_visible_to_obs_loss': pred_visible_to_obs_loss,
        'real_temporal_smoothness_loss': temporal_smoothness,
    }


def _time_ramp(num_steps: int, final_factor: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if num_steps <= 1 or abs(final_factor - 1.0) < 1e-8:
        return torch.ones(num_steps, device=device, dtype=dtype)
    return torch.linspace(1.0, final_factor, steps=num_steps, device=device, dtype=dtype)


def compute_rollout_losses(sample: Dict, rollout_output: Dict, loss_cfg: DictConfig) -> Dict[str, torch.Tensor]:
    predicted_positions = rollout_output['predicted_positions']
    predicted_flows = rollout_output['predicted_flows']
    target_positions = sample['future_particle_positions'][:predicted_positions.shape[0]]
    target_flows = sample['future_particle_flows'][:predicted_flows.shape[0]]
    delta_external_forces = rollout_output['delta_external_forces']
    delta_material_properties = rollout_output['delta_material_properties']
    predicted_material_log_E = rollout_output['predicted_material_log_E']
    predicted_material_nu = rollout_output['predicted_material_nu']

    smooth_l1_beta = float(loss_cfg.get('smooth_l1_beta', 1.0))
    default_loss_type = str(loss_cfg.get('reconstruction_loss_type', 'l1')).lower()
    position_loss_type = str(loss_cfg.get('position_loss_type', default_loss_type)).lower()
    flow_loss_type = str(loss_cfg.get('flow_loss_type', default_loss_type)).lower()
    material_loss_type = str(loss_cfg.get('material_loss_type', default_loss_type)).lower()
    dense_position_loss = _reconstruction_loss(predicted_positions, target_positions, position_loss_type, smooth_l1_beta)
    dense_flow_loss = _reconstruction_loss(predicted_flows, target_flows, flow_loss_type, smooth_l1_beta)
    is_real_world = bool(sample.get('is_real_world', False))
    real_cfg = loss_cfg.get('real_world_reconstruction', {})
    use_real_reconstruction = is_real_world and bool(real_cfg.get('enabled', True))
    if use_real_reconstruction:
        real_losses = _compute_real_world_reconstruction_loss(
            sample,
            predicted_positions,
            predicted_flows,
            loss_cfg,
            position_loss_type,
            smooth_l1_beta,
        )
    else:
        zero = _zero_from(predicted_positions)
        real_losses = {
            'real_recon_loss': zero,
            'real_correspondence_loss': zero,
            'real_obs_to_pred_loss': zero,
            'real_pred_visible_to_obs_loss': zero,
            'real_temporal_smoothness_loss': zero,
        }
    if use_real_reconstruction:
        position_loss = real_losses['real_recon_loss']
        flow_loss = (
            float(real_cfg.get('track_flow_weight', 0.0))
            * dense_flow_loss
        )
    else:
        position_loss = dense_position_loss
        flow_loss = dense_flow_loss
    material_time_weights = _time_ramp(
        predicted_material_log_E.shape[0],
        float(loss_cfg.get('material_ramp_final_factor', 2.0)),
        predicted_material_log_E.device,
        predicted_material_log_E.dtype,
    )
    force_time_weights = _time_ramp(
        delta_external_forces.shape[0],
        float(loss_cfg.get('force_reg_ramp_final_factor', 4.0)),
        delta_external_forces.device,
        delta_external_forces.dtype,
    )

    target_material_log_E = sample['gt_material_log_E'].unsqueeze(0).expand_as(predicted_material_log_E)
    target_material_nu = sample['gt_material_nu'].unsqueeze(0).expand_as(predicted_material_nu)
    material_log_E_loss = _reconstruction_loss(
        predicted_material_log_E,
        target_material_log_E,
        material_loss_type,
        smooth_l1_beta,
        time_weights=material_time_weights,
    )
    material_nu_loss = _reconstruction_loss(
        predicted_material_nu,
        target_material_nu,
        material_loss_type,
        smooth_l1_beta,
        time_weights=material_time_weights,
    )
    material_loss = 0.5 * (material_log_E_loss + material_nu_loss)
    if use_real_reconstruction and not bool(real_cfg.get('supervise_material', False)):
        material_log_E_loss = _zero_from(predicted_material_log_E)
        material_nu_loss = _zero_from(predicted_material_nu)
        material_loss = _zero_from(predicted_material_log_E)
    force_regularization = _time_weighted_mean(delta_external_forces.pow(2).mean(dim=(-1, -2)), force_time_weights)
    material_delta_regularization = _time_weighted_mean(
        delta_material_properties.pow(2).mean(dim=(-1, -2)),
        material_time_weights,
    )

    total = (
        float(loss_cfg.get('position_weight', 1.0)) * position_loss
        + float(loss_cfg.get('flow_weight', 0.5)) * flow_loss
        + float(loss_cfg.get('material_weight', 0.1)) * material_loss
        + float(loss_cfg.get('force_reg_weight', 1.0e-4)) * force_regularization
        + float(loss_cfg.get('material_delta_reg_weight', 1.0e-4)) * material_delta_regularization
    )

    return {
        'total': total,
        'position_loss': position_loss.detach(),
        'flow_loss': flow_loss.detach(),
        'force_reg_loss': force_regularization.detach(),
        'material_loss': material_loss.detach(),
        'material_log_E_loss': material_log_E_loss.detach(),
        'material_nu_loss': material_nu_loss.detach(),
        'material_delta_reg_loss': material_delta_regularization.detach(),
        'dense_position_loss': dense_position_loss.detach(),
        'dense_flow_loss': dense_flow_loss.detach(),
        'real_recon_loss': real_losses['real_recon_loss'].detach(),
        'real_correspondence_loss': real_losses['real_correspondence_loss'].detach(),
        'real_obs_to_pred_loss': real_losses['real_obs_to_pred_loss'].detach(),
        'real_pred_visible_to_obs_loss': real_losses['real_pred_visible_to_obs_loss'].detach(),
        'real_temporal_smoothness_loss': real_losses['real_temporal_smoothness_loss'].detach(),
        'real_sample': torch.as_tensor(float(use_real_reconstruction), device=predicted_positions.device),
    }
