"""
Small rollout runtime helpers used by the training loop.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Optional

from .rollout import DifferentiableRolloutEngine


@contextmanager
def temporary_rollout_timestep(
    rollout_engine: DifferentiableRolloutEngine,
    *,
    dt: Optional[float] = None,
    steps_per_frame: Optional[int] = None,
    ground_height: Optional[float] = None,
):
    saved_dt = rollout_engine.dt
    saved_steps_per_frame = rollout_engine.steps_per_frame
    saved_ground_height = getattr(rollout_engine, 'ground_height', None)
    if dt is not None:
        rollout_engine.dt = float(dt)
    if steps_per_frame is not None:
        rollout_engine.steps_per_frame = max(int(steps_per_frame), 1)
    if ground_height is not None and saved_ground_height is not None:
        rollout_engine.ground_height = float(ground_height)
    try:
        yield
    finally:
        rollout_engine.dt = saved_dt
        rollout_engine.steps_per_frame = saved_steps_per_frame
        if saved_ground_height is not None:
            rollout_engine.ground_height = saved_ground_height


def real_world_chunk_dt(
    episode: Dict[str, object],
    rollout_engine: DifferentiableRolloutEngine,
    chunk_steps: int,
) -> Optional[float]:
    if not bool(episode.get('is_real_world', False)):
        return None
    frame_dt = float(episode.get('frame_dt', 0.0))
    if frame_dt <= 0.0:
        return None
    return frame_dt / max(int(chunk_steps) * int(rollout_engine.steps_per_frame), 1)
