"""
Farthest Point Sampling (FPS) utility.
"""

import torch
from torch import Tensor


def farthest_point_sampling(points: Tensor, n_samples: int) -> Tensor:
    """Farthest point sampling on a point cloud.

    Args:
        points: (N, 3) point cloud
        n_samples: number of points to sample

    Returns:
        indices: (n_samples,) long tensor of selected indices
    """
    N = points.shape[0]
    if n_samples >= N:
        return torch.arange(N, device=points.device)

    device = points.device
    selected = torch.zeros(n_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), float('inf'), device=device)

    # Start from a random point
    idx = torch.randint(0, N, (1,), device=device).item()
    selected[0] = idx

    for i in range(1, n_samples):
        dist_to_last = ((points - points[selected[i - 1]]) ** 2).sum(dim=1)
        distances = torch.min(distances, dist_to_last)
        selected[i] = distances.argmax()

    return selected
