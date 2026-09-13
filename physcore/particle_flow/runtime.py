"""
Runtime helpers shared by PhysWM training entrypoints.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def should_use_live_tqdm() -> bool:
    return is_main_process() and sys.stderr.isatty()


def get_local_world_size() -> int:
    return int(os.environ.get('LOCAL_WORLD_SIZE', os.environ.get('WORLD_SIZE', 1)))


def get_visible_cuda_device_count() -> int:
    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count <= 0:
        raise RuntimeError('Distributed training requires at least one visible CUDA device.')
    return visible_gpu_count


def setup_distributed() -> Tuple[int, int, int]:
    """Initialize distributed training. Returns (rank, local_rank, world_size)."""
    if 'RANK' not in os.environ:
        return 0, 0, 1

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', device_id=torch.device(f'cuda:{local_rank}'))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    visible_gpu_count = get_visible_cuda_device_count()
    local_world_size = get_local_world_size()
    if local_world_size > visible_gpu_count:
        raise RuntimeError(
            'torch.distributed.run launched more local ranks than visible GPUs. '
            'NCCL does not support multiple distributed ranks on the same GPU for this trainer. '
            'Keep --nproc_per_node equal to the number of visible GPUs and use '
            'train.concurrent_episodes_per_gpu to run multiple episodes concurrently on each GPU.'
        )
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def broadcast_parameters(model: torch.nn.Module, src: int = 0) -> None:
    """Broadcast all model parameters from src rank to all other ranks."""
    if not is_distributed():
        return
    for param in model.parameters():
        dist.broadcast(param.data, src=src)
    for buf in model.buffers():
        dist.broadcast(buf.data, src=src)


def allreduce_gradients(model: torch.nn.Module) -> None:
    """Average gradients across all ranks."""
    if not is_distributed():
        return
    world_size = get_world_size()
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)
