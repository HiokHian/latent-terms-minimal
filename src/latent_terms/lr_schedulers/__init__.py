"""LR-scheduler registry."""

from __future__ import annotations

from typing import Callable, Mapping

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..registry import build_from_registry
from .cosine import cosine_with_warmup
from .linear_warmup import linear_warmup

__all__ = ["build_lr_scheduler", "cosine_with_warmup", "linear_warmup"]

_LR_SCHEDULER_REGISTRY: dict[str, Callable[..., LRScheduler]] = {
    "linear_warmup": linear_warmup,
    "cosine_with_warmup": cosine_with_warmup,
}


def build_lr_scheduler(optimizer: Optimizer, cfg: Mapping) -> LRScheduler:
    return build_from_registry(
        _LR_SCHEDULER_REGISTRY, cfg, optimizer, label="lr_scheduler"
    )
