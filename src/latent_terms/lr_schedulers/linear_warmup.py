"""Linear warmup from 0 to 1, then flat at 1."""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def linear_warmup(optimizer: Optimizer, warmup_steps: int) -> LambdaLR:
    warmup_steps = int(warmup_steps)
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")

    def _lr_lambda(step: int) -> float:
        return min(1.0, step / max(1, warmup_steps))

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)
