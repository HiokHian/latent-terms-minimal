"""Linear warmup then cosine decay to ``min_lr_ratio``."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """LambdaLR: 0 -> 1 over ``warmup_steps``, then cosine to ``min_lr_ratio``.

    Past ``total_steps`` the multiplier stays at ``min_lr_ratio``. The cosine is
    a multiplier on the base lr, so the peak (at ``step == warmup_steps``) equals
    the configured base lr and the floor equals ``base_lr * min_lr_ratio``.
    """
    warmup_steps = int(warmup_steps)
    total_steps = int(total_steps)
    min_lr_ratio = float(min_lr_ratio)
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if total_steps <= warmup_steps:
        raise ValueError(
            f"cosine_with_warmup requires total_steps ({total_steps}) > "
            f"warmup_steps ({warmup_steps})."
        )

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if step >= total_steps:
            return min_lr_ratio
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)
