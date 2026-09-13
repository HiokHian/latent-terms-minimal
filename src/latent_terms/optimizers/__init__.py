"""Optimizer registry."""

from __future__ import annotations

from typing import Callable, Iterable, Mapping

import torch

from ..registry import build_from_registry


def adamw(
    params: Iterable,
    lr: float,
    weight_decay: float = 0.01,  # AdamW's own standard default; not specified by the paper
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
    )


_OPTIMIZER_REGISTRY: dict[str, Callable[..., torch.optim.Optimizer]] = {
    "adamw": adamw,
}

__all__ = ["build_optimizer", "adamw"]


def build_optimizer(params: Iterable, cfg: Mapping) -> torch.optim.Optimizer:
    return build_from_registry(_OPTIMIZER_REGISTRY, cfg, params, label="optimizer")
