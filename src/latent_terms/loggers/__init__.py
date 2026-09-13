"""Logger registry."""

from __future__ import annotations

from typing import Callable, Mapping

from lightning.pytorch.loggers import Logger

from ..registry import build_from_registry
from .csv_logger import csv_logger
from .wandb_logger import wandb_logger

__all__ = ["build_logger", "csv_logger", "wandb_logger"]

_LOGGER_REGISTRY: dict[str, Callable[..., Logger]] = {
    "wandb": wandb_logger,
    "csv": csv_logger,
}


def build_logger(cfg: Mapping) -> Logger:
    return build_from_registry(_LOGGER_REGISTRY, cfg, label="logger")
