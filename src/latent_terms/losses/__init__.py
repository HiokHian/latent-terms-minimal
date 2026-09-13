"""Loss registry. New loss = one import + one registry line + a config name."""

from __future__ import annotations

from typing import Mapping

from ..registry import build_from_registry
from .base import Loss
from .latent_terms_loss import LatentTermsLoss

__all__ = ["Loss", "LatentTermsLoss", "build_loss"]

_LOSS_REGISTRY: dict[str, type[Loss]] = {
    "latent_terms": LatentTermsLoss,
}


def build_loss(cfg: Mapping) -> Loss:
    return build_from_registry(_LOSS_REGISTRY, cfg, label="loss")
