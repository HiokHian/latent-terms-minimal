"""SAE layer registry.

Every variant returns SAEOutput so the loss and LitModule stay variant-agnostic.
hidden_dim must be present in the cfg dict alongside name/n_latents/topk.
"""

from __future__ import annotations

from typing import Mapping

from ...registry import build_from_registry
from .layer import SAELayer, SAEOutput

__all__ = ["SAELayer", "SAEOutput", "build_sae"]

_SAE_REGISTRY: dict[str, type[SAELayer]] = {
    "top_k": SAELayer,
}


def build_sae(cfg: Mapping) -> SAELayer:
    """Build an SAE layer from config. ``cfg`` must include ``hidden_dim``."""
    return build_from_registry(_SAE_REGISTRY, cfg, label="sae")
