"""Frozen backbone embedder registry.

Only used by the streaming training path (StreamingLitDataModule): the precomputed-.npy
path needs no embedder since tokens are already extracted. New embedder = one import + one
registry line + a config name.
"""

from __future__ import annotations

from typing import Mapping

from ...registry import build_from_registry
from .hf_backbone import HFBackboneEmbedder

__all__ = ["HFBackboneEmbedder", "build_embedder"]

_EMBEDDER_REGISTRY: dict[str, type[HFBackboneEmbedder]] = {
    "hf_backbone": HFBackboneEmbedder,
}


def build_embedder(cfg: Mapping) -> HFBackboneEmbedder:
    return build_from_registry(_EMBEDDER_REGISTRY, cfg, label="embedder")
