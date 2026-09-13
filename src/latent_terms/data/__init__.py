"""Datamodule and collator registries.

New datamodule/collator = one import + one registry line + a config name, same pattern as
every other pluggable component in this repo (see registry.py).
"""

from __future__ import annotations

from typing import Any, Mapping

import lightning as L

from ..registry import build_from_registry
from .collator import EmbeddingCollator
from .datamodule import EmbeddingDataset, LitDataModule
from .streaming_fineweb import FineWebEduStreamDataset, StreamingLitDataModule, StreamingTextCollator

__all__ = [
    "EmbeddingCollator",
    "EmbeddingDataset",
    "LitDataModule",
    "FineWebEduStreamDataset",
    "StreamingLitDataModule",
    "StreamingTextCollator",
    "build_datamodule",
    "build_collator",
]

_DATAMODULE_REGISTRY: dict[str, type[L.LightningDataModule]] = {
    "precomputed": LitDataModule,
    "streaming_fineweb": StreamingLitDataModule,
}

_COLLATOR_REGISTRY: dict[str, type] = {
    "embedding": EmbeddingCollator,
    "streaming_text": StreamingTextCollator,
}


def build_datamodule(cfg: Mapping[str, Any], collator, seed: int) -> L.LightningDataModule:
    return build_from_registry(_DATAMODULE_REGISTRY, cfg, collator, seed, label="datamodule")


def build_collator(cfg: Mapping[str, Any]):
    return build_from_registry(_COLLATOR_REGISTRY, cfg, label="collator")
