"""Collator for flat embedding datasets."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class EmbeddingCollator:
    """Collates token-level or pooled embedding records.

    Each record is a 1-D float tensor or numpy array. Returns ``{"x": Tensor[batch, dim]}``.
    """

    def __call__(self, batch: list[Any]) -> dict[str, torch.Tensor]:
        tensors = [
            torch.from_numpy(item).float() if isinstance(item, np.ndarray) else item
            for item in batch
        ]
        return {"x": torch.stack(tensors)}
