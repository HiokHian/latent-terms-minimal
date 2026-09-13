"""LightningDataModule for token-level embedding files.

Reads flat .npy arrays of shape [total_tokens, hidden_dim]. Each row is an
independent token hidden state; the dataset treats them as i.i.d. samples for
SAE reconstruction training.

Includes mid-epoch exact resume via StatefulDataLoader: combined with
seed_everything(seed, workers=True) in train.py and Lightning ckpt restore,
a crash resumes on the exact batch it was about to consume.
"""

from __future__ import annotations

from typing import Any

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from .loader_utils import loader_kwargs


class EmbeddingDataset(Dataset):
    """Memory-mapped dataset for large .npy files (token embeddings or pooled embeddings)."""

    def __init__(self, path: str) -> None:
        self.data = np.load(path, mmap_mode='r')

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.data[idx], dtype=torch.float32)


class LitDataModule(L.LightningDataModule):
    """``data_cfg`` keys are all required — no ``.get`` fallbacks.

    ``seed`` seeds a ``torch.Generator`` handed to the train sampler so the
    shuffled batch order is reproducible across resumes.
    """

    def __init__(self, collator, seed: int, **data_cfg) -> None:
        super().__init__()
        self.data_cfg = data_cfg
        self.collator = collator
        self.seed = int(seed)
        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None
        self._train_loader: StatefulDataLoader | None = None
        self._pending_train_loader_state: dict[str, Any] | None = None

    def setup(self, stage: str | None = None) -> None:
        cfg = self.data_cfg
        if stage in (None, "fit", "train"):
            self.train_ds = EmbeddingDataset(cfg["train_path"])
        if stage in (None, "fit", "validate", "val"):
            val_path = cfg.get("val_path")
            self.val_ds = EmbeddingDataset(val_path) if val_path else None

    def train_dataloader(self) -> StatefulDataLoader:
        assert self.train_ds is not None, "call setup('fit') first"
        if self._train_loader is None:
            generator = torch.Generator().manual_seed(self.seed)
            self._train_loader = StatefulDataLoader(
                self.train_ds,
                batch_size=self.data_cfg["batch_size"],
                collate_fn=self.collator,
                shuffle=True,
                generator=generator,
                **loader_kwargs(
                    self.data_cfg["num_workers"],
                    self.data_cfg["prefetch_factor"],
                    self.data_cfg["pin_memory"],
                ),
            )
            if self._pending_train_loader_state is not None:
                self._train_loader.load_state_dict(self._pending_train_loader_state)
                self._pending_train_loader_state = None
        return self._train_loader

    def val_dataloader(self) -> DataLoader:
        # Return an empty loader so Trainer.fit sees 0 val batches and no-ops, rather than crashing.
        if self.val_ds is None:
            return DataLoader([], batch_size=1, collate_fn=self.collator)
        return DataLoader(
            self.val_ds,
            batch_size=self.data_cfg["batch_size"],
            collate_fn=self.collator,
            shuffle=False,
            **loader_kwargs(
                self.data_cfg["num_workers"],
                self.data_cfg["prefetch_factor"],
                self.data_cfg["pin_memory"],
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        if self._train_loader is None:
            return {}
        return {"train_loader": self._train_loader.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pending = state_dict.get("train_loader")
        if pending is None:
            return
        if self._train_loader is not None:
            self._train_loader.load_state_dict(pending)
        else:
            self._pending_train_loader_state = pending
