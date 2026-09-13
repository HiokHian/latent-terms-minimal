"""Streaming FineWeb-Edu text source for the training-code.
"""

from __future__ import annotations

from typing import Any

import fsspec
import lightning as L
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import AutoTokenizer

from .loader_utils import loader_kwargs


class FineWebEduStreamDataset(IterableDataset):
    """Yields raw document text strings from FineWeb-Edu parquet shards on the HF Hub.

    Args:
        repo_id: HF dataset repo, e.g. "HuggingFaceFW/fineweb-edu".
        subset: shard directory prefix within the repo, e.g. "sample/10BT".
        num_shards: how many shard files (from the start of the sorted listing) to use.
        max_row_groups_per_shard: cap on parquet row groups read per shard.
        max_docs: total document cap across all workers combined (smoke-test sizing knob).
    """

    def __init__(
        self,
        repo_id: str,
        subset: str,
        num_shards: int,
        max_row_groups_per_shard: int,
        max_docs: int,
    ) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.subset = subset.rstrip("/")
        self.num_shards = int(num_shards)
        self.max_row_groups_per_shard = int(max_row_groups_per_shard)
        self.max_docs = int(max_docs)

        api = HfApi()
        files = api.list_repo_files(repo_id, repo_type="dataset")
        prefix = f"{self.subset}/"
        shards = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
        if not shards:
            raise ValueError(f"No parquet shards found under {prefix!r} in {repo_id!r}")
        self.shard_paths = shards[: self.num_shards]

    def _shard_url(self, path: str) -> str:
        return f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/{path}"

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        my_shards = self.shard_paths[worker_id::num_workers]
        per_worker_cap = max(1, self.max_docs // max(num_workers, 1))

        fs = fsspec.filesystem("http")
        yielded = 0
        for shard_path in my_shards:
            if yielded >= per_worker_cap:
                break
            with fs.open(self._shard_url(shard_path), mode="rb") as f:
                pf = pq.ParquetFile(f)
                n_groups = min(self.max_row_groups_per_shard, pf.num_row_groups)
                for rg in range(n_groups):
                    if yielded >= per_worker_cap:
                        break
                    for batch in pf.iter_batches(row_groups=[rg], columns=["text"], batch_size=256):
                        for text in batch.column("text"):
                            if yielded >= per_worker_cap:
                                break
                            text_str = text.as_py()
                            if text_str:
                                yielded += 1
                                yield text_str


class StreamingTextCollator:
    """Tokenizes a batch of raw document strings. No backbone forward here — that runs
    inside LatentTermsLitModule so it executes on the trainer's device/precision, not in
    a CPU DataLoader worker.
    """

    def __init__(self, tokenizer_name: str, max_length: int) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = int(max_length)

    def __call__(self, batch: list[str]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            batch, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}


class StreamingLitDataModule(L.LightningDataModule):
    """``data_cfg`` keys are all required — no ``.get`` fallbacks. Train-only (no val split;
    see module docstring for smoke-test-only limitations: no resume, no shuffle).
    """

    def __init__(self, collator: StreamingTextCollator, seed: int, **data_cfg: Any) -> None:
        super().__init__()
        self.data_cfg = data_cfg
        self.collator = collator
        self.seed = int(seed)
        self.train_ds: FineWebEduStreamDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit", "train"):
            cfg = self.data_cfg
            self.train_ds = FineWebEduStreamDataset(
                repo_id=cfg["repo_id"],
                subset=cfg["subset"],
                num_shards=cfg["num_shards"],
                max_row_groups_per_shard=cfg["max_row_groups_per_shard"],
                max_docs=cfg["max_docs"],
            )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None, "call setup('fit') first"
        return DataLoader(
            self.train_ds,
            batch_size=self.data_cfg["batch_size"],
            collate_fn=self.collator,
            **loader_kwargs(
                self.data_cfg["num_workers"],
                self.data_cfg["prefetch_factor"],
                self.data_cfg["pin_memory"],
            ),
        )

    def val_dataloader(self) -> DataLoader:
        # See LitDataModule.val_dataloader: Lightning raises on None from an overridden
        # method, so an empty loader stands in for "no validation split" here.
        return DataLoader([], batch_size=1, collate_fn=self.collator)
