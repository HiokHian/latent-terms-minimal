"""Shared DataLoader kwarg construction for real (non-empty) train/val loaders.

``prefetch_factor`` and ``persistent_workers`` are only valid together with
``num_workers > 0`` (PyTorch raises if either is passed with ``num_workers == 0``),
so this centralizes that branch rather than duplicating it in every
LightningDataModule.
"""

from __future__ import annotations


def loader_kwargs(num_workers: int, prefetch_factor: int, pin_memory: bool) -> dict[str, int | bool]:
    kwargs: dict[str, int | bool] = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = True
    return kwargs
