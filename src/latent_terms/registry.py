"""Config-driven factory shared by every component registry in this package."""

from __future__ import annotations

from typing import Any, Callable, Mapping, TypeVar

T = TypeVar("T")


def build_from_registry(
    registry: Mapping[str, Callable[..., T]],
    cfg: Mapping,
    *args: Any,
    label: str = "component",
) -> T:
    """Look up ``cfg["name"]`` in ``registry`` and call it with the rest.

    ``args`` is prepended so callers can pass leading positionals the builders
    need (e.g. ``optimizer`` for LR schedulers, ``params`` for optimizers);
    everything left in ``cfg`` after popping ``name`` is splatted as kwargs.

    Missing or unknown ``name`` raises ``KeyError`` naming the label and listing
    the registered keys — the fail-loud convention this repo uses everywhere.
    """
    cfg = dict(cfg)
    name = cfg.pop("name", None)
    if name not in registry:
        raise KeyError(
            f"unknown {label} name {name!r}; valid names: {sorted(registry)}"
        )
    return registry[name](*args, **cfg)
