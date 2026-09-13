"""Abstract loss.

Keep device movement and any cross-device reduction inside the subclass so
the LightningModule stays a passthrough.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class _ABCModuleMeta(ABCMeta, type(nn.Module)):
    """Fuse ABCMeta with nn.Module's metaclass so subclasses can stay abstract."""


class Loss(nn.Module, metaclass=_ABCModuleMeta):
    @abstractmethod
    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, Any]:  # pragma: no cover
        """Return (scalar_loss, log). `log` should be a dataclass, not a bare dict."""
        raise NotImplementedError
