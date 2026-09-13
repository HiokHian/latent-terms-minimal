"""Latent Terms SAE loss.

forward() returns (scalar_loss, LatentTermsLossOutput):
  loss_recon     — MSE reconstruction loss
  loss_sparsity  — L1 term (0.0 when l1_coef=0)
  loss           — total weighted loss
  recon_norm_mse — MSE / mean(input variance), scale-free diagnostic
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .base import Loss
from ..model.sae import SAEOutput


@dataclass
class LatentTermsLossOutput:
    loss_recon: torch.Tensor
    loss_sparsity: torch.Tensor | float
    loss: torch.Tensor
    recon_norm_mse: float


def _recon_norm_mse(loss_recon: torch.Tensor, h_raw: torch.Tensor) -> float:
    """Scale-free diagnostic: reconstruction MSE relative to input variance."""
    with torch.no_grad():
        var = h_raw.var(dim=0).mean()
        return (loss_recon / var.clamp(min=1e-8)).item()


class LatentTermsLoss(Loss):
    """L2 reconstruction + L1 sparsity for the Top-K SAE.

    Args:
        l1_coef: coefficient for the L1 sparsity term. With Top-K selection the
                 sparsity pattern is already fixed; l1_coef acts on feature magnitudes.
                 Set to 0.0 to use reconstruction only.
    """

    def __init__(self, l1_coef: float = 0.0) -> None:
        super().__init__()
        self.l1_coef = float(l1_coef)

    def forward(self, out: SAEOutput) -> tuple[torch.Tensor, LatentTermsLossOutput]:
        loss_recon = F.mse_loss(out.recons, out.h_raw)
        loss_sparsity = self.l1_coef * out.latents.abs().sum(dim=-1).mean() if self.l1_coef > 0.0 else 0.0
        loss = loss_recon + loss_sparsity

        log = LatentTermsLossOutput(
            loss_recon=loss_recon,
            loss_sparsity=loss_sparsity,
            loss=loss,
            recon_norm_mse=_recon_norm_mse(loss_recon, out.h_raw),
        )
        return loss, log
