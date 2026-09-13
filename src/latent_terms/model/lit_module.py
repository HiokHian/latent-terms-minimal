"""LightningModule for Latent Terms SAE training.

Logs per step: train/loss_recon, train/loss_sparsity, train/loss, train/mean_active,
train/batch_tokens, train/tokens_seen.
Retrieval eval requires a backbone not present here — run scripts/eval_retrieval.py post-training.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import lightning as L
import torch

from ..losses import build_loss
from ..lr_schedulers import build_lr_scheduler
from ..optimizers import build_optimizer
from .embedder import build_embedder
from .sae import SAELayer, SAEOutput, build_sae


@dataclass
class StepOutput:
    loss: torch.Tensor
    log: Any  # concrete loss's log dataclass, e.g. LatentTermsLossOutput
    mean_active: torch.Tensor
    n_tokens: int 


def _log_fields(log: Any) -> dict[str, Any]:
    """Shallow field extraction for a loss's log dataclass.

    Not dataclasses.asdict(): that deep-copies every field via copy.deepcopy, which
    raises on non-leaf tensors still attached to the autograd graph (e.g. loss_recon).
    self.log_dict handles such tensors directly, so a shallow copy is all that's needed.
    """
    return {f.name: getattr(log, f.name) for f in dataclasses.fields(log)}


class LatentTermsLitModule(L.LightningModule):
    """``model.backbone`` is optional (like the ``eval:`` block) — present only for the
    streaming path, where the batch carries raw tokenized text (``input_ids``,
    ``attention_mask``) instead of precomputed hidden states (``x``). The embedder is
    frozen and never appears in ``configure_optimizers``.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)
        # Embedder weights are stripped from checkpoints (see on_save_checkpoint); loading
        # must not raise on the resulting missing embedder.* keys.
        self.strict_loading = False

        self.sae: SAELayer = build_sae(cfg["model"]["sae"])
        self.embedder = build_embedder(cfg["model"]["backbone"]) if "backbone" in cfg["model"] else None
        self.loss_fn = build_loss(cfg["loss"])
        self.optim_cfg: dict[str, Any] = cfg["optimizer"]
        self.lr_scheduler_cfg: dict[str, Any] = cfg["lr_scheduler"]
        self.tokens_seen: int = 0

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Save only the SAE
        state_dict = checkpoint["state_dict"]
        checkpoint["state_dict"] = {
            k: v for k, v in state_dict.items() if not k.startswith("embedder.")
        }
        checkpoint.pop("optimizer_states", None)
        checkpoint.pop("lr_schedulers", None)
        checkpoint.pop("loops", None)

    def forward(self, x: torch.Tensor) -> SAEOutput:
        return self.sae(x)

    def _step(self, batch: dict[str, torch.Tensor]) -> StepOutput:
        x = self.embedder(batch["input_ids"], batch["attention_mask"]) if self.embedder is not None else batch["x"]
        out: SAEOutput = self.sae(x)
        loss, log = self.loss_fn(out)
        mean_active = (out.latents != 0).float().sum(-1).mean()
        return StepOutput(loss=loss, log=log, mean_active=mean_active, n_tokens=x.shape[0])

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        step_out = self._step(batch)
        self.tokens_seen += step_out.n_tokens
        metrics = _log_fields(step_out.log)
        metrics["mean_active"] = step_out.mean_active
        metrics["batch_tokens"] = step_out.n_tokens
        metrics["tokens_seen"] = self.tokens_seen
        self.log_dict({f"train/{k}": v for k, v in metrics.items()}, on_step=True, on_epoch=False)
        return step_out.loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        step_out = self._step(batch)
        metrics = _log_fields(step_out.log)
        metrics["mean_active"] = step_out.mean_active
        metrics["batch_tokens"] = step_out.n_tokens
        self.log_dict({f"val/{k}": v for k, v in metrics.items()}, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = build_optimizer(self.sae.parameters(), self.optim_cfg)
        scheduler = build_lr_scheduler(optimizer, self.lr_scheduler_cfg)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
