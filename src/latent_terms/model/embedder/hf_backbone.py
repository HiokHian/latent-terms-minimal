"""Frozen HF encoder wrapper for the streaming training path.

Converts a tokenized text batch into a flat [total_valid_tokens, hidden_dim] tensor —
the same shape SAELayer.forward expects, whether those tokens came from a precomputed
.npy file or were produced live by this embedder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class HFBackboneEmbedder(nn.Module):
    """Frozen HF backbone. Registered as ``"hf_backbone"``.

    Always stays in eval mode regardless of the owning LightningModule's train()/eval()
    calls, so the backbone's own dropout never introduces noise into a supposedly frozen
    representation. Never register this module's parameters with an optimizer.
    """

    def __init__(self, pretrained_model_name: str) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_model_name, dtype=torch.float32)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)

    def train(self, mode: bool = True) -> "HFBackboneEmbedder":
        return super().train(False)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state.float()
        return hidden[attention_mask.bool()]
