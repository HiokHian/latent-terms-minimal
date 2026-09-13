"""Top-K Sparse Autoencoder layer.

Architecture (from arXiv:2605.29384 — Latent Terms):
  encoder : Linear(hidden_dim, n_latents, bias=False) + encoder_bias
  top-k   : keep the k largest positive pre-activations, zero the rest
  decoder : Linear(n_latents, hidden_dim, bias=False) — NOT tied to encoder
  pre_bias: subtracted before encode, added after decode (centring term)

The decoder is kept as a separate (untied) parameter. At inference only the
encoder path is used; the decoder is discarded post-training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEOutput:
    h_raw: torch.Tensor      # original input [batch, hidden_dim]
    latents: torch.Tensor    # sparse activations [batch, n_latents], k nonzeros per row
    recons: torch.Tensor     # decoder output [batch, hidden_dim]


class SAELayer(nn.Module):
    """Top-K Sparse Autoencoder. Registered as ``"top_k"``."""

    def __init__(self, hidden_dim: int, n_latents: int, topk: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_latents = int(n_latents)
        self.topk = int(topk)

        self.pre_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.encoder = nn.Linear(hidden_dim, n_latents, bias=False)
        self.encoder_bias = nn.Parameter(torch.zeros(n_latents))
        self.decoder = nn.Linear(n_latents, hidden_dim, bias=False)
        self.register_buffer("pre_bias_initialized", torch.tensor(False))

        self._init_weights()

    def _init_weights(self) -> None:
        # Paper (§4.1): decoder gets Kaiming init, encoder = transposed decoder weights.
        nn.init.kaiming_uniform_(self.decoder.weight)
        with torch.no_grad():
            self.encoder.weight.copy_(self.decoder.weight.T)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """Sparse encode: returns [batch, n_latents] with exactly topk nonzeros per row."""
        h_centered = h - self.pre_bias
        pre_act = self.encoder(h_centered) + self.encoder_bias
        pre_act = F.relu(pre_act)

        topk_vals, topk_idx = torch.topk(pre_act, self.topk, dim=-1)
        latents = torch.zeros_like(pre_act)
        latents.scatter_(-1, topk_idx, topk_vals)
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents) + self.pre_bias

    def forward(self, h: torch.Tensor) -> SAEOutput:
        # Centre bias. Matters for backbones whose raw hidden states carry
        # large, near-constant outlier dimensions (e.g. ModernBERT "rogue" activations).
        if not bool(self.pre_bias_initialized):
            with torch.no_grad():
                self.pre_bias.copy_(h.mean(dim=0))
            self.pre_bias_initialized.fill_(True)

        latents = self.encode(h)
        recons = self.decode(latents)
        return SAEOutput(h_raw=h, latents=latents, recons=recons)

    def encode_only(self, h: torch.Tensor) -> torch.Tensor:
        """Returns sparse codes only. Caller is responsible for torch.no_grad() context."""
        return self.encode(h)
