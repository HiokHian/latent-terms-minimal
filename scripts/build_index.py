"""Build a BM25 inverted index from a trained SAE checkpoint.

Steps:
  1. Load the trained SAE from checkpoint.
  2. Load pre-extracted token hidden states (.npy) and doc-length boundaries.
  3. For each document: encode tokens through the SAE (encode_only), sum-pool,
     apply TF transform (apply_tf_transform from evaluation.indexer).
  4. Build BM25Index over the resulting sparse doc vectors.
  5. Save index + doc_ids to the output directory via BM25Index.save().

The output directory contains:
  doc_ids.json     — list of corpus _id strings in index order
  doc_sparse.npz   — scipy sparse CSR [n_docs, n_latents]
  idf.npy          — [n_latents] IDF weights
  doc_norms.npy    — [n_docs] document lengths for BM25 normalization
  meta.json        — BM25 hyperparameters + dimensions

Usage:
    python scripts/build_index.py \
        --ckpt outputs/checkpoints/last.ckpt \
        --config src/latent_terms/configs/latent_terms_streaming_200k.yaml \
        --corpus data/beir/fiqa/corpus.jsonl \
        --token-emb data/token_embeddings/fiqa_tokens.npy \
        --lengths data/token_embeddings/fiqa_doc_lengths.npy \
        --output data/index/fiqa/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from latent_terms.evaluation.indexer import BM25Index, apply_tf_transform
from latent_terms.model.lit_module import LatentTermsLitModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Lightning checkpoint path.")
    parser.add_argument("--config", required=True, help="YAML config used for training.")
    parser.add_argument("--corpus", required=True, help="corpus.jsonl (BEIR format) for doc IDs.")
    parser.add_argument("--token-emb", required=True, help="Flat token embeddings .npy [total_tokens, D].")
    parser.add_argument("--lengths", required=True, help="Per-doc token counts .npy [n_docs].")
    parser.add_argument("--output", required=True, help="Output directory for index files.")
    parser.add_argument("--sae-batch-size", type=int, default=4096, help="Tokens per SAE forward pass.")
    parser.add_argument("--transform", default="sqrt", choices=["sqrt", "log1p", "none"],
                        help="Sublinear TF transform applied after sum-pooling.")
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("Loading checkpoint...")
    module = LatentTermsLitModule.load_from_checkpoint(args.ckpt, cfg=cfg, map_location=device)
    module.eval()
    sae = module.sae.to(device)

    token_embs = np.load(args.token_emb, mmap_mode="r")   # [total_tokens, D]
    doc_lengths = np.load(args.lengths).astype(np.int32)   # [n_docs]
    n_docs = len(doc_lengths)

    doc_ids: list[str] = []
    with open(args.corpus) as f:
        for line in f:
            if line.strip():
                doc_ids.append(json.loads(line)["_id"])
    assert len(doc_ids) == n_docs, f"corpus has {len(doc_ids)} docs but lengths has {n_docs}"

    n_latents = sae.n_latents
    total_tokens = int(doc_lengths.sum())
    token_doc_ids = torch.from_numpy(
        np.repeat(np.arange(n_docs, dtype=np.int64), doc_lengths)
    ).to(device)

    doc_sums = torch.zeros((n_docs, n_latents), dtype=torch.float32, device=device)
    for batch_start in tqdm(range(0, total_tokens, args.sae_batch_size), desc="Encoding tokens"):
        batch_end = min(batch_start + args.sae_batch_size, total_tokens)
        batch = torch.tensor(
            token_embs[batch_start:batch_end], dtype=torch.float32, device=device,
        )
        with torch.no_grad():
            codes = sae.encode_only(batch)                        # [batch_tokens, n_latents]
        doc_sums.index_add_(0, token_doc_ids[batch_start:batch_end], codes)

    doc_vecs = apply_tf_transform(doc_sums.cpu().numpy(), args.transform)

    print("Building BM25 index...")
    index = BM25Index(k1=args.k1, b=args.b)
    index.build(doc_vecs)

    out_dir = Path(args.output)
    index.save(out_dir)
    with open(out_dir / "doc_ids.json", "w") as f:
        json.dump(doc_ids, f)

    print(f"Index written to {out_dir}/")
    print(f"  n_docs={n_docs}  n_latents={n_latents}")
    nnz = int((doc_vecs != 0).sum())
    print(f"  Sparsity: {nnz / (n_docs * n_latents) * 100:.4f}%")


if __name__ == "__main__":
    main()
