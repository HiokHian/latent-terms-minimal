"""Evaluate a frozen dense retriever's native cosine-similarity retrieval on a BEIR dataset.

This is a pre-flight check, not part of the Latent Terms pipeline: before training any SAE,
we must confirm our encode/pool/score path reproduces the paper's reported native-backbone
numbers (Table 1, arXiv:2605.29384). Reuses LatentTermsEvaluator with a FlatIndex (exact
cosine via FAISS IndexFlatIP over L2-normalized mean-pooled embeddings) instead of BM25Index.

Usage:
    python scripts/eval_dense_baseline.py \
        --model nthakur/contriever-base-msmarco \
        --corpus data/beir/scifact/corpus.jsonl \
        --queries data/beir/scifact/queries.jsonl \
        --qrels data/beir/scifact/qrels.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from latent_terms.evaluation.evaluator import LatentTermsEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF backbone, e.g. nthakur/contriever-base-msmarco")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask[..., None].bool()
    summed = last_hidden_state.masked_fill(~mask, 0.0).sum(dim=1)
    counts = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
    return summed / counts


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading backbone {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    backbone = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32)
    backbone.eval().to(device)
    for param in backbone.parameters():
        param.requires_grad_(False)

    def encode_fn(texts: list[str]) -> np.ndarray:
        encoded = tokenizer(
            texts, padding=True, truncation=True,
            max_length=args.max_length, return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            hidden = backbone(**encoded).last_hidden_state.float()
            pooled = mean_pool(hidden, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        return pooled.cpu().numpy().astype(np.float32)

    evaluator = LatentTermsEvaluator(
        corpus_path=args.corpus,
        queries_path=args.queries,
        qrels_path=args.qrels,
        index_cfg={"name": "flat"},
        metrics=["ndcg", "recall"],
        k_values=[10, 100],
        batch_size=args.batch_size,
    )

    print(f"Corpus: {len(evaluator.corpus_ids)}  Queries: {len(evaluator.query_ids)}")
    metrics = evaluator.run(encode_fn)

    print("\n--- Native dense (cosine) baseline ---")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
