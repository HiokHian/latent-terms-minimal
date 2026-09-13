"""Evaluate Latent Terms retrieval on a BEIR dataset.

Loads a pre-built BM25 index (from build_index.py), encodes queries through
the backbone + SAE at eval time, and computes nDCG@10, Recall@100.

Usage:
    python scripts/eval_retrieval.py \
        --ckpt outputs/checkpoints/last.ckpt \
        --config src/latent_terms/configs/latent_terms_streaming_200k.yaml \
        --queries data/beir/fiqa/queries.jsonl \
        --qrels data/beir/fiqa/qrels.json \
        --index data/index/fiqa/ \
        --backbone nthakur/contriever-base-msmarco \
        --output results/fiqa_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from latent_terms.evaluation.indexer import BM25Index, apply_tf_transform
from latent_terms.evaluation.metrics import build_ranked_results, compute_metrics
from latent_terms.model.lit_module import LatentTermsLitModule
from latent_terms.model.sae import SAELayer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--queries", required=True, help="queries.jsonl (BEIR format).")
    parser.add_argument("--qrels", required=True, help="qrels.json.")
    parser.add_argument("--index", required=True, help="Index directory from build_index.py.")
    parser.add_argument("--backbone", required=True, help="HF model used for token extraction.")
    parser.add_argument("--output", default=None, help="Optional JSON path for metric results.")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--transform", default="sqrt", choices=["sqrt", "log1p", "none"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def encode_queries(
    query_texts: list[str],
    backbone: AutoModel,
    tokenizer: AutoTokenizer,
    sae: SAELayer,
    transform: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    """Returns [n_queries, n_latents] sparse SAE vectors via sum-pool + transform."""
    all_vecs = []
    for start in tqdm(range(0, len(query_texts), batch_size), desc="Encoding queries"):
        batch = query_texts[start : start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            hidden = backbone(**encoded).last_hidden_state.float()
            mask = encoded["attention_mask"].bool()

            for i in range(len(batch)):
                n_tokens = int(mask[i].sum())
                tok_codes = sae.encode_only(hidden[i, :n_tokens, :]).cpu().numpy()
                all_vecs.append(apply_tf_transform(tok_codes.sum(axis=0), transform))

    return np.stack(all_vecs, axis=0)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("Loading SAE checkpoint...")
    module = LatentTermsLitModule.load_from_checkpoint(args.ckpt, cfg=cfg, map_location=device)
    module.eval()
    sae = module.sae.to(device)

    print(f"Loading backbone {args.backbone}...")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    backbone = AutoModel.from_pretrained(args.backbone, torch_dtype=torch.float32)
    backbone.eval().to(device)
    for param in backbone.parameters():
        param.requires_grad_(False)

    index = BM25Index.load(args.index)
    with open(Path(args.index) / "doc_ids.json") as f:
        doc_ids = json.load(f)

    with open(args.queries) as f:
        query_records = [json.loads(line) for line in f if line.strip()]
    with open(args.qrels) as f:
        qrels_raw = json.load(f)
    qrels = {qid: {did: int(float(rel)) for did, rel in docs.items()}
             for qid, docs in qrels_raw.items()}

    valid = set(qrels.keys())
    pairs = [(r["_id"], r["text"]) for r in query_records if r["_id"] in valid]
    if not pairs:
        raise RuntimeError("No matching query IDs found in qrels. Check query/qrels alignment.")
    query_ids, query_texts = zip(*pairs)

    print(f"Queries: {len(query_ids)}  Corpus: {len(doc_ids)}")

    query_sparse = encode_queries(
        list(query_texts), backbone, tokenizer, sae,
        args.transform, args.batch_size, args.max_length, device,
    )

    # +1: over-fetch so a filtered-out self-match (see build_ranked_results) still leaves
    # top_k genuine candidates.
    scores, indices = index.search(query_sparse, top_k=args.top_k + 1)

    results = build_ranked_results(query_ids, doc_ids, scores, indices, args.top_k)
    metrics = compute_metrics(qrels, results, metrics=["ndcg", "recall"], k_values=[10, 100])

    print("\n--- Results ---")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
