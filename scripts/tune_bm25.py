"""Grid-search BM25 k1/b for a trained SAE checkpoint's index, to find a best-effort
answer to "tuned BM25" against the paper's own tuned-parameter numbers

Usage:
    python scripts/tune_bm25.py \
        --ckpt outputs/checkpoints/streaming_200k/epoch=epoch=00.ckpt \
        --config src/latent_terms/configs/latent_terms_streaming_200k.yaml \
        --queries data/beir/scifact/queries.jsonl \
        --qrels data/beir/scifact/qrels.json \
        --index data/index/scifact_200k/ \
        --backbone nthakur/contriever-base-msmarco \
        --output results/scifact_200k_bm25_tuned.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModel, AutoTokenizer

from eval_retrieval import encode_queries
from latent_terms.evaluation.indexer import BM25Index
from latent_terms.evaluation.metrics import build_ranked_results, compute_metrics
from latent_terms.model.lit_module import LatentTermsLitModule

K1_GRID = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.6, 3.0]
B_GRID = [0.0, 0.2, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--transform", default="sqrt", choices=["sqrt", "log1p", "none"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


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
    backbone = AutoModel.from_pretrained(args.backbone, dtype=torch.float32)
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
    query_ids, query_texts = zip(*pairs)

    print(f"Queries: {len(query_ids)}  Corpus: {len(doc_ids)}")
    print("Encoding queries once (k1/b don't affect this step)...")
    query_sparse = encode_queries(
        list(query_texts), backbone, tokenizer, sae,
        args.transform, args.batch_size, args.max_length, device,
    )

    grid_results: list[dict[str, float]] = []
    best: dict[str, float] | None = None
    for k1 in K1_GRID:
        for b in B_GRID:
            index.k1 = k1
            index.b = b
            scores, indices = index.search(query_sparse, top_k=args.top_k + 1)
            results = build_ranked_results(query_ids, doc_ids, scores, indices, args.top_k)
            metrics = compute_metrics(qrels, results, metrics=["ndcg", "recall"], k_values=[10, 100])
            row = {"k1": k1, "b": b, **metrics}
            grid_results.append(row)
            if best is None or metrics["ndcg@10"] > best["ndcg@10"]:
                best = row
            print(f"  k1={k1:.1f} b={b:.2f} -> ndcg@10={metrics['ndcg@10']:.4f}")

    print("\n--- Best ---")
    for k, v in sorted(best.items()):
        print(f"  {k}: {v}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"best": best, "grid": grid_results}, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
