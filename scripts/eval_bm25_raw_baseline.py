"""Pure lexical (word-level) BM25 baseline over raw corpus/query text — no model, no SAE.

Two independent purposes share this one implementation:
  1. Isolate whether evaluation/metrics.py's compute_metrics/build_ranked_results path is
     behaving correctly, with zero training or GPU needed (see AGENTS.md).
  2. Calibrate a new dataset split against a paper's reported lexical-BM25 number before
     spending GPU time on the SAE pipeline (e.g. confirming data/beir/limit/ matches
     arXiv:2605.29384's Table 2 BM25 row) — pass --top-k 10 for this use.

Kept fully sparse throughout (scipy.sparse for vocabulary/term-frequency/IDF/scoring) rather
than materializing a dense [n_docs, vocab_size] array the way evaluation/indexer.py's
BM25Index does for the SAE's fixed-size latent dictionary — a raw lexical vocabulary is
unbounded and can be far larger than n_latents, so a dense array here would be wasteful (or
outright too large) on bigger corpora. Independent of BM25Index by design, not just for
scale: an independent reimplementation of the same Robertson BM25 formula is a genuine
cross-check against BM25Index's own scoring, not merely a second call into the same code.

Usage:
    python scripts/eval_bm25_raw_baseline.py \
        --corpus data/beir/scifact/corpus.jsonl \
        --queries data/beir/scifact/queries.jsonl \
        --qrels data/beir/scifact/qrels.json \
        --output results/scifact_bm25_raw_baseline.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from latent_terms.evaluation.metrics import build_ranked_results, compute_metrics

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="corpus.jsonl (BEIR format).")
    parser.add_argument("--queries", required=True, help="queries.jsonl (BEIR format).")
    parser.add_argument("--qrels", required=True, help="qrels.json.")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--output", default=None, help="Optional JSON path for metric results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    doc_ids: list[str] = []
    doc_tokens: list[list[str]] = []
    with open(args.corpus) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            doc_ids.append(rec["_id"])
            text = (rec.get("title", "") + " " + rec["text"]).strip()
            doc_tokens.append(tokenize(text))

    n_docs = len(doc_ids)
    print(f"Corpus: {n_docs} docs")

    vocab: dict[str, int] = {}
    rows, cols, vals = [], [], []
    doc_lens = np.zeros(n_docs, dtype=np.float32)
    for doc_idx, tokens in enumerate(doc_tokens):
        doc_lens[doc_idx] = len(tokens)
        counts = Counter(tokens)
        for tok, tf in counts.items():
            term_id = vocab.setdefault(tok, len(vocab))
            rows.append(doc_idx)
            cols.append(term_id)
            vals.append(tf)

    n_terms = len(vocab)
    print(f"Vocabulary: {n_terms} terms")
    doc_tf = sp.csr_matrix((vals, (rows, cols)), shape=(n_docs, n_terms), dtype=np.float32)
    doc_tf_csc = doc_tf.tocsc()

    df = np.diff(doc_tf_csc.indptr)  # docs containing each term
    idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
    avg_doc_len = float(doc_lens.mean())

    # BM25 saturated term weight per (doc, term) nonzero, precomputed once, sparse throughout.
    doc_tf_coo = doc_tf.tocoo()
    len_norm = args.k1 * (1 - args.b + args.b * doc_lens[doc_tf_coo.row] / avg_doc_len)
    tf_sat = doc_tf_coo.data * (args.k1 + 1) / (doc_tf_coo.data + len_norm)
    weighted = sp.csr_matrix(
        (tf_sat * idf[doc_tf_coo.col], (doc_tf_coo.row, doc_tf_coo.col)),
        shape=(n_docs, n_terms), dtype=np.float32,
    )

    query_records: list[dict] = []
    with open(args.queries) as f:
        for line in f:
            if line.strip():
                query_records.append(json.loads(line))
    with open(args.qrels) as f:
        qrels_raw = json.load(f)
    qrels = {qid: {did: int(float(rel)) for did, rel in docs.items()} for qid, docs in qrels_raw.items()}

    valid = set(qrels.keys())
    pairs = [(r["_id"], r["text"]) for r in query_records if r["_id"] in valid]
    if not pairs:
        raise RuntimeError("No matching query IDs found in qrels. Check query/qrels alignment.")
    query_ids, query_texts = zip(*pairs)
    print(f"Queries: {len(query_ids)}")

    q_rows, q_cols, q_vals = [], [], []
    for q_idx, text in enumerate(query_texts):
        for tok, tf in Counter(tokenize(text)).items():
            if tok in vocab:
                q_rows.append(q_idx)
                q_cols.append(vocab[tok])
                q_vals.append(tf)
    query_tf = sp.csr_matrix((q_vals, (q_rows, q_cols)), shape=(len(query_ids), n_terms), dtype=np.float32)

    # scores[q, d] = sum_t query_tf[q, t] * weighted[d, t]  (BM25 query term frequency left
    # unsaturated). Densifying here is fine -- this is [n_queries, n_docs], not [n_docs, vocab],
    # so it never hits the scale problem the sparse vocab/TF machinery above was built to avoid.
    scores_sparse = query_tf @ weighted.T
    scores = np.asarray(scores_sparse.todense(), dtype=np.float32)

    # +1: over-fetch so a filtered-out self-match still leaves top_k genuine candidates.
    k = min(args.top_k + 1, n_docs)
    top_k_idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    top_k_scores = np.take_along_axis(scores, top_k_idx, axis=1)
    sort_order = np.argsort(-top_k_scores, axis=1)
    top_k_idx = np.take_along_axis(top_k_idx, sort_order, axis=1)
    top_k_scores = np.take_along_axis(top_k_scores, sort_order, axis=1)

    results = build_ranked_results(list(query_ids), doc_ids, top_k_scores, top_k_idx, args.top_k)
    k_values = [k for k in (10, 100) if k <= args.top_k] or [args.top_k]
    metrics = compute_metrics(qrels, results, metrics=["ndcg", "recall"], k_values=k_values)

    print("\n--- Results (lexical BM25, no model) ---")
    for key, v in sorted(metrics.items()):
        print(f"  {key}: {v:.4f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
