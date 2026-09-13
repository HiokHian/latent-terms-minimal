"""BEIR-format retrieval evaluator for Latent Terms.

Corpus and query files follow the BEIR/MTEB schema:
  corpus.jsonl   — one doc per line: {"_id": str, "text": str, "title": str}
  queries.jsonl  — one query per line: {"_id": str, "text": str}
  qrels.json     — {query_id: {doc_id: relevance_int}}

The encode_fn passed to run() receives a list of texts and returns either:
  - Dense vectors [n, hidden_dim] for a FlatIndex baseline, or
  - Sparse SAE vectors [n, n_latents] for BM25Index retrieval.

For Latent Terms, the encode_fn extracts token-level hidden states (one per
token), encodes each through the SAE, aggregates per-document via sum-pool +
sqrt-transform, and returns the [n_docs, n_latents] sparse matrix.
"""

from __future__ import annotations

import json
from typing import Callable, Mapping

import numpy as np

from .indexer import BaseIndex, build_index
from .metrics import build_ranked_results, compute_metrics


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_qrels(path: str) -> dict[str, dict[str, int]]:
    with open(path) as f:
        raw = json.load(f)
    return {qid: {did: int(float(rel)) for did, rel in docs.items()} for qid, docs in raw.items()}


class LatentTermsEvaluator:
    """Encodes corpus + queries, builds index, searches, computes IR metrics.

    Args:
        corpus_path:   path to corpus.jsonl
        queries_path:  path to queries.jsonl
        qrels_path:    path to qrels.json
        index_cfg:     dict passed to build_index (e.g. {"name": "bm25"})
        metrics:       e.g. ["ndcg", "recall"]
        k_values:      e.g. [10, 100]
        batch_size:    texts per encode_fn call
        text_field:    "title_text" concatenates title + text; or a single field name
    """

    def __init__(
        self,
        corpus_path: str,
        queries_path: str,
        qrels_path: str,
        index_cfg: Mapping,
        metrics: list[str],
        k_values: list[int],
        batch_size: int = 256,
        text_field: str = "title_text",
    ) -> None:
        corpus_records = _load_jsonl(corpus_path)
        query_records = _load_jsonl(queries_path)

        self.qrels = _load_qrels(qrels_path)
        self.metrics = metrics
        self.k_values = k_values
        self.batch_size = batch_size
        self.index_cfg = dict(index_cfg)

        def _get_text(record: dict) -> str:
            if text_field == "title_text":
                return (record.get("title", "") + " " + record["text"]).strip()
            return record[text_field]

        self.corpus_ids: list[str] = [r["_id"] for r in corpus_records]
        self.corpus_texts: list[str] = [_get_text(r) for r in corpus_records]

        query_ids = [r["_id"] for r in query_records]
        query_texts = [_get_text(r) for r in query_records]
        valid = set(self.qrels.keys())
        pairs = [(qid, qt) for qid, qt in zip(query_ids, query_texts) if qid in valid]
        self.query_ids, self.query_texts = (list(x) for x in zip(*pairs)) if pairs else ([], [])

    def _encode_all(
        self,
        texts: list[str],
        encode_fn: Callable[[list[str]], np.ndarray],
    ) -> np.ndarray:
        if not texts:
            raise ValueError("encode_all received an empty text list")
        chunks = []
        for start in range(0, len(texts), self.batch_size):
            chunks.append(encode_fn(texts[start : start + self.batch_size]))
        return np.concatenate(chunks, axis=0)

    def run(self, encode_fn: Callable[[list[str]], np.ndarray]) -> dict[str, float]:
        max_k = max(self.k_values)

        corpus_vecs = self._encode_all(self.corpus_texts, encode_fn)
        index: BaseIndex = build_index(self.index_cfg)
        index.build(corpus_vecs)

        query_vecs = self._encode_all(self.query_texts, encode_fn)
        # +1: over-fetch so a filtered-out self-match (see build_ranked_results) still
        # leaves max_k genuine candidates.
        scores, indices = index.search(query_vecs, top_k=max_k + 1)

        results = build_ranked_results(self.query_ids, self.corpus_ids, scores, indices, max_k)

        return compute_metrics(self.qrels, results, self.metrics, self.k_values)
