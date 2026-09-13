"""Standard IR metrics over ranked retrieval results.

Computed via BEIR's own EvaluateRetrieval (beir.retrieval.evaluation) rather than a
hand-rolled reimplementation — the reference implementation BEIR leaderboard numbers
are computed with.

All functions share the same signature:
  qrels   : {query_id: {doc_id: relevance_int}}
  results : {query_id: {doc_id: score_float}}   (higher score = more relevant)
  k       : cutoff

EvaluateRetrieval.evaluate mutates `results` in place (drops any doc id == query id,
its own `ignore_identical_ids` default) — a harmless no-op here since build_ranked_results
already filtered these, and no caller in this repo reuses `results` afterward.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from beir.retrieval.evaluation import EvaluateRetrieval


def ndcg_at_k(
    qrels: Mapping[str, Mapping[str, int]],
    results: Mapping[str, Mapping[str, float]],
    k: int,
) -> float:
    ndcg, _, _, _ = EvaluateRetrieval.evaluate(qrels, results, [k])
    return ndcg[f"NDCG@{k}"]


def recall_at_k(
    qrels: Mapping[str, Mapping[str, int]],
    results: Mapping[str, Mapping[str, float]],
    k: int,
) -> float:
    _, _, recall, _ = EvaluateRetrieval.evaluate(qrels, results, [k])
    return recall[f"Recall@{k}"]


def mrr_at_k(
    qrels: Mapping[str, Mapping[str, int]],
    results: Mapping[str, Mapping[str, float]],
    k: int,
) -> float:
    mrr = EvaluateRetrieval.evaluate_custom(qrels, results, [k], metric="mrr")
    return mrr[f"MRR@{k}"]


def build_ranked_results(
    query_ids: Sequence[str],
    doc_ids: Sequence[str],
    scores: Sequence[Sequence[float]],
    indices: Sequence[Sequence[int]],
    k: int,
) -> dict[str, dict[str, float]]:
    """Converts index.search() output to {query_id: {doc_id: score}}, excluding
    self-matches (matches BEIR's exact_search.py) and truncating to k. Requires k+1
    candidates per query to compensate for a filtered self-match.
    """
    results: dict[str, dict[str, float]] = {}
    for i, qid in enumerate(query_ids):
        ranked: list[tuple[str, float]] = []
        for idx, score in zip(indices[i], scores[i]):
            if idx < 0:
                continue
            did = doc_ids[int(idx)]
            if did == qid:
                continue
            ranked.append((did, float(score)))
            if len(ranked) == k:
                break
        results[qid] = dict(ranked)
    return results


_METRIC_FNS = {
    "ndcg": ndcg_at_k,
    "recall": recall_at_k,
    "mrr": mrr_at_k,
}


def compute_metrics(
    qrels: Mapping[str, Mapping[str, int]],
    results: Mapping[str, Mapping[str, float]],
    metrics: list[str],
    k_values: list[int],
) -> dict[str, float]:
    """Compute all requested metrics at all k values.

    Returns keys like "ndcg@10", "recall@100", "mrr@10".
    """
    out: dict[str, float] = {}
    for metric in metrics:
        fn = _METRIC_FNS[metric]
        for k in k_values:
            out[f"{metric}@{k}"] = fn(qrels, results, k)
    return out
