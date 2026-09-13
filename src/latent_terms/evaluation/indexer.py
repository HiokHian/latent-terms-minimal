"""Index abstractions for Latent Terms retrieval.

Two index types:
  BM25Index   — inverted index with BM25 scoring over SAE latent feature IDs.
                This is the primary Latent Terms retrieval path. No external
                dependencies: built on scipy sparse matrices + numpy.
  FlatIndex   — exact inner product via FAISS, for comparison against the dense
                baseline. Same interface as BM25Index.

BM25 scoring:
  Robertson BM25 with k1=1.5, b=0.75 (BM25+ floor for IDF to avoid negative scores).
  Latent feature indices are treated as vocabulary terms; feature activation
  values (post sqrt-transform) are the term frequencies.

Shared utility:
  apply_tf_transform(vec, transform) — sublinear TF scaling for both corpus and query
  vectors. Must be called consistently on both sides.
"""

from __future__ import annotations

import json
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Mapping

import numpy as np
import scipy.sparse as sp

from ..registry import build_from_registry


def apply_tf_transform(vec: np.ndarray, transform: str) -> np.ndarray:
    """Apply sublinear TF scaling to a dense or sparse vector in-place equivalent.

    Args:
        vec: float32 array of any shape.
        transform: "sqrt" | "log1p" | "none".

    Returns a new array of the same shape and dtype.
    """
    if transform == "sqrt":
        return (np.sign(vec) * np.sqrt(np.abs(vec))).astype(np.float32)
    if transform == "log1p":
        return (np.sign(vec) * np.log1p(np.abs(vec))).astype(np.float32)
    if transform == "none":
        return np.asarray(vec, dtype=np.float32)
    raise ValueError(f"Unknown transform {transform!r}. Expected 'sqrt', 'log1p', or 'none'.")


class BaseIndex(metaclass=ABCMeta):
    @abstractmethod
    def build(self, doc_sparse_vecs: np.ndarray) -> None:
        """Index corpus. doc_sparse_vecs: [n_docs, n_latents] sparse or dense."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vecs: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (scores [n_queries, top_k], doc_indices [n_queries, top_k])."""
        raise NotImplementedError


class BM25Index(BaseIndex):
    """Inverted index with BM25 scoring over SAE latent feature IDs.

    Registered as ``"bm25"``.

    doc_sparse_vecs passed to build() should already be aggregated per document
    (sum-pool over tokens + apply_tf_transform) and have shape [n_docs, n_latents].
    Nonzero values are treated as term frequencies.

    k1: BM25 term-frequency saturation (default 1.5)
    b:  BM25 document-length normalization (default 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self._n_docs: int = 0
        self._n_latents: int = 0
        self._idf: np.ndarray | None = None
        self._doc_norms: np.ndarray | None = None
        self._avg_doc_len: float = 0.0
        self._posting_lists: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def build(self, doc_sparse_vecs: np.ndarray) -> None:
        n_docs, n_latents = doc_sparse_vecs.shape
        self._n_docs = n_docs
        self._n_latents = n_latents

        doc_lengths = np.abs(doc_sparse_vecs).sum(axis=1)
        self._avg_doc_len = float(doc_lengths.mean()) if n_docs > 0 else 1.0
        self._doc_norms = doc_lengths

        doc_freq = np.zeros(n_latents, dtype=np.int64)
        self._posting_lists = {}

        for feat_id in range(n_latents):
            feat_column = doc_sparse_vecs[:, feat_id]
            nonzero_mask = feat_column != 0
            if not nonzero_mask.any():
                continue
            doc_ids = np.where(nonzero_mask)[0].astype(np.int32)
            tf_vals = feat_column[nonzero_mask].astype(np.float32)
            self._posting_lists[feat_id] = (doc_ids, tf_vals)
            doc_freq[feat_id] = len(doc_ids)

        self._idf = np.log1p((n_docs - doc_freq + 0.5) / (doc_freq + 0.5))

    def search(self, query_vecs: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._idf is not None, "call build() first"
        n_queries = query_vecs.shape[0]
        all_scores = np.zeros((n_queries, self._n_docs), dtype=np.float32)

        for q_idx in range(n_queries):
            query_vec = query_vecs[q_idx]
            for feat_id in np.where(query_vec != 0)[0]:
                if feat_id not in self._posting_lists:
                    continue
                doc_ids, tf_vals = self._posting_lists[feat_id]
                query_tf = float(query_vec[feat_id])
                posting_doc_lens = self._doc_norms[doc_ids]
                tf_norm = tf_vals * (self.k1 + 1) / (
                    tf_vals + self.k1 * (1 - self.b + self.b * posting_doc_lens / self._avg_doc_len)
                )
                all_scores[q_idx, doc_ids] += query_tf * float(self._idf[feat_id]) * tf_norm

        top_k_idx = np.argpartition(-all_scores, min(top_k, self._n_docs - 1), axis=1)[:, :top_k]
        top_k_scores = np.take_along_axis(all_scores, top_k_idx, axis=1)
        sort_order = np.argsort(-top_k_scores, axis=1)
        return (
            np.take_along_axis(top_k_scores, sort_order, axis=1),
            np.take_along_axis(top_k_idx, sort_order, axis=1),
        )

    def save(self, out_dir: str | Path) -> None:
        """Serialize index state to out_dir/ (doc_sparse.npz, idf.npy)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        doc_id_rows, feat_id_cols, tf_vals = [], [], []
        for feat_id, (doc_ids, feat_tf_vals) in self._posting_lists.items():
            doc_id_rows.extend(doc_ids.tolist())
            feat_id_cols.extend([feat_id] * len(doc_ids))
            tf_vals.extend(feat_tf_vals.tolist())
        doc_sparse = sp.csr_matrix(
            (np.array(tf_vals, dtype=np.float32), (np.array(doc_id_rows), np.array(feat_id_cols))),
            shape=(self._n_docs, self._n_latents),
        )
        sp.save_npz(str(out_dir / "doc_sparse.npz"), doc_sparse)
        np.save(out_dir / "idf.npy", self._idf)
        np.save(out_dir / "doc_norms.npy", self._doc_norms)
        (out_dir / "meta.json").write_text(
            json.dumps({"k1": self.k1, "b": self.b,
                        "n_docs": self._n_docs, "n_latents": self._n_latents,
                        "avg_doc_len": self._avg_doc_len})
        )

    @classmethod
    def load(cls, index_dir: str | Path) -> "BM25Index":
        """Load a previously saved BM25Index."""
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "meta.json").read_text())
        index = cls(k1=meta["k1"], b=meta["b"])
        index._n_docs = meta["n_docs"]
        index._n_latents = meta["n_latents"]
        index._avg_doc_len = meta["avg_doc_len"]
        index._idf = np.load(index_dir / "idf.npy")
        index._doc_norms = np.load(index_dir / "doc_norms.npy")

        doc_sparse = sp.load_npz(str(index_dir / "doc_sparse.npz")).tocsc()
        index._posting_lists = {}
        for feat_id in range(index._n_latents):
            feat_column = doc_sparse.getcol(feat_id)
            if feat_column.nnz == 0:
                continue
            index._posting_lists[feat_id] = (
                feat_column.indices.astype(np.int32),
                np.asarray(feat_column.data, dtype=np.float32),
            )
        return index


class FlatIndex(BaseIndex):
    """Exact inner product via FAISS IndexFlatIP. Registered as ``"flat"``."""

    def __init__(self) -> None:
        self._index = None

    def build(self, doc_sparse_vecs: np.ndarray) -> None:
        import faiss
        vecs = np.ascontiguousarray(doc_sparse_vecs, dtype=np.float32)
        self._index = faiss.IndexFlatIP(vecs.shape[1])
        self._index.add(vecs)

    def search(self, query_vecs: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._index is not None, "call build() first"
        queries = np.ascontiguousarray(query_vecs, dtype=np.float32)
        return self._index.search(queries, top_k)


_INDEX_REGISTRY: dict = {
    "bm25": BM25Index,
    "flat": FlatIndex,
}


def build_index(cfg: Mapping) -> BaseIndex:
    return build_from_registry(_INDEX_REGISTRY, cfg, label="index")
