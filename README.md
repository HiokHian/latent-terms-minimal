# Latent Terms

A replication of [**Latent Terms** (arXiv:2605.29384)](https://arxiv.org/abs/2605.29384) results for Contriever on LIMIT and selected BEIR benchmarks — a token-level Sparse Autoencoder (SAE) post-trained on frozen dense retriever hidden states, producing sparse latent codes compatible with BM25 retrieval.

Latent Terms operates on per-token hidden states, producing a learned vocabulary of semantic features. Retrieval is then BM25-compatible via an inverted index.

## Install

```bash
# Clone the repo
git clone https://github.com/HiokHian/latent_terms.git
cd latent_terms

# Install uv (if not present)
curl -Lsf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

## End-to-end quickstart (Contriever, 5 BEIR datasets — replicates Table 1)

This reproduces the paper's Contriever + Latent Terms numbers on SciFact, FiQA, NFCorpus,
ArguAna, and SciDocs. There is one config on this branch —
`src/latent_terms/configs/latent_terms_streaming_200k.yaml` — and one backbone,
`nthakur/contriever-base-msmarco`.


### 1. Train the SAE (no BEIR data needed for this step)

```bash
# Smoke test — 1 train batch + exit, catches config/import errors cheaply
python train.py --config src/latent_terms/configs/latent_terms_streaming_200k.yaml --fast-dev-run

# Full run — streams FineWeb-Edu live over HTTPS, ~20-50 min on a single GPU depending on
# network speed and HF cache warmth. Needs network access for the whole run, not just at
# startup. If this run crashes and is resumed from a checkpoint, note the streaming
# datamodule has no exact-resume support — it restarts from the first shard, re-exposing
# already-seen documents, rather than continuing where it left off.
python train.py --config src/latent_terms/configs/latent_terms_streaming_200k.yaml
```

Checkpoint lands at `outputs/checkpoints/streaming_200k/epoch=epoch=00.ckpt` (Lightning
appends `-v1`, `-v2`, ... if a checkpoint with that name already exists — check
`ls outputs/checkpoints/streaming_200k/` after training and use whichever file it produced).

### 2. Download the 5 BEIR datasets and extract per-token hidden states

```bash
BACKBONE=nthakur/contriever-base-msmarco

for ds in scifact fiqa nfcorpus arguana scidocs; do
    python scripts/prepare_beir.py --dataset $ds --output-dir data/beir --split test

    python scripts/generate_token_embeddings.py \
        --model $BACKBONE \
        --corpus data/beir/$ds/corpus.jsonl \
        --output data/token_embeddings/${ds}_tokens.npy \
        --lengths data/token_embeddings/${ds}_doc_lengths.npy
done
```

> **Disk note**: `${ds}_tokens.npy` holds one 768-dim float32 row per corpus token and can
> get large on bigger corpora (FiQA's ~9.6M tokens is ~30GB). If you're disk-constrained,
> process one dataset at a time and delete its `_tokens.npy` right after step 3 builds that
> dataset's index — the index itself (`doc_sparse.npz`, `idf.npy`, `doc_norms.npy`) is a few
> MB and is all `eval_retrieval.py` needs afterward.

### 3. Build a BM25 inverted index per dataset from the trained checkpoint

```bash
CKPT=outputs/checkpoints/streaming_200k/epoch=epoch=00.ckpt
CONFIG=src/latent_terms/configs/latent_terms_streaming_200k.yaml

for ds in scifact fiqa nfcorpus arguana scidocs; do
    python scripts/build_index.py \
        --ckpt $CKPT \
        --config $CONFIG \
        --corpus data/beir/$ds/corpus.jsonl \
        --token-emb data/token_embeddings/${ds}_tokens.npy \
        --lengths data/token_embeddings/${ds}_doc_lengths.npy \
        --output data/index/$ds/
done
```

### 4. Evaluate retrieval (nDCG@10/100, Recall@10/100)

```bash
for ds in scifact fiqa nfcorpus arguana scidocs; do
    python scripts/eval_retrieval.py \
        --ckpt $CKPT \
        --config $CONFIG \
        --queries data/beir/$ds/queries.jsonl \
        --qrels data/beir/$ds/qrels.json \
        --index data/index/$ds/ \
        --backbone $BACKBONE \
        --output results/${ds}_200k_eval.json
done
```

Expect nDCG@10 within a few percent of the paper's Table 1 (SciFact 0.713, FiQA 0.317,
NFCorpus 0.340, ArguAna 0.436, SciDocs 0.165) — exact numbers vary run to run since the
streaming training data order isn't deterministic.

### 5. Optional: comparison and diagnostic baselines

These aren't part of the Latent Terms pipeline and aren't needed to reproduce Table 1 — they
exist to sanity-check the pipeline or provide the "no SAE" comparison row.

```bash
# Native Contriever, no SAE — the "before" number the SAE improves on. No checkpoint needed.
for ds in scifact fiqa nfcorpus arguana scidocs; do
    python scripts/eval_dense_baseline.py \
        --model $BACKBONE \
        --corpus data/beir/$ds/corpus.jsonl \
        --queries data/beir/$ds/queries.jsonl \
        --qrels data/beir/$ds/qrels.json \
        --output results/${ds}_contriever_dense_baseline.json
done

# Pure lexical BM25 over raw text, no model at all — isolates whether the metrics
# implementation itself (scripts/eval_retrieval.py's compute_metrics path) is behaving,
# independent of any trained checkpoint.
python scripts/eval_bm25_raw_baseline.py \
    --corpus data/beir/scifact/corpus.jsonl \
    --queries data/beir/scifact/queries.jsonl \
    --qrels data/beir/scifact/qrels.json

```

### 6. Optional: LIMIT benchmark

[LIMIT](https://arxiv.org/abs/2508.21038) is an adversarial dataset built to expose a
structural limitation of single-vector dense retrieval: with enough queries mapped to
distinct top-k document sets, no fixed-size embedding can represent every ranking, so
recall saturates at a low ceiling no matter how good the model is. Latent Terms'
SAE+BM25 scoring recovers most of the gap by routing through a sparse latent vocabulary
instead of a single dense vector, sidestepping the bottleneck entirely rather than
training around it.

LIMIT isn't available via `prepare_beir.py`; pull it directly from LIMIT's HF
source (`orionweller/LIMIT`):

```bash
mkdir -p data/beir/limit
python - <<'PY'
import json
from datasets import load_dataset

corpus = load_dataset("orionweller/LIMIT", "corpus")["corpus"]
queries = load_dataset("orionweller/LIMIT", "queries")["queries"]
qrels = load_dataset("orionweller/LIMIT", "default")["test"]

with open("data/beir/limit/corpus.jsonl", "w") as f:
    for row in corpus:
        f.write(json.dumps({"_id": row["_id"], "title": row["title"], "text": row["text"]}) + "\n")

with open("data/beir/limit/queries.jsonl", "w") as f:
    for row in queries:
        f.write(json.dumps({"_id": row["_id"], "text": row["text"]}) + "\n")

nested: dict[str, dict[str, int]] = {}
for row in qrels:
    nested.setdefault(row["query-id"], {})[row["corpus-id"]] = row["score"]
with open("data/beir/limit/qrels.json", "w") as f:
    json.dump(nested, f)
PY
```

From here it's the same pipeline as any BEIR dataset — reuse the checkpoint from step 1,
no new training needed. The paper reports Recall@10 (not nDCG) for this benchmark, so
pass `--top-k 10` to match:

```bash
# LIMIT's 50k docs produce a large per-token embeddings file (~25GB) 
python scripts/generate_token_embeddings.py \
    --model $BACKBONE \
    --corpus data/beir/limit/corpus.jsonl \
    --output /dev/shm/limit_tokens.npy \
    --lengths /dev/shm/limit_doc_lengths.npy

python scripts/build_index.py \
    --ckpt $CKPT --config $CONFIG \
    --corpus data/beir/limit/corpus.jsonl \
    --token-emb /dev/shm/limit_tokens.npy \
    --lengths /dev/shm/limit_doc_lengths.npy \
    --output data/index/limit_contriever/
rm -f /dev/shm/limit_tokens.npy /dev/shm/limit_doc_lengths.npy

# Native Contriever baseline (no SAE) — the "before" number.
python scripts/eval_dense_baseline.py \
    --model $BACKBONE \
    --corpus data/beir/limit/corpus.jsonl \
    --queries data/beir/limit/queries.jsonl \
    --qrels data/beir/limit/qrels.json \
    --output results/limit_contriever_native.json

# Contriever + Latent Terms — the "after" number.
python scripts/eval_retrieval.py \
    --ckpt $CKPT --config $CONFIG \
    --queries data/beir/limit/queries.jsonl \
    --qrels data/beir/limit/qrels.json \
    --index data/index/limit_contriever/ \
    --backbone $BACKBONE \
    --top-k 10 \
    --output results/limit_contriever_lt.json

# Calibration check: confirms this repo's LIMIT split matches the paper's before trusting
# the two numbers above (word-level BM25, independent of the SAE pipeline entirely).
python scripts/eval_bm25_raw_baseline.py \
    --corpus data/beir/limit/corpus.jsonl \
    --queries data/beir/limit/queries.jsonl \
    --qrels data/beir/limit/qrels.json \
    --top-k 10
```

| | Paper (Table 2) | This repo |
|---|---|---|
| Contriever (native) R@10 | 0.021 | 0.031 |
| **Contriever + Latent Terms** R@10 | **0.414** | **0.396** |
| Lexical BM25 (calibration) R@10 | 0.944 | 0.946 |


## Architecture

```
backbone (frozen)         per-token hidden states [tokens, D]
     ↓
SAELayer (trained)        top-k sparse codes [tokens, n_latents]
     ↓
sum-pool + sqrt           doc sparse vector [n_latents]
     ↓
BM25 inverted index       retrieval via latent feature IDs as vocabulary terms
```

The SAE has three parameters (following the paper):
- `pre_bias` — pre-activation shift subtracted before encoding
- `encoder` — linear map to latent space (bias-free)
- `encoder_bias` — per-latent threshold (added before top-k)
- `decoder` — untied linear reconstruction (Kaiming-initialized; `encoder` is then set to `decoder.weight.T`, per the paper)

Training is self-supervised reconstruction: `MSE(h, decoder(encode(h - pre_bias)))`.

## Key scripts

| Script | Purpose |
|---|---|
| `scripts/generate_token_embeddings.py` | Extract per-token hidden states from HF backbone |
| `scripts/build_index.py` | Encode corpus with trained SAE, build BM25 inverted index |
| `scripts/eval_retrieval.py` | End-to-end BEIR-format retrieval eval (SAE + BM25) |
| `scripts/prepare_beir.py` | Download BEIR datasets + convert qrels to JSON |
| `scripts/bootstrap_pod.sh` | One-time RunPod pod setup (uv, HF cache, env vars) |
| `scripts/eval_dense_baseline.py` | *(optional)* Native backbone cosine retrieval — the "no SAE" comparison row |
| `scripts/eval_bm25_raw_baseline.py` | *(optional)* Pure lexical BM25 over raw text, no model — isolates the metrics code from the trained checkpoint |
| `scripts/tune_bm25.py` | *(optional, exploratory)* k1/b grid search on a built index — not how Table 1's numbers were produced |

## Running on RunPod

```bash
# After provisioning a pod:
git clone https://github.com/HiokHian/latent_terms.git /workspace/latent_terms
cd /workspace/latent_terms
bash scripts/bootstrap_pod.sh
```


## Cost & resource profile

Measured on the actual hardware this project runs on — a single RunPod RTX 3090
(~$0.50/hr on-demand) — replicating the training run plus all 5 BEIR datasets in
[Quickstart](#end-to-end-quickstart-contriever-5-beir-datasets--replicates-table-1) above.

**VRAM**:

| Stage | Peak VRAM |
|---|---|
| Training (`train.py`, bf16-mixed, batch_size=16, max_length=512) | ~4.7 GB |
| Index building (`build_index.py`, SAE-only forward, batch=4096 tokens) | ~2.3 GB |
| Query-time inference (`eval_retrieval.py`, backbone + SAE, batch_size=32) | ~1.7 GB |

Any GPU with ≥8GB VRAM should run this end-to-end.

**How much more VRAM does using this method actually cost**
At batch_size=32, max_length=512, 608 real non-padded tokens:

| | Peak VRAM | |
|---|---|---|
| Contriever alone (plain dense retrieval) | 1010.0 MB | *(100%, baseline)* |
| Contriever + Latent Terms SAE | 1211.5 MB | **+201.5 MB → +19.95% (≈20%)** |

**Running this method costs about 20% more VRAM than plain Contriever dense retrieval**

**Disk**:

| Dataset | Tokens | `*_tokens.npy` size |
|---|---|---|
| SciFact | 1.69M | 5.2 GB |
| FiQA | 9.62M | 29.5 GB |
| NFCorpus | 1.26M | 3.9 GB |
| ArguAna | 1.78M | 5.5 GB |
| SciDocs | 5.79M | 17.8 GB |

Generating all 5 up front means ~61.9GB on disk simultaneously. If you're disk-constrained
at all, two ways around it:
1. Process one dataset at a time, deleting `*_tokens.npy` right after that dataset's
   `build_index.py` step (the built index itself is only a few MB — see the disk note in
   the quickstart above).
2. Point `--output`/`--token-emb` at `/dev/shm` (tmpfs) instead of `data/token_embeddings/`
   — useful specifically when disk is quota-limited but RAM isn't, at the cost of using
   that much RAM instead (FiQA's 29.5GB peak is well inside a typical 64GB+ machine).

Steady-state footprint (after cleaning up `*_tokens.npy`) is small: ~1.8GB total —
201MB checkpoint (post `on_save_checkpoint` stripping optimizer state), ~420MB across
all 5 built indices, ~350MB of raw BEIR text, ~840MB one-time HF cache for the backbone.

**Time and cost** — wall-clock for a single RTX 3090, this session's actual run:

| Stage | Time |
|---|---|
| Train the SAE (`train.py`, 12,500 steps) | ~21–50 min* |
| `prepare_beir.py` × 5 datasets | ~2–5 min |
| `generate_token_embeddings.py` + `build_index.py` × 5 datasets | ~30 min |
| `eval_retrieval.py` × 5 datasets | ~5–10 min |
| **Total** | **~58–95 min** |

\* Training streams FineWeb-Edu for the entire run, so wall-clock is network-bound, not
just compute-bound, on top of being compute-bound by sequence length: ~20:45 was this
session's actual time at `max_length=512` with a warm HF cache (up from ~12:28 measured
at the old `max_length=256` — longer attention sequences cost more per step, as expected);
budget more on a slower connection or cold cache.

At **$0.50/hr**, a full from-scratch replication (fresh training + all 5 BEIR datasets)
costs **roughly $0.48–$0.80** in GPU time. This is a cheap experiment to rerun or verify.

## Paper reference

> Latent Terms: Sparse Autoencoders for Interpretable and Efficient Retrieval (2025).
> arXiv:2605.29384.
