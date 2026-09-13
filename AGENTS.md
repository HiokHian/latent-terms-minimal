# AGENTS.md

**This is the single source of truth for agents working in this repo.**

| Where to look | For what |
|---|---|
| **this file** | everything: project conventions, how to add components, workspace layout |
| [`README.md`](README.md) | human-facing overview, install/run commands, end-to-end workflow |
---

# Part 1 — Codebase conventions

## What this project is

Replication and extension of **Latent Terms** (arXiv:2605.29384) — a token-level Sparse
Autoencoder (SAE) post-trained on frozen dense retriever hidden states, producing sparse latent
codes over a learned vocabulary of semantic features.

- Input: per-token hidden states `[tokens, D]`, not a pooled embedding
- SAE target: reconstruct each token's hidden state
- Retrieval: BM25 (sum-pool → sqrt → inverted index)
- Labels needed: none (self-supervised reconstruction)

## The objective is RETRIEVAL. Do not optimise reconstruction MSE.

Always evaluate with nDCG@10 from `scripts/eval_retrieval.py`. Report `recon_norm_mse` as
a diagnostic, never as a selection criterion.

## Project structure

```
AGENTS.md                      THIS FILE (single source of truth)
CLAUDE.md                      thin pointer to AGENTS.md, for tools that look for it by name
README.md                      human-facing overview + quickstart

train.py                       entry point: parse config, build module+datamodule, fit
scripts/
  generate_token_embeddings.py  HF backbone → flat [total_tokens, D] .npy per-token hiddens
  build_index.py               trained SAE → BM25 inverted index (doc_sparse.npz, idf.npy)
  eval_retrieval.py            end-to-end retrieval eval (SAE + BM25; BEIR format)
  eval_dense_baseline.py       native backbone cosine retrieval — no-SAE comparison row
  eval_bm25_raw_baseline.py    lexical BM25 over raw text — no model at all
  tune_bm25.py                 exploratory k1/b grid search on a built index
  prepare_beir.py              BEIR download + qrels TSV → JSON conversion
  bootstrap_pod.sh             one-time pod setup (uv, HF cache, env vars)

src/latent_terms/
  registry.py                  shared factory fn used by every registry
  data/
    datamodule.py              LitDataModule + EmbeddingDataset (.npy memmap)
    collator.py                EmbeddingCollator only (no text/image collators)
  model/
    embedder/                  frozen backbone wrappers: precomputed.py (identity pass)
    sae/                       the sparse autoencoder layer: layer.py (SAELayer)
    model.py                   LatentTermsModel = embedder (frozen) + sae
    lit_module.py              LatentTermsLitModule: training step, optimizer, val
  losses/
    latent_terms_loss.py       LatentTermsLoss: MSE recon + optional L1 sparsity
  optimizers/                  adamw
  lr_schedulers/               cosine_with_warmup, linear_warmup
  loggers/                     csv, wandb
  evaluation/
    evaluator.py               LatentTermsEvaluator (BEIR format)
    indexer.py                 BM25Index (Robertson BM25) + FlatIndex (FAISS)
    metrics.py                 ndcg_at_k, recall_at_k, mrr_at_k, compute_metrics
  configs/
    latent_terms_streaming_200k.yaml   the config; experiments are diffs against it
```

## The one pattern everything follows

Every pluggable component (SAE variant, embedder, loss, optimizer, LR scheduler, index)
is built the same way:

1. **An abstract base class** defines the interface.
2. **Concrete subclasses** live one-per-file in the component's directory.
3. **A `_XXX_REGISTRY` dict** in that directory's `__init__.py` maps a string name to the class.
4. **A `build_xxx(cfg)` function** wraps `registry.build_from_registry`.

Config shape is always `{"name": "...", **kwargs}`. **No `.get()` fallbacks anywhere** — a
missing or misspelled config key must raise loudly, not silently default.

Read `src/latent_terms/registry.py` — it's ~37 lines and is the entire dispatch mechanism.
**Never write a second dispatch mechanism** (if/elif chains, `getattr` lookups).

## Launching a training run

Training is self-supervised on streamed FineWeb-Edu text and does **not** depend on any BEIR
dataset — steps 1 and 2 below are independent of each other; only steps 3-4 need both.

```bash
# 1. Train the SAE (streams FineWeb-Edu live; no BEIR data needed for this step)
python train.py --config src/latent_terms/configs/latent_terms_streaming_200k.yaml --fast-dev-run
python train.py --config src/latent_terms/configs/latent_terms_streaming_200k.yaml
# Checkpoint lands at outputs/checkpoints/streaming_200k/epoch=epoch=00.ckpt

# 2. Download a BEIR dataset and extract per-token hidden states from the same backbone
python scripts/prepare_beir.py --dataset fiqa --output-dir data/beir --split test
python scripts/generate_token_embeddings.py \
    --model nthakur/contriever-base-msmarco \
    --corpus data/beir/fiqa/corpus.jsonl \
    --output data/token_embeddings/fiqa_tokens.npy \
    --lengths data/token_embeddings/fiqa_doc_lengths.npy

# 3. Build BM25 index from trained SAE
python scripts/build_index.py \
    --ckpt outputs/checkpoints/streaming_200k/epoch=epoch=00.ckpt \
    --config src/latent_terms/configs/latent_terms_streaming_200k.yaml \
    --corpus data/beir/fiqa/corpus.jsonl \
    --token-emb data/token_embeddings/fiqa_tokens.npy \
    --lengths data/token_embeddings/fiqa_doc_lengths.npy \
    --output data/index/fiqa/

# 4. Evaluate retrieval
python scripts/eval_retrieval.py \
    --ckpt outputs/checkpoints/streaming_200k/epoch=epoch=00.ckpt \
    --config src/latent_terms/configs/latent_terms_streaming_200k.yaml \
    --queries data/beir/fiqa/queries.jsonl \
    --qrels data/beir/fiqa/qrels.json \
    --index data/index/fiqa/ \
    --backbone nthakur/contriever-base-msmarco
```

See `README.md`'s quickstart for the full 5-BEIR-dataset loop and the exact commands used to
replicate the paper's Table 1.

There is no CLI override mechanism — to run a different experiment, copy
`latent_terms_streaming_200k.yaml` to a new file and edit it, then point `--config` at that.

## Adding a new SAE variant

1. Add `model/sae/<name>.py`, subclass `nn.Module`, return `SAEOutput` from `forward`.
   Populate every field of `SAEOutput` (use `None` for unused optional fields — not a
   missing attribute).
2. Register in `model/sae/__init__.py`: import the class, add `"<name>": YourSAE` to
   `_SAE_REGISTRY`, add to `__all__`.
3. Wire via config: `model.sae: {name: <name>, ...kwargs}`.
4. Only `model.sae.parameters()` should be trained — `configure_optimizers` in
   `lit_module.py` already does this; don't change it unless you add a learnable pooler.

## SAEOutput contract

`SAEOutput` is defined in `model/sae/layer.py`:

```python
@dataclass
class SAEOutput:
    h_raw: Tensor      # [batch, hidden_dim] — the input (pre-bias subtracted)
    latents: Tensor    # [batch, n_latents] — sparse codes, k nonzeros per row
    recons: Tensor     # [batch, hidden_dim] — reconstruction (decoder output)
```

Losses receive `SAEOutput` and must not reach into model internals.

## Adding a new loss

1. Subclass `Loss` (`losses/base.py`). `forward(*args, **kwargs) -> tuple[Tensor, dataclass]`
   — return `(loss_tensor, log)`, where `log` is a dataclass (not a bare dict — see
   `LatentTermsLossOutput` for the pattern; `lit_module.py` extracts its fields for logging).
2. Register in `losses/__init__.py`.
3. Wire via `loss: {name: <name>, ...kwargs}` in config.

## Adding a new index type

1. Subclass `BaseIndex` (`evaluation/indexer.py`). Implement `build(doc_vecs)` and
   `search(query_vecs, top_k) -> (scores, indices)`.
2. Register in `evaluation/indexer.py`'s `_INDEX_REGISTRY`.
3. Wire via `eval.index_cfg: {name: <name>, ...kwargs}` in config.

## Before adding new code

- Smoke test with a tiny corpus (e.g., 10 docs) before running at scale.
- No `.get()` config fallbacks, no silent defaults for anything that isn't genuinely optional.
- `model.backbone` in `latent_terms_streaming_200k.yaml` is the one legitimate optional-block
  pattern (checked with `"backbone" in cfg["model"]` in `lit_module.py`) — present only for
  the streaming path, absent for a precomputed-embeddings config.

## Where documentation goes

Two content files, no `plans/` directory on this branch (`CLAUDE.md` is a pointer, not content):

- **Conventions, patterns, pitfalls → this file.**
- **Human-facing setup, run commands, results → `README.md`.**
