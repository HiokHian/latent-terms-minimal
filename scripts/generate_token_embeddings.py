"""Extract per-token hidden states from a frozen HF text backbone.

Outputs a flat .npy file of shape [total_tokens, hidden_dim] where each row
is one token's last-layer hidden state. All documents' tokens are concatenated
— the SAE treats each token as an independent sample during training.

A companion doc_lengths.npy records the token count per document so the flat
array can be reconstructed into per-document boundaries at inference time:
  doc_boundaries = np.cumsum([0] + doc_lengths.tolist())
  doc_i_tokens = flat[doc_boundaries[i]:doc_boundaries[i+1]]

Usage:
    python scripts/generate_token_embeddings.py \
        --model nthakur/contriever-base-msmarco \
        --corpus data/beir/fiqa/corpus.jsonl \
        --output data/token_embeddings/fiqa_tokens.npy \
        --lengths data/token_embeddings/fiqa_doc_lengths.npy \
        --batch-size 64 \
        --max-length 512

Only the last layer's hidden states are saved. For token packing (fitting
multiple short docs into one context window) use --pack-sequences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model name or path.")
    parser.add_argument("--corpus", required=True, help="corpus.jsonl (BEIR format).")
    parser.add_argument("--output", required=True, help="Output .npy for token embeddings.")
    parser.add_argument("--lengths", required=True, help="Output .npy for per-doc token counts.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--text-field", default="title_text",
                        help="'title_text' to concatenate; or a single field name.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_corpus(path: str, text_field: str) -> list[str]:
    texts = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if text_field == "title_text":
                text = (rec.get("title", "") + " " + rec["text"]).strip()
            else:
                text = rec[text_field]
            texts.append(text)
    return texts


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)

    texts = load_corpus(args.corpus, args.text_field)
    print(f"Corpus: {len(texts)} documents")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.lengths).parent.mkdir(parents=True, exist_ok=True)

    print("Pass 1/2: tokenizing to determine per-doc lengths...")
    doc_lengths: list[int] = []
    for start in tqdm(range(0, len(texts), args.batch_size), desc="Counting tokens"):
        batch_texts = texts[start : start + args.batch_size]
        encoded = tokenizer(batch_texts, truncation=True, max_length=args.max_length)
        doc_lengths.extend(len(ids) for ids in encoded["input_ids"])

    doc_lengths_arr = np.array(doc_lengths, dtype=np.int32)
    total_tokens = int(doc_lengths_arr.sum())
    hidden_dim = model.config.hidden_size
    boundaries = np.concatenate([[0], np.cumsum(doc_lengths_arr)])

    flat = np.lib.format.open_memmap(
        args.output, mode="w+", dtype=np.float32, shape=(total_tokens, hidden_dim),
    )

    print(f"Pass 2/2: encoding {total_tokens} tokens across {len(texts)} docs...")
    for start in tqdm(range(0, len(texts), args.batch_size), desc="Encoding"):
        batch_texts = texts[start : start + args.batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            out = model(**encoded)
        hidden = out.last_hidden_state.float().cpu().numpy()
        mask = encoded["attention_mask"].cpu().numpy()

        for i in range(len(batch_texts)):
            doc_idx = start + i
            n_tokens = int(mask[i].sum())
            flat[boundaries[doc_idx]:boundaries[doc_idx + 1]] = hidden[i, :n_tokens, :]

    flat.flush()
    np.save(args.lengths, doc_lengths_arr)

    print(f"Saved: {flat.shape} token embeddings → {args.output}")
    print(f"Saved: {len(doc_lengths)} doc lengths → {args.lengths}")
    print(f"Mean tokens per doc: {np.mean(doc_lengths):.1f}")


if __name__ == "__main__":
    main()
