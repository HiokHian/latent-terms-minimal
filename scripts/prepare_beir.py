"""Download a BEIR dataset and convert it into the format evaluation/evaluator.py expects.

BEIR's own download (beir.util.download_and_unzip) already produces
corpus.jsonl and queries.jsonl matching our schema ({"_id", "text", "title"}
and {"_id", "text"}), but ships qrels as a TSV (qrels/<split>.tsv:
query-id, corpus-id, score) rather than the nested qrels.json
({query_id: {doc_id: relevance}}) CSRRetrievalEvaluator reads. This script
downloads the dataset and writes that one converted file; corpus.jsonl and
queries.jsonl are used as BEIR produces them, unmodified.

Usage:
    python scripts/prepare_beir.py --dataset fiqa --output-dir data/beir --split test
"""

from __future__ import annotations

import argparse
import json
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download + prepare a BEIR dataset.")
    parser.add_argument("--dataset", default="fiqa", help="BEIR dataset name, e.g. fiqa, scifact, nfcorpus")
    parser.add_argument("--output-dir", default="data/beir")
    parser.add_argument("--split", default="test", help="qrels split to convert (train/dev/test)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from beir import util

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.dataset}.zip"
    data_path = util.download_and_unzip(url, args.output_dir)

    qrels_tsv = os.path.join(data_path, "qrels", f"{args.split}.tsv")
    qrels: dict[str, dict[str, int]] = {}
    with open(qrels_tsv) as f:
        next(f)  # header: query-id  corpus-id  score
        for line in f:
            qid, did, score = line.strip().split("\t")
            qrels.setdefault(qid, {})[did] = int(score)

    qrels_json_path = os.path.join(data_path, "qrels.json")
    with open(qrels_json_path, "w") as f:
        json.dump(qrels, f)

    print(f"corpus:  {os.path.join(data_path, 'corpus.jsonl')}")
    print(f"queries: {os.path.join(data_path, 'queries.jsonl')}")
    print(f"qrels:   {qrels_json_path}  ({len(qrels)} queries)")


if __name__ == "__main__":
    main()
