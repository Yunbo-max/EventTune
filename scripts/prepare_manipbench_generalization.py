#!/usr/bin/env python3
"""Convert official simplified ManipBench Q1 folders to fixed domain splits."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


OPTIONS = ("A", "B", "C", "D")


def normalize_answer(text: str) -> str:
    hits = re.findall(r"(?<![A-Z])[ABCD](?![A-Z])", text.upper())
    if not hits:
        raise ValueError(f"cannot parse A/B/C/D answer from {text!r}")
    return hits[-1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/manipbench-simplified/Q1")
    parser.add_argument("--output-dir", default="data/prepared/manipbench_q1")
    parser.add_argument("--domains", nargs="+", default=["bridge_pick_place", "droid_pick_place", "droid_arti"])
    parser.add_argument("--support-per-class", type=int, default=8)
    parser.add_argument("--query-target", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    for domain in args.domains:
        grouped = defaultdict(list)
        for folder in sorted((Path(args.root) / domain).glob("question_*")):
            required = [folder / "question.txt", folder / "answer.txt", folder / "image.png"]
            if not all(path.is_file() for path in required):
                continue
            answer = normalize_answer(required[1].read_text(encoding="utf-8", errors="replace"))
            grouped[answer].append({
                "sample_id": f"{domain}-{folder.name}", "domain_id": domain,
                "group_id": folder.name, "image": str(required[2].resolve()),
                "label": answer, "label_id": OPTIONS.index(answer),
                "question": required[0].read_text(encoding="utf-8", errors="replace").strip(),
                "dataset": "manipbench-q1", "metadata": {"source_folder": str(folder.resolve())},
            })
        if set(grouped) != set(OPTIONS):
            raise RuntimeError(f"{domain}: expected A/B/C/D, found {sorted(grouped)}")
        support, pool = [], []
        for option in OPTIONS:
            rng.shuffle(grouped[option])
            support.extend(grouped[option][: args.support_per_class])
            pool.extend(grouped[option][args.support_per_class :])
        per_class = min(args.query_target // 4, *(len(grouped[x]) - args.support_per_class for x in OPTIONS))
        query = []
        for option in OPTIONS:
            query.extend([r for r in pool if r["label"] == option][:per_class])
        rng.shuffle(support); rng.shuffle(query)
        output = Path(args.output_dir) / domain / f"seed_{args.seed}"
        write_jsonl(output / "support.jsonl", support)
        write_jsonl(output / "query.jsonl", query)
        print(json.dumps({"domain": domain, "support": len(support), "query": len(query)}))


if __name__ == "__main__":
    main()
