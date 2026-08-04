#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eventttt.io import read_samples, write_json
from eventttt.metrics import metrics_by_event
from eventttt.qwen import DEFAULT_MODEL, load_model, preflight, score_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an adapter with D4 product-of-experts")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--d4-views", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=448)
    args = parser.parse_args()

    print(json.dumps(preflight(require_gpu=True), indent=2))
    samples = read_samples(args.manifest)
    model, processor = load_model(
        args.model_id,
        source_adapter=args.adapter,
        gradient_checkpointing=False,
    )
    rows = score_samples(model, processor, samples, args.d4_views, args.crop_size)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    write_json(output / "metrics.json", metrics_by_event(rows))


if __name__ == "__main__":
    main()
