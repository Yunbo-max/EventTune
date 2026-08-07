#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm

from eventttt.io import iter_jsonl, read_samples, write_json
from eventttt.metrics import metrics_by_event
from eventttt.qwen import DEFAULT_MODEL, load_model, preflight, score_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an adapter with D4 product-of-experts")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adapter", default="", help="LoRA adapter path; leave empty for base model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--no-lora", action="store_true", help="Evaluate raw base model without any LoRA")
    parser.add_argument("--d4-views", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=448)
    args = parser.parse_args()

    print(json.dumps(preflight(require_gpu=True), indent=2))
    samples = read_samples(args.manifest)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions = output / "predictions.jsonl"
    completed_rows = list(iter_jsonl(predictions)) if predictions.exists() else []
    completed_ids = {row["sample_id"] for row in completed_rows}
    if len(completed_ids) != len(completed_rows):
        raise ValueError(f"Duplicate sample IDs in partial predictions: {predictions}")
    pending = [sample for sample in samples if sample.sample_id not in completed_ids]
    if completed_rows:
        print(f"resume: {len(completed_rows)} predictions complete, {len(pending)} remaining")
    model, processor = load_model(
        args.model_id,
        source_adapter=args.adapter or None,
        gradient_checkpointing=False,
        use_lora=not args.no_lora,
    )
    with predictions.open("a", encoding="utf-8", buffering=1) as handle:
        for sample in tqdm(pending, desc="Scoring", dynamic_ncols=True):
            row = score_sample(model, processor, sample, args.d4_views, args.crop_size)
            handle.write(json.dumps(row) + "\n")
    rows = list(iter_jsonl(predictions))
    if len(rows) != len(samples):
        raise RuntimeError(f"Expected {len(samples)} predictions, found {len(rows)}")
    write_json(output / "metrics.json", metrics_by_event(rows))


if __name__ == "__main__":
    main()
