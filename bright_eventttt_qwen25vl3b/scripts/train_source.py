#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eventttt.io import read_samples, write_json
from eventttt.qwen import DEFAULT_MODEL, fit_steps, load_model, preflight, trainable_parameter_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Source-event SFT with Qwen2.5-VL-3B QLoRA")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=448)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-qlora", action="store_true")
    args = parser.parse_args()

    print(json.dumps(preflight(require_gpu=True, require_qlora=not args.no_qlora), indent=2))
    samples = read_samples(args.train_manifest)
    model, processor = load_model(args.model_id, qlora=not args.no_qlora)
    print(json.dumps(trainable_parameter_report(model), indent=2))
    losses = fit_steps(
        model,
        processor,
        samples,
        args.steps,
        args.learning_rate,
        args.batch_size,
        args.gradient_accumulation,
        args.crop_size,
        seed=args.seed,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    processor.save_pretrained(output)
    write_json(
        output / "train_summary.json",
        {
            "model_id": args.model_id,
            "examples": len(samples),
            "steps": args.steps,
            "final_loss": losses[-1] if losses else None,
            "arguments": vars(args),
        },
    )


if __name__ == "__main__":
    main()
