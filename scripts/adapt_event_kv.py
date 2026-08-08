#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from eventttt.io import read_samples, write_json
from eventttt.kv_ttt import (
    build_controller_from_state,
    build_post_image_mask_fn,
    default_layers,
    discover_language_decoder_kv,
    extract_kv_subspace,
    fit_kv_coefficients,
    freeze_model,
    image_token_id_of,
    load_kv_state,
    save_kv_state,
)
from eventttt.qwen import DEFAULT_MODEL, load_model, preflight, trainable_parameter_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-Visual Residual KV-TTT: extract a correctness-gradient KV "
        "subspace from target support and learn 32 event coefficients"
    )
    parser.add_argument("--support-manifest", required=True)
    parser.add_argument("--source-adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--alpha-max", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--crop-size", type=int, default=448)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(json.dumps(preflight(require_gpu=True), indent=2))
    torch.manual_seed(args.seed)

    support = read_samples(args.support_manifest)
    if len(support) != 12:
        print(f"warning: standard Hawaii support is 12 examples, found {len(support)}")

    model, processor = load_model(
        args.model_id, source_adapter=args.source_adapter, gradient_checkpointing=False
    )
    freeze_model(model)
    modules, num_layers = discover_language_decoder_kv(model)
    selected_layers = args.layers or default_layers(num_layers)
    module_slice = [
        (layer_id, kind, module)
        for layer_id, kind, module in modules
        if layer_id in set(selected_layers)
    ]
    missing = [layer for layer in selected_layers if layer not in {l for l, _, _ in modules}]
    if missing:
        raise ValueError(f"Requested layers not present in the decoder: {missing}")
    if len({kind for _, kind, _ in module_slice}) != 2:
        raise RuntimeError("Selected layers must include both k_proj and v_proj")

    device = next(model.parameters()).device
    build_post_mask = build_post_image_mask_fn(image_token_id_of(model))

    report = trainable_parameter_report(model)
    print(json.dumps(
        {
            "num_layers": num_layers,
            "selected_layers": selected_layers,
            "rank": args.rank,
            "trainable_after_freeze": report,
        },
        indent=2,
    ))
    if report["trainable"] != 0:
        raise RuntimeError("VLM/LoRA parameters must be frozen before KV extraction")

    t0 = time.time()
    bases, spectra = extract_kv_subspace(
        model,
        processor,
        support,
        module_slice,
        build_post_mask,
        rank=args.rank,
        crop_size=args.crop_size,
    )
    extraction_seconds = time.time() - t0

    controller = build_controller_from_state(
        module_slice, {"bases": bases, "rank": args.rank, "alpha_max": args.alpha_max}, device=device
    )
    num_scalars = controller.num_scalars()
    print(f"KV-TTT trainable scalars: {num_scalars}")
    if num_scalars != len(module_slice) * args.rank:
        raise RuntimeError("Unexpected number of KV-TTT scalars")

    t0 = time.time()
    losses = fit_kv_coefficients(
        model,
        controller,
        processor,
        support,
        build_post_mask,
        steps=args.steps,
        learning_rate=args.learning_rate,
        l2=args.l2,
        crop_size=args.crop_size,
    )
    adaptation_seconds = time.time() - t0

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_kv_state(
        output / "kv_state.pt",
        controller,
        args.model_id,
        metadata={
            "source_adapter": args.source_adapter,
            "num_layers": num_layers,
            "crop_size": args.crop_size,
            "seed": args.seed,
        },
    )

    write_json(
        output / "extraction.json",
        {
            "model_id": args.model_id,
            "num_decoder_layers": num_layers,
            "selected_layers": selected_layers,
            "rank": args.rank,
            "support_examples": len(support),
            "gradient_passes": len(support),
            "crop_size": args.crop_size,
            "trainable_before_extraction": report,
            "kv_ttt_scalars": num_scalars,
            "spectra": spectra,
            "extraction_seconds": round(extraction_seconds, 2),
            "arguments": vars(args),
        },
    )
    write_json(
        output / "adaptation.json",
        {
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "steps": args.steps,
            "coefficient_l2": args.l2,
            "alpha_max": args.alpha_max,
            "gradient_clip": 1.0,
            "losses": losses,
            "final_loss": losses[-1] if losses else None,
            "effective_gamma": controller.effective_gamma(),
            "coefficient_norm": {
                key: float(torch.linalg.vector_norm(parameter.detach().cpu()))
                for key, parameter in controller.coefficients.items()
            },
            "adaptation_seconds": round(adaptation_seconds, 2),
        },
    )
    print(json.dumps(
        {
            "output_dir": str(output),
            "extraction_seconds": round(extraction_seconds, 2),
            "adaptation_seconds": round(adaptation_seconds, 2),
            "final_loss": losses[-1] if losses else None,
            "effective_gamma": controller.effective_gamma(),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
