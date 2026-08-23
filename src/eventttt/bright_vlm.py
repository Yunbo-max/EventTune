"""Shared frozen BRIGHT scorer for Gemma 3 and Llama-backed LLaVA models.

The model families use different multimodal chat conventions, but both are
evaluated with the same paired pre/post crop and candidate-label likelihood
protocol.  This module intentionally does not add a generation fallback.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from .aggregation import product_of_experts
from .prompts import SYSTEM_PROMPT, messages, question_for
from .qwen import _find_last_subsequence
from .schemas import DAMAGE_LABELS, Sample
from .vision import crop_pair, load_image


Family = Literal["phi", "gemma", "llama"]


def load_bright_vlm(model_id: str, family: Family):
    """Load a Gemma 3 or Llama-backed LLaVA checkpoint for frozen scoring."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if family == "phi":
        from .task_phi import load_phi

        return load_phi(model_id, efficient_attention=True)

    processor = AutoProcessor.from_pretrained(model_id)
    if family == "llama":
        # Older LLaVA checkpoints omit these processor fields even though the
        # CLIP vision config has a 14px patch and a CLS token.
        vision = getattr(processor.image_processor, "patch_size", None)
        processor.patch_size = vision or 14
        processor.num_additional_image_tokens = 1
        processor.vision_feature_select_strategy = "default"
        dtype = torch.float16
    elif family == "gemma":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unknown BRIGHT VLM family: {family}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=dtype, device_map="auto"
    )
    return model, processor


def _variant(sample: Sample, label: str) -> Sample:
    return Sample.from_dict(
        {**sample.to_dict(), "label": label, "label_id": DAMAGE_LABELS.index(label)}
    )


def _llama_text(sample: Sample, label: str) -> str:
    # The public checkpoint is an LLaVA wrapper around Llama 3 and has no chat
    # template in its processor.  Keep the paired-image order explicit.
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "<image><image>\n"
        f"{SYSTEM_PROMPT} {question_for(sample)}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{label}"
    )


def _phi_text(sample: Sample, label: str) -> str:
    return (
        "<|user|>\n<|image_1|>\n<|image_2|>\n"
        f"{SYSTEM_PROMPT} {question_for(sample)}"
        f"<|end|>\n<|assistant|>\n{label}<|end|>\n"
    )


def _inputs_for_candidate(
    processor, model_family: Family, sample: Sample, label: str, pre, post
):
    if model_family == "gemma":
        text = processor.apply_chat_template(
            messages(_variant(sample, label), True, pre, post),
            tokenize=False,
            add_generation_prompt=False,
        )
    elif model_family == "llama":
        text = _llama_text(sample, label)
    else:
        text = _phi_text(sample, label)
    text_arg = text if model_family == "phi" else [text]
    batch = processor(text=text_arg, images=[pre, post], return_tensors="pt")
    answer_ids = processor.tokenizer(label, add_special_tokens=False)["input_ids"]
    start = _find_last_subsequence(batch["input_ids"][0].tolist(), answer_ids)
    return batch, (start, start + len(answer_ids))


def bright_labeled_batch(processor, model_family: Family, sample: Sample, crop_size: int = 448):
    """Build a BRIGHT pair with loss restricted to the true class tokens."""
    pre, post = crop_pair(
        load_image(sample.pre_image), load_image(sample.post_image),
        sample.bbox_xyxy, size=crop_size,
    )
    batch, (start, end) = _inputs_for_candidate(
        processor, model_family, sample, sample.label, pre, post
    )
    labels = torch.full_like(batch["input_ids"], -100)
    labels[0, start:end] = batch["input_ids"][0, start:end]
    batch["labels"] = labels
    return batch, (start, end)


def enable_bright_lora(model, model_family: Family, rank: int = 16,
                       alpha: int = 32, dropout: float = 0.05):
    """Attach the pre-registered LoRA-TTA modules for a BRIGHT pilot."""
    from peft import LoraConfig, get_peft_model

    targets = ["qkv_proj", "o_proj"] if model_family == "phi" else [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ]
    config = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=targets,
    )
    adapted = get_peft_model(model, config)
    checkpoint = getattr(adapted, "gradient_checkpointing_enable", None)
    if callable(checkpoint):
        checkpoint()
    enable = getattr(adapted, "enable_input_require_grads", None)
    if callable(enable):
        enable()
    return adapted


def fit_bright_lora(model, processor, model_family: Family, samples,
                    crop_size: int = 448, passes: int = 4,
                    learning_rate: float = 2e-4, seed: int = 0):
    """Support-only supervised LoRA fit; query samples never enter this loop."""
    if passes < 0:
        raise ValueError("passes must be non-negative")
    if not samples or passes == 0:
        return []
    generator = torch.Generator().manual_seed(seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    device = next(model.parameters()).device
    model.train()
    losses = []
    for _ in range(passes):
        total = 0.0
        order = torch.randperm(len(samples), generator=generator).tolist()
        for index in order:
            batch, _ = bright_labeled_batch(
                processor, model_family, samples[index], crop_size
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach())
        losses.append(total / len(samples))
    return losses


@torch.inference_mode()
def score_bright_sample(
    model,
    processor,
    model_family: Family,
    sample: Sample,
    crop_size: int = 448,
    controller=None,
    mask_builder=None,
) -> dict:
    """Score one BRIGHT pair by normalized candidate completion likelihood."""
    model.eval()
    pre, post = crop_pair(
        load_image(sample.pre_image),
        load_image(sample.post_image),
        sample.bbox_xyxy,
        size=crop_size,
    )
    device = next(model.parameters()).device
    scores = []
    for label in DAMAGE_LABELS:
        batch, (start, end) = _inputs_for_candidate(
            processor, model_family, sample, label, pre, post
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        if controller is not None:
            if mask_builder is None:
                raise ValueError("mask_builder is required with a KV controller")
            controller.set_mask(mask_builder(batch["input_ids"]))
        try:
            logits = model(**batch).logits.float()
        finally:
            if controller is not None:
                controller.clear_mask()
        targets = batch["input_ids"][0, start:end]
        token_log_probs = torch.log_softmax(logits[0, start - 1 : end - 1], dim=-1)
        scores.append(float(token_log_probs.gather(-1, targets.unsqueeze(-1)).sum()))
        del batch, logits, token_log_probs
    probabilities = product_of_experts([scores])
    prediction_id = int(np.argmax(probabilities))
    return {
        "sample_id": sample.sample_id,
        "event_id": sample.event_id,
        "tile_id": sample.tile_id,
        "label": sample.label,
        "label_id": sample.label_id,
        "prediction": DAMAGE_LABELS[prediction_id],
        "probabilities": probabilities.tolist(),
        "mean_log_scores": scores,
    }
