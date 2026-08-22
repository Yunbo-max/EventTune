"""Minimal Phi-3.5-Vision single-image candidate scorer.

Phi uses numbered image tags and a negative image-token representation rather
than Qwen's chat-template image objects; keeping this adapter separate makes
the task protocol comparable without changing the established Qwen path.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
import torch
from PIL import Image
from .aggregation import product_of_experts
from .schemas import TaskSample

def load_phi(model_id):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    # The pinned Phi config requests FlashAttention2, which is not part of the
    # reproducible environment; eager attention is numerically equivalent for
    # this evaluation path.
    config._attn_implementation = "eager"
    config._attn_implementation_internal = "eager"
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True,
        config=config, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
    return model, processor

def load_task_image(sample: TaskSample, max_size=448):
    image=Image.open(sample.image).convert('RGB')
    if max(image.size)>max_size:
        scale=max_size/max(image.size)
        image=image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),Image.Resampling.LANCZOS)
    return image

def _batch(processor, sample, image, label):
    image=image if image is not None else load_task_image(sample)
    text=("<|user|>\n<|image_1|>\n"+sample.question+
          "<|end|>\n<|assistant|>\n"+label+"<|end|>\n")
    batch=processor(text=text, images=[image], padding=True, return_tensors='pt')
    ids=processor.tokenizer(label, add_special_tokens=False)['input_ids']
    row=batch['input_ids'][0].tolist()
    start=-1
    for i in range(len(row)-len(ids),-1,-1):
        if row[i:i+len(ids)]==ids: start=i; break
    if start<0: raise ValueError('Phi answer tokens not found')
    return batch,(start,start+len(ids))

def candidate_scores(model, processor, sample, device=None):
    device=device or next(model.parameters()).device; scores=[]; model.eval()
    with torch.inference_mode():
        image=load_task_image(sample)
        for label in sample.candidate_labels:
            batch,(start,end)=_batch(processor,sample,image,label)
            batch={k:v.to(device) for k,v in batch.items()}
            logits=model(**batch).logits.float()[0]
            targets=batch['input_ids'][0,start:end]
            lp=torch.log_softmax(logits[start-1:end-1],dim=-1)
            scores.append(float(lp.gather(-1,targets[:,None]).sum()))
    return np.asarray(scores,dtype=np.float64)

def score_task_sample(model, processor, sample, device=None):
    scores=candidate_scores(model,processor,sample,device)
    probs=product_of_experts([scores.tolist()]); pred=int(np.argmax(probs))
    return {'sample_id':sample.sample_id,'domain_id':sample.domain_id,'group_id':sample.group_id,
            'label':sample.label,'label_id':sample.label_id,'prediction':sample.candidate_labels[pred],
            'probabilities':probs.tolist(),'mean_log_scores':scores.tolist()}
