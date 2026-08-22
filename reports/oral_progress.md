# Oral cross-task experiment progress

This file is append-only. BRIGHT is frozen and is not rerun by the oral
generalization experiment.

## 2026-08-22 — protocol and backend

- Models are pinned locally: Qwen2.5-VL-7B-Instruct and Phi-3.5-Vision-Instruct.
- Camelyon17 and ManipBench manifests use query seed `1729`; support seed only
  changes the support set. Query IDs are identical across support seeds.
- CPU test suite: 52 passed.
- Qwen single-image candidate-likelihood backend passes Camelyon binary and
  ManipBench four-option GPU smoke tests on RTX 3090.
- Phi-3.5-Vision numbered-image-tag candidate scorer passes a two-example
  Camelyon GPU smoke (eager attention fallback is pinned because flash-attn is
  not installed).
- Qwen seed-0 Camelyon17 Hospital 2 query300, fixed support16:

| Arm | Macro-F1 | Balanced accuracy | NLL |
|---|---:|---:|---:|
| Frozen | 0.5474 | 0.5967 | 0.6668 |
| LoRA-TTA, 4 support passes | 0.4991 | 0.5833 | 2.6356 |
| Random-KV, rank16 Full alpha3 | 0.5405 | 0.5800 | 0.6691 |
| Gradient-Cov KV, rank16 Full alpha3 | **0.6909** | **0.6967** | **0.5715** |

These are seed-0 gate results, not the final multi-seed estimate. The LoRA
negative result and Random-KV control are retained. Exact artifacts are under
`runs/oral/qwen/camelyon17/hospital_2/seed_0/`.

## 2026-08-22 — ManipBench Q1 Qwen seed-0

All three domains use the same 400-example fixed query and 32-example support
protocol.  Frozen is the resized 448px scorer; the older native-resolution
bridge run is excluded.  LoRA uses four support passes with checkpointing;
KV uses rank 16, full coefficients, alpha 3, and four coefficient steps.

| Domain / arm | Macro-F1 | Balanced accuracy | NLL |
|---|---:|---:|---:|
| bridge / Frozen | 0.2312 | 0.3200 | 1.6511 |
| bridge / LoRA-TTA | **0.7205** | **0.7225** | 4.3489 |
| bridge / Random-KV | 0.2271 | 0.3175 | 1.5663 |
| bridge / Gradient-Cov KV | 0.3561 | 0.3900 | 1.2586 |
| bridge / Hidden Residual | 0.3795 | 0.4125 | 1.2625 |
| droid-pick-place / Frozen | 0.1580 | 0.2725 | 1.8821 |
| droid-pick-place / LoRA-TTA | 0.7124 | 0.7125 | 2.6714 |
| droid-pick-place / Gradient-Cov KV | 0.2944 | 0.3275 | 1.4236 |
| droid-arti / Frozen | 0.1471 | 0.2650 | 1.9514 |
| droid-arti / LoRA-TTA | 0.7854 | 0.7875 | 1.6893 |
| droid-arti / Gradient-Cov KV | 0.2731 | 0.3075 | 1.4384 |

The droid-arti frozen gate is below the pre-registered chance+0.02 threshold
(0.27); it is retained and explicitly reported rather than silently dropped.
The bridge hidden-residual arm and all three Gradient-Cov KV arms are now
executed. Random-KV is currently recorded for bridge; Phi full evaluation,
additional support seeds, and paired multi-seed statistics remain.

The fixed-seed Frozen rerun for ManipBench support seed 1 is complete:
bridge/droid-pick-place/droid-arti balanced accuracies are 0.3200/0.2725/0.2675
(macro-F1 0.2333/0.1553/0.1483). Query IDs and image paths are identical to
seed 0, and a 20-example deterministic check produced identical probabilities
across seeds.

Camelyon17 Hospital 2 is now complete for support seeds 0/1/2. The paired
summaries are in `reports/camelyon_*_multiseed.json`:

| Arm | Mean macro-F1 | Mean balanced accuracy | Mean NLL |
|---|---:|---:|---:|
| Frozen | 0.5538 | 0.6011 | 0.6658 |
| LoRA-TTA | 0.4054 | 0.5356 | 5.7935 |
| Random-KV | 0.5360 | 0.5856 | 0.6736 |
| Gradient-Cov KV | **0.6781** | **0.6822** | **0.5895** |
| Hidden Residual | 0.4696 | 0.5500 | 0.6902 |
