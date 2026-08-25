# Guardian/FailCoT execution-verification results

This experiment uses the official execution-verification metadata from the Guardian/FailCoT OOD bundle. It is a fixed-semantics binary task: `success` versus `failure`. Each sample is converted to a composite image with the before observation on top and the after observation on the bottom; for UR5-Fail, the front-camera before/after pair is used. This is the single-view-pair pilot, not the full multi-view Guardian model.

Protocol: 8 support examples per class (16 total) and all remaining examples as query, with three stratified support seeds. InternVL3 uses the corrected `Answer: <label>` format. LoRA is rank 16, alpha 32, dropout 0.05, targeting Q/K/V/O. Ours is covariance KV residual, rank 16, full coefficients, alpha 3, layers 14 and 27, four support updates at learning rate 0.05. Query labels are never used for fitting or selection.

| OOD split | Samples | Frozen Macro-F1 | LoRA Macro-F1 | Ours Macro-F1 |
|---|---:|---:|---:|---:|
| UR5-Fail | 140 | 0.4513±0.0067 | 0.4432±0.0510 | 0.4375±0.0688 |
| RoboFail | 153 | 0.4476±0.0263 | **0.5010±0.0645** | 0.4231±0.0505 |
| RoboVQA execution | 357 | 0.4043±0.0068 | **0.5159±0.0087** | 0.4917±0.0275 |

Ours has substantially better NLL than LoRA on all three splits (UR5-Fail 0.7734 vs 3.0313, RoboFail 0.8340 vs 7.5011, RoboVQA 0.8158 vs 4.3961), but does not yet win Macro-F1. This is informative: fixed success/failure semantics reduce the language-selection burden, yet the current two-layer visual KV update remains weaker than LoRA on OOD classification.

The raw dataset is intentionally not tracked. Recreate it with `hf download paulpacaud/Guardian-FailCoT-OOD-datasets --repo-type dataset --local-dir data/guardian_ood/_ood_tmp` followed by the extraction layout described in `scripts/prepare_guardian_failure.py`.
