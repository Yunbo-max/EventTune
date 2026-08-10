# Balanced same-event inference progress

Last updated: 2026-08-10 22:32 UTC.

Status: **complete**. All four uniform events, expanded supervised ablations,
and multiseed robustness runs have finished. The publication-facing primary outputs are
`balanced_inference_results.md` and `balanced_inference_results.json` in this
directory.

The NeurIPS-level supervised ablation expansion is 40/40 complete. It covers support budget 12/24/48,
layers 14/27/14+27, ranks 5/8/16/32, and update steps 1/2/4/8 using raw VLM,
same-event labeled support, full KV-TTT, and alpha 3. No run has failed or hit
CUDA OOM.

A multiseed robustness suite is also complete (16/16). It adds two independently sampled, balanced,
query-tile-disjoint support24 seeds for Full and Diagonal alpha-3 KV-TTT on all
four events (16 new runs). The final three-seed paired analysis is in
`multiseed_significance.md` and `multiseed_significance.json`: Full versus raw
VLM has mean delta Macro-F1 +0.0457 (95% CI +0.0046 to +0.0847, p=0.0167),
and Diagonal versus raw VLM has +0.0399 (95% CI +0.0063 to +0.0734,
p=0.0061). Full versus Diagonal is not significant (p=0.7347).

This is a live execution snapshot, not the final results table. The experiment
uses only the four strictly uniform BRIGHT events (support 8/8/8 and query
100/100/100). Every adaptation starts independently from raw
`Qwen/Qwen2.5-VL-7B-Instruct` and uses the same event's support24. All deltas
below are against raw-VLM `original_eval` on the identical 300 query IDs.

## Completed raw-VLM baselines

| Event | Query N | Macro-F1 | Balanced accuracy | NLL |
|---|---:|---:|---:|---:|
| Hawaii wildfire | 300 | 0.1992 | 0.3433 | 1.2491 |
| Libya flood | 300 | 0.2585 | 0.3633 | 1.3331 |
| Noto earthquake | 300 | 0.2087 | 0.3333 | 1.2779 |
| Turkey earthquake | 300 | 0.2273 | 0.3400 | 1.4333 |

## Completed adapted results

| Event | Arm | Query N | Macro-F1 | Delta F1 | Delta balanced accuracy | NLL reduction |
|---|---|---:|---:|---:|---:|---:|
| Hawaii wildfire | support24 LoRA | 300 | 0.3375 | +0.1383 | +0.0200 | +0.1540 |
| Hawaii wildfire | supervised full KV-TTT, alpha 0.5 | 300 | 0.2163 | +0.0171 | +0.0033 | +0.0374 |
| Hawaii wildfire | unsupervised diagonal KV-TTT, alpha 0.5 | 300 | 0.1819 | -0.0173 | -0.0067 | +0.0034 |

The Hawaii supervised-diagonal run is not a completed negative result. Its
first attempt was interrupted by CUDA OOM when an unrelated GPU experiment
overlapped the suite, so it remains pending and will be rerun in isolation.

## Primary queue status

- Completed: raw-VLM originals and support24 LoRA for all four events.
- Completed primary KV-TTT: supervised full alpha 3, supervised diagonal
  alpha 3, and unsupervised diagonal alpha 3 for all four events.
- Completed ablations: alpha-0.5 amplitude variants and unsupervised
  full-controller variants for all four events.
- No suite process remains active and all result comparisons contain 300
  query examples per event and arm.

## Resume and failure behavior

The suite is artifact-resumable: completed adapters, KV states, evaluations,
and comparisons are skipped only when their completion artifacts exist, while
partial evaluations resume through the configuration-aware evaluation path.
Restarting

```bash
PYTHON_BIN=.venv/bin/python bash scripts/run_balanced_inference_suite.sh
```

therefore continues the queue without discarding completed results. A network
failure does not affect local training or evaluation because the model and
BRIGHT data are cached locally. It can delay only GitHub/Hugging Face
publication; the local Git commit and run artifacts remain available for a
later retry.

The final complete suite will generate
`reports/balanced_inference_results.json` and
`reports/balanced_inference_results.md` using
`scripts/summarize_balanced_inference.py`.
