# Balanced same-event VLM inference suite

This is the execution protocol for the inference experiments added after
`4b8ed53`. It supersedes the older cross-event source-LoRA interpretation for
this experiment only.

## Comparison contract

Every event uses the same `target_support.jsonl` (24 examples, eight per class)
and the same balanced `target_query.jsonl` as its existing `original_eval`.
Every adaptation starts independently from
`Qwen/Qwen2.5-VL-7B-Instruct`; no adapter trained on another event is loaded.

The primary baseline is the existing raw-VLM `original_eval`. The runner
requires its predictions and `compare_predictions.py` refuses to compare arms
unless the sample-ID sets are identical.

| Arm | Starting model | Use of the event's support24 |
|---|---|---|
| `original_eval` | raw VLM | none |
| `support24_lora` | raw VLM | labeled, support-only CV and final LoRA fit |
| `support24_kv_full_a0p5` | raw VLM | labeled correctness-gradient basis and bounded full mixing |
| `support24_kv_unsupervised` | raw VLM | images only; identity-view pseudo-label consistency |
| `support24_kv_diagonal_a0p5` | raw VLM | labeled diagonal-controller ablation |
| `support24_kv_full_a3` | raw VLM | labeled amplitude ablation |
| `support24_kv_unsupervised_full` | raw VLM | images only; full-controller ablation |

The locked KV defaults are rank 5, four updates, learning rate 0.05, L2
`1e-3`, decoder layers 14 and 27, crop size 448, and one evaluation view.
Unsupervised adaptation uses two D4 views. Query labels are used only after an
arm has been saved.

## Execution and resume

```bash
PYTHON_BIN=.venv/bin/python bash scripts/run_balanced_inference_suite.sh
```

Set `EVENTS` to a space-separated subset when debugging one or more folds.
Every adapter, KV state, evaluation, and comparison is skipped only when its
completion artifact exists. Partial evaluations use the existing strict
configuration-aware resume path.

The final command writes:

- `reports/balanced_inference_results.json`, the machine-readable rows and
  macro-event means;
- `reports/balanced_inference_results.md`, the publication-facing table.

Both report each arm relative to raw-VLM original inference. Source-trained
cross-event LoRA results are not substituted for this baseline.
