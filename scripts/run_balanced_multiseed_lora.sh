#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python3}"
PREP_ROOT="${PREP_ROOT:-data/prepared/neurips}"
RUNS_ROOT="${RUNS_ROOT:-runs/neurips}"
EVENTS="${EVENTS:-hawaii-wildfire libya-flood noto-earthquake turkey-earthquake}"

"${PYTHON_BIN}" scripts/prepare_balanced_support_seeds.py --prep-root "${PREP_ROOT}" \
  --events ${EVENTS} --seeds 1 2
for event in ${EVENTS}; do
  split_dir="${PREP_ROOT}/${event}"; fold_dir="${RUNS_ROOT}/${event}"
  for seed in 1 2; do
    arm_dir="${fold_dir}/support24_seed${seed}_lora"
    eval_dir="${fold_dir}/support24_seed${seed}_lora_eval"
    [[ -f "${arm_dir}/selection.json" ]] || "${PYTHON_BIN}" scripts/adapt_event.py \
      --support-manifest "${split_dir}/target_support_seed${seed}.jsonl" \
      --output-dir "${arm_dir}" --crop-size 448
    [[ -f "${eval_dir}/metrics.json" ]] || "${PYTHON_BIN}" scripts/evaluate.py \
      --manifest "${split_dir}/target_query.jsonl" --adapter "${arm_dir}" \
      --d4-views 1 --crop-size 448 --output-dir "${eval_dir}"
    [[ -f "${arm_dir}/gain_vs_original.json" ]] || "${PYTHON_BIN}" scripts/compare_predictions.py \
      --baseline "${fold_dir}/original_eval/predictions.jsonl" \
      --adapted "${eval_dir}/predictions.jsonl" --output "${arm_dir}/gain_vs_original.json"
  done
done
