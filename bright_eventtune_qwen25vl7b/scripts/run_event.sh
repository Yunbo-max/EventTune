#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 SPLIT_DIR RUN_DIR [SOURCE_STEPS] [FIXED_EVENT_STEPS]" >&2
  exit 2
fi

SPLIT_DIR="$1"
RUN_DIR="$2"
SOURCE_STEPS="${3:-1000}"
FIXED_EVENT_STEPS="${4:-}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SOURCE_GRADIENT_ACCUMULATION="${SOURCE_GRADIENT_ACCUMULATION:-8}"
CROP_SIZE="${CROP_SIZE:-448}"
EVAL_D4_VIEWS="${EVAL_D4_VIEWS:-8}"

mkdir -p "${RUN_DIR}"

"${PYTHON_BIN}" scripts/train_source.py \
  --train-manifest "${SPLIT_DIR}/source_train.jsonl" \
  --steps "${SOURCE_STEPS}" \
  --gradient-accumulation "${SOURCE_GRADIENT_ACCUMULATION}" \
  --crop-size "${CROP_SIZE}" \
  --output-dir "${RUN_DIR}/source_adapter"

"${PYTHON_BIN}" scripts/evaluate.py \
  --manifest "${SPLIT_DIR}/target_query.jsonl" \
  --adapter "${RUN_DIR}/source_adapter" \
  --d4-views "${EVAL_D4_VIEWS}" \
  --crop-size "${CROP_SIZE}" \
  --output-dir "${RUN_DIR}/source_eval"

ADAPT_ARGUMENTS=()
if [[ -n "${FIXED_EVENT_STEPS}" ]]; then
  ADAPT_ARGUMENTS+=(--fixed-steps "${FIXED_EVENT_STEPS}")
fi
"${PYTHON_BIN}" scripts/adapt_event.py \
  --support-manifest "${SPLIT_DIR}/target_support.jsonl" \
  --source-adapter "${RUN_DIR}/source_adapter" \
  --output-dir "${RUN_DIR}/event_adapter" \
  --crop-size "${CROP_SIZE}" \
  "${ADAPT_ARGUMENTS[@]}"

"${PYTHON_BIN}" scripts/evaluate.py \
  --manifest "${SPLIT_DIR}/target_query.jsonl" \
  --adapter "${RUN_DIR}/event_adapter" \
  --d4-views "${EVAL_D4_VIEWS}" \
  --crop-size "${CROP_SIZE}" \
  --output-dir "${RUN_DIR}/event_eval"

"${PYTHON_BIN}" scripts/compare_predictions.py \
  --baseline "${RUN_DIR}/source_eval/predictions.jsonl" \
  --adapted "${RUN_DIR}/event_eval/predictions.jsonl" \
  --output "${RUN_DIR}/adaptation_gain.json"
