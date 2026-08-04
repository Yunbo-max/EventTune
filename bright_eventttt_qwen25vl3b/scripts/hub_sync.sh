#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-pull}"
DATASET_REPO="${EVENTTUNE_DATASET_REPO:-humanlong/EventTune-BRIGHT}"
MODEL_REPO="${EVENTTUNE_MODEL_REPO:-humanlong/EventTune-Qwen2.5-VL-3B}"
HF="${HF_CLI:-hf}"

case "${ACTION}" in
  pull)
    mkdir -p data/hub artifacts/model
    "${HF}" download "${DATASET_REPO}" --repo-type dataset --local-dir data/hub
    "${HF}" download "${MODEL_REPO}" --repo-type model --local-dir artifacts/model
    ;;
  push-data)
    test -d data/hub || { echo "data/hub does not exist" >&2; exit 1; }
    "${HF}" upload "${DATASET_REPO}" data/hub . --repo-type dataset
    ;;
  push-model)
    test -d artifacts/model || { echo "artifacts/model does not exist" >&2; exit 1; }
    "${HF}" upload "${MODEL_REPO}" artifacts/model . --repo-type model
    ;;
  *)
    echo "usage: $0 {pull|push-data|push-model}" >&2
    exit 2
    ;;
esac
