# EventTune

Few-shot event-time adaptation of a small vision-language model for building-damage assessment from paired remote-sensing images.

## Quick start on a fresh GPU node

GitHub is the source of truth for code and reproducibility instructions. The
private Hugging Face collections hold durable dataset derivatives and trained
adapters:

- [EventTune-BRIGHT dataset](https://huggingface.co/datasets/humanlong/EventTune-BRIGHT)
- [EventTune Qwen2.5-VL-7B adapters](https://huggingface.co/humanlong/EventTune-Qwen2.5-VL-7B)
- [Qwen2.5-VL-7B base model](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [BRIGHT official release](https://zenodo.org/records/20072020)

Authenticate once, clone the code, choose a PyTorch wheel compatible with the
node, and restore the persistent artifacts:

```bash
git clone https://github.com/Yunbo-max/EventTune.git
cd EventTune

python3 -m venv .venv
# Example for a CUDA 12.1-compatible node; select another official PyTorch
# wheel index when the host driver requires it.
.venv/bin/python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install -e '.[train,test]'

.venv/bin/hf auth login
HF_CLI=.venv/bin/hf bash scripts/hub_sync.sh pull
.venv/bin/python -m pytest
.venv/bin/python scripts/preflight.py
```

`hub_sync.sh pull` restores `data/manifests/`, `data/splits/`, checksums, and
available model adapters. Transformers obtains the immutable base model directly from
`Qwen/Qwen2.5-VL-7B-Instruct` on the first model run. If the dataset Hub does
not yet contain a required raw asset, follow [Data preparation](#data-preparation)
to acquire it from the official source and rebuild the derivative.

Run a prepared event split with:

```bash
bash scripts/run_event.sh \
  data/splits/bright_4shot/TARGET_EVENT/seed_0 \
  runs/TARGET_EVENT/seed_0 \
  1000
```

Before releasing an ephemeral node, publish approved outputs and push code:

```bash
.venv/bin/python scripts/export_run.py \
  --run-dir runs/TARGET_EVENT/seed_0 \
  --destination runs/TARGET_EVENT/seed_0/run-v1
HF_CLI=.venv/bin/hf bash scripts/hub_sync.sh push-data
HF_CLI=.venv/bin/hf bash scripts/hub_sync.sh push-model
git add -A && git commit -m "Describe the experiment update" && git push
```

## What this project studies

Disaster-damage models often face a large domain shift at deployment. A new event can have a different hazard, country, sensor, season, image resolution, viewing geometry, or post-event modality. A model trained on earlier disasters may therefore perform poorly on the new event.

This project tests one focused hypothesis:

> Can a source-trained Qwen2.5-VL-7B model improve on an unseen disaster by fitting a small temporary adapter to only a few labeled buildings from that event?

Each example contains:

- a pre-event optical crop;
- a post-event optical or SAR crop of the same building;
- the building bounding box and event/tile identifiers;
- one normalized label: `intact`, `damaged`, or `destroyed`.

The primary dataset is BRIGHT. Dataset adapters also exist for xBD and DisasterM3 so the same protocol can later be evaluated across datasets.

## What method is implemented

The method is **support-selected event-time LoRA**, named EventTune. It has seven stages. The Python package retains the historical `eventttt` namespace for compatibility.

### 1. Leakage-safe event split

One disaster is held out as the target event. All other events form `source_train`. A small class-balanced target support set is sampled, and the remaining target buildings form `target_query`.

The split is tile-disjoint: every tile used to construct the support pool is removed from the query set. Consequently, nearby buildings from the same image tile cannot appear on both sides of evaluation.

### 2. Source supervised fine-tuning

`Qwen/Qwen2.5-VL-7B-Instruct` is trained on the source events with label-only supervised loss. The prompt supplies the paired pre/post images and asks for exactly one of the three severity labels.

Training uses standard BF16 LoRA:

- LoRA rank 16, alpha 32, dropout 0.05;
- adapters on `q_proj`, `k_proj`, `v_proj`, and `o_proj`;
- the 7B base VLM remains frozen in bfloat16;
- only the approximately 10.1 million LoRA parameters are optimized.
- source examples use inverse-frequency sampling to prevent the heavily
  imbalanced BRIGHT labels from collapsing short runs to `intact`.

No 4-bit quantization or QLoRA path is used. On the reference RTX A5000 24 GB,
one 448-pixel paired-image training step peaks at approximately 17.0 GiB
allocated and 18.5 GiB reserved CUDA memory.

### 3. Source-only target baseline

Before seeing target support examples, the source adapter is evaluated on the held-out target query set. This produces the baseline against which event adaptation is compared.

Classification uses candidate-label likelihood instead of unconstrained text generation. The model scores the full assistant answer for each possible label and normalizes the three scores into probabilities.

### 4. Support-only adaptation selection

The target query set is never used to choose hyperparameters. Candidate update counts `[0, 4, 8, 16, 32]` are evaluated with stratified cross-validation entirely inside the labeled support set.

Selection maximizes mean support-CV macro-F1, then breaks ties using lower NLL, lower ECE, and fewer update steps. Including zero steps lets the selector reject harmful adaptation.

### 5. Final temporary event adapter

After selection, the source adapter is reset to its original state and fitted on all target support examples for the selected number of updates. This produces a separate, temporary adapter for that event.

### 6. Paired D4 inference

At evaluation time, the same rotation/reflection is applied to both pre- and post-event images. Per-view label evidence is combined with a product of experts: log-scores are averaged across views and then normalized.

### 7. Adaptation-gain evaluation

The source and adapted adapters are evaluated on exactly the same unseen query buildings. `adaptation_gain.json` reports:

- gains where higher is better: macro-F1, balanced accuracy, quadratic weighted kappa;
- error reductions where positive is better: ordinal MAE, NLL, Brier score, and ECE.

This design evaluates **supervised few-shot event-time adaptation**. The current implementation is not unlabeled test-time training, reinforcement learning, or self-distillation.

## Experimental protocol

The recommended study uses:

- leave-one-disaster-event-out evaluation;
- 1, 2, 4, and 8 labeled examples per class;
- seeds 0–4;
- operational support budgets of 12, 24, and 48 examples;
- target events with different hazards and sensors;
- a source-only LoRA baseline and an adapted LoRA model under identical inference settings.

The current two-GPU validation run assigns one independent event to each GPU:

- GPU 0: Hawaii wildfire, seed 0;
- GPU 1: Libya flood, seed 0;
- four support examples per class (12 total);
- 100 source updates and four D4 evaluation views.

This initial run is a proof-of-function check. A single positive event/seed is not sufficient for a paper-level conclusion.

## Repository layout

```text
configs/                 Experiment configuration
data/README.md           Dataset locations and checksums
scripts/download_bright.sh
                         BRIGHT downloader
scripts/prepare_dataset.py
                         BRIGHT, xBD, and DisasterM3 normalization
scripts/make_splits.py   Event-held-out, tile-disjoint splits
scripts/train_source.py  Source-event LoRA SFT
scripts/adapt_event.py   Support-only CV and final event adapter
scripts/evaluate.py      Candidate scoring and D4 evaluation
scripts/compare_predictions.py
                         Baseline/adaptation comparison
scripts/run_event.sh     Complete single-event pipeline
scripts/launch_long_job.py
                         Detached, logged one-GPU launcher
src/eventttt/            Reusable Python implementation
tests/                   Unit tests
```

## Detailed setup

Python 3.10+ and an NVIDIA GPU are expected.

Use a PyTorch CUDA build compatible with the host NVIDIA driver. Do not assume
every compute node has the same CUDA stack. For example, a CUDA 12.1-compatible
node can use:

```bash
python -m venv .venv
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install -e '.[train,test]'
.venv/bin/python scripts/preflight.py
```

The preflight command stops early if CUDA, bfloat16, Qwen-VL, or PEFT is unavailable.

## Ephemeral compute and persistent assets

GitHub is the source of truth for code, configuration, and reproduction
instructions. Hugging Face stores durable dataset derivatives and model
artifacts. A GPU node is disposable and should be recoverable from those two
services.

The private Hub repositories are:

- dataset derivatives: `humanlong/EventTune-BRIGHT`;
- model adapters and run summaries: `humanlong/EventTune-Qwen2.5-VL-7B`.

On a fresh node, clone/pull GitHub, install a PyTorch build compatible with the
node, and then restore persistent assets:

```bash
HF_CLI=.venv/bin/hf bash scripts/hub_sync.sh pull
```

Before a node is released, keep approved manifests/splits under `data/`, and
export each completed run into `artifacts/model/`, then run:

```bash
.venv/bin/python scripts/export_run.py \
  --run-dir runs/TARGET_EVENT/seed_0 \
  --destination runs/TARGET_EVENT/seed_0/run-v1
HF_CLI=.venv/bin/hf bash scripts/hub_sync.sh push-data
HF_CLI=.venv/bin/hf bash scripts/hub_sync.sh push-model
git status
```

The exporter requires both evaluations and `adaptation_gain.json`, omits raw
imagery and caches, rewrites machine-local paths in run metadata, records a
SHA-256 manifest, and refuses to overwrite a different exported run.

Override either Hub destination with `EVENTTUNE_DATASET_REPO` or
`EVENTTUNE_MODEL_REPO`. Do not upload BRIGHT, xBD, or DisasterM3 raw archives
unless their distribution terms explicitly permit it. The official acquisition
locations and checksums are documented in `data/README.md`; the generated
manifests, splits, checksums, and permitted derivatives belong in the dataset
repository.

## Data preparation

Download only the public BRIGHT instance labels:

```bash
bash scripts/download_bright.sh labels data/raw/bright
```

Download the complete official BRIGHT pre/post/target release:

```bash
bash scripts/download_bright.sh full data/raw/bright
```

Create the preferred instance-level manifest and crops:

```bash
.venv/bin/python scripts/prepare_dataset.py bright-coco \
  --root data/raw/bright \
  --annotations data/raw/bright/target_instance_level \
  --output data/manifests/bright_instances.jsonl \
  --output-crops data/crops/bright_instances
```

`prepare_dataset.py` also exposes `bright-raster`, `xbd`, and `disasterm3` subcommands. xBD minor and major damage are merged into `damaged` so all datasets share the three-class label space.

## Create splits

The following command creates five four-shot-per-class splits for all target events:

```bash
.venv/bin/python scripts/make_splits.py \
  --manifest data/manifests/bright_instances.jsonl \
  --output-dir data/splits/bright_4shot \
  --shots-per-class 4 \
  --seeds 0 1 2 3 4
```

Each split directory contains:

```text
source_train.jsonl
target_support.jsonl
target_query.jsonl
split.json
```

## Run one complete event experiment

```bash
SOURCE_GRADIENT_ACCUMULATION=8 \
CROP_SIZE=448 \
EVAL_D4_VIEWS=8 \
bash scripts/run_event.sh \
  data/splits/bright_4shot/TARGET_EVENT/seed_0 \
  runs/TARGET_EVENT/seed_0 \
  1000
```

The pipeline automatically performs source training, baseline evaluation, support-only selection, final event adaptation, adapted evaluation, and comparison.

For one-shot-per-class support, cross-validation is not identifiable. Use a step count pre-registered on source/meta-events:

```bash
bash scripts/run_event.sh SPLIT_DIR RUN_DIR 1000 8
```

## Run two events on two GPUs

Each event uses one GPU; this is not one model sharded across two GPUs.

```bash
.venv/bin/python scripts/launch_long_job.py \
  --gpu 0 \
  --split-dir data/splits/bright_4shot/hawaii-wildfire/seed_0 \
  --run-dir runs/hawaii-wildfire_seed0

.venv/bin/python scripts/launch_long_job.py \
  --gpu 1 \
  --split-dir data/splits/bright_4shot/libya-flood/seed_0 \
  --run-dir runs/libya-flood_seed0
```

## Outputs and interpretation

```text
RUN_DIR/
  source_adapter/train_summary.json
  source_eval/metrics.json
  source_eval/predictions.jsonl
  event_adapter/support_cv.json
  event_adapter/selection.json
  event_eval/metrics.json
  event_eval/predictions.jsonl
  adaptation_gain.json
```

A basic initial success signal requires positive values in:

```text
gain.macro_f1
gain.balanced_accuracy
error_reduction.nll
```

Loss alone does not validate the method. A low training loss only shows that an adapter fitted its training examples. The main claim must be based on adaptation gains on the untouched query set and should be confirmed across target events and seeds.

## CPU/unit smoke test

The model training path requires CUDA, but data/split/metric components can be tested without downloading BRIGHT:

```bash
.venv/bin/python scripts/make_smoke_data.py --output-dir data/smoke
.venv/bin/python scripts/make_splits.py \
  --manifest data/smoke/manifest.jsonl \
  --output-dir data/smoke/splits \
  --target-events flood-alpha \
  --shots-per-class 1 \
  --seeds 0
.venv/bin/python -m pytest
```

## Scope and limitations

- The code classifies/reclassifies known building instances; it does not generate instance masks.
- For the BRIGHT instance task, use official detector/segmenter proposals and `reclassify_coco.py` to replace their severity classes.
- Support examples are labeled, so results should not be described as fully unsupervised TTT.
- Support-CV is noisy with only 12 examples and must be repeated across seeds.
- D4 candidate scoring is deliberately conservative but computationally expensive.
- Cross-sensor and cross-hazard conclusions require evaluation on more than one target event.

## Source ZIP policy

The shareable source archive contains the README, configs, scripts, package source, tests, and metadata. It excludes all datasets (including generated smoke data), `.venv`, downloaded raw imagery, generated manifests/splits, model caches, adapters, and run outputs. Those artifacts are large or reproducible from the commands above and may have separate distribution terms.
