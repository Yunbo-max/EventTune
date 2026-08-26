# Task subspace and actuation-site diagnostics

These experiments use query labels to analyze mechanism.  They are explicitly
**oracle diagnostics**, not valid test-time adaptation estimates and not part
of the clean main comparison.  All runs use InternVL3-8B, rank 16 Full
controllers, layers 14 and 27, and the fixed seed-0 support/query manifests.

The diagnostics test three hypotheses:

1. whether a basis estimated from support captures query correctness-gradient
   energy;
2. whether replacing the support basis with a query-label oracle basis closes
   the LoRA gap; and
3. whether residuals on Q or O, alone or together with K/V, reveal an
   actuation-site limitation.

For support basis `B_s` and query gradient second moment `C_q`, the reported
energy overlap is

```
rho = sum_m tr(B_s,m^T C_q,m B_s,m) / sum_m tr(C_q,m),
```

where the sum is over the selected layers and projection modules.

## Cross-task seed-0 results

| Dataset / site | support-to-query rho | Macro-F1 | NLL |
|---|---:|---:|---:|
| Camelyon17 / KV | **0.9362** | 0.5994 | 0.6629 |
| Camelyon17 / Q | 0.5942 | 0.5125 | 0.7057 |
| Camelyon17 / O | 0.4884 | 0.4172 | 0.7590 |
| Camelyon17 / QKVO | 0.7786 | **0.6664** | **0.6333** |
| RoboFail / KV | **0.7441** | 0.4389 | **0.7219** |
| RoboFail / Q | 0.5039 | **0.4811** | 0.8908 |
| RoboFail / O | 0.2876 | 0.4506 | 0.9488 |
| RoboFail / QKVO | 0.6077 | 0.4538 | 0.7446 |
| ManipBench droid-pick-place / KV | **0.7242** | 0.2193 | 1.4898 |
| ManipBench droid-pick-place / Q | 0.4873 | 0.1548 | 1.8415 |
| ManipBench droid-pick-place / O | 0.3306 | 0.1631 | 1.8497 |
| ManipBench droid-pick-place / QKVO | 0.6122 | **0.2625** | **1.4610** |

The Camelyon run uses the formal four-step learning-rate-0.05 setting.  The
RoboFail run uses the eight-step learning-rate-0.01 follow-up.  ManipBench uses
its formal eight-step learning-rate-0.05 follow-up.  Values should therefore be
interpreted within dataset; the projection-site comparison is controlled
within each row block.

## Query-basis oracle

The query-label oracle changes only the basis: controller coefficients are
still fitted on the original support set and have the same rank and scalar
count.

| Dataset | support-basis KV F1 | query-basis KV F1 | Change |
|---|---:|---:|---:|
| Camelyon17 | 0.5994 | **0.6632** | +0.0638 |
| RoboFail | **0.4389** | 0.4119 | -0.0270 |
| ManipBench droid-pick-place | 0.2193 | **0.2482** | +0.0289 |

This falsifies the strongest version of the subspace-estimation-only
explanation.  Better query geometry helps Camelyon and slightly helps
ManipBench, but the ManipBench oracle remains far below the formal LoRA mean
of 0.4304.  RoboFail becomes worse with the query basis.  Therefore the
remaining gaps cannot be attributed only to `B_s != B_q`.

## Guardian overlap boundary check

The same full-query KV overlap diagnostic on the three Guardian seed-0 folds
gives:

| Split | KV rho | KV Macro-F1 |
|---|---:|---:|
| UR5-Fail | 0.6862 | 0.4601 |
| RoboFail | **0.7441** | 0.4389 |
| RoboVQA execution | 0.6868 | **0.5128** |

Aggregate `rho` does not rank performance within Guardian: RoboFail has the
highest overlap but the lowest F1 of these three diagnostic runs.  The proposed
claim that overlap alone predicts adaptation gain is therefore not supported.
Per-module results point to a more specific issue: on RoboFail, layer-14 K
overlap is 0.8564 while layer-14 V overlap is only 0.3940.  Future analysis
should retain module-wise geometry and class-conditional gradients instead of
compressing them to one scalar.

## Interpretation

The results support a narrower mechanism story:

- Camelyon, where the target capability is already present and the shift is
  largely visual evidence interpretation, has very high KV transfer.  A query
  oracle improves it, and QKVO gives a further bounded diagnostic gain.
- On ManipBench, query-basis KV remains far below LoRA.  QKVO improves over KV
  by 0.0432, showing some actuation-site mismatch, but it does not close the
  much larger gap.  This is consistent with a task/decision-function and
  capacity mismatch rather than an optimization-only failure.
- RoboFail is not explained by low aggregate overlap.  Q-only is strongest on
  seed 0, whereas Q-only is harmful on ManipBench.  A universal switch from KV
  to Q is therefore not justified.

These are seed-0 mechanism diagnostics selected with query information.  They
must be replicated on held-out support seeds before supporting a statistical
claim.  In particular, the QKVO rows are diagnostic upper-bound candidates,
not a replacement for the preregistered KV method.

## Reproduction

The reusable runner is `scripts/diagnose_task_subspace.py`.  Example:

```bash
PYTHONPATH=src python3 scripts/diagnose_task_subspace.py \
  --family internvl3 \
  --model-id artifacts/models/InternVL3-8B-Instruct \
  --support data/prepared/guardian_execution/robofail/seed_0/support.jsonl \
  --query data/prepared/guardian_execution/robofail/seed_0/query.jsonl \
  --output runs/diagnostics/robofail_seed0_full.json \
  --sites KV Q O QKVO --steps 8 --learning-rate 0.01
```

Large covariances and prediction artifacts remain under ignored `runs/`.
