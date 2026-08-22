# Cross-task completion audit

| Requirement | Evidence | Status |
|---|---|---|
| Do not rerun BRIGHT | No BRIGHT jobs launched; all new runs are under `runs/oral/{qwen,phi}` | PASS |
| Fixed query / support-only seeds | `configs/oral_generalization.yaml`; all Qwen paired JSON reports have `query_ids_identical: true` | PASS |
| Camelyon17 three-seed matrix | `reports/camelyon_*_multiseed.json`, five Qwen arms | PASS |
| ManipBench Q1 three-domain, three-seed matrix | `reports/manipbench_*_multiseed.json`, five Qwen arms per domain | PASS |
| Integrity and reproducibility audit | `reports/task_run_audit.json`: 67/67 Qwen prediction directories passed; pytest 52/52 | PASS |
| Six-hour execution contract | YAML budget, `scripts/run_oral_batch.sh`, debug allowance | PASS |
| Phi single-image backend and gates | Four Frozen gate runs; Camelyon and two droid domains fail gate, bridge barely passes | PASS (gate-limited) |
| Phi adapted LoRA/KV matrix | Fused-qkv Gradient-Cov prototype OOMs on 24-GiB card during support backward; no adapted Phi numbers reported | INCOMPLETE |

The Qwen primary cross-task study is complete and reproducible. The only
remaining gap is Phi adapted evaluation after the preregistered gate-limited
diagnostics; resolving it requires a lower-memory attention implementation or
a larger GPU, not a query-informed change to the protocol.
