# End-to-end query inference efficiency

Measured on the same RTX 3090 and the same 300 Hawaii queries. These timings
exclude adaptation and measure only the final query pass.

| Method | Total seconds | Seconds/sample | Peak GPU memory |
|---|---:|---:|---:|
| Pure VLM | 402.154 | 1.3405 | 17,470 MiB |
| support24 LoRA | 411.743 | 1.3725 | 17,508 MiB |
| Full KV-TTT | **401.465** | **1.3382** | 17,470 MiB |
| Diagonal KV-TTT | 401.522 | 1.3384 | 17,470 MiB |

KV-TTT inference is effectively identical to the pure VLM. LoRA is about 2.4%
slower and uses 38 MiB more peak memory in this benchmark. The main KV-TTT
efficiency advantage is adaptation/storage: Full learns 100 scalars and
Diagonal 20 (each serialized state about 44 KiB), versus 10,092,544 trainable
LoRA parameters (about 39.5 MiB). Observed KV extraction averaged about 17 s
and coefficient fitting about 53 s.

Accuracy and efficiency should be stated together: KV-TTT significantly beats
the raw VLM, while fixed support24 LoRA remains more accurate on these four
events. KV-TTT offers a substantially smaller adaptation state and essentially
zero query-time overhead.
