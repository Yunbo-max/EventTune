# InternVL3 ManipBench steps=8 follow-up

This is the formal InternVL3 ManipBench setup with only the Ours update count
changed from 4 to 8. Rank=16, alpha=3, learning rate=0.05, L2=1e-3, layers
14 and 27, support/query IDs, and all three seeds are unchanged. LoRA is the
same formal rank-16 baseline.

| Domain | Ours steps=4 | Ours steps=8 | Change | LoRA |
|---|---:|---:|---:|---:|
| bridge_pick_place | 0.2993±0.0366 | 0.3347±0.0101 | +0.0354 | 0.5551±0.0159 |
| droid_arti | 0.3307±0.0691 | 0.4064±0.0404 | +0.0757 | 0.6759±0.0150 |
| droid_pick_place | 0.2839±0.0729 | 0.3328±0.0948 | +0.0489 | 0.4304±0.2223 |

Eight updates improve all three domains, especially droid_arti, but Ours
still remains below LoRA on every domain. The run uses the corrected
`Answer: <label>` scoring path and the same formal query sets; no query-based
selection was performed.
