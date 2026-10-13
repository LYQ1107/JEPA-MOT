# UA legacy-track common-state replay (seed 17)

This is a bounded CPU extension of the identity audit. It uses the existing
17-video UA-DETRAC development `predictive_ranking` track files, the formal
1024-dimensional `dense_current` detector caches, and the raw Train/Test XML
labels mounted under `/data1/liuyeqiang/nas`. It selects only frames where the
old track assignment differs from the top candidate produced by a transparent
last-box IoU geometry replay. The selected state, history, and reconstructed
top-six candidates are then shared by every control.

The replay produced 4,753 candidate-change cases. 4,199 mapped to the dense
cache and were scored; the other 554 were retained as mapping exclusions in
the machine-readable report. Candidate coverage was 99.05% for every control.
The models are the existing UA source-domain seed17 MLP and Transformer
checkpoints; no training, threshold sweep, or extra seed was run.

| control | Top-1 | MRR | pairwise AUC | local net correction |
|---|---:|---:|---:|---:|
| geometry / native replay | 99.64% | 0.9979 | 0.9987 | 0 |
| history last | 83.55% | 0.8805 | 0.7420 | -669 |
| history mean4 | 94.88% | 0.9683 | 0.7580 | -198 |
| history EMA | 94.61% | 0.9670 | 0.7575 | -209 |
| MLP CE | 93.03% | 0.9606 | 0.7958 | -275 |
| MLP JEPA predictive | 90.91% | 0.9465 | 0.7451 | -363 |
| MLP JEPA + ranking | 92.79% | 0.9591 | 0.7861 | -285 |
| Transformer CE | 90.82% | 0.9481 | 0.7886 | -367 |
| Transformer JEPA predictive | 87.09% | 0.9198 | 0.7278 | -522 |
| Transformer JEPA + ranking | 88.72% | 0.9316 | 0.7373 | -454 |
| feature shuffle | 29.55% | 0.5384 | 0.5053 | -2,915 |
| feature random | 35.99% | 0.6008 | 0.5023 | -2,647 |
| constant / affine | 2.26% | 0.2550 | 0.5000 | -4,050 |

The replay therefore shows the same direction as the BDD common-state audit:
the current absolute-latent predictive objective is below its CE and ranking
controls on this UA state. It is negative evidence against the present JEPA
identity target, not evidence that every JEPA target is impossible.

This artifact does not replace the registered GPU Wave-A. It reuses an old
closed-loop output, reconstructs native candidates with last-box IoU rather
than the original GeneralTrack cost matrix, and labels only after state
selection. It cannot support an online HOTA/IDF1 claim or a claim that the
full H1 protocol is complete. The full provenance, per-domain metrics,
per-video hashes, exclusions, and limitations are in
[`UA_LEGACY_TRACK_REPLAY_SEED17.json`](UA_LEGACY_TRACK_REPLAY_SEED17.json).

