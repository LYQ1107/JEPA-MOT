# Common-state identity metrics

This report is a CPU re-evaluation of 1,394 evaluable events from the existing
BDD100K seed17 `predictive_residual_audit` rollout. Every control sees the
same four-row history and the same tracer top-candidate subset. The feature
cache is joined by box geometry; all 1,394 events mapped successfully with
mean minimum candidate IoU above 0.99999999.

The shared candidate coverage is 1,177/1,394 = 84.43%. Conditional Top-1,
MRR, pairwise AUC, AP, and margin are computed only when the correct identity
is present. Uncalibrated NLL/Brier/ECE are diagnostics, not calibrated
probability claims.

| control | Top-1 | MRR | AUC | AP | margin |
|---|---:|---:|---:|---:|---:|
| native recorded cost | 97.37% | 0.9850 | 0.9884 | 0.9846 | 0.7569 |
| history last | 75.36% | 0.8525 | 0.9012 | 0.8539 | 0.0673 |
| history mean4 | 71.96% | 0.8337 | 0.8913 | 0.8366 | 0.0605 |
| history EMA | 71.11% | 0.8283 | 0.8866 | 0.8319 | 0.0583 |
| MLP CE | 76.13% | 0.8638 | 0.9191 | 0.8651 | 0.1035 |
| MLP JEPA predictive | 73.49% | 0.8428 | 0.8956 | 0.8451 | 0.0529 |
| MLP JEPA + ranking | 76.21% | 0.8637 | 0.9176 | 0.8644 | 0.0953 |
| Transformer CE | 73.32% | 0.8493 | 0.9126 | 0.8517 | 0.0945 |
| Transformer JEPA predictive | 73.92% | 0.8457 | 0.8959 | 0.8478 | 0.0522 |
| Transformer JEPA + ranking | 74.34% | 0.8538 | 0.9110 | 0.8554 | 0.0921 |
| feature shuffle | 19.37% | 0.4410 | 0.4966 | 0.4325 | -0.1625 |
| feature random | 25.66% | 0.4986 | 0.4892 | 0.4765 | -0.0256 |
| constant/affine | 13.68% | 0.3969 | 0.5000 | 0.3970 | 0.0000 |

The predictor query used only cached history, geometry metadata, and source
checkpoints. Target labels entered only in `event_metric` after scores were
computed. The native score is retained as a full matcher reference; the
feature-only comparison is the relevant test for the current JEPA target.

The result selects two small structural hypotheses for the next GPU window:

1. **Identity-residual JEPA:** predict a residual around a stable identity
   anchor and compare with a same-capacity residual CE head.
2. **Candidate-relative JEPA:** predict the track-to-candidate relation or
   ranking margin directly, with matched pair-MLP/Transformer controls.

No threshold sweep or multi-seed expansion is justified before one of these
changes produces a held-out common-state gain.

