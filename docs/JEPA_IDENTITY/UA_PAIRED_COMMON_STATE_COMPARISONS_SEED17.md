# UA paired common-state comparisons (seed 17)

This report reuses the 4,199 cases from the existing UA-DETRAC legacy-track
replay and compares controls on each fixed event and reconstructed candidate
list. The expected identity is present in 4,159 cases (99.05%); 40 cases are
kept as candidate-coverage loss. Positive deltas mean the candidate is better;
the NLL/Brier convention is documented in the JSON report. Intervals and
permutation p-values are descriptive one-seed paired diagnostics.

| candidate vs reference | Top-1 delta | MRR delta | pairwise AUC delta | net-correction delta |
|---|---:|---:|---:|---:|
| MLP predictive vs MLP CE | -2.12 pp [-2.98, -1.25], p<.0001 | -0.0141 [-.0193, -.0089], p<.0001 | -0.0507 [-.0562, -.0454], p<.0001 | -88 events |
| MLP predictive vs history-last | +7.36 pp [+6.49, +8.22], p<.0001 | +0.0660 [+.0596, +.0729], p<.0001 | +0.0031 [+.0001, +.0063], p=.0506 | +306 events |
| MLP predictive vs MLP ranking | -1.88 pp [-2.64, -1.11], p<.0001 | -0.0126 [-.0172, -.0082], p<.0001 | -0.0410 [-.0459, -.0363], p<.0001 | -78 events |
| Transformer predictive vs Transformer CE | -3.73 pp [-4.76, -2.74], p<.0001 | -0.0283 [-.0346, -.0221], p<.0001 | -0.0608 [-.0671, -.0547], p<.0001 | -155 events |
| Transformer predictive vs history-last | +3.53 pp [+2.84, +4.28], p<.0001 | +0.0393 [+.0343, +.0448], p<.0001 | -0.0141 [-.0175, -.0109], p<.0001 | +147 events |
| Transformer predictive vs Transformer ranking | -1.64 pp [-2.43, -0.89], p<.0001 | -0.0118 [-.0168, -.0071], p<.0001 | -0.0095 [-.0130, -.0059], p<.0001 | -68 events |

The UA replay therefore agrees with BDD on the important comparison: the
current predictive target is below matched CE and ranking controls, while it
can exceed simple last-history aggregation. This is supplementary identity
evidence, not a fresh closed-loop event tracer. The replay reuses existing
predictive-ranking tracks, reconstructs geometry candidates, and uses labels
only after candidate selection for evaluability; it cannot establish an
online HOTA/IDF1 claim or a general H1 theorem.

The machine-readable source is
`UA_PAIRED_COMMON_STATE_COMPARISONS_SEED17.json`.
