# Paired common-state comparisons (BDD seed 17)

This report compares controls on the same 1,394 recorded BDD development
events and the same candidate lists used by
`COMMON_STATE_IDENTITY_METRICS.json`. The correct identity is present in 1,177
events (84.43%) and absent in 217; absent cases are excluded from ranking
metrics and are not treated as model errors. No training, rollout, new seed,
or sealed data was used.

The reported improvement is candidate minus reference for Top-1, MRR, and
pairwise AUC. For NLL and Brier it is reference minus candidate, so positive
always means the candidate is better. Intervals are paired bootstrap 95% CIs;
the p-values are two-sided paired sign-permutation diagnostics. They are
descriptive one-seed uncertainty measures, without a multiple-comparison
claim.

| candidate vs reference | Top-1 delta | MRR delta | pairwise AUC delta | net-correction delta |
|---|---:|---:|---:|---:|
| MLP predictive vs MLP CE | -2.63 pp [-4.50, -0.76], p=.0064 | -0.0210 [-.0321, -.0101], p=.0003 | -0.0235 [-.0330, -.0149], p<.0001 | -31 events, p=.0081 |
| MLP predictive vs history-last | -1.87 pp [-3.23, -0.51], p=.0105 | -0.0097 [-.0178, -.0018], p=.0194 | -0.0057 [-.0125, +.0010], p=.0996 | -22 events, p=.0111 |
| MLP predictive vs MLP ranking | -2.72 pp [-4.25, -1.19], p=.0010 | -0.0209 [-.0300, -.0119], p=.0002 | -0.0221 [-.0297, -.0148], p<.0001 | -32 events, p=.0008 |
| Transformer predictive vs Transformer CE | +0.59 pp [-1.27, +2.46], p=.6033 | -0.0036 [-.0147, +.0075], p=.5163 | -0.0167 [-.0257, -.0078], p=.0002 | +7 events, p=.5908 |
| Transformer predictive vs history-last | -1.44 pp [-2.80, -.08], p=.0553 | -0.0068 [-.0147, +.0011], p=.0994 | -0.0053 [-.0117, +.0010], p=.1042 | -17 events, p=.0564 |
| Transformer predictive vs Transformer ranking | -0.42 pp [-1.95, +1.19], p=.6793 | -0.0081 [-.0169, +.0007], p=.0839 | -0.0151 [-.0223, -.0081], p<.0001 | -5 events, p=.6684 |

The paired result strengthens the direction already visible in the aggregate
table: the MLP predictive objective is below CE, history-last, and ranking on
the shared event set, including net correction. The Transformer predictive
model has no reliable Top-1 or correction advantage; its pairwise AUC remains
below both Transformer CE and Transformer ranking. This is negative evidence
against the current absolute-latent predictive target, not a proof that every
possible JEPA target fails.

The machine-readable report is
`PAIRED_COMMON_STATE_COMPARISONS_SEED17.json`. It records the paired event-key
digest, finite-metric counts, control summaries, bootstrap/permutation seeds,
and the full limitations statement.
