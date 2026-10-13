# Domain-stratified paired H1 audit (BDD seed 17)

This audit reuses the fixed 1,394-event common-state cases and compares each
control within each BDD domain. The correct candidate is present in 612/717
clear, 248/300 night, 165/208 overcast, and 152/169 rainy events. The table
shows candidate minus reference for Top-1 and pairwise AUC; positive means the
predictive candidate is better. Net correction is the event-count difference.
Intervals and p-values are paired diagnostics; the strata are descriptive and
have no multiple-comparison correction.

| domain (events; present) | MLP predictive vs CE: Top-1 / AUC / net | MLP predictive vs ranking: Top-1 / AUC / net | Transformer predictive vs CE: Top-1 / AUC / net |
|---|---:|---:|---:|
| clear (717; 612) | -1.47 pp / -0.0231 / -9 | -0.65 pp / -0.0209 / -4 | -0.49 pp / -0.0187 / -3 |
| night (300; 248) | -6.05 pp / -0.0233 / -15 | -5.24 pp / -0.0188 / -13 | +1.61 pp / +0.0018 / +4 |
| overcast (208; 165) | -2.42 pp / -0.0289 / -4 | -3.03 pp / -0.0235 / -5 | +3.64 pp / -0.0307 / +6 |
| rainy (169; 152) | -1.97 pp / -0.0197 / -3 | -6.58 pp / -0.0302 / -10 | 0.00 pp / -0.0204 / 0 |

The MLP predictive direction is negative in every domain against both CE and
ranking. Its night Top-1 and correction differences have intervals excluding
zero, while the smaller clear/overcast/rainy Top-1 strata are less precise;
their AUC differences remain negative. Transformer predictive Top-1 and
correction vary by domain, but its pairwise AUC is below Transformer CE in
clear, overcast, and rainy and below Transformer ranking in clear, night, and
rainy. No domain provides a consistent positive identity-ranking signal for
the current predictive target.

The machine-readable report is
`H1_DOMAIN_STRATIFIED_PAIRED_SEED17.json`. Rare observation gaps and the small
rainy/overcast strata are diagnostic only; this remains one development seed
and the same recorded rollout state rather than an all-opportunity rerun.
