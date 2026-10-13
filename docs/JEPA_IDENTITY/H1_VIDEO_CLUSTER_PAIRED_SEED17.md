# Video-cluster paired H1 audit (seed 17)

Frame-level cases within a video are correlated, so this report resamples the
video as the unit. Each number is a video-balanced mean improvement; positive
means the predictive candidate is better. The existing event-level reports
remain the primary fixed-case summaries, while these cluster intervals show
how much the direction depends on a few long videos.

## BDD100K development (19 videos, 1,177 candidate-present events)

| comparison | Top-1 delta | pairwise AUC delta | net-correction delta |
|---|---:|---:|---:|
| MLP predictive vs MLP CE | -3.63 pp [-5.95, -1.46] | -0.0264 [-0.0376, -0.0152] | -0.0363 [-0.0601, -0.0146] |
| MLP predictive vs history-last | -1.95 pp [-3.89, +0.04] | +0.0026 [-0.0114, +0.0207] | -0.0195 [-0.0383, +0.0005] |
| MLP predictive vs MLP ranking | -3.65 pp [-5.64, -1.68] | -0.0225 [-0.0331, -0.0116] | -0.0365 [-0.0578, -0.0173] |
| Transformer predictive vs Transformer CE | +1.84 pp [-1.23, +5.27] | -0.0166 [-0.0343, +0.0041] | +0.0184 [-0.0118, +5.32] |
| Transformer predictive vs history-last | -2.16 pp [-4.31, -0.34] | -0.0043 [-0.0157, +0.0054] | -0.0216 [-0.0431, -0.34] |
| Transformer predictive vs Transformer ranking | -0.34 pp [-2.41, +1.80] | -0.0194 [-0.0314, -0.0084] | -0.0034 [-0.0240, +0.0168] |

The MLP predictive disadvantage against CE and ranking remains negative after
video clustering. The Transformer result is mixed for Top-1 and correction,
but its AUC is below Transformer ranking in the cluster-balanced audit.

## UA-DETRAC legacy replay (17 videos, 4,159 candidate-present events)

| comparison | Top-1 delta | pairwise AUC delta | net-correction delta |
|---|---:|---:|---:|
| MLP predictive vs MLP CE | -1.90 pp [-4.56, +0.92] | -0.0564 [-0.0783, -0.0385] | -0.0190 [-4.53, +0.89] |
| MLP predictive vs history-last | +8.81 pp [+6.36, +11.30] | +0.0060 [-0.0041, +0.0162] | +0.0881 [+6.35, +11.35] |
| MLP predictive vs MLP ranking | -1.80 pp [-4.21, +0.61] | -0.0466 [-0.0646, -0.0313] | -0.0180 [-4.18, +0.66] |
| Transformer predictive vs Transformer CE | -3.66 pp [-5.97, -1.39] | -0.0711 [-0.0931, -0.0529] | -0.0366 [-6.08, -1.32] |
| Transformer predictive vs history-last | +5.18 pp [+2.82, +7.94] | -0.0107 [-0.0203, -0.0004] | +0.0518 [+2.72, +7.89] |
| Transformer predictive vs Transformer ranking | -0.39 pp [-2.74, +2.75] | +0.0027 [-0.0157, +0.0309] | -0.0039 [-2.76, +2.71] |

The UA cluster result preserves the important distinction seen in the
event-level report: predictive JEPA can beat last-history aggregation while
remaining below CE/ranking. Its AUC disadvantage against CE and ranking is
negative across nearly every video. This is still a legacy-track replay with
reconstructed candidates, not a fresh closed-loop tracer.

Machine-readable inputs are `BDD_VIDEO_CLUSTER_PAIRED_SEED17.json` and
`UA_VIDEO_CLUSTER_PAIRED_SEED17.json`. Cluster p-values are descriptive,
one-seed diagnostics without a multiple-comparison correction.
