# H1: unique identity information

## Operational claim

The relevant claim is incremental identity evidence under the same candidate
set, visible history, geometry, association constraints, and comparable model
capacity. A lower latent prediction loss, a higher cosine score, or an HOTA
change by itself is insufficient.

## Wave-A protocol

The audit reuses
`runs/JEPA_CMOT/development_rank_reward_v1/bdd100k_seed17` on all 19 BDD100K
development videos. Its five controls are `predictive_residual_audit`,
`predictive_rank_reward`, `discriminative_rank_reward`,
`frozen_last_rank_reward`, and `random_rank_reward`. The tracer records the
native assignment and the intervention assignment on the same local state;
the offline auditor then matches boxes to native-group-compatible labels at
IoU >= 0.5 and uses a dominant prior identity with purity >= 0.75.

The event set is restricted to recorded assignment changes. It must therefore
be read together with candidate coverage and the count of absent event logs.

## Observed event evidence

| method | changed | evaluable | candidate present | corrected | harmful | net | corrected/harmful | intervention Top-1 (conditional) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| predictive residual | 1,732 | 1,394 | 1,177 (84.43%) | 354 | 300 | **+54** | 1.18 | 97.451% |
| predictive rank reward | 23 | 16 | 11 (68.75%) | 1 | 1 | 0 | 1.00 | 81.818% |
| discriminative rank reward | 13 | 10 | 8 (80.00%) | 1 | 1 | 0 | 1.00 | 75.000% |
| frozen last rank reward | 17 | 12 | 7 (58.33%) | 0 | 0 | 0 | n/a | 85.714% |
| random rank reward | 0 | 0 | n/a | 0 | 0 | 0 | n/a | n/a |

For `predictive_residual_audit`, the native/intervention conditional Top-1
identity accuracies are 97.366% and 97.451%; MRR is 0.98505 versus 0.98547.
The intervention pairwise AUC is 0.98997 versus 0.98842 for native, while the
uncalibrated candidate NLL is 0.98319 versus 0.94352 and the mean positive vs
best-negative cost margin is 0.6606 versus 0.7569. These mixed directions do
not support a broad claim of new identity information.

The local residual audit does show a positive net count on its changed events,
but that count is a lower-bound, rollout-conditioned quantity. It does not
measure all opportunities, does not compare independent trajectories, and
does not rule out score-scale, history-state, or selection explanations.

## Current H1 judgement

**H1 is unresolved.** The data justify retaining predictive residual as a
candidate for a common-state test, while rejecting any statement that the
current Wave-A result proves JEPA-specific identity information. The next
experiment must freeze one pre-decision state and expose A0--A10 scores on one
candidate set, including EMA/last/mean, native appearance/geometry, temporal
MLP, temporal Transformer, JEPA predictive, ranking, random/shuffled, and
constant/affine score controls.

Detailed machine-readable outputs are in
`IDENTITY_ERROR_TAXONOMY.json`, `ASSOCIATION_HEADROOM.json`, and
`IDENTITY_INFORMATION_METRICS.json`. The raw 2.4 MB case JSONL is retained
locally at `runs/JEPA_IDENTITY/h1_wave_a/identity_event_cases.jsonl` and is not
part of the source/data commit.

