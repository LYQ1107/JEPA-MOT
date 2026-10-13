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

**H1 is unresolved.** The data reject any statement that the current Wave-A
result proves JEPA-specific identity information. The common-state audit froze
one pre-decision state and exposed A0--A10 scores on one candidate set,
including EMA/last/mean, native appearance/geometry, temporal MLP, temporal
Transformer, JEPA predictive, ranking, random/shuffled, and constant/affine
controls. The registered identity-residual and conditional candidate-relative
structural candidates were both tested once with matched controls; neither
passed the H1 gate. No additional seed or threshold sweep is justified.

Detailed machine-readable outputs are in
`IDENTITY_ERROR_TAXONOMY.json`, `ASSOCIATION_HEADROOM.json`, and
`IDENTITY_INFORMATION_METRICS.json`. The raw 2.4 MB case JSONL is retained
locally at `runs/JEPA_IDENTITY/h1_wave_a/identity_event_cases.jsonl` and is not
part of the source/data commit.

## Wave-A common-state result

The second audit reconstructs every evaluable predictive-residual event from
the existing dense feature cache. All controls receive the same recorded
history, the same candidate boxes, and the same top-candidate subset; model
queries are computed before the offline identity labels are consulted. The
candidate coverage is 84.43% for every control because the candidate set is
held fixed.

| control | Top-1 | MRR | pairwise AUC | local net correction on eligible candidates |
|---|---:|---:|---:|---:|
| native recorded cost | 97.37% | 0.9850 | 0.9884 | +361 |
| last history | 75.36% | 0.8525 | 0.9012 | +102 |
| mean4 history | 71.96% | 0.8337 | 0.8913 | +62 |
| EMA history | 71.11% | 0.8283 | 0.8866 | +52 |
| MLP CE | 76.13% | 0.8638 | 0.9191 | +111 |
| MLP JEPA predictive | 73.49% | 0.8428 | 0.8956 | +80 |
| MLP JEPA + ranking | 76.21% | 0.8637 | 0.9176 | +112 |
| Transformer CE | 73.32% | 0.8493 | 0.9126 | +78 |
| Transformer JEPA predictive | 73.92% | 0.8457 | 0.8959 | +85 |
| Transformer JEPA + ranking | 74.34% | 0.8538 | 0.9110 | +90 |
| feature shuffle / random | 19.37% / 25.66% | 0.4410 / 0.4986 | 0.4966 / 0.4892 | -559 / -483 |

The native score is a full GeneralTrack association cost and is therefore a
strong reference rather than a feature-only identity probe. Within the
feature-only controls, pure predictive JEPA is below last, CE, and ranking
controls on this common state. The same direction appears in clear, overcast,
rainy, and night strata; the largest sample gap groups are 1 frame (1,037
events) and 2--4 frames (284 events). This is negative evidence for the
current absolute-latent predictive target, not evidence that every possible
JEPA target is impossible.

The full report, including confidence intervals, calibration diagnostics,
domain/gap breakdowns, checkpoint hashes, and causal limitations, is
`COMMON_STATE_IDENTITY_METRICS.json` with its companion
`COMMON_STATE_IDENTITY_METRICS.md`.

## Structural seed17 follow-up

The pre-registered structural candidates were trained once on the source-only
cache with seed17, 1,000 updates, batch 64, and the same video-disjoint split.
The host had no usable NVIDIA driver, so these runs used CPU. The evaluator
scored the same 1,394 common-state events for every method, with 84.43%
candidate coverage:

| candidate / method | Top-1 | MRR | pairwise AUC | net correction |
|---|---:|---:|---:|---:|
| identity-residual predictive | 74.17% | 0.8453 | 0.8954 | +88 |
| identity-residual ranking-only | 74.51% | 0.8555 | 0.9148 | +92 |
| candidate-relative predictive | 42.31% | 0.6355 | 0.7227 | -287 |
| candidate-relative ranking-only | 43.84% | 0.6495 | 0.7416 | -269 |

Residual predictive does not beat its matched control, history-last, or MLP CE;
candidate-relative is substantially worse. Both candidates therefore fail the
registered H1 gate. `STRUCTURAL_FORMAL_SEED17.json` records all checkpoint
hashes, provenance checks, training code hashes, and the discarded numerical
stability attempts. H1 remains unresolved as a general JEPA question, but
these two structural mechanisms are negative on this development audit.

## UA legacy-track extension

The mounted NAS made the raw UA-DETRAC XML labels available, so one bounded
CPU replay was added without starting another seed or retraining. It covers 17
existing UA development videos and 4,199 scored states after excluding 554
history/candidate joins that did not map to the formal dense cache. The replay
holds a reconstructed top-six candidate set fixed and gives every control the
same old `predictive_ranking` history. Coverage is 99.05%.

| control | Top-1 | MRR | pairwise AUC |
|---|---:|---:|---:|
| MLP CE | 93.03% | 0.9606 | 0.7958 |
| MLP JEPA predictive | 90.91% | 0.9465 | 0.7451 |
| MLP JEPA + ranking | 92.79% | 0.9591 | 0.7861 |
| Transformer CE | 90.82% | 0.9481 | 0.7886 |
| Transformer JEPA predictive | 87.09% | 0.9198 | 0.7278 |
| Transformer JEPA + ranking | 88.72% | 0.9316 | 0.7373 |

This supplementary UA replay points in the same direction as BDD: the current
predictive JEPA objective does not beat its CE, ranking, or stable-history
references. It is not the registered GPU event tracer: it reuses an existing
closed-loop output and reconstructs the geometry candidate cost. The H1 gate
therefore remains unresolved, and the replay cannot support an online tracking
claim. Full provenance is in
`UA_LEGACY_TRACK_REPLAY_SEED17.json` and the readable summary is in
`UA_LEGACY_TRACK_REPLAY_SEED17.md`.

