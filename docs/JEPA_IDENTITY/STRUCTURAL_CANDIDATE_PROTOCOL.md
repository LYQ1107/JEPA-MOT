# Structural H1 candidate protocol

This document freezes the small structural experiment and records its result.
The CPU implementation is in `jepa_identity/structural_candidates.py`; source
training used only the audited BDD100K cache and no BDD100K or UA-DETRAC target
stream was used for updates or gating.

## Why these two candidates

The common-state BDD seed17 audit found that the current absolute-latent
predictive target is below its CE and ranking controls. It therefore does not
justify another temperature or fusion sweep. The next falsifiable mechanisms
are:

1. **Identity-residual JEPA (first):** build a stable identity anchor from
   past observations and predict only the identity-conditioned change,
   `q = normalize(anchor + Delta(history))`. The first control uses the same
   residual architecture and initialization but ranking-only training.
2. **Candidate-relative JEPA (conditional):** build a relation query from the
   past and score the current same-frame hard-negative set only after that
   query exists. Its predictive loss matches the future candidate relation
   distribution; ranking-only and pair-MLP/Transformer controls remain
   required. This candidate runs only if residual JEPA is insufficient or if
   its failure diagnosis points to absolute-target mismatch.

Neither name is a claim of MOT-specific novelty. A candidate passes H1 only if
it improves held-out identity ranking and local correction metrics over its
matched non-predictive control under the same common state.

## Frozen training budget

The first formal window is one seed (`17`), one source-only cache, one
video-disjoint source fit/validation split, and at most 1,000 update steps with
batch 64. The residual candidate and its ranking-only control use the same
architecture, initialization seed, sampler, cache, batch, and update count.
No multi-seed sweep, full-video target retraining, or new dense feature cache
is permitted for this gate. The candidate-relative run is a second one-seed
window only when the residual decision rule permits it.

The source cache must pass the existing metadata checks:

- `source_domain` is the only domain read by training;
- `target_domains_read` is absent or false;
- `gt_read` is true only for source supervision;
- target-domain labels are never used to update weights or select a gate.

## Causal and common-state requirements

The history encoder receives `[history_features, history_metadata,
history_mask]` only. Every active history frame must satisfy
`history_frame < current_frame`; `validate_causal_history` checks this before
training or audit construction. The query is formed before candidate features
are read. Candidate features, candidate masks, and the positive index enter
only the scorer or the offline source loss. The model forward signatures do
not accept `positive_index`, `target_gt`, or a candidate tensor.

For the common-state audit:

- reuse the existing BDD residual rollout state and detector candidate subset;
- preserve detector order at evaluation (`stable_candidate_order` with no
  score argument);
- do not use the correct identity to add, remove, order, or gate candidates;
- report candidate-absent events separately and keep coverage identical across
  controls;
- compute identity labels only after every score has been produced;
- retain native, last/mean/EMA, CE, ranking, shuffled/random, and
  constant/affine controls.

Source hard-negative selection may use a recorded source-only reference score
with stable input-index tie breaking. That ordering is not reused as a target
domain oracle.

## Required measurements and decision rule

For the held-out common state, record Top-1, MRR, Recall@K, pairwise AUC/AP,
positive-vs-hard-negative margin, candidate-level NLL/Brier/ECE, candidate
coverage, corrected and harmful events, and net correction. Report event and
candidate-present counts and domain/gap strata. A lower prediction loss alone
does not pass the gate.

Residual JEPA advances only when it beats the matched residual CE control and
strong history controls on held-out ranking/proper-scoring evidence, survives
shuffle/random controls, and has a positive correction advantage. If it fails,
retain the failure evidence and consider the candidate-relative protocol; do
not tune around the result.

## CPU smoke and provenance

The random-tensor implementation smoke test is:

```bash
./run_python.sh -m unittest \
  jepa_identity.test_identity_event_audit \
  jepa_identity.test_structural_candidates
```

The tests cover finite residual/relation losses, candidate-independent query
construction, strict-past validation, absence of target labels from the
forward contract, candidate masks, and deterministic candidate ordering. A
successful smoke test is engineering evidence only; it is not identity
evidence.

The same interface was also exercised on 16 real BDD common-state events with
the existing `formal_detection_streams_v1` dense cache:

```bash
./run_python.sh -m jepa_identity.structural_candidate_smoke \
  --cases runs/JEPA_IDENTITY/h1_wave_b/common_state_cases.jsonl \
  --inputs-dir runs/JEPA_CMOT/development_rank_reward_v1/bdd100k_seed17/inputs \
  --out runs/JEPA_IDENTITY/h1_wave_c/structural_candidate_smoke.json \
  --max-events 16
```

All 16 cases mapped, with 1024-dimensional features, 14 candidate-present
offline loss checks, history match IoU 1.0, and candidate match IoU above
0.99999994. The report is versioned at
`STRUCTURAL_CANDIDATE_SMOKE.json`. It records no identity metric and makes no
H1 claim; it only verifies integration and the GT boundary on real cached
inputs.

The source-only training entry point was then exercised for one CPU step per
candidate with reduced hidden dimensions. Both jobs used the same existing
BDD source cache, seed17, the video-disjoint split, and no target-domain
labels; their compact summaries are in
`STRUCTURAL_TRAINING_ENTRY_SMOKE.json`. The residual smoke reached source
rank-1 0.4602 and the relation smoke 0.0951 after one uninformative step. These
numbers are harness diagnostics, not a comparison or an H1 result.

The checkpoint evaluator was also run on those two smoke checkpoints. It
scored the same 16 common-state events and retained 14 candidate-present
events for each. The residual smoke produced local net correction `+1`, while
the relation smoke produced `-8`; this expectedly negative reduced smoke is
recorded in `STRUCTURAL_EVALUATOR_SMOKE.json` and is not used to select a
research candidate.

Before a formal checkpoint is evaluated, run the preflight wrapper:

```bash
./run_python.sh -m jepa_identity.preflight_structural_candidate \
  --checkpoint <checkpoint>/final.pt \
  --metrics <checkpoint>/metrics.json \
  --cache runs/JEPA_CMOT/source_identity_cache_v1/bdd100k/features_dense_current.npz \
  --candidate identity_residual --source-domain clear \
  --cases runs/JEPA_IDENTITY/h1_wave_b/common_state_cases.jsonl \
  --inputs-dir runs/JEPA_CMOT/development_rank_reward_v1/bdd100k_seed17/inputs \
  --out-cases <out>/common_state_cases.jsonl \
  --out-report <out>/preflight.json
```

It rejects mismatched checkpoint/cache hashes, candidate or feature dimensions,
non-causal source metadata, target-domain provenance, and candidate input to
the history encoder before scoring. `--allow-engineering-smoke` is required
for reduced smoke checkpoints and must not be used for a formal result. The
real-cache preflight smoke is summarized in `STRUCTURAL_PREFLIGHT_SMOKE.json`.

## Formal seed17 comparison

The frozen one-seed budget was executed on CPU because the host NVIDIA driver
was unavailable. This changes the recorded device, not the source split,
candidate set, seed, or 1,000-step budget. Each candidate used the same
source-only cache, video-disjoint fit/validation split, hidden size 128, batch
64, and matched predictive/ranking-only pair. All four final checkpoints passed
preflight and scored all 1,394 common-state events with identical 84.43%
candidate coverage.

| candidate / method | Top-1 | MRR | pairwise AUC | net correction |
|---|---:|---:|---:|---:|
| identity-residual predictive | 74.17% | 0.8453 | 0.8954 | +88 |
| identity-residual ranking-only | 74.51% | 0.8555 | 0.9148 | +92 |
| candidate-relative predictive | 42.31% | 0.6355 | 0.7227 | -287 |
| candidate-relative ranking-only | 43.84% | 0.6495 | 0.7416 | -269 |
| history-last reference | 75.36% | 0.8525 | 0.9012 | +102 |
| MLP CE reference | 76.13% | 0.8638 | 0.9191 | +111 |

Neither structural candidate passes the H1 gate. Residual predictive is below
its matched ranking-only control and the strongest feature-only references;
candidate-relative is far below them and has negative net correction. The full
machine-readable comparison, checkpoint hashes, provenance, and discarded
non-finite-gradient diagnostics are in `STRUCTURAL_FORMAL_SEED17.json`. These
results are a seed17 development decision and do not prove that every JEPA
target is ineffective.

During the first candidate-relative attempts, finite but very large gradients
overflowed the default float32 norm accumulator and a near-constant candidate
encoder LayerNorm produced NaN backward values. The final implementation uses
an affine candidate encoder, guarded relation normalization, and float64
gradient-norm accumulation with explicit finite checks. The final four runs
are the only candidate-relative identity evidence retained for the H1 gate.

## Current status

The protocol, CPU module, formal seed17 comparison, and preflight path are
complete. The current host still has no usable NVIDIA driver, but the frozen
one-seed structural comparison was completed on CPU. No structural candidate
passes the H1 gate, so H1 remains unresolved and H2 stays frozen; no multi-seed
or online target update has been started.

