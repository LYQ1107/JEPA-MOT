# Structural H1 candidate protocol

This document freezes the next small experiment before a GPU becomes
available. It is a protocol and an implementation audit, not a result. The
CPU implementation is in
`jepa_identity/structural_candidates.py`; it has not been trained on the
BDD100K or UA-DETRAC target streams.

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

The first GPU window is one seed (`17`), one source-only cache, one
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

The implementation smoke test is:

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

## Current status

The protocol and CPU module are complete. GPU training is pending because the
current host has no usable NVIDIA driver. No structural checkpoint or H1 gain
is claimed, and H2 remains frozen until the H1 gate is resolved.

