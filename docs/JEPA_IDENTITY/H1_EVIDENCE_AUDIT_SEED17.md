# H1 evidence audit (seed 17)

The machine-readable gate
`H1_EVIDENCE_AUDIT_SEED17.json` checks the compact H1 evidence bundle without
rerunning training or reading raw labels. All 26 static checks pass.

It verifies:

- the BDD common-state audit has 1,394 paired events, identical candidate
  coverage of 1,177 present and 217 absent cases, and all registered controls:
  geometry, last/mean/EMA history, native cost, MLP/Transformer CE, predictive
  JEPA, JEPA+ranking, shuffled/random features, and constant score;
- ranking and correction metrics are present for every control, and the paired
  uncertainty report covers the same event keys;
- the two structural candidates and their matched ranking controls are seed 17,
  1,000-step CPU runs on the fixed common state, with candidate-independent
  history encoders, no target GT in training/gating, and both H1 claims marked
  false;
- the supplementary UA replay has 17 videos and 4,199 scored states, with GT
  excluded from candidate selection and used only for post-score evaluability;
- the event taxonomy, headroom, and ranking reports are all complete and do
  not use target GT for training or gating.
- the domain-stratified paired report covers clear, night, overcast, and rainy
  and sums back to the same 1,394 common-state events.
- the UA paired replay covers 4,199 cases with 4,159 candidate-present cases
  and preserves its legacy-track/evaluability limitations.

The audit status is `COMPLETE_STATIC_H1_EVIDENCE_AUDIT_WITH_RUNTIME_LIMITS`.
That status means the registered seed17 evidence is internally consistent; it
does not mean the general H1 claim is proven. The fresh UA event tracer still
requires restored NAS/GPU runtime, the BDD state is a recorded top-candidate
subset rather than an all-opportunity independent rollout, the structural
comparison has one seed, sealed evaluation remains frozen, and the registered
H2 Wave-A factorial remains unexecuted.
