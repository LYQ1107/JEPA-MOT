# Final research report (interim)

**Research state:** `ACTIVE / H1_DIAGNOSIS`  
**Scientific conclusion:** insufficient evidence for the final JEPA claim.

The first new stage reused existing BDD100K development seed17 outputs instead
of retraining. It classified 1,785 recorded association changes using offline
labels and measured candidate-level ranking diagnostics. Predictive residual
produced 354 local corrections and 300 local harms, a net +54 among 1,394
evaluable changed events. Its intervention conditional Top-1 was 97.451%
versus 97.366% for the native score. Predictive rank reward and the
discriminative control each had one correction and one harm; frozen-last had
no local correction or harm; random had no changed rows.

This is a useful selection signal, not a proof. The event set is conditioned
on changed assignments, the methods do not share one propagated state, the
tracer records a top candidate subset, and the NLL/Brier/ECE values are
uncalibrated softmax diagnostics. Candidate coverage and detector presence
also limit the recoverable error set. The evidence therefore does not yet
answer whether JEPA contains identity information beyond EMA, geometry,
temporal controls, or score-scale effects.

The prior development HOTA and continual-learning reports remain valid within
their registered scopes, but they do not identify H1 or H2 causal effects. In
particular, different source initializations cannot be used as evidence for an
online JEPA advantage. The same-initialization H2 factorial is still pending.

The next falsifiable step is the common-state evaluator described in
`NEXT_ACTION.md`. If it does not show held-out identity-ranking evidence after
strong controls, the project will retain the negative result and test a
structural change such as identity-residual or candidate-relative prediction
before any large continual-learning sweep.

