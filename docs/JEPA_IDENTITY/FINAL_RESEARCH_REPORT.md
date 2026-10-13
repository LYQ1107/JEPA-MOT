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

The common-state CPU audit then mapped all 1,394 evaluable residual events to
the existing dense feature cache and scored the same candidate subset with
history, geometry, MLP/Transformer CE, predictive JEPA, JEPA+ranking,
shuffle/random, and constant controls. Native recorded cost had 97.37%
conditional Top-1; MLP CE had 76.13%; pure MLP predictive JEPA had 73.49%;
MLP JEPA+ranking had 76.21%; Transformer predictive JEPA had 73.92%. The
candidate coverage was identical at 84.43% for every control. This common-state
result is negative evidence against the current absolute-latent predictive
target as the source of incremental identity ranking.

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

The common-state evaluator, source-only structural trainer, and structural
checkpoint evaluator are now implemented and pass real-cache engineering
smokes. The reduced smoke checkpoints are explicitly excluded from the H1
evidence because they use one CPU step and reduced dimensions; one of them
produces negative local correction, which the evaluator preserves. The next
falsifiable step is one small identity-residual JEPA experiment followed, only
if needed, by one candidate-relative experiment. Neither will be expanded to
multiple seeds or continual-learning matrices until a held-out identity-ranking
gain appears.

