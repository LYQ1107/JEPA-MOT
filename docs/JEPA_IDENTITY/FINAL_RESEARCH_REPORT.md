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

The registered structural follow-up was completed with one seed (17), 1,000
source-only updates, batch 64, and a video-disjoint source split. The host had
no usable NVIDIA driver, so the device was CPU; the seed, cache, candidate set,
and budget were unchanged. Every final checkpoint passed provenance preflight
and scored the same 1,394 common-state events:

| candidate / method | Top-1 | MRR | pairwise AUC | net correction |
|---|---:|---:|---:|---:|
| identity-residual predictive | 74.17% | 0.8453 | 0.8954 | +88 |
| identity-residual ranking-only | 74.51% | 0.8555 | 0.9148 | +92 |
| candidate-relative predictive | 42.31% | 0.6355 | 0.7227 | -287 |
| candidate-relative ranking-only | 43.84% | 0.6495 | 0.7416 | -269 |

Residual predictive is below its matched ranking-only control and below the
history-last (75.36% / 0.8525 / 0.9012 / +102) and MLP CE (76.13% / 0.8638 /
0.9191 / +111) references. Candidate-relative is far below both and has
negative net correction. Neither structural candidate passes the H1 gate. The
complete machine-readable comparison is `STRUCTURAL_FORMAL_SEED17.json`.

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
checkpoint evaluator are implemented and pass real-cache engineering smokes.
The reduced smoke checkpoints remain excluded from H1 evidence. The formal
seed17 structural comparison is now complete and negative under its registered
gate, so no threshold sweep or extra H1 seed is justified. H2 remains a
separate controlled follow-up protocol rather than a result; its execution is
deferred by the unavailable GPU and current single-seed resource boundary, not
by the H1 outcome. A bounded H2-Wave-B fixed-teacher diagnostic has since run
on BDD seed17: all four cells shared 153 recorded pseudo-pairs and 128 update
steps. D-to-D/D-to-P gained 7.19/4.58 fixed-bank Top-1 points from Source-D;
P-to-D/P-to-P gained 3.27/0.65 points from Source-P. Predictive updates
reduced prediction error more, but did not yield the larger ranking gain. This
diagnostic has no full-video HOTA or closed-loop evidence and does not support
an H2 claim; the registered Wave-A factorial remains pending.
On the first 16 shared batches, predictive/discriminative gradient cosine
averaged 0.255 from Source-D and 0.637 from Source-P, while predictive
gradient norms were 0.818 and 0.082 of the discriminative norms. This makes
prediction/ranking gradient misalignment the next falsifiable mechanism to
test.

