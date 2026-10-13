# Failure diagnosis after H1-Wave A

The current result is a diagnosis of evidence quality, not a declaration that
JEPA has failed.

## Observations

1. The residual audit has enough events to show 354 local corrections and 300
   local harms, but those events are selected because the intervention changed
   the assignment.
2. The rank-reward controls have only 13--23 recorded changes, and the random
   control has none. They cannot support a stable cross-control ranking claim.
3. Correct candidates are absent from 15.6% of residual evaluable events and
   from 31.3% of predictive-rank-reward events. Detector coverage is lower in
   clear BDD than in the other recorded weather domains.
4. Residual intervention changes conditional Top-1 and MRR only slightly, and
   its uncalibrated NLL/margin move in opposite directions.

## Most likely bottlenecks to test

- the event tracer exposes only a top candidate subset;
- the predictor is evaluated on different propagated histories across methods;
- the existing intervention is gated by native viability and therefore cannot
  recover detector or lifecycle errors;
- matcher costs are not calibrated probabilities;
- residual prediction may contain useful information that is not aligned with
  the ranking decision.

The common-state result sharpens the diagnosis: pure predictive JEPA is below
MLP CE and JEPA+ranking on the same candidate rows, while shuffle/random are
near chance. The current absolute-latent target therefore does not show
incremental identity ranking in this audit. The two structural follow-ups are
identity-residual prediction and candidate-relative prediction; both require
matched same-budget controls and held-out event scoring.

## Structural seed17 result

The two follow-ups were run once at seed17 with 1,000 source-only updates and
their matched ranking-only controls. All four final checkpoints passed
provenance preflight and used the same 1,394-event common state and 84.43%
candidate coverage:

| candidate / method | Top-1 | MRR | AUC | net correction |
|---|---:|---:|---:|---:|
| identity-residual predictive | 74.17% | 0.8453 | 0.8954 | +88 |
| identity-residual ranking-only | 74.51% | 0.8555 | 0.9148 | +92 |
| candidate-relative predictive | 42.31% | 0.6355 | 0.7227 | -287 |
| candidate-relative ranking-only | 43.84% | 0.6495 | 0.7416 | -269 |

Residual predictive is below its matched control, history-last, and MLP CE.
Candidate-relative is substantially worse and has negative net correction. The
final evidence is recorded in `STRUCTURAL_FORMAL_SEED17.json`; it does not
support an H1 claim. Early candidate-relative attempts that failed with
non-finite gradients are retained only as engineering diagnostics, while the
four stable final runs are the registered comparison.

The common-state evaluator is complete and remains the fixed path for every
structural checkpoint. The registered structural options have now been
exhausted at seed17 without an H1 gain; no threshold sweep or extra seed is
justified. H2 remains independently unresolved and is retained as the next
causal continual-learning test. The H1 result does not gate it; execution is
deferred by the unavailable GPU and the current single-seed resource boundary.

The bounded fixed-teacher H2 diagnostic then isolated the objective on one
shared BDD pseudo-pair bank. Discriminative updates produced larger candidate
ranking gains than predictive updates within both source initializations,
while predictive updates reduced cosine prediction error more strongly in the
D initialization. This is a falsifiable warning about prediction/ranking
misalignment, not a full-video H2 conclusion; the next test must preserve the
same controls in the registered causal stream.

On the first 16 shared batches, the predictive/discriminative gradient cosine
was 0.255 from Source-D and 0.637 from Source-P, with predictive gradient
norms 0.818 and 0.082 of the discriminative norms. The next mechanism should
therefore test a controlled identity-ranking component or a calibrated
predictive target, rather than simply increasing replay.

