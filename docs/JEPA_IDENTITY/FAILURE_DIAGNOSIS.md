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

The next action is a common-state evaluator, not a threshold sweep. Only after
that evaluator separates these explanations should a residual JEPA,
identity-target JEPA, or candidate-relative JEPA be trained.

