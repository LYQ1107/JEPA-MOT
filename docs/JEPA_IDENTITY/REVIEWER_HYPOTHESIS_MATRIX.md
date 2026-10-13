# Reviewer hypothesis matrix

| reviewer question | evidence currently available | current answer | remaining discriminating test |
|---|---|---|---|
| Q1: does JEPA add evidence beyond EMA? | Wave-A residual events and frozen-last control | local residual corrections are positive, but the event set and rollout states differ; not answered | common-state one-step ranking with EMA/last/mean and paired confidence intervals |
| Q2: is the result only score-scale or random-feature behavior? | predictive, discriminative, frozen-last, and random rank-reward controls; random has no changed rows | no scale-invariant conclusion; random is uninformative in this event subset | constant/affine controls plus repeated shuffle on the same event state |
| Q3: does lower prediction error improve identity ranking? | local ranking metrics and old prediction-error reports | not established; residual intervention slightly changes ranking and worsens uncalibrated NLL | held-out proper scoring and hard-negative ranking on common candidates |
| Q4: is continual gain initialization or online learning? | old seed17 continual matrices use different source families | cannot attribute online gain from those comparisons | Source-D/Source-P x Update-D/Update-P with same teacher, budget, and frozen snapshots |
| Q5: why JEPA over temporal CE/Transformer? | old strong-control reports, no common-state Wave-A matrix | unanswered | capacity-matched temporal MLP/Transformer and JEPA with identical candidate access |
| Q6: is low forgetting just no update? | old stagewise update counts and frozen controls | must measure parameter/update magnitude with new-domain gain | same-initialization factorial plus update norm and frozen baseline |
| Q7: what is MOT-specific? | current tracer operates on track histories and candidate competition | the current evidence only shows a tracker intervention point | candidate-relative and identity-residual tests with MOT hard negatives |
| Q8: are corrections reproducible across domains? | BDD domain-stratified Wave-A counts | domain sample sizes are sparse outside residual audit | UA common-state audit after GPU availability |
| Q9: are HOTA/AssA gains causal? | complete old development matrices | those metrics are full-rollout results, not causal identity proof | pair event evidence with full-rollout parity and frozen snapshots |
| Q10: is the paper claim ready? | BDD seed17 offline evidence only | no; H1 common-state and H2 factorial evidence are missing | complete registered H1/H2 gates before sealed evaluation |

