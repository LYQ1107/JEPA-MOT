# Reviewer hypothesis matrix

| reviewer question | evidence currently available | current answer | remaining discriminating test |
|---|---|---|---|
| Q1: does JEPA add evidence beyond EMA? | common-state audit on 1,394 shared residual states | current absolute-latent predictive JEPA is below EMA/last on Top-1 and AUC; H1 remains unsupported | identity-residual or candidate-relative common-state test |
| Q2: is the result only score-scale or random-feature behavior? | common-state shuffle/random/constant controls are near chance; candidate coverage is identical | random/scale alone cannot explain a positive JEPA result because the current JEPA is not positive | calibration and repeated shuffle in the next structural experiment |
| Q3: does lower prediction error improve identity ranking? | common-state predictive JEPA below CE and JEPA+ranking; old prediction-error reports | no; current prediction target is not identity-ranking evidence | held-out proper scoring for identity-residual target |
| Q4: is continual gain initialization or online learning? | old seed17 continual matrices use different source families | cannot attribute online gain from those comparisons | Source-D/Source-P x Update-D/Update-P with same teacher, budget, and frozen snapshots |
| Q5: why JEPA over temporal CE/Transformer? | common-state MLP/Transformer CE and JEPA controls | unanswered; CE/ranking are at least as strong on this audit | candidate-relative JEPA versus matched pair/Transformer controls |
| Q6: is low forgetting just no update? | old stagewise update counts and frozen controls | must measure parameter/update magnitude with new-domain gain | same-initialization factorial plus update norm and frozen baseline |
| Q7: what is MOT-specific? | current tracer operates on track histories and candidate competition; absolute target underperforms | no demonstrated MOT-specific JEPA value yet | candidate-relative prediction with one-to-one hard-negative competition |
| Q8: are corrections reproducible across domains? | BDD domain-stratified Wave-A counts | domain sample sizes are sparse outside residual audit | UA common-state audit after GPU availability |
| Q9: are HOTA/AssA gains causal? | complete old development matrices | those metrics are full-rollout results, not causal identity proof | pair event evidence with full-rollout parity and frozen snapshots |
| Q10: is the paper claim ready? | BDD seed17 offline evidence only | no; H1 common-state and H2 factorial evidence are missing | complete registered H1/H2 gates before sealed evaluation |

