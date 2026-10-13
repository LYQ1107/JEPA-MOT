# Reviewer hypothesis matrix

| reviewer question | evidence currently available | current answer | remaining discriminating test |
|---|---|---|---|
| Q1: does JEPA add evidence beyond EMA? | common-state audit on 1,394 shared residual states | current absolute-latent predictive JEPA is below EMA/last on Top-1 and AUC; H1 remains unsupported | identity-residual or candidate-relative common-state test |
| Q2: is the result only score-scale or random-feature behavior? | common-state shuffle/random/constant controls are near chance; candidate coverage is identical | random/scale alone cannot explain a positive JEPA result because the current JEPA is not positive | calibration and repeated shuffle in the next structural experiment |
| Q3: does lower prediction error improve identity ranking? | common-state predictive JEPA below CE and JEPA+ranking; old prediction-error reports | no; current prediction target is not identity-ranking evidence | held-out proper scoring for identity-residual target |
| Q4: is continual gain initialization or online learning? | old seed17 continual matrices use different source families; Wave-B and Wave-D fixed-teacher diagnostics share one pseudo-pair bank | ranking partly rescues predictive updates, but the P-cell hybrid is almost collinear with D and there is no full-video H2 gain | Source-D/Source-P x Update-D/Update-P with same teacher, budget, frozen snapshots, and stagewise metrics |
| Q5: why JEPA over temporal CE/Transformer? | common-state MLP/Transformer CE and JEPA controls | unanswered; CE/ranking are at least as strong on this audit | candidate-relative JEPA versus matched pair/Transformer controls |
| Q6: is low forgetting just no update? | Wave-B/Wave-D fixed-bank parameter distances plus 512-example retention subset | bounded updates are measurable, but retention is not a stagewise continual result | same-initialization factorial plus update norm and frozen baseline |
| Q7: what is MOT-specific? | current tracer operates on track histories and candidate competition; absolute target underperforms | no demonstrated MOT-specific JEPA value yet | candidate-relative prediction with one-to-one hard-negative competition |
| Q8: are corrections reproducible across domains? | BDD domain-stratified Wave-A counts | domain sample sizes are sparse outside residual audit | UA common-state audit after GPU availability |
| Q9: are HOTA/AssA gains causal? | complete old development matrices | those metrics are full-rollout results, not causal identity proof | pair event evidence with full-rollout parity and frozen snapshots |
| Q10: is the paper claim ready? | BDD seed17 H1 audit plus Wave-B/Wave-D fixed-teacher H2 diagnostics | no; full H1 identity proof and H2 closed-loop factorial evidence are missing | complete registered H1/H2 gates before sealed evaluation |

