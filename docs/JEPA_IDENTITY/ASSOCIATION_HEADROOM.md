# Association recovery headroom

`ASSOCIATION_HEADROOM.json` reports observed local headroom among evaluable
changed assignments. A positive `corrected_minus_harmful` is useful for
selecting the next candidate, but it is not a global upper bound because
unchanged decisions, omitted candidates, gate failures, births, deaths, and
missing event logs are outside this table.

| method | evaluable changes | candidate present | local corrected | local harmful | net | observed headroom rate |
|---|---:|---:|---:|---:|---:|---:|
| predictive residual | 1,394 | 1,177 | 354 | 300 | +54 | 25.39% |
| predictive rank reward | 16 | 11 | 1 | 1 | 0 | 6.25% |
| discriminative rank reward | 10 | 8 | 1 | 1 | 0 | 10.00% |
| frozen last rank reward | 12 | 7 | 0 | 0 | 0 | 0.00% |
| random rank reward | 0 | 0 | 0 | 0 | 0 | n/a |

The residual audit is therefore the only current control with enough recorded
events to justify a follow-up common-state analysis. Its positive local net is
also paired with lower intervention cost margin and higher uncalibrated NLL in
the candidate metrics, so the follow-up must include proper held-out scoring
and calibration controls.

