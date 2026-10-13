# Identity error taxonomy

The JSON companion is the source of record. This document defines the labels
and scope so a reviewer can distinguish a missing candidate from a ranking
mistake.

## Event labels

- `corrected_local_identity`: native choice is not the dominant prior identity
  and the intervention choice is that identity.
- `harmful_local_identity`: native choice is the dominant prior identity and
  the intervention changes away from it.
- `both_correct`: both choices match the dominant prior identity.
- `neither_correct`: neither choice matches it.
- `mixed_identity_history`: the observed history has purity below 0.75.
- `unscored_history_or_current`: no valid historical identity or the current
  frame is outside the offline evaluation frames.

## Candidate and detector coverage

The event-level `correct_candidate_present` flag means the dominant identity
has at least one matching row in the tracer's recorded candidate subset. It is
not the same as detector recall or matcher eligibility. Across all BDD
development frames, detector-level same-native-group IoU>=0.5 coverage is:

| domain | GT objects | covered | coverage |
|---|---:|---:|---:|
| clear | 2,507 | 1,772 | 70.68% |
| overcast | 2,998 | 2,466 | 82.25% |
| rainy | 1,876 | 1,560 | 83.16% |
| night | 2,338 | 1,859 | 79.51% |

This detector statistic is a presence test, not a post-matcher candidate
eligibility test. Missing `association_changes.jsonl` files are reported in
the JSON and are never treated as zero opportunities.

## Scope limits

The 1,785 recorded events are one-step decisions on each method's own rollout
state. The labels are read only by offline diagnosis. They cannot establish
full-video IDSW/AssA causality or a global upper bound on recoverable errors.

