# Next action

1. Implement a common-state one-step evaluator that snapshots one native
   tracker state and one candidate set, then scores A0--A10 under identical
   inputs and eligibility.
2. Add leakage tests for past-only history, candidate-independent queries,
   target-GT exclusion, and deterministic candidate ordering.
3. Reuse the existing BDD cache for a CPU smoke test and retain the full local
   case audit. Do not start another full-video or multi-seed training job.
4. When a GPU is available, run one small registered common-state comparison
   with predictive residual, EMA, temporal control, random/shuffled, and
   constant/affine calibration controls.
5. Select at most one structural H1 candidate from that result. Only then run
   the same-initialization H2 factorial with one seed and one domain order.

The gate for claiming H1 is a held-out candidate-ranking/proper-scoring gain,
non-random hard-negative improvement, and a net correction advantage under a
common state. The gate for claiming H2 is positive online gain against its own
frozen checkpoint plus an interpretable forgetting/adaptation trade-off.

