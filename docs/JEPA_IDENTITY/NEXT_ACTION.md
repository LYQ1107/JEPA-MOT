# Next action

1. The common-state one-step evaluator is complete: it snapshots one native
   tracker state and one candidate set, then scores A0--A10 under identical
   inputs and eligibility.
2. Leakage tests now cover past-only history, candidate-independent queries,
   target-GT exclusion from the forward contract, and deterministic candidate
   ordering (`jepa_identity.test_structural_candidates`).
3. The existing BDD cache and full local case audit are retained; the new
   structural interfaces pass a 16-event real-cache CPU smoke. No new
   full-video or multi-seed training job has been started.
4. When a GPU is available, run one registered common-state comparison using
   the structural candidate protocol in
   `STRUCTURAL_CANDIDATE_PROTOCOL.md`, with residual JEPA and its matched
   ranking-only control first.
5. The common-state result selected two structural H1 candidates, in this
   order: identity-residual JEPA, then candidate-relative JEPA. Train at most
   one of them in the next GPU window, together with its matched CE control.
6. Only after a structural candidate produces a held-out common-state gain,
   run the same-initialization H2 factorial with one seed and one domain order.

The gate for claiming H1 is a held-out candidate-ranking/proper-scoring gain,
non-random hard-negative improvement, and a net correction advantage under a
common state. The gate for claiming H2 is positive online gain against its own
frozen checkpoint plus an interpretable forgetting/adaptation trade-off.

