# Next action

1. The common-state one-step evaluator is complete: it snapshots one native
   tracker state and one candidate set, then scores A0--A10 under identical
   inputs and eligibility.
2. Leakage tests now cover past-only history, candidate-independent queries,
   target-GT exclusion from the forward contract, and deterministic candidate
   ordering (`jepa_identity.test_structural_candidates`).
3. The existing BDD cache and full local case audit are retained; the
   structural interfaces pass real-cache smoke, source-only training-entry
   smoke, and full common-state preflight.
4. The frozen seed17 structural comparison is complete on CPU: residual and
   candidate-relative predictive methods were each paired with a same-budget
   ranking-only control. Neither passed the H1 gate; see
   `STRUCTURAL_FORMAL_SEED17.json`.
5. Do not start a threshold sweep or additional seeds. The candidate-relative
   numerical-stability failures are fixed in the final implementation and the
   discarded attempts remain engineering diagnostics only.
6. Keep H2 as an independent controlled, same-initialization protocol. The
   negative H1 result does not cancel it. When a usable GPU or an explicitly
   bounded CPU window is available, run one seed and one forward order first;
   do not expand to reverse order or more seeds yet.
7. The bounded BDD seed17 fixed-teacher Wave-B diagnostic is now complete. It
   used the same 153 recorded pseudo-pairs and 128-step schedule in all four
   cells; D updates produced larger fixed-bank ranking gains than P updates.
   Treat this as objective-level evidence only, not as a full H2 result. The
   next experiment is the registered full-video Wave-A factorial with frozen
   controls and stagewise metrics.

The gate for claiming H1 is a held-out candidate-ranking/proper-scoring gain,
non-random hard-negative improvement, and a net correction advantage under a
common state. The gate for claiming H2 is positive online gain against its own
frozen checkpoint plus an interpretable forgetting/adaptation trade-off.

