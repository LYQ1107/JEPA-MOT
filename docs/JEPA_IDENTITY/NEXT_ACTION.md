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
5. A supplementary UA legacy-track CPU replay is complete for 17 existing
   development videos. It scored 4,199 shared states and again put predictive
   JEPA below CE and ranking controls. It is explicitly not the registered
   GPU event tracer; see `UA_LEGACY_TRACK_REPLAY_SEED17.md`.
6. Do not start a threshold sweep or additional seeds. The candidate-relative
   numerical-stability failures are fixed in the final implementation and the
   discarded attempts remain engineering diagnostics only.
7. Keep H2 as an independent controlled, same-initialization protocol. The
   negative H1 result does not cancel it. When a usable GPU or an explicitly
   bounded CPU window is available, run one seed and one forward order first;
   do not expand to reverse order or more seeds yet.
8. The bounded BDD seed17 fixed-teacher Wave-B diagnostic is complete. It
   used the same 153 recorded pseudo-pairs and 128-step schedule in all four
   cells; D updates produced larger fixed-bank ranking gains than P updates.
   Treat this as objective-level evidence only, not as a full H2 result.
9. The Wave-D mechanism probe added the equal-weight ranking term to the
   predictive update in two additional cells without adding a seed. It raised
   the fixed-bank gain to +7.84 points from Source-D and +1.96 from Source-P,
   but the P-cell hybrid gradient was almost identical to the discriminative
   gradient. Treat it as a partial ranking-objective rescue, not a predictive
   H2 advantage; see `H2_WAVE_D_PREDICTIVE_RANKING_SEED17.json`.
10. The next execution is the registered full-video Wave-A factorial with
   frozen controls and stagewise metrics, using one seed and one forward order
   when a usable GPU or explicitly bounded CPU window is available. Keep
   reverse order and additional seeds deferred.

The gate for claiming H1 is a held-out candidate-ranking/proper-scoring gain,
non-random hard-negative improvement, and a net correction advantage under a
common state. The gate for claiming H2 is positive online gain against its own
frozen checkpoint plus an interpretable forgetting/adaptation trade-off.

