# JEPA-MOT identity and continual-learning research

**Status:** `ACTIVE / H1_DIAGNOSIS`  
**Branch:** `research/jepa-identity-evidence-20261014`  
**Repository:** `LYQ1107/JEPA-MOT`

The research goal is to determine whether a track-conditioned JEPA predictor
contains identity-discriminative information that remains after controlling
for geometry, simple history aggregation, discriminative temporal models,
candidate coverage, score calibration, and model capacity. A separate goal is
to test whether the predictive objective improves unlabeled sequential domain
adaptation under the same initialization and update budget.

The two claims are treated as independent hypotheses:

- **H1:** JEPA contributes incremental identity evidence that improves candidate
  ranking and can be converted into reliable association corrections.
- **H2:** predictive online updating gives a better new-domain/old-domain
  trade-off than matched discriminative updating, replay, and retention
  controls.

The current H1-Wave-A result is an offline audit of existing BDD development
events. It is evidence for where the existing intervention can act, not a
proof of either hypothesis. H2 has not been re-run in this branch.

## Evidence policy

All target labels in this phase are read only by the offline diagnosis. The
existing protocol records `GT_read=false`, has no online updates, and does not
use sealed data. A missing event log is counted as missing coverage rather than
as zero association opportunities. The event diagnosis is a one-step local
counterfactual on the method's own rollout state; it is not an independent
full-baseline trajectory.

The work remains seed17 development-only because the user requested that we do
not start a large multi-seed job while the H1 mechanism is unresolved. Large
weights, image caches, and raw event cases remain local; only compact code,
summaries, and reproducibility metadata are versioned.

## Phase status

| Phase | State | Evidence |
|---|---|---|
| Phase 0: protocol and environment inheritance | complete | old JEPA-CMOT reports, protocol/manifest hashes, no active process, GPU driver unavailable, NAS reachable |
| H1-Wave A: event taxonomy and candidate headroom | complete for BDD development seed17 | `IDENTITY_ERROR_TAXONOMY.json`, `ASSOCIATION_HEADROOM.json` |
| H1-Wave A: candidate identity metrics | complete on recorded changed events | `IDENTITY_INFORMATION_METRICS.json` |
| H1 common-state evaluator | complete for BDD development seed17 residual states | `COMMON_STATE_IDENTITY_METRICS.json`; 1,394/1,394 cases mapped |
| H1 structural candidate selection | protocol implemented, not trained | `STRUCTURAL_CANDIDATE_PROTOCOL.md`; residual and candidate-relative hypotheses in `EXPERIMENT_REGISTRY.csv` |
| H2 same-initialization factorial | pending | do not infer from different-source continual results |
| Final validation | pending | sealed and multi-seed remain frozen |

