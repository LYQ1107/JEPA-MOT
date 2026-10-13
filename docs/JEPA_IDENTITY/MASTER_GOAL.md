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
proof of either hypothesis. H2 has only bounded fixed-teacher diagnostics in
this branch; the registered full-video factorial has not been re-run.

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
| Phase 0: protocol and environment inheritance | complete with runtime blockers | old JEPA-CMOT reports, protocol/manifest hashes, no active process; NVIDIA driver unavailable and NAS SSHFS endpoint disconnected |
| H1-Wave A: event taxonomy and candidate headroom | complete for BDD development seed17 | `IDENTITY_ERROR_TAXONOMY.json`, `ASSOCIATION_HEADROOM.json` |
| H1-Wave A: candidate identity metrics | complete on recorded changed events | `IDENTITY_INFORMATION_METRICS.json` |
| H1 common-state evaluator | complete for BDD development seed17 residual states | `COMMON_STATE_IDENTITY_METRICS.json`; 1,394/1,394 cases mapped |
| H1 paired uncertainty audit | complete for the same BDD common-state events | `PAIRED_COMMON_STATE_COMPARISONS_SEED17.json`; 1,177 candidate-present paired cases |
| H1 evidence bundle audit | static checks pass with explicit runtime limits | `H1_EVIDENCE_AUDIT_SEED17.json`; 22/22 checks pass and H1 claim remains unproven |
| H1 structural candidate selection | one seed17 CPU comparison complete; both structural candidates fail the held-out H1 gate | `STRUCTURAL_FORMAL_SEED17.json`, `STRUCTURAL_CANDIDATE_PROTOCOL.md`; all four final checkpoints passed preflight, but neither candidate beats its matched control and strong history/CE references |
| H1 UA extension | supplementary legacy-track CPU replay complete for 17 existing development videos; not the registered GPU tracer | `UA_LEGACY_TRACK_REPLAY_SEED17.json`; 4,199/4,753 selected cases scored, 99.05% candidate coverage, predictive JEPA below CE/ranking controls |
| H2 same-initialization factorial | Wave-B fixed-teacher and Wave-D mechanism diagnostics complete; registered full-video Wave-A not executed | `H2_WAVE_B_FIXED_TEACHER_SEED17.json`, `H2_WAVE_D_PREDICTIVE_RANKING_SEED17.json`, `H2_PROTOCOL_READINESS.json`; ranking partly rescues predictive updates but the P-cell hybrid is discriminative-dominated and no H2 claim is made |
| Final validation | pending | sealed and multi-seed remain frozen |

