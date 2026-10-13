# Current results

**Snapshot:** 2026-10-13, branch `research/jepa-identity-evidence-20261014`.

Phase 0 confirmed that the mount table still lists `/data1/liuyeqiang/nas/A100`,
but the latest preflight receives `Transport endpoint is not connected` from
that SSHFS endpoint. No tracked experiment process is running, and
`nvidia-smi` cannot communicate with the driver. `/data1` has about 35 GB
free. No new dense cache was created;
the four local structural checkpoints and their compact reports use about
32 MB under the ignored `runs/` tree.

Phase 1 reused the completed BDD100K development seed17 rank-reward protocol:

- 19 videos and five controls;
- `sealed_test_used=false`, `GT_read=false`, and updates disabled;
- 1,785 recorded assignment-change cases, of which 1,432 are evaluable;
- predictive residual: 354 corrected vs 300 harmful local events;
- predictive residual conditional intervention Top-1: 97.451%, native: 97.366%;
- predictive residual candidate coverage: 84.43%;
- random rank-reward has zero changed-event rows.

The common-state CPU follow-up mapped all 1,394 evaluable residual events and
held the candidate subset fixed. Native recorded cost reached 97.37% Top-1;
MLP CE reached 76.13%, MLP pure predictive JEPA 73.49%, MLP JEPA+ranking
76.21%, and feature shuffle/random 19.37%/25.66%. This is negative evidence
against the current absolute-latent predictive target as an incremental
identity-ranking mechanism. The selected next hypotheses are identity-residual
JEPA and candidate-relative JEPA; both were then tested once at seed17 with
matched ranking-only controls, with the negative result below.

A paired uncertainty audit now compares the same event and candidate list
across controls. Among 1,177 candidate-present events, MLP predictive JEPA is
2.63 Top-1 points below MLP CE (paired 95% CI -4.50 to -0.76), 0.0210 MRR
below it, and 31 net corrections lower; it is also below history-last and MLP
ranking. Transformer predictive has no reliable Top-1 or correction gain and
has lower pairwise AUC than both Transformer CE and Transformer ranking. This
adds uncertainty evidence to the negative H1 direction without changing the
one-seed scope or claiming a confirmatory multi-seed result. See
`PAIRED_COMMON_STATE_COMPARISONS_SEED17.json`.

The complete seed17 evidence bundle now passes a 29-check machine audit in
`H1_EVIDENCE_AUDIT_SEED17.json`. The gate verifies every required control,
shared candidate coverage, ranking/correction fields, paired-event identity, the two
structural candidates, the UA supplement, and domain-stratified coverage while keeping runtime and
generalization limits explicit. Its status is static consistency only;
`H1_GENERAL_CLAIM_NOT_ESTABLISHED` remains the scientific claim.

The UA legacy replay now also has a paired report:
`UA_PAIRED_COMMON_STATE_COMPARISONS_SEED17.json`. On 4,159 candidate-present
cases, MLP predictive is below CE and ranking but above history-last, while
Transformer predictive is below CE, ranking, and pairwise-AUC references. The
replay remains supplementary because its tracks and geometry candidates are
reconstructed from an existing rollout.

Because frames within a video are correlated, BDD and UA now also have
video-cluster paired audits. The BDD MLP predictive disadvantage versus CE and
ranking remains negative after cluster resampling; UA retains a robust AUC
disadvantage versus CE/ranking while predictive JEPA remains above last-history
aggregation. See `H1_VIDEO_CLUSTER_PAIRED_SEED17.md`.

The structural source-only trainer and checkpoint evaluator first passed
real-cache engineering smokes, including a negative reduced candidate-relative
smoke. The frozen one-seed structural comparison was then completed on CPU
because the host NVIDIA driver remains unavailable. All four final
checkpoints passed preflight and scored the same 1,394 common-state events at
84.43% candidate coverage:

| candidate / method | Top-1 | MRR | pairwise AUC | net correction |
|---|---:|---:|---:|---:|
| identity-residual predictive | 74.17% | 0.8453 | 0.8954 | +88 |
| identity-residual ranking-only | 74.51% | 0.8555 | 0.9148 | +92 |
| candidate-relative predictive | 42.31% | 0.6355 | 0.7227 | -287 |
| candidate-relative ranking-only | 43.84% | 0.6495 | 0.7416 | -269 |

Residual predictive is below its matched ranking-only control and below the
existing history-last and MLP CE references. Candidate-relative is much worse
than both. These runs therefore do not pass the H1 gate; the complete hashes,
provenance, and numerical-stability diagnostics are in
`STRUCTURAL_FORMAL_SEED17.json`.

The mounted UA-DETRAC XML labels also enabled one supplementary legacy-track
CPU replay on 17 existing development videos. It scored 4,199 shared states at
99.05% candidate coverage after 554 cache-join exclusions. UA MLP CE,
predictive JEPA, and JEPA+ranking reached 93.03%, 90.91%, and 92.79% Top-1;
the corresponding Transformer values were 90.82%, 87.09%, and 88.72%. This
reuses old `predictive_ranking` tracks and reconstructs geometry candidates, so
it is negative supporting evidence for the present predictive target rather
than a registered GPU Wave-A result. See `UA_LEGACY_TRACK_REPLAY_SEED17.md`
and `UA_LEGACY_TRACK_REPLAY_SEED17.json`.

These results are real and reproducible, but they are not a common-state
all-opportunity benchmark and do not prove H1. H2 is an independent hypothesis
rather than an H1-gated follow-up. Its bounded BDD seed17 fixed-teacher
diagnostic is complete: the four cells shared 153 recorded pseudo-pairs and a
128-step schedule, with fixed-bank Top-1 gains of +7.19/+4.58 points for
D-to-D/D-to-P and +3.27/+0.65 for P-to-D/P-to-P. This objective-level result
has no full-video HOTA or closed-loop evidence. The registered full-video
factorial remains pending; readiness and the diagnostic report are in
`H2_PROTOCOL_READINESS.json` and `H2_WAVE_B_FIXED_TEACHER_SEED17.json`.
The first 16 shared batches had predictive/discriminative gradient cosine
0.255 from Source-D and 0.637 from Source-P, with predictive gradient norms
0.818 and 0.082 of the discriminative norms. The Wave-D follow-up added the
equal-weight ranking term to the predictive update while keeping the bank and
schedule fixed. Its fixed-bank Top-1 gains were +7.84 points from Source-D
and +1.96 from Source-P, versus +4.58/+0.65 for pure predictive updates. The
P-cell hybrid gradient had cosine 0.998 with the discriminative gradient, so
this is a partial ranking-objective rescue rather than evidence of a uniquely
predictive H2 advantage. The full-video Wave-A factorial remains pending; the
mechanism record is `H2_WAVE_D_PREDICTIVE_RANKING_SEED17.json`.

The zero-training H2 Wave-A preflight confirms that all local BDD detection and
dense-feature inputs and both source snapshots are ready, but it did not launch
the formal rollout because the NAS endpoint and NVIDIA driver are unavailable;
see `H2_WAVE_A_PREFLIGHT_SEED17.json`.

