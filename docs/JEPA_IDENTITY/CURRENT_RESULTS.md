# Current results

**Snapshot:** 2026-10-13, branch `research/jepa-identity-evidence-20261014`.

Phase 0 confirmed that NAS is mounted at `/data1/liuyeqiang/nas/A100`, no
tracked experiment process is running, and `nvidia-smi` cannot communicate
with the driver. `/data1` has about 35 GB free. No new dense cache was created;
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

These results are real and reproducible, but they are not a common-state
all-opportunity benchmark and do not prove H1. H2 remains unexecuted in this
branch. See the JSON reports beside this file and the final report for the
decision and next gate.

