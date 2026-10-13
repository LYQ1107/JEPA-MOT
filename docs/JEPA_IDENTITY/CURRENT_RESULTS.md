# Current results

**Snapshot:** 2026-10-13, branch `research/jepa-identity-evidence-20261014`.

Phase 0 confirmed that the repository is clean before the new files, NAS is
mounted at `/data1/liuyeqiang/nas/A100`, no tracked experiment process is
running, and `nvidia-smi` cannot communicate with the driver. `/data1` has
about 36 GB free, so no new cache or large checkpoint is being created.

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
JEPA and candidate-relative JEPA; neither has been trained yet.

The structural source-only trainer and checkpoint evaluator now pass real-cache
engineering smokes, including a negative reduced candidate-relative smoke. No
formal structural checkpoint has been produced; these smoke numbers are
plumbing evidence only.

These results are real and reproducible, but they are not a common-state
all-opportunity benchmark and do not prove H1. H2 remains unexecuted in this
branch. See the JSON reports beside this file and the final report for the
decision and next gate.

