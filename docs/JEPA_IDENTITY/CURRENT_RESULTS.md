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

These results are real and reproducible, but they are not a common-state
all-opportunity benchmark and do not prove H1. H2 remains unexecuted in this
branch. See the JSON reports beside this file and the final report for the
decision and next gate.

