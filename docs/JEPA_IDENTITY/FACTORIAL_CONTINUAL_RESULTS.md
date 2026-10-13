# Factorial continual-learning results

**Status:** Wave-B fixed-teacher diagnostic complete; registered full-video
Wave-A factorial remains frozen and unexecuted on this branch.

The old seed17 continual reports remain under `docs/JEPA_CMOT/`. They show
real development matrices and update counts, but they mix source objectives
and therefore do not identify the online objective effect. This file is kept
as the explicit placeholder for the same-initialization D-to-D/D-to-P/P-to-D/
P-to-P experiment. The readiness and gating decision is machine-readable in
`H2_PROTOCOL_READINESS.json`.

Required artifact per cell:

1. source checkpoint and teacher SHA;
2. exact pseudo-label and update-step counts;
3. frozen same-initialization snapshot;
4. stage0--3 full-domain matrix;
5. online gain, forgetting, BWT/FWT, IDSW, and update magnitude;
6. leakage audit showing target GT was not used for training or gating.

## Wave-B fixed-teacher diagnostic (seed17)

The first bounded H2 experiment used one BDD overcast recorded pseudo-pair
bank (153 examples) from a single prior unlabeled stream checkpoint. D-to-D,
D-to-P, P-to-D, and P-to-P replayed the same 128-step, batch-8 index schedule
from their respective Source-D or Source-P snapshots. The machine-readable
record is `H2_WAVE_B_FIXED_TEACHER_SEED17.json`.

Within this fixed bank, D-to-D improved Top-1 by 7.19 percentage points and
D-to-P by 4.58 points from the same Source-D initialization. P-to-D improved
Top-1 by 3.27 points and P-to-P by 0.65 points from Source-P. Predictive
updates reduced prediction error more strongly for D-to-P, but did not improve
the fixed-bank ranking as much as the matched discriminative update. The source
validation subset was only a 512-example retention diagnostic, and the run
produced no HOTA, IDF1, IDSW, or closed-loop result. It therefore does not
support an H2 advantage claim; it identifies a concrete objective-versus-
ranking trade-off for the next full-video causal run.

The study is retained as an independent H2 test. It may start with one seed
and one forward order when a usable GPU or an explicitly bounded CPU window is
available; positive H1 evidence is not a prerequisite. Reverse order and
additional seeds remain reserved.

