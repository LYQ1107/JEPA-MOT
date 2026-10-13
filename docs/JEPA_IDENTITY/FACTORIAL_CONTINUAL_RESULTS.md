# Factorial continual-learning results

**Status:** registered and explicitly frozen; not executed on this branch.

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

The study will start with one seed and one forward order only after a selected
mechanism produces positive, reproducible common-state H1 evidence. Reverse
order and additional seeds remain reserved.

