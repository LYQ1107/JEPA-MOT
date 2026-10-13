# Factorial continual-learning results

**Status:** Wave-B fixed-teacher and Wave-D mechanism diagnostics complete;
registered full-video Wave-A factorial remains frozen and unexecuted on this
branch.

The old seed17 continual reports remain under `docs/JEPA_CMOT/`. They show
real development matrices and update counts, but they mix source objectives
and therefore do not identify the online objective effect. This file is kept
as the explicit placeholder for the same-initialization D-to-D/D-to-P/P-to-D/
P-to-P experiment. The readiness and gating decision is machine-readable in
`H2_PROTOCOL_READINESS.json`.

The exact executable four-cell registration is
`H2_WAVE_A_PROTOCOL_SEED17.json`. It fixes the anchor teacher, delay-1
pseudo-pair rule, confidence gate, replay/batch/interval/optimizer schedule,
source calibrations, and the stagewise evidence required for D-to-D, D-to-P,
P-to-D, and P-to-P. It is a protocol artifact only; no cell has been run.

## Wave-A preflight (seed17)

The zero-training preflight is recorded in
`H2_WAVE_A_PREFLIGHT_SEED17.json`. All 63 BDD Wave-A streams (44 adaptation
and 19 development) are locally structurally valid: 12,655 declared frames
have `native_full_frames=true`, `gt_read=false`, zero GT annotations, and
1024-D float32 dense caches whose offsets agree with the prepared streams.
The D/P source checkpoints and source calibrations are present. Execution is still blocked because the
NAS SSHFS mount returns `Transport endpoint is not connected` for the native
images and the host NVIDIA driver is unavailable. No Wave-A process was
started; the legacy retention-graph protocol remains excluded from the causal
2x2.

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

The same batches also give a gradient-direction diagnostic. The mean
predictive/discriminative gradient cosine was 0.255 from Source-D and 0.637
from Source-P; the predictive gradient norm was 0.818 and 0.082 of the
discriminative norm, respectively. This supports the next falsifiable
mechanism test: prediction-only updates can be too weak or misaligned with
candidate ranking, especially after a predictive source initialization.

## Wave-D predictive-plus-ranking mechanism probe (seed17)

The bounded follow-up kept the Wave-B teacher bank, source checkpoints,
128-step schedule, batch size, CPU device, and source validation subset fixed.
It changed only the online objective to the existing equal-weight
`predictive_ranking` loss and ran the two source-initialization cells. The
machine-readable record is `H2_WAVE_D_PREDICTIVE_RANKING_SEED17.json`.

| source | online objective | fixed-bank Top-1 delta | MRR delta | prediction-error delta | source validation Top-1 before -> after |
|---|---|---:|---:|---:|---:|
| D | predictive + ranking | +7.84 pp | +0.0462 | -0.3527 | 67.77% -> 65.43% |
| P | predictive + ranking | +1.96 pp | +0.0175 | +0.0488 | 59.18% -> 60.94% |

Relative to Wave-B, adding ranking raised the D-cell fixed-bank gain above
both D-to-P (+4.58 pp) and D-to-D (+7.19 pp). It also raised the P-cell gain
above P-to-P (+0.65 pp), but it stayed below P-to-D (+3.27 pp). On the first
16 shared batches, the hybrid gradient had cosine 0.841 with D and 0.719 with
P from Source-D; from Source-P it had cosine 0.998 with D and 0.679 with P.
The last number shows that the P-cell hybrid update is effectively dominated
by the discriminative component. This is mechanism evidence that ranking can
repair part of the prediction/ranking mismatch, not evidence for a uniquely
predictive online advantage.

The probe used no target GT for training or gating and produced no full-video
HOTA, IDF1, IDSW, forgetting, BWT, FWT, or closed-loop result. It therefore
does not replace the registered 2x2 Wave-A factorial.

The study is retained as an independent H2 test. It may start with one seed
and one forward order when a usable GPU or an explicitly bounded CPU window is
available; positive H1 evidence is not a prerequisite. Reverse order and
additional seeds remain reserved.

