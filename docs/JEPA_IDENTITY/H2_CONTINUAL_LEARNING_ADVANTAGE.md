# H2: continual-learning advantage

H2 is a separate hypothesis. The previous branch contains complete seed17
development matrices for several continual-learning families, but their source
initializations and update objectives are not a 2x2 causal factorial. Those
results are inherited engineering evidence and cannot be used to claim that
the predictive update itself is superior.

The registered next design is:

| source initialization | online update | name |
|---|---|---|
| discriminative | discriminative | D to D |
| discriminative | predictive | D to P |
| predictive | discriminative | P to D |
| predictive | predictive | P to P |

Frozen D and frozen P controls must use the same snapshots. The teacher,
pseudo-pair protocol, domain order, update budget, replay budget, optimizer
schedule, and stagewise all-domain evaluation must be shared. Report online
gain relative to the frozen checkpoint with the same initialization, along
with HOTA, AssA, IDF1, IDSW, average/signed forgetting, BWT, FWT, update
count, and update magnitude.

H2 remains independently unresolved. The current host has no usable NVIDIA
driver, and the registered full-video Wave-A run has not been allocated in the
present resource window. The negative H1 structural result does not close H2
or act as a prerequisite: a model can have limited frozen identity evidence
yet still have a useful predictive online update, and that possibility must be
tested directly. The protocol-readiness audit is recorded in
`H2_PROTOCOL_READINESS.json`; it lists the four cells and shared invariants
without creating a result.

As a bounded Wave-B diagnostic, one BDD seed17 fixed-teacher comparison has
now been executed. All four cells used the same 153 recorded pseudo-pairs and
the same 128 update indices. From Source-D, D-to-D gained 7.19 percentage
points of fixed-bank Top-1 while D-to-P gained 4.58 points; from Source-P,
P-to-D gained 3.27 points while P-to-P gained 0.65 points. The predictive
updates lowered cosine prediction error more strongly in the D initialization,
but their candidate ranking gain was smaller. The run is recorded in
`H2_WAVE_B_FIXED_TEACHER_SEED17.json` and is a diagnostic only: it has no
full-video HOTA/IDF1/IDSW or closed-loop evidence.

On the first 16 shared batches, the predictive/discriminative gradient cosine
averaged 0.255 for Source-D and 0.637 for Source-P, while the predictive
gradient norm averaged 0.818 and 0.082 of the discriminative norm. This points
to a testable prediction/ranking-gradient mismatch rather than evidence that
the online JEPA hypothesis is closed.

A Wave-D mechanism probe then kept that bank and schedule fixed and added the
existing equal-weight ranking term to the predictive update. D-to-
predictive-plus-ranking gained 7.84 fixed-bank Top-1 points, compared with
4.58 for D-to-P and 7.19 for D-to-D. P-to-predictive-plus-ranking gained 1.96
points, compared with 0.65 for P-to-P and 3.27 for P-to-D. The P-cell hybrid
gradient had cosine 0.998 with the discriminative gradient and only 0.679 with
the predictive gradient, so the apparent rescue is largely a discriminative
ranking effect. The source validation subset moved from 67.77% to 65.43% for
Source-D and from 59.18% to 60.94% for Source-P. This remains objective-level
mechanism evidence only; it has no full-video, HOTA, IDF1, IDSW, forgetting,
BWT, FWT, or closed-loop evidence. The compact record is
`H2_WAVE_D_PREDICTIVE_RANKING_SEED17.json`.

