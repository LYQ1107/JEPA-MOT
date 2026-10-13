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

No H2 training was started in this phase because the GPU driver is unavailable
and the user requested that multi-seed work remain limited.

