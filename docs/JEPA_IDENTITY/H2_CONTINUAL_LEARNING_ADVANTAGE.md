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

H2 is frozen in this phase for two independent reasons: the current host has
no usable NVIDIA driver, and the registered H1 structural candidates did not
produce a held-out identity-ranking gain that would open the H2 gate. The
protocol-readiness audit is recorded in `H2_PROTOCOL_READINESS.json`; it lists
the four cells and shared invariants without creating a result.

