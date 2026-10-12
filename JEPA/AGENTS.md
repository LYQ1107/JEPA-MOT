# Project execution rules

These rules are requested by the project owner and apply to every execution in
this workspace.

1. Conserve tokens. Do not continuously poll running processes or stream their
   output. Check status only at sensible milestones, on an expected completion
   boundary, or when there is evidence of an error.
2. This is not a security/offensive-defense paper. Validation and ordinary
   correctness checks are allowed, but do not over-defend the work or turn it
   into a security exercise.
3. Do not add defensive code for repeatedly discussed cases that are basic and
   genuinely implausible in this project.
4. Where a rubric is useful, apply it as practical guidance rather than as a
   rigid mechanical gate. Use engineering judgment and explain material
   trade-offs.

## Experiment-specific habits

- Preserve useful failure evidence and provenance, but avoid redundant copies
  and redundant logs.
- Prefer milestone summaries over frequent status updates.
- Keep the MOT/JEPA experiment focused on the requested scientific question;
  do not expand scope merely to satisfy a checklist.
