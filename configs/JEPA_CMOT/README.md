# Registered comparison configurations

Run from the repository root with `./run_python.sh`. Paths below `runs/JEPA_CMOT` refer to local source checkpoints and source-only calibrations documented in `docs/JEPA_CMOT`. These copies preserve the actually launched configurations and replace only the machine-specific repository prefix with relative paths. Dataset images and large checkpoints are kept locally.

Frozen comparisons use `jepa_cmot.development_controls`; current object crops additionally select `--feature-name features_crop_current.npz`. Continual comparisons use `jepa_cmot.continual_controls`; H2 and competition controls pass the source `frozen_last.json` as `--teacher-calibration`. Use `--reverse` for the reversed target order and `--compact-intermediate` to keep full state at each domain boundary plus every intermediate model weight.

A matrix is final only after every native adaptation video and every frozen evaluation stage has completed. Multi-seed configs vary source-head initialization/sampling and online replay RNG while sharing the seed17 detector and fixed feature cache. The competition family uses the same native-group negative filter and nine memory fields across all four controls.
