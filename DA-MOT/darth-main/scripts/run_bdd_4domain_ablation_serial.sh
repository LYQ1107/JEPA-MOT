#!/usr/bin/env bash
# Serial driver: run the three first-round ablations one after another.
# Strictly serial (one episode at a time) to avoid OOM at batch=10 if run
# concurrently. Each experiment runs full train + eval before the next starts.
#
# Order (default): causal_only -> spa_only -> spa_causal
#   causal_only first: highest-risk module (PCA/trim/new loss) validated earliest.
#
# Usage:
#   nohup bash scripts/run_bdd_4domain_ablation_serial.sh > work_dirs/bdd_4domain_ablation/serial_driver.log 2>&1 &
#
# Env overrides honored by the inner runner (GPU_IDS, GPU_NUM, SAMPLES_PER_GPU,
# WORKERS_PER_GPU, TEST_WORKERS_PER_GPU, FORCE).
set -uo pipefail

EXPS=${EXPS:-"causal_only spa_only spa_causal"}
RUNNER="scripts/run_bdd_4domain_spa_causal_ablation.sh"
ROOT="work_dirs/bdd_4domain_ablation"
mkdir -p "$ROOT"

if [[ ! -f "$RUNNER" ]]; then
    echo "ERROR: runner not found: $RUNNER (run from project root)" >&2; exit 1
fi

echo "==========================================================="
echo "SERIAL ablation driver"
echo "experiments (in order): $EXPS"
echo "started: $(date)"
echo "==========================================================="

for EXP in $EXPS; do
    echo ""
    echo "########################################################"
    echo "### START EXP=$EXP   $(date)"
    echo "########################################################"
    EXP="$EXP" bash "$RUNNER"
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "### EXP=$EXP FAILED (rc=$rc) at $(date). Stopping serial driver." >&2
        exit $rc
    fi
    echo "### DONE EXP=$EXP   $(date)"
done

echo ""
echo "==========================================================="
echo "All experiments complete: $(date)"
echo "Collect results with: python scripts/collect_ablation_results.py"
echo "==========================================================="
