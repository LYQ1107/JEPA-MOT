#!/usr/bin/env bash
# Master orchestrator for the SPA-fix battery (autonomous, sequential on 4 GPU).
# Rationale: the SPA failure mode is a CASCADE collapse (fine on the first
# domain, degrades stage by stage, night collapses). A single-stage clear->night
# run CANNOT reproduce that (the original 00:32 clear->night looked fine at
# MOTA 12.5 yet its 4-domain night collapsed to 4.85). So we skip the
# clear->night filter and validate each fix directly on the 4-domain cascade.
#
# Order: threshold sweep (night, eval-only, on the real 4-domain night ckpt)
#   -> conf70 4domain -> det_only 4domain -> mild_A 4domain -> pipeline 4domain
#   -> collect. Fault-tolerant + resumable (per-scheme markers).
set -uo pipefail
cd "$(dirname "$0")/.."

export GPU_IDS=${GPU_IDS:-"9,8,7,6"}
export GPU_NUM=${GPU_NUM:-4}
export SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-16}
export TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU:-6}
export PYTHON=${PYTHON:-/home/lwr/anaconda3/envs/DARTH/bin/python}

LOG_ROOT="work_dirs/bdd_4domain/spa_fix"
mkdir -p "$LOG_ROOT"
MASTER_LOG="$LOG_ROOT/orchestrator.log"

step(){
    local name="$1" script="$2"
    echo "===== [$(date '+%F %T')] START $name =====" | tee -a "$MASTER_LOG"
    if bash "$script" >> "$MASTER_LOG" 2>&1; then
        echo "===== [$(date '+%F %T')] DONE  $name =====" | tee -a "$MASTER_LOG"
    else
        echo "##### [$(date '+%F %T')] FAILED $name (continuing) #####" | tee -a "$MASTER_LOG"
    fi
    "$PYTHON" tools/collect_spa_fix_results.py >> "$MASTER_LOG" 2>&1 || true
}

echo "########## SPA-FIX BATTERY START $(date '+%F %T') ##########" | tee -a "$MASTER_LOG"

# Step 1: threshold sweep (night, eval-only, zero training risk)
step "scheme1_threshold_sweep"      scripts/run_spa_threshold_sweep_4gpu.sh

# Steps 2-5: 4-domain cascade validations, in spec priority order.
step "scheme3_conf70_4domain"       scripts/run_spa_mild_conf70_4domain_5e_4gpu.sh
step "scheme4_det_only_4domain"     scripts/run_spa_det_only_4domain_5e_4gpu.sh
step "scheme2A_mild_A_4domain"      scripts/run_spa_mild_A_4domain_5e_4gpu.sh
step "scheme5_pipeline_4domain"     scripts/run_spa_pipeline_mild_4domain_5e_4gpu.sh

echo "########## SPA-FIX BATTERY DONE $(date '+%F %T') ##########" | tee -a "$MASTER_LOG"
