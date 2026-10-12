#!/usr/bin/env bash
set -eo pipefail
cd /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main
export SOURCE_EPOCHS=${SOURCE_EPOCHS:-20}
export TTA_EPOCHS=${TTA_EPOCHS:-12}
export GPU_NUM=${GPU_NUM:-10}
export WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
if [ "${SOURCE_SAMPLES_PER_GPU:-auto}" = "auto" ] || [ "${TTA_SAMPLES_PER_GPU:-auto}" = "auto" ]; then
  bash scripts/auto_tune_bdd_7domain_batch.sh
fi
bash scripts/init_bdd_7domain_jobs.sh
bash scripts/resume_bdd_7domain_pipeline.sh --status-only
echo "Full matrix initialized. Start one job at a time with: bash scripts/resume_bdd_7domain_pipeline.sh --resume-next"
