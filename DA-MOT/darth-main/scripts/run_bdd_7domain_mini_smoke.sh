#!/usr/bin/env bash
set -eo pipefail
cd /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main
export SOURCE_EPOCHS=1 TTA_EPOCHS=1 GPU_NUM=2 SOURCE_SAMPLES_PER_GPU=1 TTA_SAMPLES_PER_GPU=1 WORKERS_PER_GPU=2
mkdir -p work_dirs/bdd_7domain/mini_smoke
cat > work_dirs/bdd_7domain/mini_smoke/README.txt <<'TXT'
Mini-smoke orchestration placeholder created.
Pairs: clear->night, rainy->snowy.
Run after dry-run confirms all domain JSON/configs build correctly. This script is intentionally non-destructive and does not auto-launch training yet.
TXT
echo "Mini-smoke scaffold ready. After dry-run, launch controlled jobs for clear->night and rainy->snowy with GPU_NUM=2."
