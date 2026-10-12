#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHON_BIN="${PYTHON_BIN:-/home/lwr/anaconda3/envs/ByteTrack/bin/python}"
export GPU_IDS="${GPU_IDS:-9 8 7 6}"
export MAX_PARALLEL="${MAX_PARALLEL:-4}"
export DROP_MISSING_IMAGES="${DROP_MISSING_IMAGES:-1}"
export DET_SOURCE="${DET_SOURCE:-source_detector}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/generaltrack}"
export PREPARED_ROOT="${PREPARED_ROOT:-datasets/generaltrack_prepared}"
export INPUT_CHECK="${INPUT_CHECK:-outputs/generaltrack_input_check.csv}"
export SUMMARY_CSV="${SUMMARY_CSV:-outputs/generaltrack/generaltrack_multidataset_summary.csv}"

echo "[generaltrack] resume missing ua_detrac+vkitti2: $(date '+%F %T')"
echo "[generaltrack] gpu ids: $GPU_IDS"
echo "[generaltrack] summary: $SUMMARY_CSV"

DATASETS="ua_detrac vkitti2" DOMAINS="all" bash scripts/run_generaltrack_multidataset.sh

echo "[generaltrack] completed missing ua_detrac+vkitti2: $(date '+%F %T')"
