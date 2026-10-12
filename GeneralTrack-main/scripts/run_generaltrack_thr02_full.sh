#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHON_BIN="${PYTHON_BIN:-/home/lwr/anaconda3/envs/ByteTrack/bin/python}"
export GPU_IDS="${GPU_IDS:-0 1}"
export MAX_PARALLEL="${MAX_PARALLEL:-2}"
export DROP_MISSING_IMAGES="${DROP_MISSING_IMAGES:-1}"
export DET_SOURCE="${DET_SOURCE:-source_detector_thr02}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/generaltrack_thr02}"
export PREPARED_ROOT="${PREPARED_ROOT:-datasets/generaltrack_prepared_thr02}"
export INPUT_CHECK="${INPUT_CHECK:-outputs/generaltrack_thr02_input_check.csv}"
export SUMMARY_CSV="${SUMMARY_CSV:-outputs/generaltrack_thr02/generaltrack_thr02_summary.csv}"

echo "[generaltrack-thr02] start full run: $(date '+%F %T')"
echo "[generaltrack-thr02] python: $PYTHON_BIN"
echo "[generaltrack-thr02] gpu ids: $GPU_IDS"
echo "[generaltrack-thr02] max parallel: $MAX_PARALLEL"
echo "[generaltrack-thr02] det source: $DET_SOURCE"
echo "[generaltrack-thr02] output root: $OUTPUT_ROOT"
echo "[generaltrack-thr02] prepared root: $PREPARED_ROOT"

echo "[generaltrack-thr02] group 1/4: bdd100k all"
DATASETS="bdd100k" DOMAINS="all" bash scripts/run_generaltrack_multidataset.sh

echo "[generaltrack-thr02] group 2/4: motsynth cloudy dark fog rain snow"
DATASETS="motsynth" DOMAINS="cloudy dark fog rain snow" bash scripts/run_generaltrack_multidataset.sh

echo "[generaltrack-thr02] group 3/4: ua_detrac all"
DATASETS="ua_detrac" DOMAINS="all" bash scripts/run_generaltrack_multidataset.sh

echo "[generaltrack-thr02] group 4/4: vkitti2 all"
DATASETS="vkitti2" DOMAINS="all" bash scripts/run_generaltrack_multidataset.sh

echo "[generaltrack-thr02] completed full run: $(date '+%F %T')"
echo "[generaltrack-thr02] summary: $SUMMARY_CSV"
echo "[generaltrack-thr02] skipped: motsynth/indoor has no source_detector_thr02 detections_coco.json."
