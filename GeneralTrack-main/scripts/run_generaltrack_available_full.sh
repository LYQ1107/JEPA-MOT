#!/usr/bin/env bash
set -euo pipefail

export PYTHON_BIN="${PYTHON_BIN:-/home/lwr/anaconda3/envs/ByteTrack/bin/python}"
export GPU_IDS="${GPU_IDS:-9 8 7 6}"
export MAX_PARALLEL="${MAX_PARALLEL:-4}"
export DET_SOURCE="${DET_SOURCE:-source_detector}"
export DROP_MISSING_IMAGES="${DROP_MISSING_IMAGES:-1}"

run_group() {
  local datasets="$1"
  local domains="$2"
  DATASETS="$datasets" DOMAINS="$domains" bash scripts/run_generaltrack_multidataset.sh
}

run_group "bdd100k" "all"
run_group "motsynth" "cloudy dark fog rain snow"
run_group "ua_detrac" "all"
run_group "vkitti2" "all"

echo "Available-domain full run complete."
echo "Skipped: motsynth/indoor has no source_detector detections_coco.json."
echo "Summary: outputs/generaltrack/generaltrack_multidataset_summary.csv"
