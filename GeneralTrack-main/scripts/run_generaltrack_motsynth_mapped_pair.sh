#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/lwr/anaconda3/envs/ByteTrack/bin/python}"
DOMAINS="${DOMAINS:-cloudy dark fog rain snow}"
export DROP_MISSING_IMAGES=1
export KEEP_DET_MAPPED_FRAMES_ONLY=1
export MAX_PARALLEL="${MAX_PARALLEL:-4}"

run_group() {
  local det_source="$1"
  local gpu_ids="$2"
  local root_name="$3"

  echo "[generaltrack-mapped] start ${det_source}: $(date '+%F %T')"
  PYTHON_BIN="$PYTHON_BIN" \
  DATASETS="motsynth" \
  DOMAINS="$DOMAINS" \
  DET_SOURCE="$det_source" \
  GPU_IDS="$gpu_ids" \
  OUTPUT_ROOT="outputs/generaltrack_mapped/${root_name}" \
  PREPARED_ROOT="datasets/generaltrack_prepared_mapped/${root_name}" \
  INPUT_CHECK="outputs/generaltrack_mapped/${root_name}_input_check.csv" \
  SUMMARY_CSV="outputs/generaltrack_mapped/${root_name}/generaltrack_motsynth_mapped_summary.csv" \
  bash scripts/run_generaltrack_multidataset.sh
  echo "[generaltrack-mapped] completed ${det_source}: $(date '+%F %T')"
}

run_group "source_detector" "${SOURCE_GPU_IDS:-9 8 7 6}" "source_detector" &
source_pid="$!"

run_group "source_detector_thr02" "${THR02_GPU_IDS:-0 1}" "source_detector_thr02" &
thr02_pid="$!"

wait "$source_pid"
wait "$thr02_pid"

echo "[generaltrack-mapped] all mapped MOTSynth runs completed: $(date '+%F %T')"
