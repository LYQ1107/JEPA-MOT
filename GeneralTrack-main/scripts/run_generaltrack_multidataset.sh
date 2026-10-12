#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/damot_generaltrack_datasets.yaml}"
DATASETS="${DATASETS:-motsynth ua_detrac vkitti2}"
DOMAINS="${DOMAINS:-all}"
MAX_VIDEOS="${MAX_VIDEOS:-}"
DET_SOURCE="${DET_SOURCE:-source_detector}"
DROP_MISSING_IMAGES="${DROP_MISSING_IMAGES:-0}"
KEEP_DET_MAPPED_FRAMES_ONLY="${KEEP_DET_MAPPED_FRAMES_ONLY:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/generaltrack}"
PREPARED_ROOT="${PREPARED_ROOT:-datasets/generaltrack_prepared}"
INPUT_CHECK="${INPUT_CHECK:-outputs/generaltrack_input_check.csv}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_ROOT}/generaltrack_multidataset_summary.csv}"

mkdir -p "$OUTPUT_ROOT" "$(dirname "$INPUT_CHECK")" "$(dirname "$SUMMARY_CSV")"

"$PYTHON_BIN" tools/inspect_generaltrack_inputs.py \
  --config "$CONFIG" \
  --datasets "$DATASETS" \
  --domains "$DOMAINS" \
  --det-source "$DET_SOURCE" \
  --output "$INPUT_CHECK"

domain_list() {
  local dataset="$1"
  "$PYTHON_BIN" - "$CONFIG" "$dataset" "$DOMAINS" <<'PY'
import json
import sys
from pathlib import Path

config_path, dataset, domains = sys.argv[1:4]
try:
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
except Exception:
    cfg = json.loads(Path(config_path).read_text())
targets = cfg["datasets"][dataset]["targets"]
if domains == "all":
    print(" ".join(targets))
else:
    requested = [d for d in domains.replace(",", " ").split() if d]
    missing = [d for d in requested if d not in targets]
    if missing:
        raise SystemExit("Unknown domains for {}: {}".format(dataset, ", ".join(missing)))
    print(" ".join(requested))
PY
}

run_domain() {
    local dataset="$1"
    local domain="$2"
    local gpu="${3:-}"
    prepared_dir="${PREPARED_ROOT}/${dataset}/${domain}"
    output_dir="${OUTPUT_ROOT}/${dataset}/${domain}"
    convert_args=(
      "$PYTHON_BIN" tools/convert_cocomot_to_generaltrack.py
      --config "$CONFIG"
      --dataset "$dataset"
      --domain "$domain"
      --det-source "$DET_SOURCE"
      --output-dir "$prepared_dir"
    )
    if [[ -n "$MAX_VIDEOS" ]]; then
      convert_args+=(--max-videos "$MAX_VIDEOS")
    fi
    if [[ "$DROP_MISSING_IMAGES" == "1" ]]; then
      convert_args+=(--drop-missing-images)
    fi
    if [[ "$KEEP_DET_MAPPED_FRAMES_ONLY" == "1" ]]; then
      convert_args+=(--keep-det-mapped-frames-only)
    fi
    "${convert_args[@]}"

    track_args=(
      "$PYTHON_BIN" tools/track_damot_domain.py
      --dataset "$dataset"
      --domain "$domain"
      --prepared-dir "$prepared_dir"
      --output-dir "$output_dir"
    )
    if [[ -n "$MAX_VIDEOS" ]]; then
      track_args+=(--max-videos "$MAX_VIDEOS")
    fi
    if [[ -n "$gpu" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" "${track_args[@]}"
    else
      "${track_args[@]}"
    fi

    "$PYTHON_BIN" tools/eval_damot_generaltrack.py \
      --config "$CONFIG" \
      --dataset "$dataset" \
      --domain "$domain" \
      --prepared-dir "$prepared_dir" \
      --track-results "$output_dir/track_results" \
      --output-dir "$output_dir" \
      --det-source "$DET_SOURCE" \
      --summary-csv "$SUMMARY_CSV"
}

gpu_array=($GPU_IDS)
task_index=0
for dataset in $DATASETS; do
  for domain in $(domain_list "$dataset"); do
    if [[ ${#gpu_array[@]} -gt 0 ]]; then
      gpu="${gpu_array[$((task_index % ${#gpu_array[@]}))]}"
      run_domain "$dataset" "$domain" "$gpu" &
      task_index=$((task_index + 1))
      while [[ $(jobs -rp | wc -l) -ge "$MAX_PARALLEL" ]]; do
        wait -n
      done
    else
      run_domain "$dataset" "$domain"
    fi
  done
done
wait

echo "Summary: $SUMMARY_CSV"
