#!/usr/bin/env bash
# Run source-detector detections on all target domains.
# Usage:
#   bash scripts/run_source_detector_targets.sh
#   CUDA_VISIBLE_DEVICES=6,7,8,9 MAX_VIDEOS=1 DATASETS="motsynth" DOMAINS="cloudy" bash scripts/run_source_detector_targets.sh
set -euo pipefail

DAMOT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DAMOT_ROOT"

PYTHON="${PYTHON:-/home/lwr/anaconda3/envs/DARTH/bin/python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7,8,9}"
DATASETS="${DATASETS:-all}"
DOMAINS="${DOMAINS:-all}"
MAX_VIDEOS="${MAX_VIDEOS:-}"
SCORE_THR="${SCORE_THR:-0.001}"
GPUS="${GPUS:-4}"
LAUNCHER="${LAUNCHER:-pytorch}"
GENERALTRACK_ROOT="${GENERALTRACK_ROOT:-/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/source_detector}"
MASTER_PORT="${MASTER_PORT:-29510}"
WORKERS="${WORKERS:-4}"
ALLOW_ADAPTED="${ALLOW_ADAPTED:-}"

AUDIT_MD="${OUTPUT_ROOT}/source_ckpt_target_audit.md"
EVAL_PLAN="${OUTPUT_ROOT}/source_detector_eval_plan.yaml"

mkdir -p "$OUTPUT_ROOT"

echo "============================================================"
echo "  DAMOT Source Detector Pipeline"
echo "  CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "  DATASETS  = $DATASETS"
echo "  DOMAINS   = $DOMAINS"
echo "  MAX_VIDEOS= ${MAX_VIDEOS:-all}"
echo "  SCORE_THR = $SCORE_THR"
echo "  GPUS      = $GPUS"
echo "  LAUNCHER  = $LAUNCHER"
echo "============================================================"

# ── Step 1: audit ────────────────────────────────────────────────────────────
echo ""
echo "[1/2] Running checkpoint & annotation audit..."
"$PYTHON" tools/damot_audit_source_ckpts_and_targets.py \
    --damot-root "$DAMOT_ROOT" \
    --out        "$AUDIT_MD" \
    --config-out "$EVAL_PLAN"

echo ""
echo "Audit report : $AUDIT_MD"
echo "Eval plan    : $EVAL_PLAN"
echo "Cat mapping  : ${OUTPUT_ROOT}/category_mapping_audit.md"

# ── Step 2: run detections ───────────────────────────────────────────────────
echo ""
echo "[2/2] Running source-detector inference on target domains..."

MAX_VIDEOS_ARG=""
if [ -n "$MAX_VIDEOS" ]; then
    MAX_VIDEOS_ARG="--max-videos $MAX_VIDEOS"
fi

ALLOW_ADAPTED_ARG=""
if [ -n "$ALLOW_ADAPTED" ]; then
    ALLOW_ADAPTED_ARG="--allow-adapted-ckpt"
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
"$PYTHON" tools/run_source_detector_on_targets.py \
    --plan       "$EVAL_PLAN" \
    --dataset    "$DATASETS" \
    --domains    "$DOMAINS" \
    --output-root           "$OUTPUT_ROOT" \
    --generaltrack-output-root "$GENERALTRACK_ROOT/outputs/source_detector" \
    --score-thr  "$SCORE_THR" \
    --gpus       "$GPUS" \
    --launcher   "$LAUNCHER" \
    --master-port "$MASTER_PORT" \
    --workers    "$WORKERS" \
    $MAX_VIDEOS_ARG \
    $ALLOW_ADAPTED_ARG

echo ""
echo "============================================================"
echo "  ALL DONE"
echo "  Summary CSV: ${OUTPUT_ROOT}/source_detector_detection_summary.csv"
echo "  GeneralTrack dets: ${GENERALTRACK_ROOT}/outputs/source_detector/"
echo "============================================================"
