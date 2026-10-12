#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHON_BIN="${PYTHON_BIN:-/home/lwr/anaconda3/envs/ByteTrack/bin/python}"
export GPU_IDS="${GPU_IDS:-9 8 7 6}"
export MAX_PARALLEL="${MAX_PARALLEL:-4}"
export DROP_MISSING_IMAGES="${DROP_MISSING_IMAGES:-1}"

echo "[generaltrack] start remaining run: $(date '+%F %T')"
echo "[generaltrack] python: $PYTHON_BIN"
echo "[generaltrack] gpu ids: $GPU_IDS"
echo "[generaltrack] max parallel: $MAX_PARALLEL"
echo "[generaltrack] drop missing images: $DROP_MISSING_IMAGES"

echo "[generaltrack] group 1/3: motsynth cloudy dark fog rain snow"
DATASETS="motsynth" DOMAINS="cloudy dark fog rain snow" bash scripts/run_generaltrack_multidataset.sh

echo "[generaltrack] group 2/3: ua_detrac all"
DATASETS="ua_detrac" DOMAINS="all" bash scripts/run_generaltrack_multidataset.sh

echo "[generaltrack] group 3/3: vkitti2 all"
DATASETS="vkitti2" DOMAINS="all" bash scripts/run_generaltrack_multidataset.sh

echo "[generaltrack] completed remaining run: $(date '+%F %T')"
echo "[generaltrack] summary: outputs/generaltrack/generaltrack_multidataset_summary.csv"
