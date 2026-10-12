#!/usr/bin/env bash
# Scheme 4: Detection-only SPA, clear->night, 5e.
set -euo pipefail
cd "$(dirname "$0")/.."
export EXP_NAME="spa_det_only_clear_to_night_5e"
export CFG="configs/adapters/darth/bdd_4domain/spa_fix/darth_spa_det_only_clear_to_night_5e.py"
export TARGET_ORDER_STR="night"
source scripts/spa_fix_common.sh
spa_fix_run
