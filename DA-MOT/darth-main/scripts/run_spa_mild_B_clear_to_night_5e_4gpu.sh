#!/usr/bin/env bash
# Scheme 2B: SPA-Mild-B, clear->night, 5e.
set -euo pipefail
cd "$(dirname "$0")/.."
export EXP_NAME="spa_mild_B_clear_to_night_5e"
export CFG="configs/adapters/darth/bdd_4domain/spa_fix/darth_spa_mild_B_clear_to_night_5e.py"
export TARGET_ORDER_STR="night"
source scripts/spa_fix_common.sh
spa_fix_run
