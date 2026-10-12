#!/usr/bin/env bash
# Single-version wrapper: V1 top-score car only.
set -euo pipefail
export ONLY_VERSION="v1_topscore"
exec bash "$(dirname "$0")/run_roi_proto_ablation_6gpu.sh"
