#!/usr/bin/env bash
# Single-version wrapper: V3 diverse car only.
set -euo pipefail
export ONLY_VERSION="v3_diverse"
exec bash "$(dirname "$0")/run_roi_proto_ablation_6gpu.sh"
