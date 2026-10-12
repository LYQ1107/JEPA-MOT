#!/usr/bin/env bash
# Single-version wrapper: V2 conservative car only.
set -euo pipefail
export ONLY_VERSION="v2_conservative"
exec bash "$(dirname "$0")/run_roi_proto_ablation_6gpu.sh"
