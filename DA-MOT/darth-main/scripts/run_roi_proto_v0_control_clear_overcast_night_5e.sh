#!/usr/bin/env bash
# Single-version wrapper: V0 control only.
set -euo pipefail
export ONLY_VERSION="v0_control"
exec bash "$(dirname "$0")/run_roi_proto_ablation_6gpu.sh"
