#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$task_root/runtime/site-packages:$task_root:$task_root/GeneralTrack-main:$task_root/GeneralTrack-main/tools:$task_root/JEPA${PYTHONPATH:+:$PYTHONPATH}"
exec "${JEPA_PYTHON:-/home/liuyeqiang/anaconda3/envs/GMT/bin/python}" "$@"
