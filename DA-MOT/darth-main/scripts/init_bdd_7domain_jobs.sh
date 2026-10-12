#!/usr/bin/env bash
set -eo pipefail
cd /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main
set +u
eval "$(conda shell.bash hook)"
conda activate DARTH
set -u
FORCE=""
if [ "${1:-}" = "--force" ]; then FORCE="--force"; fi
python tools/bdd_7domain_manager.py init $FORCE
