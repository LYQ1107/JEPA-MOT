#!/usr/bin/env bash
set -eo pipefail
PROJECT_DIR="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main"
cd "$PROJECT_DIR"
set +u
eval "$(conda shell.bash hook)"
conda activate DARTH
set -u
mkdir -p work_dirs/bdd_7domain
LOG="work_dirs/bdd_7domain/status_check_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== DARTH 7-domain status check ==="
date
echo "PROJECT_DIR=$PROJECT_DIR"
echo "python=$(which python)"
python - <<'PY'
import torch
print('torch', torch.__version__)
try:
 import mmcv; print('mmcv', mmcv.__version__)
except Exception as e: print('mmcv ERR', e)
try:
 import mmdet; print('mmdet', mmdet.__version__)
except Exception as e: print('mmdet ERR', e)
try:
 import mmtrack; print('mmtrack', mmtrack.__version__)
except Exception as e: print('mmtrack ERR', e)
PY

echo "=== nvidia-smi ==="
nvidia-smi || true

echo "=== Related processes before cleanup ==="
PATTERN="tools/run/train.py|tools/run/test.py|torch.distributed.launch|torch.distributed.run|darth_qdtrack|qdtrack|bdd_7domain|bdd_clear|bdd_night"
ps -fu "$USER" | grep -E "$PATTERN" | grep -v grep | grep -v "check_and_clean_darth_7domain" | grep -v "tee -a" || true

echo "=== Domain directories ==="
declare -A DOMAINS=(
 [clear]="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_clear"
 [dawn_dusk]="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_dawn_dusk"
 [night]="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_night"
 [partly_cloudy]="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_partly_cloudy"
 [overcast]="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_overcast"
 [rainy]="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_rainy"
 [snowy]="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_snowy"
)
for d in clear dawn_dusk night partly_cloudy overcast rainy snowy; do
  p="${DOMAINS[$d]}"
  if [ -d "$p" ]; then videos=$(python - <<PY
from pathlib import Path
p=Path('$p')
print(sum(1 for x in p.iterdir() if x.is_dir()))
PY
); echo "OK $d $p videos=$videos"; else echo "MISSING $d $p"; fi
done

echo "=== Annotation dirs ==="
for p in "/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/bdd/annotations/box_track_20/train" "/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/bdd/annotations/box_track_20/val"; do
  if [ -d "$p" ]; then echo "OK $p files=$(find "$p" -maxdepth 1 -name '*.json' | wc -l)"; else echo "MISSING $p"; fi
done

echo "=== Conversion scripts ==="
for p in tools/convert_datasets/bdd100k/convert_bdd_weather_to_cocovid.py tools/convert_datasets/bdd100k/check_cocovid_json.py; do
  [ -f "$p" ] && echo "OK $p" || echo "MISSING $p"
done

echo "=== Existing key json/checkpoint ==="
for p in data/bdd_da/annotations/box_track_20/box_track_clear_train_cocofmt.json data/bdd_da/annotations/box_track_20/box_track_night_val_cocofmt.json work_dirs/bdd_clear/qdtrack_clear_12e/qdtrack_clear_12e/epoch_12.pth; do
  [ -e "$p" ] && echo "OK $p" || echo "MISSING $p"
done

echo "=== Optional cleanup ==="
if [ "${CLEAN_OLD_PROCS:-0}" = "1" ]; then
  PIDS=$(ps -fu "$USER" | grep -E "$PATTERN" | grep -v grep | grep -v "check_and_clean_darth_7domain" | grep -v "tee -a" | awk '{print $2}' || true)
  if [ -n "$PIDS" ]; then
    echo "Killing PIDs: $PIDS"
    kill -TERM $PIDS || true
    sleep 10
    for pid in $PIDS; do kill -0 "$pid" 2>/dev/null && kill -9 "$pid" || true; done
  else
    echo "No processes to kill."
  fi
else
  echo "CLEAN_OLD_PROCS not set to 1; no kill performed."
fi

echo "=== Done. Log: $LOG ==="
