#!/usr/bin/env bash
set -eo pipefail
PROJECT_DIR="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main"
cd "$PROJECT_DIR"
set +u
eval "$(conda shell.bash hook)"
conda activate DARTH
set -u
mkdir -p work_dirs/bdd_7domain/data_logs data/bdd_da/annotations/box_track_20
LOG="work_dirs/bdd_7domain/data_prepare_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
TRAIN_DIR="/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/bdd/annotations/box_track_20/train"
VAL_DIR="/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/bdd/annotations/box_track_20/val"
OUT_DIR="data/bdd_da/annotations/box_track_20"

declare -A ROOTS=(
 [clear]="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_clear"
 [dawn_dusk]="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_dawn_dusk"
 [night]="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_night"
 [partly_cloudy]="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_partly_cloudy"
 [overcast]="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_overcast"
 [rainy]="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_rainy"
 [snowy]="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_snowy"
)

echo "=== Prepare BDD 7-domain CocoVID ==="
for d in clear dawn_dusk night partly_cloudy overcast rainy snowy; do
  echo "=== Convert $d ==="
  python tools/convert_datasets/bdd100k/convert_bdd_weather_to_cocovid.py \
    --domain-name "$d" --image-root "${ROOTS[$d]}" --ann-train-dir "$TRAIN_DIR" --ann-val-dir "$VAL_DIR" --out-dir "$OUT_DIR"
  for split in train val all; do
    jf="$OUT_DIR/box_track_${d}_${split}_cocofmt.json"
    echo "--- Check $jf ---"
    python tools/convert_datasets/bdd100k/check_cocovid_json.py --json-file "$jf" --img-prefix "${ROOTS[$d]}" --vis-dir "work_dirs/bdd_7domain/data_logs/vis_${d}_${split}" --num-vis 0 || true
  done
  echo "--- Stats $d ---"
  python - <<PY
import json
p='$OUT_DIR/stats_${d}.json'
s=json.load(open(p))
print(json.dumps({k:s.get(k) for k in ['domain_name','num_videos','num_images','num_annotations','num_unique_instances','category_histogram','skipped_invalid_boxes']}, indent=2))
PY
done

echo "=== Data preparation complete. Log: $LOG ==="
