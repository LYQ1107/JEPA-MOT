#!/bin/bash
# DARTH Baseline: BDD100K Clear -> Night Domain Adaptation
# This script runs through the complete DARTH baseline pipeline

set -e

DARTH_ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main"
PYTHON="/home/lwr/anaconda3/envs/DARTH/bin/python"
GPU_ID="${GPU_ID:-0}"

cd ${DARTH_ROOT}

echo "========================================"
echo "DARTH Baseline: BDD100K Clear -> Night"
echo "========================================"
echo "DARTH root: ${DARTH_ROOT}"
echo "Python: ${PYTHON}"
echo "GPU: ${GPU_ID}"
echo ""

# Step 1: Convert clear domain
echo "[Step 1/8] Converting clear domain to CocoVID format..."
${PYTHON} tools/convert_datasets/bdd100k/convert_bdd_weather_to_cocovid.py \
  --domain-name clear \
  --image-root /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_clear \
  --ann-train-dir /data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/bdd/annotations/box_track_20/train \
  --ann-val-dir /data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/bdd/annotations/box_track_20/val \
  --out-dir data/bdd_da/annotations/box_track_20 \
  --classes pedestrian car truck bus motorcycle bicycle
echo ""

# Step 2: Convert night domain
echo "[Step 2/8] Converting night domain to CocoVID format..."
${PYTHON} tools/convert_datasets/bdd100k/convert_bdd_weather_to_cocovid.py \
  --domain-name night \
  --image-root /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_night \
  --ann-train-dir /data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/bdd/annotations/box_track_20/train \
  --ann-val-dir /data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/bdd/annotations/box_track_20/val \
  --out-dir data/bdd_da/annotations/box_track_20 \
  --classes pedestrian car truck bus motorcycle bicycle
echo ""

# Step 3: Validate clear JSON
echo "[Step 3/8] Validating clear CocoVID JSON..."
${PYTHON} tools/convert_datasets/bdd100k/check_cocovid_json.py \
  --json-file data/bdd_da/annotations/box_track_20/box_track_clear_val_cocofmt.json \
  --img-prefix /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_clear \
  --vis-dir work_dirs/debug_bdd_da_vis/clear_val \
  --num-vis 5
echo ""

# Step 4: Validate night JSON
echo "[Step 4/8] Validating night CocoVID JSON..."
${PYTHON} tools/convert_datasets/bdd100k/check_cocovid_json.py \
  --json-file data/bdd_da/annotations/box_track_20/box_track_night_val_cocofmt.json \
  --img-prefix /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_night \
  --vis-dir work_dirs/debug_bdd_da_vis/night_val \
  --num-vis 5
echo ""

# Step 5: Source QDTrack training (debug mode with 1 epoch)
echo "[Step 5/8] Training QDTrack on clear domain (debug: 1 epoch)..."
CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} tools/run/train.py \
  configs/mot/qdtrack/qdtrack_faster-rcnn_r50_fpn_12e_bdd_clear.py \
  --work-dir work_dirs/bdd_clear/qdtrack_clear_debug \
  --exp-name qdtrack_clear_debug \
  --no-version \
  --cfg-options data.samples_per_gpu=1 data.workers_per_gpu=2 total_epochs=1
echo ""

# Step 6: Source-only evaluation on night (using the debug checkpoint)
echo "[Step 6/8] Evaluating source-only model on night domain..."
CHECKPOINT="work_dirs/bdd_clear/qdtrack_clear_debug/latest.pth"
if [ -f "${CHECKPOINT}" ]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} tools/run/test.py \
    configs/mot/qdtrack/qdtrack_faster-rcnn_r50_fpn_12e_bdd_clear.py \
    --checkpoint ${CHECKPOINT} \
    --work-dir work_dirs/bdd_night/source_only_from_clear \
    --exp-name source_only_night_eval \
    --eval bbox track \
    --cfg-options \
      data.test.ann_file=data/bdd_da/annotations/box_track_20/box_track_night_val_cocofmt.json \
      data.test.img_prefix=/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_night/
else
  echo "WARNING: Checkpoint not found at ${CHECKPOINT}, skipping source-only eval"
fi
echo ""

# Step 7: DARTH adaptation on night domain (debug: 1 epoch)
echo "[Step 7/8] Running DARTH adaptation on night domain (debug: 1 epoch)..."
CHECKPOINT="work_dirs/bdd_clear/qdtrack_clear_debug/latest.pth"
if [ -f "${CHECKPOINT}" ]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} tools/run/train.py \
    configs/adapters/darth/darth_qdtrack_faster-rcnn_r50_fpn_10e_bdd_night_from_clear.py \
    --work-dir work_dirs/bdd_night/darth_from_clear_debug \
    --exp-name darth_night_from_clear_debug \
    --no-version \
    --cfg-options \
      model.init_cfg.checkpoint=${CHECKPOINT} \
      find_unused_parameters=True \
      data.samples_per_gpu=1 \
      data.workers_per_gpu=2 \
      evaluation.interval=1 \
      optimizer.lr=0.002 \
      optimizer_config.grad_clip.max_norm=35.0 \
      model.loss_rpn_distillation.loss_weight=0.1 \
      model.loss_roi_distillation.loss_weight=0.1 \
      custom_hooks.0.momentum=0.002 \
      model.conf_thr=0.7 \
      total_epochs=1
else
  echo "WARNING: Checkpoint not found at ${CHECKPOINT}, skipping DARTH adaptation"
fi
echo ""

# Step 8: Evaluate adapted model on night
echo "[Step 8/8] Evaluating DARTH-adapted model on night domain..."
CHECKPOINT="work_dirs/bdd_night/darth_from_clear_debug/latest.pth"
if [ -f "${CHECKPOINT}" ]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} tools/run/test.py \
    configs/adapters/darth/darth_qdtrack_faster-rcnn_r50_fpn_10e_bdd_night_from_clear.py \
    --checkpoint ${CHECKPOINT} \
    --work-dir work_dirs/bdd_night/darth_adapted_eval \
    --exp-name darth_adapted_night_eval \
    --eval bbox track
else
  echo "WARNING: Checkpoint not found at ${CHECKPOINT}, skipping adapted eval"
fi
echo ""

echo "========================================"
echo "DARTH Baseline Pipeline Complete!"
echo "========================================"
echo ""
echo "Results are saved in:"
echo "  - Source training: work_dirs/bdd_clear/qdtrack_clear_debug/"
echo "  - Source-only eval: work_dirs/bdd_night/source_only_from_clear/"
echo "  - DARTH adaptation: work_dirs/bdd_night/darth_from_clear_debug/"
echo "  - Adapted eval: work_dirs/bdd_night/darth_adapted_eval/"
