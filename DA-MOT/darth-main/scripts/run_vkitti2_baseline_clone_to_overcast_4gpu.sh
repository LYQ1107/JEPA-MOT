#!/usr/bin/env bash
# Train QDTrack source model on VKITTI2-clone (20e),
# then run DARTH baseline adaptation to VKITTI2-overcast.
# GPUs: 5,4,3,2  (4 cards)

set -euo pipefail
[[ -f tools/run/train.py ]] || { echo "ERROR: run from project root" >&2; exit 1; }

GPU_IDS=${GPU_IDS:-"5,4,3,2"}
GPU_NUM=${GPU_NUM:-4}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-2}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
PYTHON=${PYTHON:-/home/lwr/anaconda3/envs/DARTH/bin/python}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

SOURCE_CFG="configs/mot/qdtrack/vkitti2/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_clone.py"
ADAPT_CFG="configs/adapters/darth/vkitti2/darth_qdtrack_faster-rcnn_r50_fpn_5e_vkitti2_overcast_from_clone.py"

SOURCE_WORK_DIR="work_dirs/vkitti2/source/clone_20e/qdtrack_vkitti2_clone_20e"
ADAPT_WORK_DIR="work_dirs/vkitti2/adapt/darth_baseline_overcast_kdinit_10e"

get_port() {
    $PYTHON -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}

mkdir -p "$SOURCE_WORK_DIR" "$ADAPT_WORK_DIR" \
         "work_dirs/vkitti2/source/clone_20e/logs" \
         "work_dirs/vkitti2/adapt/logs"

echo "================================================================"
echo " STEP 1: Source training — VKITTI2 clone 20e"
echo "  config  : $SOURCE_CFG"
echo "  work_dir: $SOURCE_WORK_DIR"
echo "  GPUs    : $GPU_IDS  (n=$GPU_NUM)"
echo "================================================================"

SOURCE_CKPT="$SOURCE_WORK_DIR/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_clone/$(ls $SOURCE_WORK_DIR/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_clone/ 2>/dev/null | head -1)/epoch_20.pth"
SOURCE_DONE="$SOURCE_WORK_DIR/source_train_done.marker"

if [[ -f "$SOURCE_DONE" ]]; then
    echo "[SKIP] source training already done ($SOURCE_DONE)"
else
    PORT=$(get_port)
    CUDA_VISIBLE_DEVICES=${GPU_IDS} \
        $PYTHON -m torch.distributed.launch \
            --nproc_per_node=${GPU_NUM} \
            --master_addr=${MASTER_ADDR} \
            --master_port=${PORT} \
            tools/run/train.py \
            ${SOURCE_CFG} \
            --launcher pytorch \
            --work-dir ${SOURCE_WORK_DIR} \
            --cfg-options \
                data.samples_per_gpu=${SAMPLES_PER_GPU} \
                data.workers_per_gpu=${WORKERS_PER_GPU} \
        2>&1 | tee work_dirs/vkitti2/source/clone_20e/logs/train.log
    date > "$SOURCE_DONE"
fi

[[ -f "$SOURCE_CKPT" ]] || { echo "ERROR: source ckpt not found: $SOURCE_CKPT" >&2; exit 1; }

echo ""
echo "================================================================"
echo " STEP 2: DARTH baseline adaptation — VKITTI2 overcast (10e)"
echo "  config  : $ADAPT_CFG"
echo "  source  : KDPretrained init_cfg (same as previous BDD DARTH runs)"
echo "  work_dir: $ADAPT_WORK_DIR"
echo "  GPUs    : $GPU_IDS  (n=$GPU_NUM)"
echo "================================================================"

ADAPT_DONE="$ADAPT_WORK_DIR/adapt_done.marker"

if [[ -f "$ADAPT_DONE" ]]; then
    echo "[SKIP] adaptation already done ($ADAPT_DONE)"
else
    PORT=$(get_port)
    CUDA_VISIBLE_DEVICES=${GPU_IDS} \
        $PYTHON -m torch.distributed.launch \
            --nproc_per_node=${GPU_NUM} \
            --master_addr=${MASTER_ADDR} \
            --master_port=${PORT} \
            tools/run/train.py \
            ${ADAPT_CFG} \
            --launcher pytorch \
            --work-dir ${ADAPT_WORK_DIR} \
            --cfg-options \
                data.samples_per_gpu=${SAMPLES_PER_GPU} \
                data.workers_per_gpu=${WORKERS_PER_GPU} \
                total_epochs=10 \
        2>&1 | tee work_dirs/vkitti2/adapt/logs/adapt_overcast.log
    date > "$ADAPT_DONE"
fi

echo ""
echo "[DONE] VKITTI2 baseline pipeline finished."
echo "  Source ckpt : $SOURCE_CKPT"
echo "  Adapt dir   : $ADAPT_WORK_DIR"
