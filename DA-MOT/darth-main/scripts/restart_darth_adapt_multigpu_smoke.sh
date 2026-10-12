#!/bin/bash
set -eo pipefail

PROJECT_DIR="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main"
cd "$PROJECT_DIR"

# Activate DARTH conda environment (disable nounset for conda scripts)
set +u
eval "$(conda shell.bash hook)"
conda activate DARTH
set -u

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
WORK_DIR="work_dirs/bdd_night/darth_from_clear_multigpu_smoke_${TIMESTAMP}"
LOGFILE="${WORK_DIR}/restart_darth_adapt_multigpu_smoke_${TIMESTAMP}.log"
CONFIG="configs/adapters/darth/darth_qdtrack_faster-rcnn_r50_fpn_10e_bdd_night_from_clear.py"

mkdir -p "$WORK_DIR"

exec > >(tee -a "$LOGFILE") 2>&1

echo "=== DARTH Multi-GPU Smoke Test ==="
echo "Timestamp: $TIMESTAMP"
echo "Work dir: $WORK_DIR"

# --- Check & select source checkpoint ---
CKPT_DIR="work_dirs/bdd_clear/qdtrack_clear_12e/qdtrack_clear_12e"
if [ -f "${CKPT_DIR}/best_mAP_epoch_12.pth" ]; then
    CHECKPOINT="${CKPT_DIR}/best_mAP_epoch_12.pth"
elif [ -f "${CKPT_DIR}/latest.pth" ]; then
    CHECKPOINT="${CKPT_DIR}/latest.pth"
elif [ -f "${CKPT_DIR}/epoch_4.pth" ]; then
    CHECKPOINT="${CKPT_DIR}/epoch_4.pth"
else
    echo "ERROR: No source checkpoint found. Aborting."
    exit 1
fi
CHECKPOINT=$(readlink -f "$CHECKPOINT")
echo "Source checkpoint: $CHECKPOINT"

# --- Check for residual processes ---
echo ""
echo "=== Checking residual processes ==="
MY_PID=$$
MY_PPID=$(ps -o ppid= -p $MY_PID | tr -d ' ')
PROCS=$(ps -fu $USER | grep -E "tools/run/train\.py|tools/run/test\.py|torch\.distributed\.(launch|run)" | grep -v grep | grep -v "$MY_PID" | grep -v "$MY_PPID" || true)
if [ -n "$PROCS" ]; then
    echo "Found residual processes:"
    echo "$PROCS"
    echo "$PROCS" | awk '{print $2}' | xargs -r kill -TERM 2>/dev/null || true
    sleep 5
    echo "Killed residual processes."
else
    echo "No residual processes found."
fi

# --- Select free port ---
PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "Selected port: $PORT"

# --- GPU selection ---
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$PORT
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export MPLBACKEND=Agg

GPUS=2

echo ""
echo "=== Launch Configuration ==="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "MASTER_PORT=$MASTER_PORT"
echo "GPUS=$GPUS"
echo "CONFIG=$CONFIG"
echo "CHECKPOINT=$CHECKPOINT"
echo "WORK_DIR=$WORK_DIR"

# --- Launch training ---
echo ""
echo "=== Starting 2-GPU 1-epoch smoke test ==="

CMD="python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    tools/run/train.py \
    $CONFIG \
    --launcher pytorch \
    --work-dir $WORK_DIR \
    --cfg-options \
        model.init_cfg.checkpoint=$CHECKPOINT \
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
        total_epochs=1"

echo ""
echo "Full command:"
echo "$CMD"
echo ""

eval $CMD

EXIT_CODE=$?
echo ""
echo "=== Training finished with exit code: $EXIT_CODE ==="

# --- Post-training summary ---
echo ""
echo "=== Last 80 lines of training log ==="
tail -80 "$LOGFILE" 2>/dev/null || true

echo ""
echo "=== Generated files in work_dir ==="
find "$WORK_DIR" -name "*.pth" -o -name "*.log" -o -name "*.json" 2>/dev/null | sort

echo ""
echo "=== Smoke test complete ==="
