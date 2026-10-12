#!/bin/bash
# Baseline finetune for vkitti2 — fog/morning/overcast/rain/sunset, 5e, 4 GPUs
# Uses setsid + process group kill to ensure clean teardown.

set -euo pipefail
cd /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main
[[ -f tools/run/train.py ]] || { echo "ERROR: must run from project root" >&2; exit 1; }

PIDFILE="/tmp/vkitti2_baseline_finetune.pid"
if [[ -f "$PIDFILE" ]]; then
    OLD_PID=$(cat "$PIDFILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "ERROR: already running (PID $OLD_PID)" >&2; exit 1
    fi
fi
echo $$ > "$PIDFILE"
cleanup() { rm -f "$PIDFILE"; }
trap cleanup EXIT

export CUDA_VISIBLE_DEVICES=2,3,4,5
PYTHON="/home/lwr/anaconda3/envs/DARTH/bin/python"
TORCHRUN="/home/lwr/anaconda3/envs/DARTH/bin/torchrun"
CFGD="configs/mot/qdtrack/vkitti2/baseline_finetune"
EVAL_CFGD="configs/mot/qdtrack/vkitti2"
TRAIN="tools/run/train.py"
TEST="tools/run/test.py"
EVAL_OPTS="--eval bbox track --fuse-conv-bn --cfg-options data.workers_per_gpu=4 cudnn_benchmark=True model.tracker.init_score_thr=0.20 model.tracker.obj_score_thr=0.20"
ADAPT_BASE="work_dirs/vkitti2/baseline_single_target/adapt"
TEST_BASE="work_dirs/vkitti2/test/baseline_clone_to"
LOG_BASE="work_dirs/vkitti2/baseline_single_target/logs"
mkdir -p "$LOG_BASE"

get_port() {
    $PYTHON -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}

run_distributed() {
    # run_distributed <log_file> <port> <script> [args...]
    local LOG="$1"; local PORT="$2"; shift 2
    "$TORCHRUN" \
        --nproc_per_node=4 \
        --master_addr=127.0.0.1 \
        --master_port="$PORT" \
        --standalone \
        "$@" >> "$LOG" 2>&1
}

for DOM in fog morning overcast rain sunset; do
    TRAIN_CFG="$CFGD/qdtrack_faster-rcnn_r50_fpn_5e_vkitti2_${DOM}_from_clone.py"
    WORK_DIR="$ADAPT_BASE/clone_to_${DOM}_5e"
    TEST_DIR="$TEST_BASE/${DOM}_5e"
    TRAIN_DONE="$WORK_DIR/train_done.marker"
    EVAL_DONE="$TEST_DIR/eval_done.marker"
    TRAIN_LOG="$LOG_BASE/train_${DOM}.log"
    EVAL_LOG="$LOG_BASE/eval_${DOM}.log"

    echo "=== Domain: $DOM  $(date) ===" | tee -a "$LOG_BASE/run_summary.log"

    # --- Training ---
    if [[ -f "$TRAIN_DONE" ]]; then
        echo "[SKIP] training already done" | tee -a "$LOG_BASE/run_summary.log"
    else
        PORT=$(get_port)
        echo "-- Training: port=$PORT  log=$TRAIN_LOG" | tee -a "$LOG_BASE/run_summary.log"
        run_distributed "$TRAIN_LOG" "$PORT" \
            "$TRAIN" "$TRAIN_CFG" \
            --launcher pytorch \
            --work-dir "$WORK_DIR" \
            --cfg-options data.samples_per_gpu=2 data.workers_per_gpu=2
        date > "$TRAIN_DONE"
        echo "[DONE] training $DOM  $(date)" | tee -a "$LOG_BASE/run_summary.log"
    fi

    CKPT=$(find "$WORK_DIR" -name "epoch_5.pth" 2>/dev/null | sort | tail -1)
    [[ -n "$CKPT" && -f "$CKPT" ]] || { echo "ERROR: epoch_5.pth not found under $WORK_DIR" >&2; exit 1; }
    echo "  ckpt: $CKPT" | tee -a "$LOG_BASE/run_summary.log"

    # --- Evaluation ---
    if [[ -f "$EVAL_DONE" ]]; then
        echo "[SKIP] eval already done" | tee -a "$LOG_BASE/run_summary.log"
        continue
    fi

    mkdir -p "$TEST_DIR"
    PORT=$(get_port)
    echo "-- Eval: port=$PORT  log=$EVAL_LOG" | tee -a "$LOG_BASE/run_summary.log"
    run_distributed "$EVAL_LOG" "$PORT" \
        "$TEST" \
        "$EVAL_CFGD/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_${DOM}_eval.py" \
        --checkpoint "$CKPT" \
        --launcher pytorch \
        --work-dir "$TEST_DIR" \
        $EVAL_OPTS
    date > "$EVAL_DONE"
    echo "[DONE] eval $DOM  $(date)" | tee -a "$LOG_BASE/run_summary.log"
done

echo "===== ALL DONE  $(date) =====" | tee -a "$LOG_BASE/run_summary.log"
