#!/usr/bin/env bash
# Shared core runner for SPA-fix single-experiment train+eval.
# Sourced/invoked by the thin per-scheme wrappers. Handles one continuous
# DARTH episode over a given target order, then evals each target at epoch5.
#
# Required env before calling spa_fix_run():
#   EXP_NAME    e.g. spa_mild_A_clear_to_night_5e
#   CFG         path to the adapter config
#   TARGET_ORDER_STR  space-separated targets, e.g. "night" or "overcast rainy dawn_dusk night"
# Optional env: GPU_IDS GPU_NUM SAMPLES_PER_GPU WORKERS_PER_GPU
#               TEST_WORKERS_PER_GPU DRY_RUN FORCE PYTHON
set -euo pipefail

GPU_IDS=${GPU_IDS:-"9,8,7,6"}
GPU_NUM=${GPU_NUM:-4}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-16}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU:-6}
DRY_RUN=${DRY_RUN:-0}
FORCE=${FORCE:-0}
PYTHON=${PYTHON:-/home/lwr/anaconda3/envs/DARTH/bin/python}

SOURCE_WORK_DIR="work_dirs/bdd_4domain/source/clear_20e"
SOURCE_CKPT=${SOURCE_CKPT:-""}
ECFG_DIR="configs/mot/qdtrack/bdd_7domain"

get_port(){ "$PYTHON" -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()"; }

find_source_ckpt(){
    if [[ -n "$SOURCE_CKPT" && -f "$SOURCE_CKPT" ]]; then echo "$SOURCE_CKPT"; return; fi
    local p
    p=$(find "$SOURCE_WORK_DIR" -name "latest.pth" 2>/dev/null | head -1)
    [[ -z "$p" ]] && p=$(find "$SOURCE_WORK_DIR" -name "epoch_20.pth" 2>/dev/null | head -1)
    echo "$p"
}

spa_fix_run(){
    [[ -f tools/run/train_seq3_episode.py ]] || { echo "ERROR: run from project root" >&2; exit 1; }
    [[ -f "$CFG" ]] || { echo "ERROR: config not found: $CFG" >&2; exit 1; }
    SOURCE_CKPT=$(find_source_ckpt)
    [[ -n "$SOURCE_CKPT" && -f "$SOURCE_CKPT" ]] || { echo "ERROR: clear source ckpt not found under $SOURCE_WORK_DIR" >&2; exit 1; }

    local ROOT="work_dirs/bdd_4domain/spa_fix/${EXP_NAME}"
    local EPISODE="$ROOT/episode"
    local TRAIN_MARKER="$ROOT/train_done.marker"
    read -r -a TARGETS <<< "$TARGET_ORDER_STR"

    echo "==========================================================="
    echo "SPA-FIX EXPERIMENT: ${EXP_NAME}"
    echo "  config        : $CFG"
    echo "  source_ckpt   : $SOURCE_CKPT"
    echo "  target_order  : ${TARGETS[*]}"
    echo "  epochs/stage  : 5   eval each target @ epoch 5"
    echo "  GPU_IDS       : $GPU_IDS (GPU_NUM=$GPU_NUM)  SAMPLES/GPU=$SAMPLES_PER_GPU"
    echo "  test workers  : $TEST_WORKERS_PER_GPU (--fuse-conv-bn cudnn_benchmark=True)"
    echo "  work_dir      : $ROOT"
    echo "  DRY_RUN=$DRY_RUN FORCE=$FORCE"
    echo "==========================================================="

    # ---- train ----
    if [[ -f "$TRAIN_MARKER" && "$FORCE" != "1" ]]; then
        echo "SKIP train (marker exists): $TRAIN_MARKER"
    else
        [[ "$DRY_RUN" != "1" ]] && { mkdir -p "$ROOT/logs" "$EPISODE"; cp "$CFG" "$ROOT/$(basename "$CFG")"; }
        local port; port=$(get_port)
        local cmd="CUDA_VISIBLE_DEVICES=${GPU_IDS} $PYTHON -m torch.distributed.launch \
            --nproc_per_node=${GPU_NUM} --master_addr=127.0.0.1 --master_port=${port} \
            tools/run/train_seq3_episode.py \
            --base-config ${CFG} --source-ckpt ${SOURCE_CKPT} \
            --target-order ${TARGETS[*]} --eval-domains ${TARGETS[*]} \
            --work-dir ${EPISODE} --epochs-per-stage 5 \
            --samples-per-gpu ${SAMPLES_PER_GPU} --workers-per-gpu ${WORKERS_PER_GPU} \
            --launcher pytorch --no-validate > ${ROOT}/logs/train_episode.log 2>&1"
        echo "[CMD] $cmd"
        if [[ "$DRY_RUN" != "1" ]]; then
            eval "$cmd" && date > "$TRAIN_MARKER" || { echo "TRAIN FAILED: $EXP_NAME" >&2; exit 1; }
        fi
    fi

    # ---- eval each target at its own stage, epoch 5 ----
    local idx=0
    for dom in "${TARGETS[@]}"; do
        idx=$((idx+1))
        local STAGE; STAGE=$(printf "stage%02d_after_%s" "$idx" "$dom")
        local CKPT="$EPISODE/adapt/${STAGE}/epoch_5.pth"
        local EDIR="$ROOT/eval/${STAGE}/epoch_5/${dom}"
        local ECFG="$ECFG_DIR/qdtrack_faster-rcnn_r50_fpn_20e_bdd_${dom}.py"
        local EVAL_MARKER="$EDIR/eval_done.marker"
        if [[ -f "$EVAL_MARKER" && "$FORCE" != "1" ]]; then echo "SKIP eval $STAGE/$dom (marker)"; continue; fi
        [[ "$DRY_RUN" != "1" ]] && mkdir -p "$EDIR"
        if [[ ! -f "$CKPT" && "$DRY_RUN" != "1" ]]; then echo "ERROR: ckpt not found: $CKPT" >&2; exit 1; fi
        local port; port=$(get_port)
        local cmd="CUDA_VISIBLE_DEVICES=${GPU_IDS} $PYTHON -m torch.distributed.launch \
            --nproc_per_node=${GPU_NUM} --master_addr=127.0.0.1 --master_port=${port} \
            tools/run/test.py ${ECFG} --checkpoint ${CKPT} --launcher pytorch \
            --work-dir ${EDIR} --eval bbox track --fuse-conv-bn \
            --cfg-options data.workers_per_gpu=${TEST_WORKERS_PER_GPU} cudnn_benchmark=True \
            > ${EDIR}/eval.log 2>&1"
        echo "[CMD] $cmd"
        if [[ "$DRY_RUN" != "1" ]]; then
            eval "$cmd" && date > "$EVAL_MARKER" || { echo "EVAL FAILED: $STAGE/$dom" >&2; exit 1; }
        fi
    done
    echo ""
    echo "Done: $EXP_NAME -> eval dirs under $ROOT/eval/stage*/epoch_5/<domain>/"
}
