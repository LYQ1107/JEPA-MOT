#!/usr/bin/env bash
# Task1: clear source -> 4 independent single-target TTA
# source: 20e, each target: 10e, eval at epoch 5 and 10 on all 5 domains
set -euo pipefail

SOURCE_DOMAIN="clear"
TARGET_ORDER=("overcast" "rainy" "dawn_dusk" "night")
EVAL_DOMAINS=("clear" "overcast" "rainy" "dawn_dusk" "night")
EVAL_EPOCHS=(5 10)
ADAPT_EPOCHS=10
SOURCE_EPOCHS=20

GPU_IDS=${GPU_IDS:-"9,8,7,6,5,4,3,2,1,0"}
GPU_NUM=${GPU_NUM:-10}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-10}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU:-8}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}

ROOT_WORK_DIR="work_dirs/bdd_4domain/task1_single_target"
SOURCE_CKPT=${SOURCE_CKPT:-""}

SOURCE_CFG="configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear.py"
SOURCE_WORK_DIR="work_dirs/bdd_4domain/source/clear_20e"

get_port() { python3 -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()"; }
log_cmd() {
    echo "[CMD] $*"
    [[ "$DRY_RUN" == "1" ]] && return
    eval "$*"
}

find_source_ckpt() {
    [[ -n "$SOURCE_CKPT" && -f "$SOURCE_CKPT" ]] && echo "$SOURCE_CKPT" && return
    # mmtrack saves into a timestamped subdir; search recursively
    local p
    p=$(find "$SOURCE_WORK_DIR" -name "epoch_${SOURCE_EPOCHS}.pth" 2>/dev/null | head -1)
    [[ -n "$p" ]] && echo "$p" && return
    echo ""
}

if [[ ! -f tools/run/train.py ]]; then echo "ERROR: run from project root" >&2; exit 1; fi

echo "==========================================="
echo "Task1: 4 independent single-target TTA"
echo "==========================================="
echo "GPU_IDS=$GPU_IDS  GPU_NUM=$GPU_NUM  SPG=$SAMPLES_PER_GPU"
echo "SOURCE_WORK_DIR=$SOURCE_WORK_DIR"
echo "ROOT_WORK_DIR=$ROOT_WORK_DIR"
echo "ADAPT_EPOCHS=$ADAPT_EPOCHS  EVAL_EPOCHS=${EVAL_EPOCHS[*]}"
echo ""

mkdir -p "$SOURCE_WORK_DIR" "$ROOT_WORK_DIR"

# ─── Source training ─────────────────────────────────────────────────────────
SOURCE_DONE="$SOURCE_WORK_DIR/train_done.marker"
SOURCE_CKPT_FILE=$(find "$SOURCE_WORK_DIR" -name "epoch_${SOURCE_EPOCHS}.pth" 2>/dev/null | head -1)

if [[ -f "$SOURCE_DONE" && -n "$SOURCE_CKPT_FILE" && "$FORCE" != "1" ]]; then
    echo "SKIP source training (already done, ckpt=$SOURCE_CKPT_FILE)"
else
    echo "--- Source training: clear ${SOURCE_EPOCHS}e ---"
    PORT=$(get_port)
    log_cmd "CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch \
        --nproc_per_node=${GPU_NUM} --master_addr=127.0.0.1 --master_port=${PORT} \
        tools/run/train.py ${SOURCE_CFG} \
        --launcher pytorch --work-dir ${SOURCE_WORK_DIR} --no-version \
        --cfg-options total_epochs=${SOURCE_EPOCHS} runner.max_epochs=${SOURCE_EPOCHS} \
            checkpoint_config.interval=1 checkpoint_config.max_keep_ckpts=-1 \
            evaluation.interval=$((SOURCE_EPOCHS+1)) \
            data.samples_per_gpu=${SAMPLES_PER_GPU} data.workers_per_gpu=${WORKERS_PER_GPU} \
        > ${SOURCE_WORK_DIR}/train.log 2>&1"
    [[ "$DRY_RUN" != "1" ]] && date > "$SOURCE_DONE"
fi

SOURCE_CKPT=$(find_source_ckpt)
if [[ -z "$SOURCE_CKPT" || ! -f "$SOURCE_CKPT" ]]; then
    echo "ERROR: source checkpoint not found" >&2; exit 1
fi
echo "Source ckpt: $SOURCE_CKPT"
echo ""

# ─── Per-target TTA + eval ────────────────────────────────────────────────────
for tgt in "${TARGET_ORDER[@]}"; do
    ADAPT_CFG="configs/adapters/darth/bdd_7domain/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_${tgt}.py"
    TTA_DIR="$ROOT_WORK_DIR/tta_clear_to_${tgt}"
    TTA_DONE="$TTA_DIR/train_done.marker"
    mkdir -p "$TTA_DIR"

    if [[ -f "$TTA_DONE" && "$FORCE" != "1" ]]; then
        echo "SKIP TTA clear→${tgt} (already done)"
    else
        echo "--- TTA: clear → ${tgt} (${ADAPT_EPOCHS}e) ---"
        PORT=$(get_port)
        log_cmd "CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch \
            --nproc_per_node=${GPU_NUM} --master_addr=127.0.0.1 --master_port=${PORT} \
            tools/run/train.py ${ADAPT_CFG} \
            --launcher pytorch --work-dir ${TTA_DIR} --no-version \
            --cfg-options total_epochs=${ADAPT_EPOCHS} runner.max_epochs=${ADAPT_EPOCHS} \
                checkpoint_config.interval=1 checkpoint_config.max_keep_ckpts=-1 \
                evaluation.interval=$((ADAPT_EPOCHS+1)) \
                find_unused_parameters=True \
                model.init_cfg.checkpoint=${SOURCE_CKPT} resume_from=None \
                data.samples_per_gpu=${SAMPLES_PER_GPU} data.workers_per_gpu=${WORKERS_PER_GPU} \
            > ${TTA_DIR}/train.log 2>&1"
        [[ "$DRY_RUN" != "1" ]] && date > "$TTA_DONE"
    fi

    # Eval at epoch 5 and 10
    # Use the original student checkpoint saved by mmtrack (consistent with DARTH source)
    ADAPT_SUBDIR=$(basename "${ADAPT_CFG}" .py)
    for ep in "${EVAL_EPOCHS[@]}"; do
        CKPT_EP="$TTA_DIR/${ADAPT_SUBDIR}/epoch_${ep}.pth"
        if [[ ! -f "$CKPT_EP" && "$DRY_RUN" != "1" ]]; then
            echo "  WARNING: ${ADAPT_SUBDIR}/epoch_${ep}.pth not found under $TTA_DIR, skip"
            continue
        fi
        for dom in "${EVAL_DOMAINS[@]}"; do
            EVAL_CFG="configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_${dom}.py"
            EVAL_DIR="$ROOT_WORK_DIR/eval/tta_clear_to_${tgt}/epoch_${ep}/${dom}"
            EVAL_DONE="$EVAL_DIR/eval.done"
            mkdir -p "$EVAL_DIR"
            if [[ -f "$EVAL_DONE" && "$FORCE" != "1" ]]; then
                echo "  SKIP eval ${tgt}/ep${ep}/${dom}"
                continue
            fi
            echo "  Eval: clear→${tgt} ep${ep} on ${dom}"
            PORT=$(get_port)
            log_cmd "CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch \
                --nproc_per_node=${GPU_NUM} --master_addr=127.0.0.1 --master_port=${PORT} \
                tools/run/test.py ${EVAL_CFG} \
                --checkpoint ${CKPT_EP} --launcher pytorch \
                --work-dir ${EVAL_DIR} --eval bbox track --fuse-conv-bn \
                --cfg-options data.workers_per_gpu=${TEST_WORKERS_PER_GPU} cudnn_benchmark=True \
                > ${EVAL_DIR}/eval.log 2>&1"
            [[ "$DRY_RUN" != "1" ]] && date > "$EVAL_DONE"
        done
    done
    echo ""
done

echo "==========================================="
echo "Task1 complete. Results: $ROOT_WORK_DIR/eval/"
echo "==========================================="
