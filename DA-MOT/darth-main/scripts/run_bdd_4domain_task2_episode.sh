#!/usr/bin/env bash
# Task2: clear -> overcast -> rainy -> dawn_dusk -> night (full continuous episode)
# Both student and teacher are carried across stages.
# source: 20e, each target: 10e, eval at epoch 5 and 10 on all 5 domains.
set -euo pipefail

SOURCE_DOMAIN="clear"
TARGET_ORDER=("overcast" "rainy" "dawn_dusk" "night")
EVAL_DOMAINS=("overcast" "rainy" "dawn_dusk" "night")
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

ROOT_WORK_DIR="work_dirs/bdd_4domain/task2_full_episode"
SOURCE_CKPT=${SOURCE_CKPT:-""}

SOURCE_CFG="configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear.py"
SOURCE_WORK_DIR="work_dirs/bdd_4domain/source/clear_20e"
BASE_CFG="configs/adapters/darth/bdd_7domain/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_overcast.py"

get_port() { python3 -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()"; }
log_cmd() {
    echo "[CMD] $*"
    [[ "$DRY_RUN" == "1" ]] && return
    eval "$*"
}

find_source_ckpt() {
    [[ -n "$SOURCE_CKPT" && -f "$SOURCE_CKPT" ]] && echo "$SOURCE_CKPT" && return
    local p
    p=$(find "$SOURCE_WORK_DIR" -name "epoch_${SOURCE_EPOCHS}.pth" 2>/dev/null | head -1)
    [[ -n "$p" ]] && echo "$p" && return
    echo ""
}

if [[ ! -f tools/run/train_seq3_episode.py ]]; then
    echo "ERROR: run from project root" >&2; exit 1
fi

echo "==========================================="
echo "Task2: full continuous episode (student+teacher carried)"
echo "==========================================="
echo "Target order: ${TARGET_ORDER[*]}"
echo "GPU_IDS=$GPU_IDS  GPU_NUM=$GPU_NUM  SPG=$SAMPLES_PER_GPU"
echo "ROOT_WORK_DIR=$ROOT_WORK_DIR"
echo "ADAPT_EPOCHS=$ADAPT_EPOCHS  EVAL_EPOCHS=${EVAL_EPOCHS[*]}"
echo ""

mkdir -p "$SOURCE_WORK_DIR" "$ROOT_WORK_DIR"

# ─── Source training ──────────────────────────────────────────────────────────
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

# ─── Continuous episode training ─────────────────────────────────────────────
EPISODE_DIR="$ROOT_WORK_DIR/episode"
EPISODE_DONE="$EPISODE_DIR/state/train_episode.done"
TRAIN_LOG="$ROOT_WORK_DIR/logs/train_episode.log"
mkdir -p "$ROOT_WORK_DIR/logs"

if [[ -f "$EPISODE_DONE" && "$FORCE" != "1" ]]; then
    echo "SKIP episode training (already done)"
else
    echo "--- Continuous episode: ${TARGET_ORDER[*]} ---"
    PORT=$(get_port)
    log_cmd "CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch \
        --nproc_per_node=${GPU_NUM} --master_addr=127.0.0.1 --master_port=${PORT} \
        tools/run/train_seq3_episode.py \
        --base-config ${BASE_CFG} \
        --source-ckpt ${SOURCE_CKPT} \
        --target-order ${TARGET_ORDER[*]} \
        --eval-domains ${EVAL_DOMAINS[*]} \
        --work-dir ${EPISODE_DIR} \
        --epochs-per-stage ${ADAPT_EPOCHS} \
        --samples-per-gpu ${SAMPLES_PER_GPU} \
        --workers-per-gpu ${WORKERS_PER_GPU} \
        --launcher pytorch --no-validate \
        > ${TRAIN_LOG} 2>&1"
    [[ "$DRY_RUN" != "1" ]] && { mkdir -p "$EPISODE_DIR/state"; date > "$EPISODE_DONE"; }
fi

# ─── Eval at epoch 5 and 10 per stage ────────────────────────────────────────
# Stage snapshots from train_seq3_episode.py are end-of-stage (epoch 10).
# Per-epoch checkpoints are in episode/adapt/stage0N_after_<domain>/epoch_X.pth
echo ""
echo "--- Evaluating all stages at epoch 5 and 10 ---"

STAGE_NAMES=("stage00_source_only" "stage01_after_overcast" "stage02_after_rainy" "stage03_after_dawn_dusk" "stage04_after_night")
# stage00 only has source ckpt; stages 1-4 have per-epoch checkpoints
for stage_idx in "${!STAGE_NAMES[@]}"; do
    stage="${STAGE_NAMES[$stage_idx]}"
    if [[ "$stage_idx" -eq 0 ]]; then
        # source snapshot only
        ep_list=(0)
    else
        ep_list=("${EVAL_EPOCHS[@]}")
    fi

    for ep in "${ep_list[@]}"; do
        if [[ "$stage_idx" -eq 0 ]]; then
            CKPT="$EPISODE_DIR/stage_snapshots/stage00_source_only.pth"
            eval_tag="${stage}"
        else
            tgt="${TARGET_ORDER[$((stage_idx-1))]}"
            CKPT="$EPISODE_DIR/adapt/${stage}/epoch_${ep}.pth"
            eval_tag="${stage}/epoch_${ep}"
        fi

        if [[ ! -f "$CKPT" && "$DRY_RUN" != "1" ]]; then
            echo "  WARNING: checkpoint not found: $CKPT"
            continue
        fi

        for dom in "${EVAL_DOMAINS[@]}"; do
            EVAL_CFG="configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_${dom}.py"
            EVAL_DIR="$ROOT_WORK_DIR/eval/${eval_tag}/${dom}"
            EVAL_DONE="$EVAL_DIR/eval.done"
            mkdir -p "$EVAL_DIR"
            if [[ -f "$EVAL_DONE" && "$FORCE" != "1" ]]; then
                echo "  SKIP eval ${eval_tag}/${dom}"
                continue
            fi
            echo "  Eval: ${eval_tag} on ${dom}"
            PORT=$(get_port)
            log_cmd "CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch \
                --nproc_per_node=${GPU_NUM} --master_addr=127.0.0.1 --master_port=${PORT} \
                tools/run/test.py ${EVAL_CFG} \
                --checkpoint ${CKPT} --launcher pytorch \
                --work-dir ${EVAL_DIR} --eval bbox track --fuse-conv-bn \
                --cfg-options data.workers_per_gpu=${TEST_WORKERS_PER_GPU} cudnn_benchmark=True \
                > ${EVAL_DIR}/eval.log 2>&1"
            [[ "$DRY_RUN" != "1" ]] && date > "$EVAL_DONE"
        done
    done
done

echo ""
echo "==========================================="
echo "Task2 complete. Results: $ROOT_WORK_DIR/eval/"
echo "==========================================="
