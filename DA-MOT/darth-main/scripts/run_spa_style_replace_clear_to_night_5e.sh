#!/usr/bin/env bash
# Quick-validation experiment: SPA-style replacement, clear -> night, 5 epochs.
#   - teacher view = clean strong view (deterministic geometry only)
#   - student/key  = low-frequency SPA weak view  (SPA before Normalize, RGB)
#   - contrastive  = high-frequency SPA weak view (SPA before Normalize, RGB)
#   - causal DISABLED; old forward_train post-process SPA DISABLED
#   - eval night at epoch 3 and 5 only (clear NOT evaluated)
#
# Does NOT touch causal_only / baseline_task2 / task1 outputs.
#
# Usage:
#   bash scripts/run_spa_style_replace_clear_to_night_5e.sh
#   DRY_RUN=1 bash scripts/run_spa_style_replace_clear_to_night_5e.sh
set -uo pipefail

EXP_NAME="spa_style_replace_clear_to_night_5e"
BASE_CFG="configs/adapters/darth/bdd_7domain/darth_seq_spa_style_replace_clear_to_night_5e.py"

SOURCE_DOMAIN="clear"
TARGET_ORDER=("night")
EVAL_DOMAINS=("night")
EVAL_EPOCHS=(3 5)
ADAPT_EPOCHS=5

GPU_IDS=${GPU_IDS:-"9,8,7,6,5,4,3,2,1,0"}
GPU_NUM=${GPU_NUM:-10}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-10}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU:-8}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}
PYTHON=${PYTHON:-/home/lwr/anaconda3/envs/DARTH/bin/python}

ROOT_WORK_DIR="work_dirs/bdd_4domain/${EXP_NAME}"
EPISODE_DIR="$ROOT_WORK_DIR/episode"
SOURCE_WORK_DIR="work_dirs/bdd_4domain/source/clear_20e"
SOURCE_CKPT=${SOURCE_CKPT:-""}

get_port() { "$PYTHON" -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()"; }
log_cmd() { echo "[CMD] $*"; [[ "$DRY_RUN" == "1" ]] && return; eval "$*"; }

find_source_ckpt() {
    if [[ -n "$SOURCE_CKPT" && -f "$SOURCE_CKPT" ]]; then echo "$SOURCE_CKPT"; return; fi
    local p
    p=$(find "$SOURCE_WORK_DIR" -name "latest.pth" 2>/dev/null | head -1)
    [[ -z "$p" ]] && p=$(find "$SOURCE_WORK_DIR" -name "epoch_20.pth" 2>/dev/null | head -1)
    echo "$p"
}

[[ -f tools/run/train_seq3_episode.py ]] || { echo "ERROR: run from project root" >&2; exit 1; }
SOURCE_CKPT=$(find_source_ckpt)
[[ -n "$SOURCE_CKPT" && -f "$SOURCE_CKPT" ]] || { echo "ERROR: clear source ckpt not found under $SOURCE_WORK_DIR" >&2; exit 1; }

# Backup pre-existing same-name dir (never delete).
if [[ -d "$ROOT_WORK_DIR" && "$DRY_RUN" != "1" ]]; then
    if [[ -n "$(ls -A "$ROOT_WORK_DIR" 2>/dev/null | grep -v SPA_ORIGINAL_NOTES.md)" ]]; then
        BK="${ROOT_WORK_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
        echo "[BACKUP] existing $ROOT_WORK_DIR -> $BK"
        cp -r "$ROOT_WORK_DIR" "$BK"
    fi
fi

# PLACEHOLDER_BANNER
echo "==========================================================="
echo "EXPERIMENT: ${EXP_NAME}"
echo "==========================================================="
echo "use_causal               : False"
echo ""
echo "teacher view             : clean strong view"
echo "  random geometry        : disabled"
echo "  original photometric   : disabled"
echo "  SPA                    : disabled"
echo ""
echo "student/key view         : low-frequency SPA weak view"
echo "  random geometry        : disabled"
echo "  original photometric   : disabled"
echo "  SPA position           : BEFORE Normalize (RGB image space)"
echo "  low_ratio              : 0.10"
echo "  keep_prob              : 0.90   (mask_ratio=0.10)"
echo "  noise                  : 0.0"
echo ""
echo "contrastive view         : high-frequency SPA weak view"
echo "  random geometry        : disabled"
echo "  original photometric   : disabled"
echo "  SPA position           : BEFORE Normalize (RGB image space)"
echo "  noise_ratio (gamma)    : 0.03"
echo ""
echo "old forward_train SPA postprocess disabled : True"
echo "  (use_spa_frequency_aug=False, spa_forward_postprocess=False)"
echo ""
echo "base_config              : $BASE_CFG"
echo "source_checkpoint        : $SOURCE_CKPT"
echo "source_domain            : $SOURCE_DOMAIN"
echo "target_order             : ${TARGET_ORDER[*]}"
echo "eval_domains             : ${EVAL_DOMAINS[*]} (clear NOT evaluated)"
echo "epochs_per_stage         : $ADAPT_EPOCHS"
echo "eval_epochs              : ${EVAL_EPOCHS[*]}"
echo "GPU_IDS                  : $GPU_IDS"
echo "GPU_NUM                  : $GPU_NUM"
echo "SAMPLES_PER_GPU          : $SAMPLES_PER_GPU"
echo "WORKERS_PER_GPU          : $WORKERS_PER_GPU"
echo "TEST_WORKERS_PER_GPU     : $TEST_WORKERS_PER_GPU"
echo "work_dir                 : $ROOT_WORK_DIR"
echo "DRY_RUN=$DRY_RUN  FORCE=$FORCE"
echo "==========================================================="
echo "NOTE: per-view post-Normalize mean/std/min/max are printed once"
echo "      in the training log (log_view_stats=True) to confirm SPA did"
echo "      NOT act on the normalized tensor."
echo "==========================================================="
echo ""

if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$ROOT_WORK_DIR/logs" "$EPISODE_DIR/state" "$EPISODE_DIR/stage_snapshots"
    cp "$BASE_CFG" "$ROOT_WORK_DIR/$(basename "$BASE_CFG")"
fi

TRAIN_LOG="$ROOT_WORK_DIR/logs/train_episode.log"
EPISODE_DONE="$EPISODE_DIR/state/train_episode.done"

# ── train ────────────────────────────────────────────────────────────────────
if [[ -f "$EPISODE_DONE" && "$FORCE" != "1" ]]; then
    echo "SKIP training (done: $EPISODE_DONE)"
else
    echo "--- Training: clear -> night, ${ADAPT_EPOCHS} epochs ---"
    PORT=$(get_port)
    log_cmd "CUDA_VISIBLE_DEVICES=${GPU_IDS} $PYTHON -m torch.distributed.launch \
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

# ── eval night at epoch 3 and 5 ──────────────────────────────────────────────
echo ""
echo "--- Eval night at epoch ${EVAL_EPOCHS[*]} ---"
STAGE="stage01_after_night"
for ep in "${EVAL_EPOCHS[@]}"; do
    CKPT="$EPISODE_DIR/adapt/${STAGE}/epoch_${ep}.pth"
    if [[ ! -f "$CKPT" && "$DRY_RUN" != "1" ]]; then
        echo "  WARNING: ckpt not found: $CKPT"; continue
    fi
    for dom in "${EVAL_DOMAINS[@]}"; do
        EVAL_CFG="configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_${dom}.py"
        EVAL_DIR="$ROOT_WORK_DIR/eval/epoch_${ep}/${dom}"
        [[ "$DRY_RUN" != "1" ]] && mkdir -p "$EVAL_DIR"
        echo "  Eval epoch_${ep} on ${dom}"
        PORT=$(get_port)
        log_cmd "CUDA_VISIBLE_DEVICES=${GPU_IDS} $PYTHON -m torch.distributed.launch \
            --nproc_per_node=${GPU_NUM} --master_addr=127.0.0.1 --master_port=${PORT} \
            tools/run/test.py ${EVAL_CFG} \
            --checkpoint ${CKPT} --launcher pytorch \
            --work-dir ${EVAL_DIR} --eval bbox track --fuse-conv-bn \
            --cfg-options data.workers_per_gpu=${TEST_WORKERS_PER_GPU} cudnn_benchmark=True \
            > ${EVAL_DIR}/eval.log 2>&1"
    done
done

echo ""
echo "==========================================================="
echo "Done. Summarize with:"
echo "  $PYTHON scripts/collect_spa_style_replace_results.py"
echo "==========================================================="
