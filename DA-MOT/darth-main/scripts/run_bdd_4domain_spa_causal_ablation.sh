#!/usr/bin/env bash
# First-round ablation runner for the four-domain Task2 full continuous episode.
#
#   source = clear  ->  overcast -> rainy -> dawn_dusk -> night
#   epochs_per_stage = 10 ; eval at epoch 5 and 10 ; eval on the 4 targets
#   teacher + student both carried across stages (same as Task2 baseline)
#
# Selects experiment via EXP env var. Does NOT touch baseline result dirs.
#
# Usage:
#   EXP=spa_only     bash scripts/run_bdd_4domain_spa_causal_ablation.sh
#   EXP=causal_only  bash scripts/run_bdd_4domain_spa_causal_ablation.sh
#   EXP=spa_causal   bash scripts/run_bdd_4domain_spa_causal_ablation.sh
#   DRY_RUN=1 EXP=spa_only bash scripts/run_bdd_4domain_spa_causal_ablation.sh
set -euo pipefail

# ─── Experiment selection ────────────────────────────────────────────────────
EXP=${EXP:-spa_only}
case "$EXP" in
    spa_only)
        BASE_CFG="configs/adapters/darth/bdd_7domain/darth_seq_spa_only.py" ;;
    causal_only)
        BASE_CFG="configs/adapters/darth/bdd_7domain/darth_seq_causal_only.py" ;;
    spa_causal)
        BASE_CFG="configs/adapters/darth/bdd_7domain/darth_seq_spa_causal.py" ;;
    *)
        echo "ERROR: unknown EXP='$EXP' (expected spa_only|causal_only|spa_causal)" >&2
        exit 1 ;;
esac

# ─── Episode definition (identical to Task2 full continuous) ─────────────────
SOURCE_DOMAIN="clear"
TARGET_ORDER=("overcast" "rainy" "dawn_dusk" "night")
EVAL_DOMAINS=("overcast" "rainy" "dawn_dusk" "night")
EVAL_EPOCHS=(5 10)
ADAPT_EPOCHS=10

# ─── Compute / IO config (8 GPU, fixed; do NOT auto-probe batch) ─────────────
GPU_IDS=${GPU_IDS:-"9,8,7,6,3,2,1,0"}
GPU_NUM=${GPU_NUM:-8}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-10}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU:-8}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}
# Python interpreter: default to the DARTH conda env so background/nohup runs
# don't accidentally use the base env (which has no torch).
PYTHON=${PYTHON:-/home/lwr/anaconda3/envs/DARTH/bin/python}

ROOT_WORK_DIR="work_dirs/bdd_4domain_ablation/task2_${EXP}"

# ─── Source checkpoint (new-annotation clear source; recursive lookup) ───────
SOURCE_WORK_DIR="work_dirs/bdd_4domain/source/clear_20e"
SOURCE_EPOCHS=20
SOURCE_CKPT=${SOURCE_CKPT:-""}

get_port() { "$PYTHON" -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()"; }
log_cmd() {
    echo "[CMD] $*"
    [[ "$DRY_RUN" == "1" ]] && return
    eval "$*"
}

find_source_ckpt() {
    # Prefer explicit SOURCE_CKPT, else recursively find epoch_20.pth under the
    # new-annotation clear source dir. Never fall back to a repo-root ckpt.
    if [[ -n "$SOURCE_CKPT" && -f "$SOURCE_CKPT" ]]; then echo "$SOURCE_CKPT"; return; fi
    find "$SOURCE_WORK_DIR" -name "epoch_${SOURCE_EPOCHS}.pth" 2>/dev/null | head -1
}

if [[ ! -f tools/run/train_seq3_episode.py ]]; then
    echo "ERROR: run from project root" >&2; exit 1
fi

SOURCE_CKPT=$(find_source_ckpt)
if [[ -z "$SOURCE_CKPT" || ! -f "$SOURCE_CKPT" ]]; then
    echo "ERROR: new-annotation clear source checkpoint not found under $SOURCE_WORK_DIR" >&2
    echo "       (refusing to fall back to any old baseline source)" >&2
    exit 1
fi

EPISODE_DIR="$ROOT_WORK_DIR/episode"
RUN_LOG="$ROOT_WORK_DIR/logs/run.log"
TRAIN_LOG="$ROOT_WORK_DIR/logs/train_episode.log"

echo "==========================================================="
echo "First-round ablation: EXP=${EXP}"
echo "==========================================================="
echo "base_config         : $BASE_CFG"
echo "source_checkpoint   : $SOURCE_CKPT"
echo "target_order        : ${TARGET_ORDER[*]}"
echo "eval_domains        : ${EVAL_DOMAINS[*]} (clear NOT evaluated)"
echo "epochs_per_stage    : $ADAPT_EPOCHS"
echo "eval_epochs         : ${EVAL_EPOCHS[*]}"
echo "GPU_IDS             : $GPU_IDS"
echo "GPU_NUM             : $GPU_NUM"
echo "SAMPLES_PER_GPU     : $SAMPLES_PER_GPU"
echo "WORKERS_PER_GPU     : $WORKERS_PER_GPU"
echo "TEST_WORKERS_PER_GPU: $TEST_WORKERS_PER_GPU"
echo "work_dir            : $ROOT_WORK_DIR"
echo "DRY_RUN=$DRY_RUN  FORCE=$FORCE"
echo "==========================================================="
echo ""

if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$ROOT_WORK_DIR/logs" "$EPISODE_DIR/state" "$EPISODE_DIR/stage_snapshots"
    cp "$BASE_CFG" "$ROOT_WORK_DIR/$(basename "$BASE_CFG")"
fi

# ─── Continuous episode training ─────────────────────────────────────────────
EPISODE_DONE="$EPISODE_DIR/state/train_episode.done"
if [[ -f "$EPISODE_DONE" && "$FORCE" != "1" ]]; then
    echo "SKIP episode training (already done: $EPISODE_DONE)"
else
    echo "--- Continuous episode training: ${TARGET_ORDER[*]} ---"
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

# ─── Per-stage eval at epoch 5 and 10 on the 4 target domains ────────────────
echo ""
echo "--- Evaluating all stages at epoch ${EVAL_EPOCHS[*]} ---"

STAGE_NAMES=("stage01_after_overcast" "stage02_after_rainy" "stage03_after_dawn_dusk" "stage04_after_night")
for stage_idx in "${!STAGE_NAMES[@]}"; do
    stage="${STAGE_NAMES[$stage_idx]}"
    for ep in "${EVAL_EPOCHS[@]}"; do
        CKPT="$EPISODE_DIR/adapt/${stage}/epoch_${ep}.pth"
        eval_tag="${stage}/epoch_${ep}"
        if [[ ! -f "$CKPT" && "$DRY_RUN" != "1" ]]; then
            echo "  WARNING: checkpoint not found: $CKPT"; continue
        fi
        for dom in "${EVAL_DOMAINS[@]}"; do
            EVAL_CFG="configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_${dom}.py"
            EVAL_DIR="$ROOT_WORK_DIR/eval/${eval_tag}/${dom}"
            EVAL_DONE="$EVAL_DIR/eval.done"
            [[ "$DRY_RUN" != "1" ]] && mkdir -p "$EVAL_DIR"
            if [[ -f "$EVAL_DONE" && "$FORCE" != "1" ]]; then
                echo "  SKIP eval ${eval_tag}/${dom}"; continue
            fi
            echo "  Eval: ${eval_tag} on ${dom}"
            PORT=$(get_port)
            log_cmd "CUDA_VISIBLE_DEVICES=${GPU_IDS} $PYTHON -m torch.distributed.launch \
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
echo "==========================================================="
echo "EXP=${EXP} complete. Results: ${ROOT_WORK_DIR}/eval/"
echo "Summarize with: python scripts/collect_ablation_results.py"
echo "==========================================================="
