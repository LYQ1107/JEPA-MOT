#!/usr/bin/env bash
# SPA-only four-domain validation (4 GPU): clear -> overcast -> rainy ->
# dawn_dusk -> night, 5 epochs/stage, eval each target domain at epoch 5.
# Reproduces the "00:32" SPA setting (see config header).
# Does NOT touch V0/V1/V2/V3/V1a/V1b, causal_only, or old spa_only outputs.
#
#   bash scripts/run_spa_only_4domain_5e_4gpu.sh
#   DRY_RUN=1 bash scripts/run_spa_only_4domain_5e_4gpu.sh
#   FORCE=1   bash scripts/run_spa_only_4domain_5e_4gpu.sh
set -euo pipefail

GPU_IDS=${GPU_IDS:-"9,8,7,6"}
GPU_NUM=${GPU_NUM:-4}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-16}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU:-6}
DRY_RUN=${DRY_RUN:-0}
FORCE=${FORCE:-0}
PYTHON=${PYTHON:-/home/lwr/anaconda3/envs/DARTH/bin/python}

CFG="configs/adapters/darth/bdd_4domain/roi_proto/darth_seq_spa_only_4domain_5e.py"
ROOT="work_dirs/bdd_4domain/spa_only_4domain_5e"
EPISODE="$ROOT/episode"
SOURCE_WORK_DIR="work_dirs/bdd_4domain/source/clear_20e"
SOURCE_CKPT=${SOURCE_CKPT:-""}
TARGET_ORDER=("overcast" "rainy" "dawn_dusk" "night")
EVAL_EPOCH=5

get_port(){ "$PYTHON" -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()"; }

find_source_ckpt(){
    if [[ -n "$SOURCE_CKPT" && -f "$SOURCE_CKPT" ]]; then echo "$SOURCE_CKPT"; return; fi
    local p
    p=$(find "$SOURCE_WORK_DIR" -name "latest.pth" 2>/dev/null | head -1)
    [[ -z "$p" ]] && p=$(find "$SOURCE_WORK_DIR" -name "epoch_20.pth" 2>/dev/null | head -1)
    echo "$p"
}

[[ -f tools/run/train_seq3_episode.py ]] || { echo "ERROR: run from project root" >&2; exit 1; }
[[ -f "$CFG" ]] || { echo "ERROR: config not found: $CFG" >&2; exit 1; }
SOURCE_CKPT=$(find_source_ckpt)
[[ -n "$SOURCE_CKPT" && -f "$SOURCE_CKPT" ]] || { echo "ERROR: clear source ckpt not found under $SOURCE_WORK_DIR" >&2; exit 1; }

TRAIN_MARKER="$ROOT/train_done.marker"

echo "==========================================================="
echo "EXPERIMENT: spa_only_4domain_5e (00:32 SPA setting)"
echo "  config         : $CFG"
echo "  source_ckpt    : $SOURCE_CKPT"
echo "  target_order   : ${TARGET_ORDER[*]}"
echo "  epochs/stage   : 5    eval each target @ epoch $EVAL_EPOCH"
echo "  GPU_IDS        : $GPU_IDS  (GPU_NUM=$GPU_NUM)"
echo "  SAMPLES/GPU    : $SAMPLES_PER_GPU  (global batch $((SAMPLES_PER_GPU*GPU_NUM)))"
echo "  test workers   : $TEST_WORKERS_PER_GPU  (--fuse-conv-bn, cudnn_benchmark=True)"
echo "  work_dir       : $ROOT"
echo "  causal=OFF RoI-memory=OFF ; SPA student-only low_freq_mask keep_prob=0.9"
echo "  DRY_RUN=$DRY_RUN  FORCE=$FORCE"
echo "==========================================================="

# ---- train (single continuous episode over 4 target domains) ----
if [[ -f "$TRAIN_MARKER" && "$FORCE" != "1" ]]; then
    echo "SKIP train (marker exists): $TRAIN_MARKER"
else
    [[ "$DRY_RUN" != "1" ]] && { mkdir -p "$ROOT/logs" "$EPISODE"; cp "$CFG" "$ROOT/$(basename "$CFG")"; }
    PORT=$(get_port)
    CMD="CUDA_VISIBLE_DEVICES=${GPU_IDS} $PYTHON -m torch.distributed.launch \
        --nproc_per_node=${GPU_NUM} --master_addr=127.0.0.1 --master_port=${PORT} \
        tools/run/train_seq3_episode.py \
        --base-config ${CFG} --source-ckpt ${SOURCE_CKPT} \
        --target-order ${TARGET_ORDER[*]} --eval-domains ${TARGET_ORDER[*]} \
        --work-dir ${EPISODE} --epochs-per-stage 5 \
        --samples-per-gpu ${SAMPLES_PER_GPU} --workers-per-gpu ${WORKERS_PER_GPU} \
        --launcher pytorch --no-validate > ${ROOT}/logs/train_episode.log 2>&1"
    echo "[CMD] $CMD"
    if [[ "$DRY_RUN" != "1" ]]; then
        eval "$CMD" && date > "$TRAIN_MARKER" || { echo "TRAIN FAILED" >&2; exit 1; }
    fi
fi

# ---- eval each target domain at its own stage, epoch 5 ----
ECFG_DIR="configs/mot/qdtrack/bdd_7domain"
declare -i idx=0
for dom in "${TARGET_ORDER[@]}"; do
    idx+=1
    STAGE=$(printf "stage%02d_after_%s" "$idx" "$dom")
    CKPT="$EPISODE/adapt/${STAGE}/epoch_${EVAL_EPOCH}.pth"
    EDIR="$ROOT/eval/${STAGE}/epoch_${EVAL_EPOCH}/${dom}"
    ECFG="$ECFG_DIR/qdtrack_faster-rcnn_r50_fpn_20e_bdd_${dom}.py"
    EVAL_MARKER="$EDIR/eval_done.marker"
    if [[ -f "$EVAL_MARKER" && "$FORCE" != "1" ]]; then
        echo "SKIP eval $STAGE/$dom (marker exists)"; continue
    fi
    [[ "$DRY_RUN" != "1" ]] && mkdir -p "$EDIR"
    if [[ ! -f "$CKPT" && "$DRY_RUN" != "1" ]]; then
        echo "ERROR: ckpt not found: $CKPT" >&2; exit 1
    fi
    PORT=$(get_port)
    CMD="CUDA_VISIBLE_DEVICES=${GPU_IDS} $PYTHON -m torch.distributed.launch \
        --nproc_per_node=${GPU_NUM} --master_addr=127.0.0.1 --master_port=${PORT} \
        tools/run/test.py ${ECFG} --checkpoint ${CKPT} --launcher pytorch \
        --work-dir ${EDIR} --eval bbox track --fuse-conv-bn \
        --cfg-options data.workers_per_gpu=${TEST_WORKERS_PER_GPU} cudnn_benchmark=True \
        > ${EDIR}/eval.log 2>&1"
    echo "[CMD] $CMD"
    if [[ "$DRY_RUN" != "1" ]]; then
        eval "$CMD" && date > "$EVAL_MARKER" || { echo "EVAL FAILED: $STAGE/$dom" >&2; exit 1; }
    fi
done

echo ""
echo "All done. Eval dirs under: $ROOT/eval/stage*/epoch_${EVAL_EPOCH}/<domain>/"
