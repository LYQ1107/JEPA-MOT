#!/usr/bin/env bash
# RoI Prototype Memory — best version four-domain 5e (8 GPU).
# clear -> overcast -> rainy -> dawn_dusk -> night, 5 epochs/stage.
# Eval each target at its stage epoch5. causal=OFF, SPA=OFF.
# Select version via BEST_VERSION={v1,v1a,v1b}. DO NOT launch until the
# V1a/V1b quick-verify picks a winner.
#
#   BEST_VERSION=v1  bash scripts/run_roi_proto_best_4domain_5e_8gpu.sh
#   DRY_RUN=1 BEST_VERSION=v1a bash scripts/run_roi_proto_best_4domain_5e_8gpu.sh
set -euo pipefail

GPU_IDS=${GPU_IDS:-"9,8,7,6,5,4,3,2"}
GPU_NUM=${GPU_NUM:-8}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-10}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU:-6}
DRY_RUN=${DRY_RUN:-0}
FORCE=${FORCE:-0}
BEST_VERSION=${BEST_VERSION:-v1}
PYTHON=${PYTHON:-/home/lwr/anaconda3/envs/DARTH/bin/python}

CFG_DIR="configs/adapters/darth/bdd_4domain/roi_proto"
case "$BEST_VERSION" in
    v1)  CFG="$CFG_DIR/darth_seq_roi_proto_best_4domain_5e.py" ;;
    v1a) CFG="$CFG_DIR/darth_seq_roi_proto_v1a_4domain_5e.py" ;;
    v1b) CFG="$CFG_DIR/darth_seq_roi_proto_v1b_4domain_5e.py" ;;
    *) echo "ERROR: BEST_VERSION must be v1|v1a|v1b" >&2; exit 1 ;;
esac
ROOT="work_dirs/bdd_4domain/roi_proto_best_${BEST_VERSION}_4domain_5e"
EPISODE="$ROOT/episode"
SOURCE_WORK_DIR="work_dirs/bdd_4domain/source/clear_20e"
SOURCE_CKPT=${SOURCE_CKPT:-""}
TARGET_ORDER=("overcast" "rainy" "dawn_dusk" "night")
EVAL_DOMAINS=("overcast" "rainy" "dawn_dusk" "night")
EVAL_EPOCH=5

get_port(){ "$PYTHON" -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()"; }
find_source_ckpt(){
    if [[ -n "$SOURCE_CKPT" && -f "$SOURCE_CKPT" ]]; then echo "$SOURCE_CKPT"; return; fi
    local p; p=$(find "$SOURCE_WORK_DIR" -name "latest.pth" 2>/dev/null | head -1)
    [[ -z "$p" ]] && p=$(find "$SOURCE_WORK_DIR" -name "epoch_20.pth" 2>/dev/null | head -1)
    echo "$p"
}

[[ -f tools/run/train_seq3_episode.py ]] || { echo "ERROR: run from project root" >&2; exit 1; }
[[ -f "$CFG" ]] || { echo "ERROR: config not found: $CFG" >&2; exit 1; }
SOURCE_CKPT=$(find_source_ckpt)
[[ -n "$SOURCE_CKPT" && -f "$SOURCE_CKPT" ]] || { echo "ERROR: clear source ckpt not found" >&2; exit 1; }

echo "==========================================================="
echo "RoI PROTO best four-domain 5e   BEST_VERSION=$BEST_VERSION"
echo "  config       : $CFG"
echo "  source_ckpt  : $SOURCE_CKPT"
echo "  target_order : ${TARGET_ORDER[*]}"
echo "  eval_domains : ${EVAL_DOMAINS[*]} (epoch $EVAL_EPOCH)"
echo "  GPU_IDS=$GPU_IDS GPU_NUM=$GPU_NUM SAMPLES/GPU=$SAMPLES_PER_GPU"
echo "  causal=OFF SPA=OFF   output=$ROOT"
echo "==========================================================="

TRAIN_MARKER="$ROOT/train_done.marker"
if [[ -f "$TRAIN_MARKER" && "$FORCE" != "1" ]]; then
    echo "SKIP train (marker exists)"
else
    [[ "$DRY_RUN" != "1" ]] && mkdir -p "$ROOT/logs" "$EPISODE"
    [[ "$DRY_RUN" != "1" ]] && cp "$CFG" "$ROOT/$(basename "$CFG")"
    PORT=$(get_port)
    CMD="CUDA_VISIBLE_DEVICES=${GPU_IDS} $PYTHON -m torch.distributed.launch \
        --nproc_per_node=${GPU_NUM} --master_addr=127.0.0.1 --master_port=${PORT} \
        tools/run/train_seq3_episode.py --base-config ${CFG} --source-ckpt ${SOURCE_CKPT} \
        --target-order ${TARGET_ORDER[*]} --eval-domains ${EVAL_DOMAINS[*]} \
        --work-dir ${EPISODE} --epochs-per-stage 5 \
        --samples-per-gpu ${SAMPLES_PER_GPU} --workers-per-gpu ${WORKERS_PER_GPU} \
        --launcher pytorch --no-validate > ${ROOT}/logs/train_episode.log 2>&1"
    echo "[CMD] $CMD"
    if [[ "$DRY_RUN" != "1" ]]; then
        eval "$CMD" && date > "$TRAIN_MARKER" || { echo "TRAIN FAILED" >&2; exit 1; }
    fi
fi

declare -A STAGE_OF=( [overcast]=stage01_after_overcast [rainy]=stage02_after_rainy \
    [dawn_dusk]=stage03_after_dawn_dusk [night]=stage04_after_night )
for dom in "${EVAL_DOMAINS[@]}"; do
    STAGE="${STAGE_OF[$dom]}"
    CKPT="$EPISODE/adapt/${STAGE}/epoch_${EVAL_EPOCH}.pth"
    EDIR="$ROOT/eval/${STAGE}/epoch_${EVAL_EPOCH}/${dom}"
    ECFG="configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_${dom}.py"
    MARKER="$ROOT/eval_${dom}.marker"
    if [[ -f "$MARKER" && "$FORCE" != "1" ]]; then echo "SKIP eval $dom"; continue; fi
    [[ "$DRY_RUN" != "1" ]] && mkdir -p "$EDIR"
    if [[ ! -f "$CKPT" && "$DRY_RUN" != "1" ]]; then echo "ERROR: ckpt missing $CKPT" >&2; exit 1; fi
    PORT=$(get_port)
    CMD="CUDA_VISIBLE_DEVICES=${GPU_IDS} $PYTHON -m torch.distributed.launch \
        --nproc_per_node=${GPU_NUM} --master_addr=127.0.0.1 --master_port=${PORT} \
        tools/run/test.py ${ECFG} --checkpoint ${CKPT} --launcher pytorch \
        --work-dir ${EDIR} --eval bbox track --fuse-conv-bn \
        --cfg-options data.workers_per_gpu=${TEST_WORKERS_PER_GPU} cudnn_benchmark=True \
        > ${EDIR}/eval.log 2>&1"
    echo "[CMD] $CMD"
    if [[ "$DRY_RUN" != "1" ]]; then
        eval "$CMD" && date > "$MARKER" || { echo "EVAL FAILED $dom" >&2; exit 1; }
    fi
done

echo "Done four-domain 5e for $BEST_VERSION."
