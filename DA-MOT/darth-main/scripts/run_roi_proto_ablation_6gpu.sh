#!/usr/bin/env bash
# RoI Prototype Memory ablation (6 GPU): runs V0, V1, V2, V3 in order.
# Each version: clear -> overcast -> night, 5 epochs/stage, then eval night ep5.
# causal=OFF, SPA=OFF. Does NOT touch causal_only / SPA experiments.
#
#   bash scripts/run_roi_proto_ablation_6gpu.sh
#   DRY_RUN=1 bash scripts/run_roi_proto_ablation_6gpu.sh
#   RUN_V4=1  bash scripts/run_roi_proto_ablation_6gpu.sh   # if V4 implemented
set -euo pipefail

GPU_IDS=${GPU_IDS:-"9,8,7,6,2,3"}
GPU_NUM=${GPU_NUM:-6}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-10}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU:-6}
DRY_RUN=${DRY_RUN:-0}
FORCE=${FORCE:-0}
RUN_V4=${RUN_V4:-0}
PYTHON=${PYTHON:-/home/lwr/anaconda3/envs/DARTH/bin/python}

CFG_DIR="configs/adapters/darth/bdd_4domain/roi_proto"
ROOT="work_dirs/bdd_4domain/roi_proto_ablation"
SOURCE_WORK_DIR="work_dirs/bdd_4domain/source/clear_20e"
SOURCE_CKPT=${SOURCE_CKPT:-""}
TARGET_ORDER=("overcast" "night")
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
SOURCE_CKPT=$(find_source_ckpt)
[[ -n "$SOURCE_CKPT" && -f "$SOURCE_CKPT" ]] || { echo "ERROR: clear source ckpt not found" >&2; exit 1; }

ONLY_VERSION=${ONLY_VERSION:-""}

run_version(){
    local vname="$1" cfg="$2"
    if [[ -n "$ONLY_VERSION" && "$vname" != *"$ONLY_VERSION"* ]]; then
        return 0
    fi
    local wdir="$ROOT/$vname"
    local episode="$wdir/episode"
    local train_marker="$wdir/train_done.marker"
    local eval_marker="$wdir/eval_done.marker"

    echo "==========================================================="
    echo "VERSION: $vname"
    echo "  config        : $cfg"
    echo "  source_ckpt   : $SOURCE_CKPT"
    echo "  target_order  : ${TARGET_ORDER[*]}"
    echo "  GPU_IDS       : $GPU_IDS  (GPU_NUM=$GPU_NUM)"
    echo "  SAMPLES/GPU   : $SAMPLES_PER_GPU"
    echo "  causal=OFF SPA=OFF  eval night epoch $EVAL_EPOCH"
    echo "==========================================================="

    [[ -f "$cfg" ]] || { echo "ERROR: config not found: $cfg" >&2; exit 1; }

    # ---- train ----
    if [[ -f "$train_marker" && "$FORCE" != "1" ]]; then
        echo "SKIP train (marker exists): $train_marker"
    else
        mkdir -p "$wdir/logs" "$episode"
        [[ "$DRY_RUN" != "1" ]] && cp "$cfg" "$wdir/$(basename "$cfg")"
        local port; port=$(get_port)
        local cmd="CUDA_VISIBLE_DEVICES=${GPU_IDS} $PYTHON -m torch.distributed.launch \
            --nproc_per_node=${GPU_NUM} --master_addr=127.0.0.1 --master_port=${port} \
            tools/run/train_seq3_episode.py \
            --base-config ${cfg} --source-ckpt ${SOURCE_CKPT} \
            --target-order ${TARGET_ORDER[*]} --eval-domains night \
            --work-dir ${episode} --epochs-per-stage 5 \
            --samples-per-gpu ${SAMPLES_PER_GPU} --workers-per-gpu ${WORKERS_PER_GPU} \
            --launcher pytorch --no-validate > ${wdir}/logs/train_episode.log 2>&1"
        echo "[CMD] $cmd"
        if [[ "$DRY_RUN" != "1" ]]; then
            eval "$cmd" && date > "$train_marker" || { echo "TRAIN FAILED: $vname" >&2; exit 1; }
        fi
    fi

    # ---- eval night epoch5 (final stage = stage02_after_night) ----
    if [[ -f "$eval_marker" && "$FORCE" != "1" ]]; then
        echo "SKIP eval (marker exists): $eval_marker"
    else
        local ckpt="$episode/adapt/stage02_after_night/epoch_${EVAL_EPOCH}.pth"
        local edir="$wdir/eval/epoch_${EVAL_EPOCH}/night"
        local ecfg="configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_night.py"
        [[ "$DRY_RUN" != "1" ]] && mkdir -p "$edir"
        if [[ ! -f "$ckpt" && "$DRY_RUN" != "1" ]]; then
            echo "ERROR: ckpt not found: $ckpt" >&2; exit 1
        fi
        local port; port=$(get_port)
        local cmd="CUDA_VISIBLE_DEVICES=${GPU_IDS} $PYTHON -m torch.distributed.launch \
            --nproc_per_node=${GPU_NUM} --master_addr=127.0.0.1 --master_port=${port} \
            tools/run/test.py ${ecfg} --checkpoint ${ckpt} --launcher pytorch \
            --work-dir ${edir} --eval bbox track --fuse-conv-bn \
            --cfg-options data.workers_per_gpu=${TEST_WORKERS_PER_GPU} cudnn_benchmark=True \
            > ${edir}/eval.log 2>&1"
        echo "[CMD] $cmd"
        if [[ "$DRY_RUN" != "1" ]]; then
            eval "$cmd" && date > "$eval_marker" || { echo "EVAL FAILED: $vname" >&2; exit 1; }
        fi
    fi
}

V0="$CFG_DIR/darth_seq_roi_proto_v0_control_clear_overcast_night_5e.py"
V1="$CFG_DIR/darth_seq_roi_proto_v1_topscore_car_l001_clear_overcast_night_5e.py"
V2="$CFG_DIR/darth_seq_roi_proto_v2_conservative_car_l0005_clear_overcast_night_5e.py"
V3="$CFG_DIR/darth_seq_roi_proto_v3_diverse_car_l001_clear_overcast_night_5e.py"
V4="$CFG_DIR/darth_seq_roi_proto_v4_teacher_proto_car_l001_clear_overcast_night_5e.py"

run_version "v0_control_clear_overcast_night_5e"              "$V0"
run_version "v1_topscore_car_l001_clear_overcast_night_5e"   "$V1"
run_version "v2_conservative_car_l0005_clear_overcast_night_5e" "$V2"
run_version "v3_diverse_car_l001_clear_overcast_night_5e"    "$V3"
if [[ "$RUN_V4" == "1" && -f "$V4" ]]; then
    run_version "v4_teacher_proto_car_l001_clear_overcast_night_5e" "$V4"
fi

echo ""
echo "All requested versions finished. Summarize with:"
echo "  $PYTHON tools/collect_roi_proto_ablation.py"
[[ "$DRY_RUN" != "1" ]] && "$PYTHON" tools/collect_roi_proto_ablation.py || true
