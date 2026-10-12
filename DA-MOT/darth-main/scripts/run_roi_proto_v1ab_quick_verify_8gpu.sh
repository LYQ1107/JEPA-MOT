#!/usr/bin/env bash
# RoI Prototype Memory quick-verify (8 GPU): runs V1a, V1b only.
# Does NOT rerun V0/V1/V2/V3 and does NOT touch their outputs.
# Each version: clear -> overcast -> night, 3 epochs/stage, eval night ep3.
# causal=OFF, SPA=OFF.
#
#   bash scripts/run_roi_proto_v1ab_quick_verify_8gpu.sh
#   DRY_RUN=1 bash scripts/run_roi_proto_v1ab_quick_verify_8gpu.sh
set -euo pipefail

GPU_IDS=${GPU_IDS:-"9,8,7,6,5,4,3,2"}
GPU_NUM=${GPU_NUM:-8}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-10}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU:-6}
DRY_RUN=${DRY_RUN:-0}
FORCE=${FORCE:-0}
PYTHON=${PYTHON:-/home/lwr/anaconda3/envs/DARTH/bin/python}

CFG_DIR="configs/adapters/darth/bdd_4domain/roi_proto"
ROOT="work_dirs/bdd_4domain/roi_proto_ablation"
SOURCE_WORK_DIR="work_dirs/bdd_4domain/source/clear_20e"
SOURCE_CKPT=${SOURCE_CKPT:-""}
TARGET_ORDER=("overcast" "night")
EVAL_EPOCH=3

get_port(){ "$PYTHON" -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()"; }
find_source_ckpt(){
    if [[ -n "$SOURCE_CKPT" && -f "$SOURCE_CKPT" ]]; then echo "$SOURCE_CKPT"; return; fi
    local p; p=$(find "$SOURCE_WORK_DIR" -name "latest.pth" 2>/dev/null | head -1)
    [[ -z "$p" ]] && p=$(find "$SOURCE_WORK_DIR" -name "epoch_20.pth" 2>/dev/null | head -1)
    echo "$p"
}

[[ -f tools/run/train_seq3_episode.py ]] || { echo "ERROR: run from project root" >&2; exit 1; }
SOURCE_CKPT=$(find_source_ckpt)
[[ -n "$SOURCE_CKPT" && -f "$SOURCE_CKPT" ]] || { echo "ERROR: clear source ckpt not found" >&2; exit 1; }

run_version(){
    local vname="$1" cfg="$2"
    local wdir="$ROOT/$vname"
    local episode="$wdir/episode"
    local train_marker="$wdir/train_done.marker"
    local eval_marker="$wdir/eval_done.marker"

    echo "==========================================================="
    echo "VERSION: $vname"
    echo "  config           : $cfg"
    echo "  source_ckpt      : $SOURCE_CKPT"
    echo "  target_order     : ${TARGET_ORDER[*]}"
    echo "  epochs_per_stage : 3"
    echo "  eval             : night epoch $EVAL_EPOCH"
    echo "  GPU_IDS          : $GPU_IDS  (GPU_NUM=$GPU_NUM)"
    echo "  SAMPLES_PER_GPU  : $SAMPLES_PER_GPU"
    echo "  causal=False SPA=False roi_proto enabled=True"
    echo "  output_dir       : $wdir"
    echo "==========================================================="

    [[ -f "$cfg" ]] || { echo "ERROR: config not found: $cfg" >&2; exit 1; }

    # ---- train ----
    if [[ -f "$train_marker" && "$FORCE" != "1" ]]; then
        echo "SKIP train (marker exists): $train_marker"
    else
        [[ "$DRY_RUN" != "1" ]] && mkdir -p "$wdir/logs" "$wdir/state" "$wdir/summary" "$episode"
        [[ "$DRY_RUN" != "1" ]] && cp "$cfg" "$wdir/$(basename "$cfg")"
        local port; port=$(get_port)
        local cmd="CUDA_VISIBLE_DEVICES=${GPU_IDS} $PYTHON -m torch.distributed.launch \
            --nproc_per_node=${GPU_NUM} --master_addr=127.0.0.1 --master_port=${port} \
            tools/run/train_seq3_episode.py \
            --base-config ${cfg} --source-ckpt ${SOURCE_CKPT} \
            --target-order ${TARGET_ORDER[*]} --eval-domains night \
            --work-dir ${episode} --epochs-per-stage 3 \
            --samples-per-gpu ${SAMPLES_PER_GPU} --workers-per-gpu ${WORKERS_PER_GPU} \
            --launcher pytorch --no-validate > ${wdir}/logs/train_episode.log 2>&1"
        echo "[CMD] $cmd"
        if [[ "$DRY_RUN" != "1" ]]; then
            eval "$cmd" && date > "$train_marker" || { echo "TRAIN FAILED: $vname" >&2; exit 1; }
        fi
    fi

    # ---- eval night epoch3 (final stage = stage02_after_night) ----
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

V1A="$CFG_DIR/darth_seq_roi_proto_v1a_topscore_car_l0005_clear_overcast_night_3e.py"
V1B="$CFG_DIR/darth_seq_roi_proto_v1b_topscore_car_k200_topk5_l001_clear_overcast_night_3e.py"

run_version "v1a_topscore_car_l0005_clear_overcast_night_3e"          "$V1A"
run_version "v1b_topscore_car_k200_topk5_l001_clear_overcast_night_3e" "$V1B"

echo ""
echo "V1a/V1b quick-verify finished. Summarize with:"
echo "  $PYTHON tools/collect_roi_proto_ablation.py --v1ab"
[[ "$DRY_RUN" != "1" ]] && "$PYTHON" tools/collect_roi_proto_ablation.py --v1ab || true
