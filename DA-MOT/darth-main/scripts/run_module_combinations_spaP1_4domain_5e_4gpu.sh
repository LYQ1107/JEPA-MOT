#!/usr/bin/env bash
# Module combination experiments: SPA-P1 + RoI-V1 / causal / both.
# All use pre-Normalize pipeline-level mild SPA training.
# Eval uses init_score_thr=0.30 (P1) via cfg-options.
#
#   bash scripts/run_module_combinations_spaP1_4domain_5e_4gpu.sh
#   DRY_RUN=1 bash scripts/run_module_combinations_spaP1_4domain_5e_4gpu.sh
#   RUN_SPAP1_ROIV1=0 RUN_SPAP1_CAUSAL=0 RUN_SPAP1_CAUSAL_ROIV1=1 \
#     bash scripts/run_module_combinations_spaP1_4domain_5e_4gpu.sh
set -euo pipefail
[[ -f tools/run/train_seq3_episode.py ]] || { echo "ERROR: run from project root" >&2; exit 1; }

GPU_IDS=${GPU_IDS:-"9,8,7,6"}
GPU_NUM=${GPU_NUM:-4}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-16}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU:-6}
DRY_RUN=${DRY_RUN:-0}
FORCE=${FORCE:-0}
RUN_SPAP1_ROIV1=${RUN_SPAP1_ROIV1:-1}
RUN_SPAP1_CAUSAL=${RUN_SPAP1_CAUSAL:-1}
RUN_SPAP1_CAUSAL_ROIV1=${RUN_SPAP1_CAUSAL_ROIV1:-0}
PYTHON=${PYTHON:-/home/lwr/anaconda3/envs/DARTH/bin/python}

SOURCE_CKPT="work_dirs/bdd_4domain/source/clear_20e/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear/latest.pth"
ECFG_DIR="configs/mot/qdtrack/bdd_7domain"
COMBO_ROOT="work_dirs/bdd_4domain/module_combinations"
TARGET_ORDER="overcast rainy dawn_dusk night"
INIT_THR="model.tracker.init_score_thr=0.30"

[[ -f "$SOURCE_CKPT" ]] || { echo "ERROR: source ckpt not found: $SOURCE_CKPT" >&2; exit 1; }

get_port(){ "$PYTHON" -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()"; }

echo "==========================================================="
echo "MODULE COMBINATIONS — SPA-P1 series"
echo "  source_ckpt : $SOURCE_CKPT"
echo "  GPU_IDS     : $GPU_IDS  (GPU_NUM=$GPU_NUM)"
echo "  SAMPLES/GPU : $SAMPLES_PER_GPU"
echo "  eval thr    : $INIT_THR"
echo "  DRY_RUN=$DRY_RUN  FORCE=$FORCE"
echo "  RUN_SPAP1_ROIV1=$RUN_SPAP1_ROIV1"
echo "  RUN_SPAP1_CAUSAL=$RUN_SPAP1_CAUSAL"
echo "  RUN_SPAP1_CAUSAL_ROIV1=$RUN_SPAP1_CAUSAL_ROIV1"
echo "==========================================================="

run_combo() {
    local EXP_NAME="$1"
    local CFG="$2"

    local ROOT="$COMBO_ROOT/$EXP_NAME"
    local EPISODE="$ROOT/episode"
    local TRAIN_MARKER="$ROOT/train_done.marker"

    [[ -f "$CFG" ]] || { echo "ERROR: config not found: $CFG" >&2; exit 1; }

    echo ""
    echo "--- COMBO: $EXP_NAME ---"
    echo "  config  : $CFG"
    echo "  work_dir: $ROOT"

    # ── train ──────────────────────────────────────────────────────────────
    if [[ -f "$TRAIN_MARKER" && "$FORCE" != "1" ]]; then
        echo "  SKIP train (marker exists)"
    else
        [[ "$DRY_RUN" != "1" ]] && mkdir -p "$ROOT/logs" "$EPISODE" \
            && cp "$CFG" "$ROOT/$(basename "$CFG")"
        local port; port=$(get_port)
        local cmd="CUDA_VISIBLE_DEVICES=${GPU_IDS} \
${PYTHON} -m torch.distributed.launch \
    --nproc_per_node=${GPU_NUM} \
    --master_addr=127.0.0.1 \
    --master_port=${port} \
    tools/run/train_seq3_episode.py \
    --base-config ${CFG} \
    --source-ckpt ${SOURCE_CKPT} \
    --target-order ${TARGET_ORDER} \
    --eval-domains ${TARGET_ORDER} \
    --work-dir ${EPISODE} \
    --epochs-per-stage 5 \
    --samples-per-gpu ${SAMPLES_PER_GPU} \
    --workers-per-gpu ${WORKERS_PER_GPU} \
    --launcher pytorch \
    --no-validate \
    > ${ROOT}/logs/train_episode.log 2>&1"
        echo "[TRAIN CMD] $cmd"
        if [[ "$DRY_RUN" != "1" ]]; then
            if eval "$cmd"; then
                date > "$TRAIN_MARKER"
                echo "  TRAIN DONE -> $TRAIN_MARKER"
            else
                echo "  TRAIN FAILED: $EXP_NAME" >&2
                exit 1
            fi
        fi
    fi

    # ── eval each stage at epoch 5 with P1 threshold ───────────────────────
    local idx=0
    for dom in $TARGET_ORDER; do
        idx=$((idx+1))
        local STAGE; STAGE=$(printf "stage%02d_after_%s" "$idx" "$dom")
        local CKPT="$EPISODE/adapt/${STAGE}/epoch_5.pth"
        local EDIR="$ROOT/eval/${STAGE}/epoch_5/${dom}"
        local ECFG="$ECFG_DIR/qdtrack_faster-rcnn_r50_fpn_20e_bdd_${dom}.py"
        local EVAL_MARKER="$EDIR/eval_done.marker"

        if [[ -f "$EVAL_MARKER" && "$FORCE" != "1" ]]; then
            echo "  SKIP eval $STAGE/$dom (marker exists)"
            continue
        fi

        [[ "$DRY_RUN" != "1" ]] && mkdir -p "$EDIR"

        if [[ ! -f "$CKPT" && "$DRY_RUN" != "1" ]]; then
            echo "  ERROR: ckpt not found: $CKPT" >&2; exit 1
        fi

        local port; port=$(get_port)
        local cmd="CUDA_VISIBLE_DEVICES=${GPU_IDS} \
${PYTHON} -m torch.distributed.launch \
    --nproc_per_node=${GPU_NUM} \
    --master_addr=127.0.0.1 \
    --master_port=${port} \
    tools/run/test.py ${ECFG} \
    --checkpoint ${CKPT} \
    --launcher pytorch \
    --work-dir ${EDIR} \
    --eval bbox track \
    --fuse-conv-bn \
    --cfg-options \
        data.workers_per_gpu=${TEST_WORKERS_PER_GPU} \
        cudnn_benchmark=True \
        ${INIT_THR} \
    > ${EDIR}/eval.log 2>&1"
        echo "[EVAL CMD] $STAGE/$dom  thr=$INIT_THR"
        echo "           $cmd"
        if [[ "$DRY_RUN" != "1" ]]; then
            if eval "$cmd"; then
                date > "$EVAL_MARKER"
                echo "  EVAL DONE $STAGE/$dom -> $EVAL_MARKER"
            else
                echo "  EVAL FAILED: $STAGE/$dom" >&2
                exit 1
            fi
        fi
    done
}

[[ "$DRY_RUN" != "1" ]] && mkdir -p "$COMBO_ROOT"

[[ "$RUN_SPAP1_ROIV1" == "1" ]] && \
    run_combo "spaP1_roiV1_4domain_5e" \
              "configs/adapters/darth/bdd_4domain/module_combinations/darth_spaP1_roiV1_4domain_5e.py"

[[ "$RUN_SPAP1_CAUSAL" == "1" ]] && \
    run_combo "spaP1_causalOld_4domain_5e" \
              "configs/adapters/darth/bdd_4domain/module_combinations/darth_spaP1_causalOld_4domain_5e.py"

[[ "$RUN_SPAP1_CAUSAL_ROIV1" == "1" ]] && \
    run_combo "spaP1_causalOld_roiV1_4domain_5e" \
              "configs/adapters/darth/bdd_4domain/module_combinations/darth_spaP1_causalOld_roiV1_4domain_5e.py"

echo ""
echo "All requested combos complete."
echo "Collect: python tools/collect_module_components_summary.py"
