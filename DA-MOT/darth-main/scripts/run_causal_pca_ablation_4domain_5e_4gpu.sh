#!/usr/bin/env bash
# Causal PCA ablation: V1 (cov_ema) -> V2 (car_weighted_cov_ema) ->
#                      V3 (pairwise_residual), all 4-domain 5e on GPU 0-3.
# No SPA, no RoI proto, no changes to inference tracker.
#
#   bash scripts/run_causal_pca_ablation_4domain_5e_4gpu.sh
#   DRY_RUN=1 bash scripts/run_causal_pca_ablation_4domain_5e_4gpu.sh
#   RUN_V1=1 RUN_V2=1 RUN_V3=0 bash ...
set -euo pipefail

GPU_IDS=${GPU_IDS:-"0,1,2,3"}
GPU_NUM=${GPU_NUM:-4}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-16}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU:-6}
DRY_RUN=${DRY_RUN:-0}
FORCE=${FORCE:-0}
RUN_V1=${RUN_V1:-1}
RUN_V2=${RUN_V2:-1}
RUN_V3=${RUN_V3:-1}
PYTHON=${PYTHON:-/home/lwr/anaconda3/envs/DARTH/bin/python}

SOURCE_CKPT="work_dirs/bdd_4domain/source/clear_20e/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear/latest.pth"
ECFG_DIR="configs/mot/qdtrack/bdd_7domain"
OUT_ROOT="work_dirs/bdd_4domain/causal_pca_ablation"
TARGET_ORDER="overcast rainy dawn_dusk night"

[[ -f tools/run/train_seq3_episode.py ]] || { echo "ERROR: run from project root" >&2; exit 1; }
[[ -f "$SOURCE_CKPT" ]]                 || { echo "ERROR: source ckpt not found: $SOURCE_CKPT" >&2; exit 1; }

get_port(){ "$PYTHON" -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()"; }

run_one(){
    local vname="$1" cfg="$2"
    local wdir="$OUT_ROOT/${vname}"
    local episode_dir="$wdir/episode"
    local train_marker="$wdir/train_done.marker"

    echo ""
    echo "==========================================================="
    echo "CAUSAL PCA  version : ${vname}"
    echo "  config     : ${cfg}"
    echo "  source_ckpt: ${SOURCE_CKPT}"
    echo "  work_dir   : ${wdir}"
    echo "  GPU_IDS    : ${GPU_IDS}  GPU_NUM=${GPU_NUM}  SAMPLES/GPU=${SAMPLES_PER_GPU}"
    echo "  DRY_RUN=${DRY_RUN}  FORCE=${FORCE}"
    echo "==========================================================="

    # ── train ────────────────────────────────────────────────────────────────
    if [[ -f "$train_marker" && "$FORCE" != "1" ]]; then
        echo "SKIP train (marker exists)"
    else
        [[ "$DRY_RUN" != "1" ]] && mkdir -p "$wdir/logs" "$episode_dir"
        local port; port=$(get_port)
        local tcmd="CUDA_VISIBLE_DEVICES=${GPU_IDS} \
${PYTHON} -m torch.distributed.launch \
    --nproc_per_node=${GPU_NUM} \
    --master_addr=127.0.0.1 \
    --master_port=${port} \
    tools/run/train_seq3_episode.py \
    --base-config ${cfg} \
    --source-ckpt ${SOURCE_CKPT} \
    --target-order ${TARGET_ORDER} \
    --eval-domains ${TARGET_ORDER} \
    --work-dir ${episode_dir} \
    --epochs-per-stage 5 \
    --samples-per-gpu ${SAMPLES_PER_GPU} \
    --workers-per-gpu ${WORKERS_PER_GPU} \
    --launcher pytorch \
    --no-validate \
    > ${wdir}/logs/train_episode.log 2>&1"
        echo "[TRAIN CMD] $tcmd"
        if [[ "$DRY_RUN" != "1" ]]; then
            eval "$tcmd" && date > "$train_marker" \
                || { echo "TRAIN FAILED: ${vname}" >&2; exit 1; }
        fi
    fi

    # ── eval each target domain at its own stage epoch 5 ────────────────────
    local idx=0
    for dom in $TARGET_ORDER; do
        idx=$((idx+1))
        local stage; stage=$(printf "stage%02d_after_%s" "$idx" "$dom")
        local ckpt="${episode_dir}/adapt/${stage}/epoch_5.pth"
        local edir="${wdir}/eval/${stage}/epoch_5/${dom}"
        local eval_marker="${edir}/eval_done.marker"
        local ecfg="${ECFG_DIR}/qdtrack_faster-rcnn_r50_fpn_20e_bdd_${dom}.py"

        if [[ -f "$eval_marker" && "$FORCE" != "1" ]]; then
            echo "SKIP eval ${stage}/${dom} (marker exists)"
            continue
        fi
        [[ "$DRY_RUN" != "1" ]] && mkdir -p "$edir"
        if [[ ! -f "$ckpt" && "$DRY_RUN" != "1" ]]; then
            echo "ERROR: ckpt not found: $ckpt" >&2; exit 1
        fi
        local port; port=$(get_port)
        local ecmd="CUDA_VISIBLE_DEVICES=${GPU_IDS} \
${PYTHON} -m torch.distributed.launch \
    --nproc_per_node=${GPU_NUM} \
    --master_addr=127.0.0.1 \
    --master_port=${port} \
    tools/run/test.py ${ecfg} \
    --checkpoint ${ckpt} \
    --launcher pytorch \
    --work-dir ${edir} \
    --eval bbox track \
    --fuse-conv-bn \
    --cfg-options \
        data.workers_per_gpu=${TEST_WORKERS_PER_GPU} \
        cudnn_benchmark=True \
    > ${edir}/eval.log 2>&1"
        echo "[EVAL CMD] ${stage}/${dom}"
        echo "      $ecmd"
        if [[ "$DRY_RUN" != "1" ]]; then
            eval "$ecmd" && date > "$eval_marker" \
                || { echo "EVAL FAILED: ${vname}/${stage}/${dom}" >&2; exit 1; }
        fi
    done
    echo "Done: ${vname}"
}

mkdir -p "$OUT_ROOT"

[[ "$RUN_V1" == "1" ]] && run_one \
    "v1_cov_ema_4domain_5e" \
    "configs/adapters/darth/bdd_4domain/causal_pca_ablation/darth_seq_causal_pca_v1_cov_ema_4domain_5e.py"

[[ "$RUN_V2" == "1" ]] && run_one \
    "v2_car_weighted_cov_ema_4domain_5e" \
    "configs/adapters/darth/bdd_4domain/causal_pca_ablation/darth_seq_causal_pca_v2_car_weighted_cov_ema_4domain_5e.py"

[[ "$RUN_V3" == "1" ]] && run_one \
    "v3_pairwise_residual_4domain_5e" \
    "configs/adapters/darth/bdd_4domain/causal_pca_ablation/darth_seq_causal_pca_v3_pairwise_residual_4domain_5e.py"

echo ""
echo "########## CAUSAL PCA ABLATION COMPLETE $(date '+%F %T') ##########"
echo "Collect results:  python tools/collect_causal_pca_ablation.py"
