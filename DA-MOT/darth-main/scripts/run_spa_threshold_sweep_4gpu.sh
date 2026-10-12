#!/usr/bin/env bash
# Scheme 1: test-time threshold sweep on the EXISTING spa_only_4domain_5e night
# checkpoint. NO training. Overrides tracker/detector thresholds via cfg-options
# only; does NOT modify test.py, datasets, or any other test setting.
# Evaluates night first (S0..S6). Four-domain eval only if night improves.
#
#   bash scripts/run_spa_threshold_sweep_4gpu.sh
#   DRY_RUN=1 bash scripts/run_spa_threshold_sweep_4gpu.sh
#   FORCE=1   bash scripts/run_spa_threshold_sweep_4gpu.sh
set -euo pipefail

GPU_IDS=${GPU_IDS:-"9,8,7,6"}
GPU_NUM=${GPU_NUM:-4}
TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU:-6}
DRY_RUN=${DRY_RUN:-0}
FORCE=${FORCE:-0}
PYTHON=${PYTHON:-/home/lwr/anaconda3/envs/DARTH/bin/python}

SPA_ROOT="work_dirs/bdd_4domain/spa_only_4domain_5e"
NIGHT_CKPT="$SPA_ROOT/episode/adapt/stage04_after_night/epoch_5.pth"
OUT_ROOT="work_dirs/bdd_4domain/spa_fix/threshold_sweep"
ECFG_DIR="configs/mot/qdtrack/bdd_7domain"

get_port(){ "$PYTHON" -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()"; }

[[ -f tools/run/test.py ]] || { echo "ERROR: run from project root" >&2; exit 1; }
[[ -f "$NIGHT_CKPT" ]] || { echo "ERROR: night ckpt not found: $NIGHT_CKPT" >&2; exit 1; }

# Sweep definitions: name -> extra cfg-options (threshold overrides only).
# Base night eval: rcnn.score_thr=0.05, init_score_thr=0.2, obj_score_thr=0.2.
declare -A SWEEP
SWEEP[s0_original]=""
SWEEP[s1_det_plus005]="model.detector.test_cfg.rcnn.score_thr=0.10"
SWEEP[s2_det_plus010]="model.detector.test_cfg.rcnn.score_thr=0.15"
SWEEP[s3_init_plus010]="model.tracker.init_score_thr=0.30"
SWEEP[s4_track_plus010]="model.tracker.obj_score_thr=0.30"
SWEEP[s5_det_init_plus010]="model.detector.test_cfg.rcnn.score_thr=0.15 model.tracker.init_score_thr=0.30"
SWEEP[s6_det_track_plus010]="model.detector.test_cfg.rcnn.score_thr=0.15 model.tracker.obj_score_thr=0.30"

SWEEP_ORDER="s0_original s1_det_plus005 s2_det_plus010 s3_init_plus010 s4_track_plus010 s5_det_init_plus010 s6_det_track_plus010"

run_eval(){
    local sname="$1" dom="$2" ckpt="$3"
    local edir="$OUT_ROOT/$sname/$dom"
    local ecfg="$ECFG_DIR/qdtrack_faster-rcnn_r50_fpn_20e_bdd_${dom}.py"
    local marker="$edir/eval_done.marker"
    if [[ -f "$marker" && "$FORCE" != "1" ]]; then echo "SKIP $sname/$dom (marker)"; return 0; fi
    [[ "$DRY_RUN" != "1" ]] && mkdir -p "$edir"
    if [[ ! -f "$ckpt" && "$DRY_RUN" != "1" ]]; then echo "ERROR: ckpt not found: $ckpt" >&2; exit 1; fi
    local port; port=$(get_port)
    local cmd="CUDA_VISIBLE_DEVICES=${GPU_IDS} $PYTHON -m torch.distributed.launch \
        --nproc_per_node=${GPU_NUM} --master_addr=127.0.0.1 --master_port=${port} \
        tools/run/test.py ${ecfg} --checkpoint ${ckpt} --launcher pytorch \
        --work-dir ${edir} --eval bbox track --fuse-conv-bn \
        --cfg-options data.workers_per_gpu=${TEST_WORKERS_PER_GPU} cudnn_benchmark=True ${SWEEP[$sname]} \
        > ${edir}/eval.log 2>&1"
    echo "[CMD] $cmd"
    if [[ "$DRY_RUN" != "1" ]]; then
        eval "$cmd" && date > "$marker" || { echo "EVAL FAILED: $sname/$dom" >&2; exit 1; }
    fi
}

echo "==========================================================="
echo "Scheme 1: threshold sweep (EVAL ONLY) on night ckpt"
echo "  ckpt      : $NIGHT_CKPT"
echo "  out       : $OUT_ROOT"
echo "  GPU_IDS   : $GPU_IDS (GPU_NUM=$GPU_NUM)  test workers=$TEST_WORKERS_PER_GPU"
echo "  DRY_RUN=$DRY_RUN FORCE=$FORCE"
echo "==========================================================="

# Step A: night only, all sweeps.
for sname in $SWEEP_ORDER; do
    run_eval "$sname" night "$NIGHT_CKPT"
done

# Step B (optional four-domain): only if SWEEP_4DOMAIN=1 is set explicitly by
# the user after inspecting night results. Uses each stage's own ckpt.
if [[ "${SWEEP_4DOMAIN:-0}" == "1" ]]; then
    declare -A STAGE_CKPT
    STAGE_CKPT[overcast]="$SPA_ROOT/episode/adapt/stage01_after_overcast/epoch_5.pth"
    STAGE_CKPT[rainy]="$SPA_ROOT/episode/adapt/stage02_after_rainy/epoch_5.pth"
    STAGE_CKPT[dawn_dusk]="$SPA_ROOT/episode/adapt/stage03_after_dawn_dusk/epoch_5.pth"
    STAGE_CKPT[night]="$SPA_ROOT/episode/adapt/stage04_after_night/epoch_5.pth"
    SWEEP_BEST=${SWEEP_BEST:-"s2_det_plus010"}
    for dom in overcast rainy dawn_dusk night; do
        run_eval "$SWEEP_BEST" "$dom" "${STAGE_CKPT[$dom]}"
    done
fi

echo ""
echo "Done. Sweep eval dirs under: $OUT_ROOT/<sName>/<domain>/"
