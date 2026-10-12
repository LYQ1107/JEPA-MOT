#!/usr/bin/env bash
# Threshold sweep for spa_pipeline_mild_4domain_5e, stage04_after_night epoch5.
# EVAL ONLY — no training, no model changes, no test.py modification.
# Mirrors baseline_threshold_sweep logic exactly; only checkpoint/output differ.
#
#   bash scripts/run_spa_pipeline_mild_threshold_sweep_night_4gpu.sh
#   DRY_RUN=1 bash scripts/run_spa_pipeline_mild_threshold_sweep_night_4gpu.sh
#   FORCE=1   bash scripts/run_spa_pipeline_mild_threshold_sweep_night_4gpu.sh
set -euo pipefail

GPU_IDS=${GPU_IDS:-"9,8,7,6"}
GPU_NUM=${GPU_NUM:-4}
TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU:-6}
DRY_RUN=${DRY_RUN:-0}
FORCE=${FORCE:-0}
RUN_P0=${RUN_P0:-1}
RUN_P1=${RUN_P1:-1}
RUN_P2=${RUN_P2:-1}
RUN_P3=${RUN_P3:-1}
RUN_P4=${RUN_P4:-1}
RUN_P5=${RUN_P5:-0}
RUN_P6=${RUN_P6:-0}
RUN_P7=${RUN_P7:-0}
PYTHON=${PYTHON:-/home/lwr/anaconda3/envs/DARTH/bin/python}

CKPT="work_dirs/bdd_4domain/spa_fix/spa_pipeline_mild_4domain_5e/episode/adapt/stage04_after_night/epoch_5.pth"
ECFG="configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_night.py"
OUT_ROOT="work_dirs/bdd_4domain/spa_fix/pipeline_mild_threshold_sweep/night"
ORIG_LOG="work_dirs/bdd_4domain/spa_fix/spa_pipeline_mild_4domain_5e/eval/stage04_after_night/epoch_5/night/eval.log"

[[ -f tools/run/test.py ]] || { echo "ERROR: run from project root" >&2; exit 1; }
[[ -f "$CKPT" ]]           || { echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; }
[[ -f "$ECFG" ]]           || { echo "ERROR: eval config not found: $ECFG" >&2; exit 1; }

get_port(){ "$PYTHON" -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()"; }

# Field mapping (confirmed from configs):
#   init_score_thr  -> model.tracker.init_score_thr   (night default: 0.20)
#   obj_score_thr   -> model.tracker.obj_score_thr    (night default: 0.20)
#   score_thr       -> model.detector.test_cfg.rcnn.score_thr (default: 0.05)

declare -A SWEEP_OPTS
SWEEP_OPTS[P0_default]=""
SWEEP_OPTS[P1_init030]="model.tracker.init_score_thr=0.30"
SWEEP_OPTS[P2_obj030]="model.tracker.obj_score_thr=0.30"
SWEEP_OPTS[P3_score015_obj030]="model.detector.test_cfg.rcnn.score_thr=0.15 model.tracker.obj_score_thr=0.30"
SWEEP_OPTS[P4_init030_obj030]="model.tracker.init_score_thr=0.30 model.tracker.obj_score_thr=0.30"
SWEEP_OPTS[P5_init025]="model.tracker.init_score_thr=0.25"
SWEEP_OPTS[P6_obj025]="model.tracker.obj_score_thr=0.25"
SWEEP_OPTS[P7_score010_obj030]="model.detector.test_cfg.rcnn.score_thr=0.10 model.tracker.obj_score_thr=0.30"

SWEEP_ORDER="P0_default P1_init030 P2_obj030 P3_score015_obj030 P4_init030_obj030 P5_init025 P6_obj025 P7_score010_obj030"

echo "==========================================================="
echo "spa_pipeline_mild threshold sweep (EVAL ONLY) — night"
echo "  checkpoint    : $CKPT"
echo "  eval config   : $ECFG"
echo "  output root   : $OUT_ROOT"
echo "  GPU_IDS       : $GPU_IDS  (GPU_NUM=$GPU_NUM)"
echo "  workers/GPU   : $TEST_WORKERS_PER_GPU"
echo "  DRY_RUN=$DRY_RUN  FORCE=$FORCE"
echo "  orig night default: init_score_thr=0.20 obj_score_thr=0.20 score_thr=0.05"
echo "  NOT touching original spa_pipeline_mild results"
echo "==========================================================="

for pname in $SWEEP_ORDER; do
    # skip if flag not set
    flag_var="RUN_${pname%%_*}"
    # map P0..P7 to RUN_P0..RUN_P7
    flag_var="RUN_$(echo $pname | sed 's/_.*$//')"
    [[ "${!flag_var:-0}" == "1" ]] || { echo "SKIP $pname (flag $flag_var=0)"; continue; }

    edir="$OUT_ROOT/$pname"
    marker="$edir/eval_done.marker"
    opts="${SWEEP_OPTS[$pname]}"

    if [[ -f "$marker" && "$FORCE" != "1" ]]; then
        echo "SKIP $pname (eval_done.marker exists)"
        continue
    fi

    [[ "$DRY_RUN" != "1" ]] && mkdir -p "$edir"

    if [[ "$DRY_RUN" != "1" ]]; then
        echo "$opts"      > "$edir/cfg_options.txt"
        echo "$CKPT"      > "$edir/checkpoint_path.txt"
        echo "$ECFG"      > "$edir/eval_config.txt"
    fi

    port=$(get_port)
    cmd="CUDA_VISIBLE_DEVICES=${GPU_IDS} \
${PYTHON} -m torch.distributed.launch \
    --nproc_per_node=${GPU_NUM} \
    --master_addr=127.0.0.1 \
    --master_port=${port} \
    tools/run/test.py ${ECFG} \
    --checkpoint ${CKPT} \
    --launcher pytorch \
    --work-dir ${edir} \
    --eval bbox track \
    --fuse-conv-bn \
    --cfg-options \
        data.workers_per_gpu=${TEST_WORKERS_PER_GPU} \
        cudnn_benchmark=True \
        ${opts} \
    > ${edir}/eval.log 2>&1"

    echo "[CMD] $pname  opts='${opts}'"
    echo "      $cmd"

    if [[ "$DRY_RUN" != "1" ]]; then
        if eval "$cmd"; then
            date > "$marker"
            echo "DONE $pname -> $marker"
        else
            echo "FAILED $pname — marker NOT written" >&2
            exit 1
        fi
    fi
done

echo ""
echo "All pipeline_mild threshold sweep evals complete."
echo "Collect results: python tools/collect_spa_pipeline_mild_threshold_sweep.py"
