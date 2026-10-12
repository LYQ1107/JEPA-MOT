#!/usr/bin/env bash
# Baseline threshold sweep on baseline_4domain_5e night checkpoint (eval-only).
# Mirrors the SPA sweep exactly: same eval config, same test args, same
# cfg-options override pattern. Only the checkpoint and output dir differ.
#
# B0: default            (init=0.2 obj=0.2  — what the night eval config sets)
# B1: init_score_thr=0.30
# B2: obj_score_thr=0.30
# B3: score_thr=0.15 + obj_score_thr=0.30  (mirrors SPA s6, the best SPA combo)
#
#   bash scripts/run_baseline_threshold_sweep_night_4gpu.sh
#   DRY_RUN=1 bash scripts/run_baseline_threshold_sweep_night_4gpu.sh
#   FORCE=1   bash scripts/run_baseline_threshold_sweep_night_4gpu.sh
set -euo pipefail

GPU_IDS=${GPU_IDS:-"0,1,2,3"}
GPU_NUM=${GPU_NUM:-4}
TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU:-6}
DRY_RUN=${DRY_RUN:-0}
FORCE=${FORCE:-0}
PYTHON=${PYTHON:-/home/lwr/anaconda3/envs/DARTH/bin/python}

# ── fixed paths ───────────────────────────────────────────────────────────────
CKPT="work_dirs/bdd_4domain/task2_full_episode/episode/adapt/stage04_after_night/epoch_5.pth"
ECFG="configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_night.py"
OUT_ROOT="work_dirs/bdd_4domain/baseline_threshold_sweep/night"
ORIG_EVAL_LOG="work_dirs/bdd_4domain/task2_full_episode/eval/stage04_after_night/epoch_5/night/eval.log"

get_port(){ "$PYTHON" -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()"; }

# ── guards ────────────────────────────────────────────────────────────────────
[[ -f tools/run/test.py ]]  || { echo "ERROR: run from project root" >&2; exit 1; }
[[ -f "$CKPT" ]]            || { echo "ERROR: baseline ckpt not found: $CKPT" >&2; exit 1; }
[[ -f "$ECFG" ]]            || { echo "ERROR: eval config not found: $ECFG" >&2; exit 1; }
[[ -f "$ORIG_EVAL_LOG" ]]   || { echo "WARN: original eval.log not found: $ORIG_EVAL_LOG"; }

# ── sweep definitions ─────────────────────────────────────────────────────────
# name -> extra cfg-options (empty string = no overrides = baseline default)
# Night eval config already sets: init_score_thr=0.2  obj_score_thr=0.2
# The tracker fields are: model.tracker.init_score_thr  model.tracker.obj_score_thr
# The detector rcnn field is: model.detector.test_cfg.rcnn.score_thr
declare -A SWEEP_OPTS
SWEEP_OPTS[B0_default]=""
SWEEP_OPTS[B1_init030]="model.tracker.init_score_thr=0.30"
SWEEP_OPTS[B2_obj030]="model.tracker.obj_score_thr=0.30"
SWEEP_OPTS[B3_score015_obj030]="model.detector.test_cfg.rcnn.score_thr=0.15 model.tracker.obj_score_thr=0.30"

SWEEP_ORDER="B0_default B1_init030 B2_obj030 B3_score015_obj030"

# ── header ────────────────────────────────────────────────────────────────────
echo "==========================================================="
echo "Baseline threshold sweep (EVAL ONLY) — night"
echo "  checkpoint  : $CKPT"
echo "  eval config : $ECFG"
echo "  output root : $OUT_ROOT"
echo "  GPU_IDS     : $GPU_IDS  (GPU_NUM=$GPU_NUM)"
echo "  workers/GPU : $TEST_WORKERS_PER_GPU"
echo "  DRY_RUN=$DRY_RUN  FORCE=$FORCE"
echo "  NOTE: eval config night default tracker:"
echo "        init_score_thr=0.20  obj_score_thr=0.20"
echo "==========================================================="

# ── eval loop ─────────────────────────────────────────────────────────────────
for bname in $SWEEP_ORDER; do
    edir="$OUT_ROOT/$bname"
    marker="$edir/eval_done.marker"
    opts="${SWEEP_OPTS[$bname]}"

    if [[ -f "$marker" && "$FORCE" != "1" ]]; then
        echo "SKIP $bname (eval_done.marker exists)"
        continue
    fi

    [[ "$DRY_RUN" != "1" ]] && mkdir -p "$edir"

    # save cfg-options record
    if [[ "$DRY_RUN" != "1" ]]; then
        echo "$opts" > "$edir/cfg_options.txt"
        echo "$CKPT" > "$edir/checkpoint_path.txt"
        echo "$ECFG"  > "$edir/eval_config.txt"
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

    echo "[CMD] $bname  opts='${opts}'"
    echo "      $cmd"

    if [[ "$DRY_RUN" != "1" ]]; then
        if eval "$cmd"; then
            date > "$marker"
            echo "DONE $bname -> $marker"
        else
            echo "FAILED $bname — marker NOT written" >&2
            exit 1
        fi
    fi
done

echo ""
echo "All baseline threshold sweep evals complete."
echo "Results under: $OUT_ROOT/"
echo "Run: python tools/collect_baseline_threshold_sweep.py"
