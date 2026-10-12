#!/usr/bin/env bash
set -euo pipefail

SOURCE_DOMAIN=clear
TARGET_ORDER=("dawn_dusk" "night" "partly_cloudy")
EVAL_DOMAINS=("dawn_dusk" "night" "partly_cloudy")
ADAPT_EPOCHS=20
GPU_IDS=${GPU_IDS:-"9,8,7,6,5,4,3,2,1,0"}
GPU_NUM=${GPU_NUM:-10}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-14}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU:-8}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}
RUN_EVAL=${RUN_EVAL:-1}

BASE_CFG=${BASE_CFG:-"configs/adapters/darth/bdd_7domain/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_dawn_dusk.py"}
ROOT_WORK_DIR=${ROOT_WORK_DIR:-"work_dirs/bdd_seq3_episode/clear_to_dawn_dusk_night_partly_cloudy"}
SOURCE_CKPT=${SOURCE_CKPT:-""}

get_port() { python3 -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()"; }
run_or_print() { echo "[CMD] $*"; if [[ "$DRY_RUN" != "1" ]]; then eval "$*"; fi; }
find_source_ckpt() {
  if [[ -n "$SOURCE_CKPT" && -f "$SOURCE_CKPT" ]]; then echo "$SOURCE_CKPT"; return; fi
  local candidates=(
    "baseline_darth_20260624/work_dirs/bdd_7domain/source/clear/qdtrack_20e/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear/20260612_173506/epoch_20.pth"
    "work_dirs/bdd_7domain/source/clear/qdtrack_20e/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear/20260612_173506/epoch_20.pth"
  )
  for c in "${candidates[@]}"; do [[ -f "$c" ]] && echo "$c" && return; done
  local found
  found=$(find baseline_darth_20260624 work_dirs -path "*/source/clear/*/epoch_20.pth" 2>/dev/null | head -1 || true)
  [[ -n "$found" ]] && echo "$found" && return
  echo ""
}

if [[ ! -f tools/run/train_seq3_episode.py || ! -f tools/run/test.py ]]; then
  echo "ERROR: run from DARTH project root" >&2; exit 1
fi
if [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" != "DARTH" ]]; then
  echo "WARNING: expected conda env DARTH, got ${CONDA_DEFAULT_ENV}" >&2
fi
SOURCE_CKPT=$(find_source_ckpt)
if [[ -z "$SOURCE_CKPT" || ! -f "$SOURCE_CKPT" ]]; then echo "ERROR: clear source checkpoint not found" >&2; exit 1; fi
if [[ ! -f "$BASE_CFG" ]]; then echo "ERROR: base config not found: $BASE_CFG" >&2; exit 1; fi
for d in "${TARGET_ORDER[@]}"; do
  [[ -f "configs/adapters/darth/bdd_7domain/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_${d}.py" ]] || { echo "missing adapt cfg $d"; exit 1; }
done
for d in "${EVAL_DOMAINS[@]}"; do
  [[ -f "configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_${d}.py" ]] || { echo "missing eval cfg $d"; exit 1; }
done
mkdir -p "$ROOT_WORK_DIR" "$ROOT_WORK_DIR/logs"

cat <<EOF
========================================
Continuous DARTH seq3 episode
source checkpoint: $SOURCE_CKPT
base config: $BASE_CFG
target order: ${TARGET_ORDER[*]}
eval domains: ${EVAL_DOMAINS[*]}
GPU_IDS = $GPU_IDS
GPU_NUM = $GPU_NUM
SAMPLES_PER_GPU = $SAMPLES_PER_GPU
WORKERS_PER_GPU = $WORKERS_PER_GPU (train)
TEST_WORKERS_PER_GPU = $TEST_WORKERS_PER_GPU (eval)
work dir: $ROOT_WORK_DIR
DRY_RUN = $DRY_RUN
========================================
EOF

TRAIN_LOG="$ROOT_WORK_DIR/logs/train_episode.log"
TRAIN_DONE="$ROOT_WORK_DIR/state/train_episode.done"
PORT=$(get_port)
TRAIN_CMD="CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_addr=127.0.0.1 --master_port=${PORT} tools/run/train_seq3_episode.py --base-config ${BASE_CFG} --source-ckpt ${SOURCE_CKPT} --target-order ${TARGET_ORDER[*]} --eval-domains ${EVAL_DOMAINS[*]} --work-dir ${ROOT_WORK_DIR} --epochs-per-stage ${ADAPT_EPOCHS} --samples-per-gpu ${SAMPLES_PER_GPU} --workers-per-gpu ${WORKERS_PER_GPU} --launcher pytorch --no-validate > ${TRAIN_LOG} 2>&1"

echo "Stage snapshot paths:"
echo "  $ROOT_WORK_DIR/stage_snapshots/stage00_source_only.pth"
for i in 1 2 3; do idx=$(printf "%02d" "$i"); t=${TARGET_ORDER[$((i-1))]}; echo "  $ROOT_WORK_DIR/stage_snapshots/stage${idx}_after_${t}.pth"; done

if [[ -f "$TRAIN_DONE" && "$FORCE" != "1" ]]; then
  echo "SKIP training episode: $TRAIN_DONE exists"
else
  run_or_print "$TRAIN_CMD"
  if [[ "$DRY_RUN" != "1" ]]; then mkdir -p "$ROOT_WORK_DIR/state"; date > "$TRAIN_DONE"; fi
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  STAGE_NAMES=("stage00_source_only" "stage01_after_dawn_dusk" "stage02_after_night" "stage03_after_partly_cloudy")
  for stage in "${STAGE_NAMES[@]}"; do
    ckpt="$ROOT_WORK_DIR/stage_snapshots/${stage}.pth"
    echo "Evaluating snapshot: $stage -> $ckpt"
    for d in "${EVAL_DOMAINS[@]}"; do
      eval_dir="$ROOT_WORK_DIR/eval/${stage}/${d}"
      eval_log="$eval_dir/eval.log"
      eval_done="$eval_dir/eval.done"
      mkdir -p "$eval_dir"
      if [[ -f "$eval_done" && "$FORCE" != "1" ]]; then echo "  SKIP $stage/$d"; continue; fi
      eval_cfg="configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_${d}.py"
      PORT=$(get_port)
      EVAL_CMD="CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_addr=127.0.0.1 --master_port=${PORT} tools/run/test.py ${eval_cfg} --checkpoint ${ckpt} --launcher pytorch --work-dir ${eval_dir} --eval bbox track --fuse-conv-bn --cfg-options data.workers_per_gpu=${TEST_WORKERS_PER_GPU} cudnn_benchmark=True > ${eval_log} 2>&1"
      run_or_print "$EVAL_CMD"
      if [[ "$DRY_RUN" != "1" ]]; then date > "$eval_done"; fi
    done
  done
fi

SUMMARY_CMD="python tools/collect_bdd_seq3_results.py ${ROOT_WORK_DIR}"
echo "Summary command: $SUMMARY_CMD"
run_or_print "$SUMMARY_CMD"

cat <<EOF
Done.
Summary paths:
  $ROOT_WORK_DIR/summaries/seq3_results_long.csv
  $ROOT_WORK_DIR/summaries/seq3_results_summary.md
  $ROOT_WORK_DIR/summaries/seq3_matrix_mota.csv
EOF
