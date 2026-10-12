#!/usr/bin/env bash
set -euo pipefail

# Task 1: Rerun all night-related baseline experiments
# Because night dataset has changed, all rows with source==night or target==night must be regenerated

GPU_IDS=${GPU_IDS:-"9,8,7,6,5,4,3,2,1,0"}
GPU_NUM=${GPU_NUM:-10}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-10}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
# Test-only dataloader workers (higher = faster eval; does NOT affect accuracy)
TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU:-8}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}
ONLY_NIGHT_SOURCE=${ONLY_NIGHT_SOURCE:-0}
SKIP_TRAIN=${SKIP_TRAIN:-0}
ALLOW_OLD_NIGHT_SOURCE=${ALLOW_OLD_NIGHT_SOURCE:-0}

OLD_RESULTS_CSV=${OLD_RESULTS_CSV:-baseline_darth_20260624/work_dirs/bdd_7domain/summaries/complete_metrics_all_epochs.csv}
ROOT_WORK_DIR="work_dirs/bdd_baseline_rerun/night_related_all"
DOMAINS=("clear" "dawn_dusk" "night" "overcast" "partly_cloudy" "rainy" "snowy")
TTA_EVAL_EPOCHS=(6 12 20)

mkdir -p "$ROOT_WORK_DIR"/{source,source_only_eval,tta,tta_eval,state,summaries}

get_port() { python3 -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()"; }

find_source_ckpt() {
  local domain=$1
  local ckpt=""
  if [[ "$domain" == "night" && "$ALLOW_OLD_NIGHT_SOURCE" != "1" ]]; then
    # Check direct path first
    ckpt="$ROOT_WORK_DIR/source/night_20e/epoch_20.pth"
    [[ -f "$ckpt" ]] && echo "$ckpt" && return
    # Check in subdirectories (mmtrack creates timestamped subdirs)
    ckpt=$(find "$ROOT_WORK_DIR/source/night_20e" -name "epoch_20.pth" -type f 2>/dev/null | head -1 || true)
    [[ -n "$ckpt" ]] && echo "$ckpt" && return
    echo "" && return
  fi
  local candidates=(
    "$ROOT_WORK_DIR/source/${domain}_20e/epoch_20.pth"
    "baseline_darth_20260624/work_dirs/bdd_7domain/source/${domain}/qdtrack_20e/epoch_20.pth"
    "work_dirs/bdd_7domain/source/${domain}/qdtrack_20e/epoch_20.pth"
  )
  for c in "${candidates[@]}"; do [[ -f "$c" ]] && echo "$c" && return; done
  # Search in subdirectories (handles timestamped mmtrack output dirs)
  local found=""
  found=$(find "$ROOT_WORK_DIR/source/${domain}_20e" -name "epoch_20.pth" -type f 2>/dev/null | head -1 || true)
  [[ -n "$found" ]] && echo "$found" && return
  found=$(find "baseline_darth_20260624/work_dirs/bdd_7domain/source/${domain}" -name "epoch_20.pth" -type f 2>/dev/null | head -1 || true)
  [[ -n "$found" ]] && echo "$found" && return
  echo ""
}

find_source_train_cfg() {
  local domain=$1
  local cfg="configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_${domain}.py"
  [[ -f "$cfg" ]] && echo "$cfg" && return
  echo ""
}

find_tta_cfg() {
  local domain=$1
  local cfg="configs/adapters/darth/bdd_7domain/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_${domain}.py"
  [[ -f "$cfg" ]] && echo "$cfg" && return
  echo ""
}

find_eval_cfg() {
  local domain=$1
  find_source_train_cfg "$domain"
}

echo "========================================="
echo "Task 1: Rerun night-related baselines"
echo "========================================="
echo "GPU_IDS = $GPU_IDS"
echo "GPU_NUM = $GPU_NUM"
echo "SAMPLES_PER_GPU = $SAMPLES_PER_GPU"
echo "WORKERS_PER_GPU = $WORKERS_PER_GPU"
echo "TEST_WORKERS_PER_GPU = $TEST_WORKERS_PER_GPU"
echo "OLD_RESULTS_CSV = $OLD_RESULTS_CSV"
echo "ROOT_WORK_DIR = $ROOT_WORK_DIR"
echo "FORCE = $FORCE"
echo "DRY_RUN = $DRY_RUN"
echo "ONLY_NIGHT_SOURCE = $ONLY_NIGHT_SOURCE"
echo "SKIP_TRAIN = $SKIP_TRAIN"
echo ""

if [[ ! -f "$OLD_RESULTS_CSV" ]]; then
  echo "ERROR: OLD_RESULTS_CSV not found: $OLD_RESULTS_CSV" >&2
  exit 1
fi

python3 - "$OLD_RESULTS_CSV" << 'PLAN_EOF'
import pandas as pd, sys
csv=sys.argv[1]
df=pd.read_csv(csv)
night=df[(df['source']=='night')|(df['target']=='night')]
print(f"Old table: {len(df)} rows")
print(f"Night-related: {len(night)} rows")
print(f"  source_only: {len(night[night['mode']=='source_only'])}")
print(f"  source_self_eval: {len(night[night['mode']=='source_self_eval'])}")
print(f"  tta_eval: {len(night[night['mode']=='tta_eval'])}")
PLAN_EOF

# Night source training
NIGHT_SRC_DIR="$ROOT_WORK_DIR/source/night_20e"
NIGHT_SRC_CKPT=$(find_source_ckpt "night")
NIGHT_SRC_DONE="$NIGHT_SRC_DIR/train_done.marker"
NIGHT_SRC_CFG=$(find_source_train_cfg "night")

if [[ -z "$NIGHT_SRC_CFG" ]]; then
  echo "ERROR: night source config not found" >&2
  exit 1
fi

echo "Night source config: $NIGHT_SRC_CFG"

if [[ "$SKIP_TRAIN" != "1" ]]; then
  if [[ -f "$NIGHT_SRC_DONE" && -f "$NIGHT_SRC_CKPT" && "$FORCE" != "1" ]]; then
    echo "SKIP night source training: $NIGHT_SRC_DONE exists"
  else
    echo "RUN night source training: 20 epochs"
    if [[ "$DRY_RUN" != "1" ]]; then
      mkdir -p "$NIGHT_SRC_DIR"
      PORT=$(get_port)
      CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch \
        --nproc_per_node=$GPU_NUM --master_addr=127.0.0.1 --master_port=$PORT \
        tools/run/train.py "$NIGHT_SRC_CFG" --launcher pytorch --work-dir "$NIGHT_SRC_DIR" --no-version \
        --cfg-options total_epochs=20 runner.max_epochs=20 checkpoint_config.interval=1 \
        checkpoint_config.max_keep_ckpts=-1 data.samples_per_gpu=$SAMPLES_PER_GPU \
        data.workers_per_gpu=$WORKERS_PER_GPU > "$NIGHT_SRC_DIR/train.log" 2>&1
      date > "$NIGHT_SRC_DONE"
    fi
  fi
fi

if [[ "$ONLY_NIGHT_SOURCE" == "1" ]]; then
  echo "ONLY_NIGHT_SOURCE=1, stopping after night source training"
  exit 0
fi

# Source-only eval
SRCONLY_PAIRS=(
  "clear night"
  "dawn_dusk night"
  "overcast night"
  "partly_cloudy night"
  "rainy night"
  "snowy night"
  "night clear"
  "night dawn_dusk"
  "night night"
  "night overcast"
  "night partly_cloudy"
  "night rainy"
  "night snowy"
)

echo ""
echo "Source-only eval: ${#SRCONLY_PAIRS[@]} pairs"

for pair in "${SRCONLY_PAIRS[@]}"; do
  read -r src tgt <<< "$pair"
  ckpt=$(find_source_ckpt "$src")
  if [[ -z "$ckpt" ]]; then
    echo "ERROR: source checkpoint not found for $src" >&2
    [[ "$DRY_RUN" == "1" ]] && continue || exit 1
  fi
  eval_dir="$ROOT_WORK_DIR/source_only_eval/source_${src}_epoch20/${tgt}"
  eval_done="$eval_dir/eval_done.marker"
  if [[ -f "$eval_done" && "$FORCE" != "1" ]]; then
    echo "SKIP source-only $src→$tgt"
    continue
  fi
  eval_cfg=$(find_eval_cfg "$tgt")
  if [[ -z "$eval_cfg" ]]; then
    echo "ERROR: eval config not found for $tgt" >&2
    [[ "$DRY_RUN" == "1" ]] && continue || exit 1
  fi
  echo "RUN source-only $src→$tgt"
  if [[ "$DRY_RUN" != "1" && "$SKIP_TRAIN" != "1" ]]; then
    mkdir -p "$eval_dir"
    PORT=$(get_port)
    CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch \
      --nproc_per_node=$GPU_NUM --master_addr=127.0.0.1 --master_port=$PORT \
      tools/run/test.py "$eval_cfg" --checkpoint "$ckpt" --launcher pytorch \
      --work-dir "$eval_dir" --eval bbox track --fuse-conv-bn \
      --cfg-options data.workers_per_gpu=$TEST_WORKERS_PER_GPU cudnn_benchmark=True > "$eval_dir/eval.log" 2>&1
    date > "$eval_done"
  fi
done

# TTA training
TTA_PAIRS=(
  "clear night"
  "dawn_dusk night"
  "overcast night"
  "partly_cloudy night"
  "rainy night"
  "snowy night"
  "night clear"
  "night dawn_dusk"
  "night overcast"
  "night partly_cloudy"
  "night rainy"
  "night snowy"
)

echo ""
echo "TTA training: ${#TTA_PAIRS[@]} pairs × 20 epochs"

for pair in "${TTA_PAIRS[@]}"; do
  read -r src tgt <<< "$pair"
  src_ckpt=$(find_source_ckpt "$src")
  if [[ -z "$src_ckpt" ]]; then
    echo "ERROR: source checkpoint not found for $src" >&2
    [[ "$DRY_RUN" == "1" ]] && continue || exit 1
  fi
  tta_dir="$ROOT_WORK_DIR/tta/${src}_to_${tgt}_20e"
  tta_done="$tta_dir/train_done.marker"
  tta_ckpt="$tta_dir/epoch_20.pth"
  if [[ ! -f "$tta_ckpt" ]]; then
    tta_ckpt=$(find "$tta_dir" -name "epoch_20.pth" -type f 2>/dev/null | head -1 || true)
  fi
  if [[ -f "$tta_done" && -n "$tta_ckpt" && "$FORCE" != "1" ]]; then
    echo "SKIP TTA $src→$tgt"
    continue
  fi
  tta_cfg=$(find_tta_cfg "$tgt")
  if [[ -z "$tta_cfg" ]]; then
    echo "ERROR: TTA config not found for $tgt" >&2
    [[ "$DRY_RUN" == "1" ]] && continue || exit 1
  fi
  echo "RUN TTA $src→$tgt"
  if [[ "$DRY_RUN" != "1" && "$SKIP_TRAIN" != "1" ]]; then
    mkdir -p "$tta_dir"
    PORT=$(get_port)
    CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch \
      --nproc_per_node=$GPU_NUM --master_addr=127.0.0.1 --master_port=$PORT \
      tools/run/train.py "$tta_cfg" --launcher pytorch --work-dir "$tta_dir" --no-version \
      --cfg-options total_epochs=20 runner.max_epochs=20 checkpoint_config.interval=1 \
      checkpoint_config.max_keep_ckpts=-1 evaluation.interval=21 \
      find_unused_parameters=True \
      load_from="$src_ckpt" resume_from=None \
      data.samples_per_gpu=$SAMPLES_PER_GPU data.workers_per_gpu=$WORKERS_PER_GPU \
      > "$tta_dir/train.log" 2>&1
    date > "$tta_done"
  fi
done

# TTA eval
echo ""
echo "TTA eval: ${#TTA_PAIRS[@]} pairs × ${#TTA_EVAL_EPOCHS[@]} epochs"

for pair in "${TTA_PAIRS[@]}"; do
  read -r src tgt <<< "$pair"
  tta_dir="$ROOT_WORK_DIR/tta/${src}_to_${tgt}_20e"
  for ep in "${TTA_EVAL_EPOCHS[@]}"; do
    ckpt="$tta_dir/epoch_${ep}.pth"
    eval_dir="$ROOT_WORK_DIR/tta_eval/${src}_to_${tgt}/epoch$(printf '%02d' $ep)/${tgt}"
    eval_done="$eval_dir/eval_done.marker"
    if [[ -f "$eval_done" && "$FORCE" != "1" ]]; then
      echo "SKIP TTA eval $src→$tgt epoch$ep"
      continue
    fi
    if [[ ! -f "$ckpt" ]]; then
      echo "WARN: TTA checkpoint not found: $ckpt"
      [[ "$DRY_RUN" == "1" ]] && continue || continue
    fi
    eval_cfg=$(find_eval_cfg "$tgt")
    if [[ -z "$eval_cfg" ]]; then
      echo "ERROR: eval config not found for $tgt" >&2
      [[ "$DRY_RUN" == "1" ]] && continue || exit 1
    fi
    echo "RUN TTA eval $src→$tgt epoch$ep"
    if [[ "$DRY_RUN" != "1" ]]; then
      mkdir -p "$eval_dir"
      PORT=$(get_port)
      CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch \
        --nproc_per_node=$GPU_NUM --master_addr=127.0.0.1 --master_port=$PORT \
        tools/run/test.py "$eval_cfg" --checkpoint "$ckpt" --launcher pytorch \
        --work-dir "$eval_dir" --eval bbox track --fuse-conv-bn \
        --cfg-options data.workers_per_gpu=$TEST_WORKERS_PER_GPU cudnn_benchmark=True > "$eval_dir/eval.log" 2>&1
      date > "$eval_done"
    fi
  done
done

# Collect results
echo ""
echo "Collecting results..."
if [[ "$DRY_RUN" != "1" ]]; then
  python tools/collect_bdd_night_related_baselines.py "$ROOT_WORK_DIR" "$OLD_RESULTS_CSV"
fi

echo ""
echo "Done. Results in: $ROOT_WORK_DIR/summaries/"
