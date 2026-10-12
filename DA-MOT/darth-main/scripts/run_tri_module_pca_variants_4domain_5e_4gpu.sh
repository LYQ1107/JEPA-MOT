#!/usr/bin/env bash
set -euo pipefail

[[ -f tools/run/train_seq3_episode.py ]] || { echo "ERROR: run from project root" >&2; exit 1; }

GPU_IDS=${GPU_IDS:-"9,8,7,6"}
GPU_NUM=${GPU_NUM:-4}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-16}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU:-6}
DRY_RUN=${DRY_RUN:-0}
FORCE=${FORCE:-0}
PYTHON=${PYTHON:-/home/lwr/anaconda3/envs/DARTH/bin/python}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

SOURCE_CKPT="work_dirs/bdd_4domain/source/clear_20e/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear/latest.pth"
ECFG_DIR="configs/mot/qdtrack/bdd_7domain"
OUT_ROOT="work_dirs/bdd_4domain/tri_module_pca_variants"
TARGET_ORDER="overcast rainy dawn_dusk night"
INIT_THR="model.tracker.init_score_thr=0.30"
DOMAINS=(overcast rainy dawn_dusk night)
STAGES=(stage01_after_overcast stage02_after_rainy stage03_after_dawn_dusk stage04_after_night)
STAGE_DOMAINS=(overcast rainy dawn_dusk night)

METHODS=(
  "spaP1_causalProtoCenter_roiV1Weak_4domain_5e"
  "spaP1_causalShrinkage_roiV1Weak_4domain_5e"
  "spaP1_causalProtoShrink_roiV1Weak_4domain_5e"
)
CONFIGS=(
  "configs/adapters/darth/bdd_4domain/module_combinations/darth_spaP1_causalProtoCenter_roiV1Weak_4domain_5e.py"
  "configs/adapters/darth/bdd_4domain/module_combinations/darth_spaP1_causalShrinkage_roiV1Weak_4domain_5e.py"
  "configs/adapters/darth/bdd_4domain/module_combinations/darth_spaP1_causalProtoShrink_roiV1Weak_4domain_5e.py"
)

[[ -f "$SOURCE_CKPT" ]] || { echo "ERROR: source ckpt not found: $SOURCE_CKPT" >&2; exit 1; }

get_port(){ "$PYTHON" -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()"; }

run_one(){
  local method="$1" cfg="$2"
  local root="$OUT_ROOT/$method"
  local episode="$root/episode"
  local marker="$root/train_done.marker"

  [[ -f "$cfg" ]] || { echo "ERROR: config not found: $cfg" >&2; exit 1; }
  echo "============================================================"
  echo "TRI-MODULE PCA VARIANT: $method"
  echo "  config=$cfg"
  echo "  work_dir=$root"
  echo "  GPU_IDS=$GPU_IDS GPU_NUM=$GPU_NUM"
  echo "  SAMPLES_PER_GPU=$SAMPLES_PER_GPU WORKERS_PER_GPU=$WORKERS_PER_GPU"
  echo "  TEST_WORKERS_PER_GPU=$TEST_WORKERS_PER_GPU"
  echo "  FORCE=$FORCE DRY_RUN=$DRY_RUN"
  echo "============================================================"

  if [[ -f "$marker" && "$FORCE" != "1" ]]; then
    echo "[SKIP] train marker exists: $marker"
  else
    [[ "$DRY_RUN" != "1" ]] && mkdir -p "$root/logs" "$episode" && cp "$cfg" "$root/$(basename "$cfg")"
    local port; port=$(get_port)
    local cmd="CUDA_VISIBLE_DEVICES=${GPU_IDS} ${PYTHON} -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_addr=${MASTER_ADDR} --master_port=${port} tools/run/train_seq3_episode.py --base-config ${cfg} --source-ckpt ${SOURCE_CKPT} --target-order ${TARGET_ORDER} --eval-domains ${TARGET_ORDER} --work-dir ${episode} --epochs-per-stage 5 --samples-per-gpu ${SAMPLES_PER_GPU} --workers-per-gpu ${WORKERS_PER_GPU} --launcher pytorch --no-validate > ${root}/logs/train_episode.log 2>&1"
    echo "[TRAIN CMD] $cmd"
    if [[ "$DRY_RUN" != "1" ]]; then
      eval "$cmd" && date > "$marker" || { echo "TRAIN FAILED: $method" >&2; exit 1; }
    fi
  fi

  for i in "${!STAGES[@]}"; do
    local stage="${STAGES[$i]}"
    local adapted="${STAGE_DOMAINS[$i]}"
    local ckpt="$episode/adapt/${stage}/epoch_5.pth"
    for dom in "${DOMAINS[@]}"; do
      local edir="$root/forgetting_eval/${stage}/epoch_5/${dom}"
      local eval_marker="$edir/eval_done.marker"
      local ecfg="$ECFG_DIR/qdtrack_faster-rcnn_r50_fpn_20e_bdd_${dom}.py"
      if [[ -f "$eval_marker" && "$FORCE" != "1" ]]; then
        echo "[SKIP] eval $method $stage -> $dom"
        continue
      fi
      mkdir -p "$edir"
      if [[ ! -f "$ckpt" && "$DRY_RUN" != "1" ]]; then
        echo "ERROR: ckpt not found: $ckpt" >&2; exit 1
      fi
      printf "%s\n" "$method" > "$edir/method.txt"
      printf "%s\n" "$stage" > "$edir/stage.txt"
      printf "%s\n" "$adapted" > "$edir/adapted_domain.txt"
      printf "%s\n" "$dom" > "$edir/eval_domain.txt"
      printf "%s\n" "$INIT_THR" > "$edir/cfg_options.txt"
      printf "%s\n" "$ckpt" > "$edir/checkpoint_path.txt"
      local port; port=$(get_port)
      local cmd="CUDA_VISIBLE_DEVICES=${GPU_IDS} ${PYTHON} -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_addr=${MASTER_ADDR} --master_port=${port} tools/run/test.py ${ecfg} --checkpoint ${ckpt} --launcher pytorch --work-dir ${edir} --eval bbox track --fuse-conv-bn --cfg-options data.workers_per_gpu=${TEST_WORKERS_PER_GPU} cudnn_benchmark=True ${INIT_THR} > ${edir}/eval.log 2>&1"
      printf "%s\n" "$cmd" > "$edir/eval_command.txt"
      echo "[EVAL CMD] $method $stage -> $dom"
      if [[ "$DRY_RUN" != "1" ]]; then
        eval "$cmd" && date > "$eval_marker" || { echo "EVAL FAILED: $method $stage $dom" >&2; exit 1; }
      fi
    done
  done
}

mkdir -p "$OUT_ROOT"
for idx in "${!METHODS[@]}"; do
  run_one "${METHODS[$idx]}" "${CONFIGS[$idx]}"
done

if [[ "$DRY_RUN" == "0" ]]; then
  "$PYTHON" tools/collect_tri_module_pca_variants.py
fi

echo "[DONE] tri-module PCA variants finished"
