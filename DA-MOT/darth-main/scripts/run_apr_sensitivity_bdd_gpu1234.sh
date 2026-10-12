#!/usr/bin/env bash
set -u -o pipefail

[[ -f tools/run/train_seq3_episode.py ]] || {
  echo "ERROR: run from the project root" >&2
  exit 2
}

REPO="$(pwd)"
PYTHON=${PYTHON:-/home/lwr/anaconda3/envs/DARTH/bin/python}
BASE_CONFIG="configs/adapters/darth/bdd_4domain/module_combinations/darth_spaP1_causalProtoShrink_roiV1Weak_4domain_5e.py"
SOURCE_CKPT="work_dirs/bdd_4domain/source/clear_20e/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear/latest.pth"
OUT_ROOT="work_dirs/bdd_4domain/apr_sensitivity"
TARGET_ORDER=(overcast rainy dawn_dusk night)
STAGES=(stage01_after_overcast stage02_after_rainy stage03_after_dawn_dusk stage04_after_night)
SEED=1790355234
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-16}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU:-16}

# One physical GPU per non-default experiment. Kp100/Kn3 reuses the completed
# reference run, which was trained with the same seed.
VARIANTS=(
  "kp50_kn3|1|50|3"
  "kp200_kn3|2|200|3"
  "kp100_kn1|3|100|1"
  "kp100_kn5|4|100|5"
)

mkdir -p "$OUT_ROOT/logs"
QUEUE_LOG="$OUT_ROOT/logs/queue.log"
exec >>"$QUEUE_LOG" 2>&1

timestamp() { date '+%F %T'; }

run_eval() {
  local gpu="$1" variant="$2" root="$3" stage="$4" domain="$5"
  local ckpt="$root/episode/adapt/$stage/epoch_5.pth"
  local eval_dir="$root/forgetting_eval/$stage/epoch_5/$domain"
  local eval_cfg="configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_${domain}.py"

  if [[ -f "$eval_dir/eval_done.marker" ]]; then
    echo "[$(timestamp)] GPU${gpu} skip eval DONE ${variant} ${stage} -> ${domain}"
    return 0
  fi
  if [[ ! -f "$ckpt" ]]; then
    echo "[$(timestamp)] GPU${gpu} MISSING_CHECKPOINT ${ckpt}" >&2
    return 2
  fi

  mkdir -p "$eval_dir"
  rm -f "$eval_dir/eval_failed.marker"
  printf '%s\n' "$variant" >"$eval_dir/method.txt"
  printf '%s\n' "$stage" >"$eval_dir/stage.txt"
  printf '%s\n' "$domain" >"$eval_dir/eval_domain.txt"
  printf '%s\n' "det=0.05 init=0.30 obj=0.20" >"$eval_dir/thresholds.txt"
  printf '%s\n' "$ckpt" >"$eval_dir/checkpoint_path.txt"

  echo "[$(timestamp)] GPU${gpu} start eval ${variant} ${stage} -> ${domain}"
  env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH="$REPO:${PYTHONPATH:-}" \
    "$PYTHON" tools/run/test.py "$eval_cfg" \
      --checkpoint "$ckpt" \
      --launcher none \
      --gpu-id 0 \
      --work-dir "$eval_dir" \
      --eval bbox track \
      --fuse-conv-bn \
      --cfg-options \
        data.workers_per_gpu="$TEST_WORKERS_PER_GPU" \
        cudnn_benchmark=True \
        model.detector.test_cfg.rcnn.score_thr=0.05 \
        model.tracker.init_score_thr=0.30 \
        model.tracker.obj_score_thr=0.20 \
      >"$eval_dir/eval.log" 2>&1
  local rc=$?
  local metrics_file
  metrics_file="$(find "$eval_dir" -type f -name summary_metrics.csv -size +0c -print -quit)"
  if [[ "$rc" -eq 0 && -n "$metrics_file" ]]; then
    date >"$eval_dir/eval_done.marker"
    echo "[$(timestamp)] GPU${gpu} done eval ${variant} ${stage} -> ${domain}"
    return 0
  fi

  printf 'rc=%s\n' "$rc" >"$eval_dir/eval_failed.marker"
  echo "[$(timestamp)] GPU${gpu} FAILED eval rc=${rc} ${variant} ${stage} -> ${domain}" >&2
  return 1
}

run_variant() {
  local spec="$1"
  local variant gpu kp kn
  IFS='|' read -r variant gpu kp kn <<<"$spec"
  local root="$OUT_ROOT/$variant"
  local episode="$root/episode"

  mkdir -p "$root/logs" "$episode"
  printf 'gpu=%s\nkp=%s\nkn=%s\nseed=%s\n' "$gpu" "$kp" "$kn" "$SEED" >"$root/experiment.conf"

  if [[ -f "$root/train_done.marker" ]]; then
    echo "[$(timestamp)] GPU${gpu} skip train DONE ${variant}"
  else
    rm -f "$root/train_failed.marker"
    echo "[$(timestamp)] GPU${gpu} start train ${variant} Kp=${kp} Kn=${kn}"
    env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONPATH="$REPO:${PYTHONPATH:-}" \
      "$PYTHON" tools/run/train_seq3_episode.py \
        --base-config "$BASE_CONFIG" \
        --source-ckpt "$SOURCE_CKPT" \
        --target-order "${TARGET_ORDER[@]}" \
        --eval-domains "${TARGET_ORDER[@]}" \
        --work-dir "$episode" \
        --epochs-per-stage 5 \
        --samples-per-gpu "$SAMPLES_PER_GPU" \
        --workers-per-gpu "$WORKERS_PER_GPU" \
        --launcher none \
        --seed "$SEED" \
        --no-validate \
        --cfg-options \
          model.track_head.roi_proto_k_per_domain_class="$kp" \
          model.track_head.roi_proto_topk="$kn" \
        >"$root/logs/train_episode.log" 2>&1
    local rc=$?
    if [[ "$rc" -ne 0 ]]; then
      printf 'rc=%s\n' "$rc" >"$root/train_failed.marker"
      echo "[$(timestamp)] GPU${gpu} FAILED train rc=${rc} ${variant}" >&2
      return "$rc"
    fi
    date >"$root/train_done.marker"
    echo "[$(timestamp)] GPU${gpu} done train ${variant}"
  fi

  local stage domain
  for stage in "${STAGES[@]}"; do
    for domain in "${TARGET_ORDER[@]}"; do
      run_eval "$gpu" "$variant" "$root" "$stage" "$domain" || return $?
    done
  done
  date >"$root/all_done.marker"
  echo "[$(timestamp)] GPU${gpu} ALL_DONE ${variant}"
}

echo "[$(timestamp)] APR sensitivity start; fixed thresholds det=0.05 init=0.30 obj=0.20"
echo "[$(timestamp)] GPUs=1,2,3,4 samples_per_gpu=${SAMPLES_PER_GPU} test_workers=${TEST_WORKERS_PER_GPU} seed=${SEED}"

PIDS=()
for spec in "${VARIANTS[@]}"; do
  run_variant "$spec" &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || status=1
done

if [[ "$status" -eq 0 ]]; then
  "$PYTHON" tools/collect_apr_sensitivity_bdd.py >>"$OUT_ROOT/logs/collect.log" 2>&1 || status=1
fi

if [[ "$status" -eq 0 ]]; then
  date >"$OUT_ROOT/all_done.marker"
  echo "[$(timestamp)] APR sensitivity queue finished successfully"
else
  date >"$OUT_ROOT/failed.marker"
  echo "[$(timestamp)] APR sensitivity queue finished with failures" >&2
fi
exit "$status"
