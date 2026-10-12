#!/usr/bin/env bash
set -u -o pipefail

REPO="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main"
PYTHON="/home/lwr/anaconda3/envs/DARTH/bin/python"
PHASE1_ROOT="$REPO/work_dirs/limit_search_gpu23/phase1_threshold"
ROOT="$REPO/work_dirs/limit_search_gpu23/full17_threshold"
FINAL_CSV="$REPO/work_dirs/limit_search_gpu23/darth_4dataset_best_thresholds.csv"
mkdir -p "$ROOT"
cd "$REPO"

CANDIDATES=(
  "base|0.05|0.20|0.20"
  "det01|0.01|0.20|0.20"
  "det03|0.03|0.20|0.20"
  "det10|0.10|0.20|0.20"
  "init01|0.05|0.10|0.20"
  "init03|0.05|0.30|0.20"
  "init07|0.05|0.70|0.20"
  "obj01|0.05|0.20|0.10"
  "obj05|0.05|0.20|0.50"
)

TARGETS=(
  "bdd|overcast|configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_overcast.py|work_dirs/bdd_4domain/tri_module_pca_variants/spaP1_causalProtoShrink_roiV1Weak_4domain_5e/episode/adapt/stage01_after_overcast/epoch_5.pth"
  "bdd|rainy|configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_rainy.py|work_dirs/bdd_4domain/tri_module_pca_variants/spaP1_causalProtoShrink_roiV1Weak_4domain_5e/episode/adapt/stage02_after_rainy/epoch_5.pth"
  "bdd|dawn_dusk|configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_dawn_dusk.py|work_dirs/bdd_4domain/tri_module_pca_variants/spaP1_causalProtoShrink_roiV1Weak_4domain_5e/episode/adapt/stage03_after_dawn_dusk/epoch_5.pth"
  "bdd|night|configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_night.py|work_dirs/bdd_4domain/tri_module_pca_variants/spaP1_causalProtoShrink_roiV1Weak_4domain_5e/episode/adapt/stage04_after_night/epoch_5.pth"
  "motsynth|cloudy|configs/mot/qdtrack/motsynth/qdtrack_faster-rcnn_r50_fpn_20e_motsynth_cloudy_eval.py|work_dirs/motsynth/trimodule_spaP1_causalProtoShrink_roiV1Weak/episode/adapt/stage01_after_cloudy/epoch_5.pth"
  "motsynth|fog|configs/mot/qdtrack/motsynth/qdtrack_faster-rcnn_r50_fpn_20e_motsynth_fog_eval.py|work_dirs/motsynth/trimodule_spaP1_causalProtoShrink_roiV1Weak/episode/adapt/stage02_after_fog/epoch_5.pth"
  "motsynth|rain|configs/mot/qdtrack/motsynth/qdtrack_faster-rcnn_r50_fpn_20e_motsynth_rain_eval.py|work_dirs/motsynth/trimodule_spaP1_causalProtoShrink_roiV1Weak/episode/adapt/stage03_after_rain/epoch_5.pth"
  "motsynth|snow|configs/mot/qdtrack/motsynth/qdtrack_faster-rcnn_r50_fpn_20e_motsynth_snow_eval.py|work_dirs/motsynth/trimodule_spaP1_causalProtoShrink_roiV1Weak/episode/adapt/stage04_after_snow/epoch_5.pth"
  "motsynth|dark|configs/mot/qdtrack/motsynth/qdtrack_faster-rcnn_r50_fpn_20e_motsynth_dark_eval.py|work_dirs/motsynth/trimodule_spaP1_causalProtoShrink_roiV1Weak/episode/adapt/stage05_after_dark/epoch_5.pth"
  "ua_detrac|cloudy|configs/mot/qdtrack/ua_detrac/qdtrack_faster-rcnn_r50_fpn_20e_ua_detrac_cloudy_eval.py|work_dirs/ua_detrac/trimodule_spaP1_causalProtoShrink_roiV1Weak/episode/adapt/stage01_after_cloudy/epoch_5.pth"
  "ua_detrac|rainy|configs/mot/qdtrack/ua_detrac/qdtrack_faster-rcnn_r50_fpn_20e_ua_detrac_rainy_eval.py|work_dirs/ua_detrac/trimodule_spaP1_causalProtoShrink_roiV1Weak/episode/adapt/stage02_after_rainy/epoch_5.pth"
  "ua_detrac|night|configs/mot/qdtrack/ua_detrac/qdtrack_faster-rcnn_r50_fpn_20e_ua_detrac_night_eval.py|work_dirs/ua_detrac/trimodule_spaP1_causalProtoShrink_roiV1Weak/episode/adapt/stage03_after_night/epoch_5.pth"
  "vkitti2|overcast|configs/mot/qdtrack/vkitti2/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_overcast_eval.py|work_dirs/vkitti2/trimodule_spaP1_causalProtoShrink_roiV1Weak/episode/adapt/stage01_after_overcast/epoch_5.pth"
  "vkitti2|morning|configs/mot/qdtrack/vkitti2/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_morning_eval.py|work_dirs/vkitti2/trimodule_spaP1_causalProtoShrink_roiV1Weak/episode/adapt/stage02_after_morning/epoch_5.pth"
  "vkitti2|rain|configs/mot/qdtrack/vkitti2/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_rain_eval.py|work_dirs/vkitti2/trimodule_spaP1_causalProtoShrink_roiV1Weak/episode/adapt/stage03_after_rain/epoch_5.pth"
  "vkitti2|fog|configs/mot/qdtrack/vkitti2/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_fog_eval.py|work_dirs/vkitti2/trimodule_spaP1_causalProtoShrink_roiV1Weak/episode/adapt/stage04_after_fog/epoch_5.pth"
  "vkitti2|sunset|configs/mot/qdtrack/vkitti2/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_sunset_eval.py|work_dirs/vkitti2/trimodule_spaP1_causalProtoShrink_roiV1Weak/episode/adapt/stage05_after_sunset/epoch_5.pth"
)

QUEUE_LOG="$ROOT/queue.log"
exec >> "$QUEUE_LOG" 2>&1

CHILD_PID=""
stop_worker() {
  if [[ -n "$CHILD_PID" ]]; then
    kill "$CHILD_PID" 2>/dev/null || true
  fi
  exit 143
}
trap stop_worker INT TERM

run_task() {
  local gpu="$1"
  local task="$2"
  local dataset domain cfg ckpt candidate det init obj
  IFS='|' read -r dataset domain cfg ckpt candidate det init obj <<< "$task"

  local out="$ROOT/$dataset/$domain/$candidate"
  mkdir -p "$out"
  if [[ -f "$out/done.marker" ]]; then
    printf '[%s] GPU%s skip DONE %s/%s/%s\n' "$(date '+%F %T')" "$gpu" "$dataset" "$domain" "$candidate"
    return 0
  fi
  rm -f "$out/running.marker" "$out/failed.marker"
  touch "$out/running.marker"
  printf '[%s] GPU%s start %s/%s/%s det=%s init=%s obj=%s\n' \
    "$(date '+%F %T')" "$gpu" "$dataset" "$domain" "$candidate" "$det" "$init" "$obj"

  env CUDA_VISIBLE_DEVICES="$gpu" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH="$REPO:${PYTHONPATH:-}" \
    "$PYTHON" tools/run/test.py "$cfg" \
      --checkpoint "$ckpt" \
      --launcher none \
      --gpu-id 0 \
      --work-dir "$out" \
      --eval bbox track \
      --fuse-conv-bn \
      --cfg-options \
        data.workers_per_gpu=1 \
        cudnn_benchmark=True \
        model.detector.test_cfg.rcnn.score_thr="$det" \
        model.tracker.init_score_thr="$init" \
        model.tracker.obj_score_thr="$obj" \
      > "$out/run.log" 2>&1 &
  CHILD_PID=$!
  wait "$CHILD_PID"
  local rc=$?
  CHILD_PID=""

  local metrics_file
  metrics_file="$(find "$out" -type f -name summary_metrics.csv -size +0c -print -quit)"
  if [[ "$rc" -eq 0 && -n "$metrics_file" ]]; then
    rm -f "$out/running.marker" "$out/failed.marker"
    printf '%s\n' "$(date '+%F %T')" > "$out/done.marker"
    printf '[%s] GPU%s done %s/%s/%s\n' "$(date '+%F %T')" "$gpu" "$dataset" "$domain" "$candidate"
  else
    rm -f "$out/running.marker"
    printf 'rc=%s\n' "$rc" > "$out/failed.marker"
    printf '[%s] GPU%s FAILED rc=%s %s/%s/%s\n' \
      "$(date '+%F %T')" "$gpu" "$rc" "$dataset" "$domain" "$candidate"
  fi
}

ALL_TASKS=()
REUSED=0
for target in "${TARGETS[@]}"; do
  IFS='|' read -r dataset domain cfg ckpt <<< "$target"
  if [[ ! -f "$cfg" || ! -f "$ckpt" ]]; then
    printf '[%s] CONFIG_OR_CHECKPOINT_MISSING %s/%s cfg=%s ckpt=%s\n' \
      "$(date '+%F %T')" "$dataset" "$domain" "$cfg" "$ckpt"
    exit 2
  fi
  for candidate_spec in "${CANDIDATES[@]}"; do
    IFS='|' read -r candidate det init obj <<< "$candidate_spec"
    if [[ -f "$PHASE1_ROOT/$dataset/$domain/$candidate/done.marker" ]]; then
      REUSED=$((REUSED + 1))
      continue
    fi
    ALL_TASKS+=("$dataset|$domain|$cfg|$ckpt|$candidate|$det|$init|$obj")
  done
done

printf '[%s] full17 queue start total=153 reused=%d pending=%d workers_per_gpu=1 samples_per_gpu=1\n' \
  "$(date '+%F %T')" "$REUSED" "${#ALL_TASKS[@]}"

GPU2_TASKS=()
GPU3_TASKS=()
for ((i = 0; i < ${#ALL_TASKS[@]}; i++)); do
  if ((i % 2 == 0)); then
    GPU2_TASKS+=("${ALL_TASKS[$i]}")
  else
    GPU3_TASKS+=("${ALL_TASKS[$i]}")
  fi
done

worker() {
  local gpu="$1"
  shift
  for task in "$@"; do
    run_task "$gpu" "$task"
  done
  printf '[%s] GPU%s worker finished\n' "$(date '+%F %T')" "$gpu"
}

worker 2 "${GPU2_TASKS[@]}" &
GPU2_PID=$!
worker 3 "${GPU3_TASKS[@]}" &
GPU3_PID=$!
wait "$GPU2_PID"
GPU2_RC=$?
wait "$GPU3_PID"
GPU3_RC=$?

printf '[%s] workers finished gpu2_rc=%s gpu3_rc=%s; collecting CSV\n' \
  "$(date '+%F %T')" "$GPU2_RC" "$GPU3_RC"
"$PYTHON" tools/collect_limit_search_best_thresholds.py --output "$FINAL_CSV"
COLLECT_RC=$?
printf '[%s] full17 queue finished collect_rc=%s csv=%s\n' \
  "$(date '+%F %T')" "$COLLECT_RC" "$FINAL_CSV"
[[ "$GPU2_RC" -eq 0 && "$GPU3_RC" -eq 0 && "$COLLECT_RC" -eq 0 ]]
