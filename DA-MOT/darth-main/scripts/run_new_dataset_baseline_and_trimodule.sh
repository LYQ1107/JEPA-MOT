#!/usr/bin/env bash
set -euo pipefail

[[ -f tools/run/train.py && -f tools/run/test.py && -f tools/run/train_seq3_episode.py ]] || { echo "ERROR: run from project root" >&2; exit 1; }

PYTHON=${PYTHON:-/home/lwr/anaconda3/envs/DARTH/bin/python}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
GPU_IDS=${GPU_IDS:-"9,8,7,6,5,4,3,2,1,0"}
GPU_NUM=${GPU_NUM:-10}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-8}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-4}
TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU:-4}
SOURCE_EPOCHS=${SOURCE_EPOCHS:-20}
ADAPT_EPOCHS=${ADAPT_EPOCHS:-5}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}
RUN_BASELINE=${RUN_BASELINE:-1}
RUN_TRIMODULE=${RUN_TRIMODULE:-1}
RUN_MOTSYNTH=${RUN_MOTSYNTH:-1}
RUN_UA_DETRAC=${RUN_UA_DETRAC:-1}
RUN_VKITTI2=${RUN_VKITTI2:-1}

port(){ "$PYTHON" -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()"; }
run(){ echo "[CMD] $*"; [[ "$DRY_RUN" == 1 ]] || eval "$*"; }
ckpt(){ find "$1" -name "epoch_$2.pth" 2>/dev/null | sort | tail -1; }
clean_source(){ [[ "$DRY_RUN" == 1 ]] && return; find "$1" -type f \( -name 'epoch_*.pth' -o -name latest.pth \) ! -name "epoch_$2.pth" -print -delete; }
clean_adapt(){ [[ "$DRY_RUN" == 1 ]] && return; find "$1" -type f \( -name 'epoch_*.pth' -o -name latest.pth \) ! -name epoch_1.pth ! -name "epoch_$2.pth" -print -delete; }
patch_ckpt(){ [[ "$DRY_RUN" == 1 ]] && return; "$PYTHON" - "$1" "$2" <<'PY'
import re,sys
from pathlib import Path
p=Path(sys.argv[1]); ck=sys.argv[2]
s=p.read_text()
s=re.sub(r"checkpoint=['\"][^'\"]*epoch_20\.pth['\"]", f"checkpoint='{ck}'", s)
p.write_text(s)
PY
}

test_dist(){
  local cfg=$1 ck=$2 wd=$3 log=$4 tag=$5 extra=${6:-}
  mkdir -p "$wd" "$(dirname "$log")"
  local tmp="$wd/tmp_collect_${tag}_$(date +%Y%m%d_%H%M%S)_$$"; rm -rf "$tmp"; mkdir -p "$tmp"
  run "CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON -m torch.distributed.launch --nproc_per_node=$GPU_NUM --master_addr=$MASTER_ADDR --master_port=$(port) tools/run/test.py $cfg --checkpoint $ck --launcher pytorch --work-dir $wd --tmpdir $tmp --eval bbox track --fuse-conv-bn --cfg-options data.workers_per_gpu=$TEST_WORKERS_PER_GPU cudnn_benchmark=True $extra > $log 2>&1"
  [[ "$DRY_RUN" == 1 ]] || rm -rf "$tmp"
}

baseline(){
  local ds=$1 src=$2 src_cfg=$3 src_wd=$4 eval_pref=$5 adapt_pref=$6; shift 6; local domains=("$@")
  local root="work_dirs/$ds/baseline_single_target"; mkdir -p "$root/logs" "work_dirs/$ds/test/logs" "$src_wd"
  echo "==== BASELINE $ds source=$src targets=${domains[*]} ===="
  local src_ck; src_ck=$(ckpt "$src_wd" "$SOURCE_EPOCHS" || true)
  if [[ -f "$src_wd/source_train_done.marker" && -n "$src_ck" && "$FORCE" != 1 ]]; then echo "[SKIP] source $ds"; else
    run "CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON -m torch.distributed.launch --nproc_per_node=$GPU_NUM --master_addr=$MASTER_ADDR --master_port=$(port) tools/run/train.py $src_cfg --launcher pytorch --work-dir $src_wd --cfg-options total_epochs=$SOURCE_EPOCHS runner.max_epochs=$SOURCE_EPOCHS checkpoint_config.interval=1 checkpoint_config.max_keep_ckpts=-1 evaluation.interval=$((SOURCE_EPOCHS+1)) data.samples_per_gpu=$SAMPLES_PER_GPU data.workers_per_gpu=$WORKERS_PER_GPU > $root/logs/source_${src}_20e.log 2>&1"
    [[ "$DRY_RUN" == 1 ]] || date > "$src_wd/source_train_done.marker"; clean_source "$src_wd" "$SOURCE_EPOCHS"
  fi
  src_ck=$(ckpt "$src_wd" "$SOURCE_EPOCHS" || true)
  if [[ "$DRY_RUN" == 1 && -z "$src_ck" ]]; then src_ck="${src_wd}/DRY_RUN_epoch_${SOURCE_EPOCHS}.pth"; fi
  [[ "$DRY_RUN" == 1 || -f "$src_ck" ]] || { echo "ERROR: source ckpt missing: $src_wd" >&2; exit 1; }
  echo "source_ckpt=$src_ck"
  for d in "${domains[@]}"; do
    local ecfg="${eval_pref}_${d}_eval.py"
    local wd="work_dirs/$ds/test/source_${src}_to_${d}_direct"
    local done="$wd/eval_done.marker"
    if [[ -f "$done" && "$FORCE" != 1 ]]; then echo "[SKIP] direct $ds $d"; else test_dist "$ecfg" "$src_ck" "$wd" "work_dirs/$ds/test/logs/source_${src}_to_${d}_direct.log" "direct_${d}"; [[ "$DRY_RUN" == 1 ]] || date > "$done"; fi
  done
  for d in "${domains[@]}"; do
    local acfg="${adapt_pref}_${d}_from_${src}.py"
    local awd="work_dirs/$ds/baseline_single_target/adapt/${src}_to_${d}_5e"
    local adone="$awd/adapt_done.marker"
    mkdir -p "$awd"; patch_ckpt "$acfg" "$src_ck"
    if [[ -f "$adone" && "$FORCE" != 1 ]]; then echo "[SKIP] adapt $ds $d"; else
      run "CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON -m torch.distributed.launch --nproc_per_node=$GPU_NUM --master_addr=$MASTER_ADDR --master_port=$(port) tools/run/train.py $acfg --launcher pytorch --work-dir $awd --cfg-options total_epochs=$ADAPT_EPOCHS runner.max_epochs=$ADAPT_EPOCHS checkpoint_config.interval=1 checkpoint_config.max_keep_ckpts=-1 evaluation.interval=$((ADAPT_EPOCHS+1)) data.samples_per_gpu=$SAMPLES_PER_GPU data.workers_per_gpu=$WORKERS_PER_GPU > $root/logs/adapt_${src}_to_${d}_5e.log 2>&1"
      [[ "$DRY_RUN" == 1 ]] || date > "$adone"; clean_adapt "$awd" "$ADAPT_EPOCHS"
    fi
    local ack; ack=$(ckpt "$awd" "$ADAPT_EPOCHS" || true)
    if [[ "$DRY_RUN" == 1 && -z "$ack" ]]; then ack="${awd}/DRY_RUN_epoch_${ADAPT_EPOCHS}.pth"; fi
    [[ "$DRY_RUN" == 1 || -f "$ack" ]] || { echo "ERROR: adapt ckpt missing: $awd" >&2; exit 1; }
    local wd="work_dirs/$ds/test/baseline_${src}_to_${d}_5e"
    local done="$wd/eval_done.marker"
    if [[ -f "$done" && "$FORCE" != 1 ]]; then echo "[SKIP] adapted eval $ds $d"; else test_dist "${eval_pref}_${d}_eval.py" "$ack" "$wd" "work_dirs/$ds/test/logs/baseline_${src}_to_${d}_5e_eval.log" "adapt_${d}"; [[ "$DRY_RUN" == 1 ]] || date > "$done"; fi
  done
}

trimodule(){
  local ds=$1 src=$2 src_ck=$3 base_cfg=$4 tmpl=$5 eval_pref=$6; shift 6; local domains=("$@")
  local meth=spaP1_causalProtoShrink_roiV1Weak
  local root="work_dirs/$ds/trimodule_$meth"
  local ep="$root/episode"
  mkdir -p "$root/logs" "$ep"
  echo "==== TRI-MODULE $ds source=$src order=${domains[*]} ===="
  [[ "$DRY_RUN" == 1 || -f "$src_ck" ]] || { echo "ERROR: source ckpt missing: $src_ck" >&2; exit 1; }
  if [[ -f "$ep/state/train_episode.done" && "$FORCE" != 1 ]]; then echo "[SKIP] trimodule train $ds"; else
    run "CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON -m torch.distributed.launch --nproc_per_node=$GPU_NUM --master_addr=$MASTER_ADDR --master_port=$(port) tools/run/train_seq3_episode.py --base-config $base_cfg --source-ckpt $src_ck --source-domain $src --target-config-template '$tmpl' --target-order ${domains[*]} --eval-domains ${domains[*]} --work-dir $ep --epochs-per-stage $ADAPT_EPOCHS --samples-per-gpu $SAMPLES_PER_GPU --workers-per-gpu $WORKERS_PER_GPU --launcher pytorch --no-validate --cfg-options cudnn_benchmark=True > $root/logs/train_episode.log 2>&1"
    [[ "$DRY_RUN" == 1 ]] || { mkdir -p "$ep/state"; date > "$ep/state/train_episode.done"; }
  fi
  local i=1
  for d in "${domains[@]}"; do
    local st; st=$(printf 'stage%02d_after_%s' "$i" "$d")
    local c="$ep/adapt/$st/epoch_${ADAPT_EPOCHS}.pth"; [[ "$DRY_RUN" == 1 || -f "$c" ]] || { echo "ERROR: ckpt missing: $c" >&2; exit 1; }
    local wd="work_dirs/$ds/test/trimodule_$meth/$st/epoch_${ADAPT_EPOCHS}/$d"
    local done="$wd/eval_done.marker"
    if [[ -f "$done" && "$FORCE" != 1 ]]; then echo "[SKIP] trimodule eval $ds $st"; else test_dist "${eval_pref}_${d}_eval.py" "$c" "$wd" "$wd/eval.log" "tri_${d}" "model.tracker.init_score_thr=0.30"; [[ "$DRY_RUN" == 1 ]] || date > "$done"; fi
    i=$((i+1))
  done
}

echo "GPU_IDS=$GPU_IDS GPU_NUM=$GPU_NUM SPG=$SAMPLES_PER_GPU WPG=$WORKERS_PER_GPU TEST_WPG=$TEST_WORKERS_PER_GPU DRY_RUN=$DRY_RUN FORCE=$FORCE"

if [[ "$RUN_BASELINE" == 1 ]]; then
  [[ "$RUN_MOTSYNTH" == 1 ]] && baseline motsynth normal configs/mot/qdtrack/motsynth/qdtrack_faster-rcnn_r50_fpn_20e_motsynth_normal.py work_dirs/motsynth/source/normal_20e/qdtrack_motsynth_normal_20e configs/mot/qdtrack/motsynth/qdtrack_faster-rcnn_r50_fpn_20e_motsynth configs/adapters/darth/motsynth/darth_qdtrack_faster-rcnn_r50_fpn_5e_motsynth cloudy fog rain snow dark
  [[ "$RUN_UA_DETRAC" == 1 ]] && baseline ua_detrac sunny configs/mot/qdtrack/ua_detrac/qdtrack_faster-rcnn_r50_fpn_20e_ua_detrac_sunny.py work_dirs/ua_detrac/source/sunny_20e/qdtrack_ua_detrac_sunny_20e configs/mot/qdtrack/ua_detrac/qdtrack_faster-rcnn_r50_fpn_20e_ua_detrac configs/adapters/darth/ua_detrac/darth_qdtrack_faster-rcnn_r50_fpn_5e_ua_detrac cloudy rainy night
  [[ "$RUN_VKITTI2" == 1 ]] && baseline vkitti2 clone configs/mot/qdtrack/vkitti2/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_clone.py work_dirs/vkitti2/source/clone_20e/qdtrack_vkitti2_clone_20e configs/mot/qdtrack/vkitti2/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2 configs/adapters/darth/vkitti2/darth_qdtrack_faster-rcnn_r50_fpn_5e_vkitti2 overcast morning rain fog sunset
fi

if [[ "$RUN_TRIMODULE" == 1 ]]; then
  [[ "$RUN_MOTSYNTH" == 1 ]] && trimodule motsynth normal "$(ckpt work_dirs/motsynth/source/normal_20e/qdtrack_motsynth_normal_20e $SOURCE_EPOCHS || true)" configs/adapters/darth/motsynth/module_combinations/darth_spaP1_causalProtoShrink_roiV1Weak_5e_motsynth_cloudy_from_normal.py 'configs/adapters/darth/motsynth/module_combinations/darth_spaP1_causalProtoShrink_roiV1Weak_5e_motsynth_{target}_from_normal.py' configs/mot/qdtrack/motsynth/qdtrack_faster-rcnn_r50_fpn_20e_motsynth cloudy fog rain snow dark
  [[ "$RUN_UA_DETRAC" == 1 ]] && trimodule ua_detrac sunny "$(ckpt work_dirs/ua_detrac/source/sunny_20e/qdtrack_ua_detrac_sunny_20e $SOURCE_EPOCHS || true)" configs/adapters/darth/ua_detrac/module_combinations/darth_spaP1_causalProtoShrink_roiV1Weak_5e_ua_detrac_cloudy_from_sunny.py 'configs/adapters/darth/ua_detrac/module_combinations/darth_spaP1_causalProtoShrink_roiV1Weak_5e_ua_detrac_{target}_from_sunny.py' configs/mot/qdtrack/ua_detrac/qdtrack_faster-rcnn_r50_fpn_20e_ua_detrac cloudy rainy night
  [[ "$RUN_VKITTI2" == 1 ]] && trimodule vkitti2 clone work_dirs/vkitti2/source/clone_20e/qdtrack_vkitti2_clone_20e/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_clone/20260707_171736/epoch_20.pth configs/adapters/darth/vkitti2/module_combinations/darth_spaP1_causalProtoShrink_roiV1Weak_5e_vkitti2_overcast_from_clone.py 'configs/adapters/darth/vkitti2/module_combinations/darth_spaP1_causalProtoShrink_roiV1Weak_5e_vkitti2_{target}_from_clone.py' configs/mot/qdtrack/vkitti2/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2 overcast morning rain fog sunset
fi

if [[ "$DRY_RUN" != 1 ]]; then
  "$PYTHON" scripts/collect_new_dataset_experiment_results.py
fi

echo '[ALL DONE]'
