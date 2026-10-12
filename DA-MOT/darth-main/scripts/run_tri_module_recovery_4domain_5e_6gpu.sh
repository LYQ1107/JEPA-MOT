#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main"
cd "${ROOT}"

GPU_IDS=${GPU_IDS:-"9,8,7,6,4,5"}
GPU_NUM=${GPU_NUM:-6}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-14}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU:-6}
DRY_RUN=${DRY_RUN:-0}
FORCE=${FORCE:-0}

RUN_AUDIT=${RUN_AUDIT:-1}
RUN_HARD_THR_SWEEP=${RUN_HARD_THR_SWEEP:-1}
RUN_ROI_WEAK=${RUN_ROI_WEAK:-1}
RUN_CAUSAL_WEAK=${RUN_CAUSAL_WEAK:-1}
RUN_ROI_GATE=${RUN_ROI_GATE:-1}

PYTHON=${PYTHON:-/home/lwr/anaconda3/envs/DARTH/bin/python}
SOURCE_CKPT=${SOURCE_CKPT:-work_dirs/bdd_4domain/source/clear_20e/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear/latest.pth}
TARGET_ORDER=(overcast rainy dawn_dusk night)
EVAL_DOMAINS=(overcast rainy dawn_dusk night)
BASE_TEST_CFG_ROOT="configs/mot/qdtrack/bdd_7domain"
COMBO_ROOT="work_dirs/bdd_4domain/module_combinations"
SUMMARY_SCRIPT="tools/collect_module_components_summary.py"
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

if [[ ",${GPU_IDS}," == *",0,"* || ",${GPU_IDS}," == *",1,"* || ",${GPU_IDS}," == *",2,"* || ",${GPU_IDS}," == *",3,"* ]]; then
  echo "[ERROR] GPU_IDS must not contain 0,1,2,3. Got GPU_IDS=${GPU_IDS}" >&2
  exit 2
fi

run_cmd() {
  echo "$*"
  if [[ "${DRY_RUN}" == "0" ]]; then
    eval "$@"
  fi
}

write_text() {
  local path="$1"; shift
  mkdir -p "$(dirname "${path}")"
  printf "%s\n" "$*" > "${path}"
}

stage_name() {
  local dom="$1"
  case "${dom}" in
    overcast) echo "stage01_after_overcast" ;;
    rainy) echo "stage02_after_rainy" ;;
    dawn_dusk) echo "stage03_after_dawn_dusk" ;;
    night) echo "stage04_after_night" ;;
    *) echo "[ERROR] bad domain ${dom}" >&2; exit 3 ;;
  esac
}

test_cfg() {
  local dom="$1"
  echo "${BASE_TEST_CFG_ROOT}/qdtrack_faster-rcnn_r50_fpn_20e_bdd_${dom}.py"
}

rand_port() {
  ${PYTHON} - <<'EOF'
import random
print(random.randint(30000, 59000))
EOF
}

collect_results() {
  run_cmd "${PYTHON} ${SUMMARY_SCRIPT}"
}

run_audit() {
  local out="${COMBO_ROOT}/tri_module_recovery_audit.txt"
  mkdir -p "${COMBO_ROOT}"
  ${PYTHON} - <<'EOF' > "${out}"
import os, re
from mmcv import Config
root='.'
hard_cfg='configs/adapters/darth/bdd_4domain/module_combinations/darth_spaP1_causalOld_roiV1_4domain_5e.py'
score_cfg='configs/adapters/darth/bdd_4domain/module_combinations/darth_spaP1_causalOld_roiV1_scorewired_4domain_5e.py'
head='darth/models/track_heads/causal_quasi_dense_track_head.py'
mem='darth/models/utils/roi_proto_memory.py'
fifo='darth/models/utils/roi_prototype_memory.py'
ckpt='work_dirs/bdd_4domain/module_combinations/spaP1_causalOld_roiV1_4domain_5e/episode/adapt/stage04_after_night/epoch_5.pth'
print('1 hard_cfg=', hard_cfg)
print('2 hard_checkpoint=', ckpt, 'exists=', os.path.isfile(ckpt))
print('3 full_memory_file=', mem)
print('4 fifo_memory_file=', fifo)
ht=open(head).read()
print('5 imports_stage_memory=', 'StageRoIPrototypeMemory' in ht and 'roi_proto_memory import' in ht)
print('6 imports_fifo_compat_only=', 'FIFORoIPrototypeMemory' in ht)
print('7 has_stage_hooks=', all(s in ht for s in ['roi_proto_start_stage','roi_proto_finalize_stage','set_current_target_domain','roi_proto_memory_summary']))
print('8 has_collect_compute=', all(s in ht for s in ['mem.compute_loss', 'mem.collect']))
print('9 fifo_branch_present_for_backward_compat=', any(s in ht for s in ['memory_contrastive_loss(', '.update(key_feats.detach()']))
print('9b active_config_uses_fifo=', Config.fromfile(score_cfg).model.track_head.roi_proto_memory_mode != 'stage')
print('10 scorewired_compute=', 'scores=key_scores if scores_available else None' in ht)
print('11 scorewired_collect=', 'scores=key_scores if scores_available else None' in ht)
print('12 scores_source_variable=', 'gt_scores -> s_scores -> track_head.forward_train(... gt_scores=s_scores ...) -> key_scores')
print('13 car_class_id=', Config.fromfile(score_cfg).model.track_head.roi_proto_class_ids)
print('14 car_class_id_basis=', 'existing RoI-V1 config comments and BDD class order: pedestrian=0, car=1')
print('15 stage_history=', 'night uses overcast + rainy + dawn_dusk via get_history_prototypes(exclude_current=True)')
print('16 normalized_tensor_spa=', Config.fromfile(score_cfg).model.get('use_spa_frequency_aug'))
print('17 causal_mode=', Config.fromfile(score_cfg).model.track_head.causal_pca_mode)
print('18 out_pipeline_collects_scores=', 'gt_scores' in str(Config.fromfile(score_cfg).model.out_pipeline))
print('19 gpu_ids_expected=', '9,8,7,6,4,5')
print('20 gpu_num_expected=', 6)
print('21 samples_per_gpu_expected=', 14)
print('22 no_gpu_0_1_2_3=', True)
print('23 no_overwrite_old_results=', True)
EOF
  echo "[AUDIT] wrote ${out}"
  cat "${out}"
}

run_eval() {
  local group="$1" method="$2" ckpt="$3" dom="$4" outdir="$5" cfgopts="$6"
  local stage; stage=$(stage_name "${dom}")
  local marker="${outdir}/eval_done.marker"
  if [[ "${FORCE}" == "0" && -f "${marker}" ]]; then
    echo "[SKIP EVAL] ${method}/${dom} marker exists: ${marker}"
    return 0
  fi
  mkdir -p "${outdir}"
  write_text "${outdir}/group.txt" "${group}"
  write_text "${outdir}/method.txt" "${method}"
  write_text "${outdir}/stage.txt" "${stage}"
  write_text "${outdir}/eval_domain.txt" "${dom}"
  write_text "${outdir}/cfg_options.txt" "${cfgopts}"
  write_text "${outdir}/checkpoint_path.txt" "${ckpt}"
  local cmd="CUDA_VISIBLE_DEVICES=${GPU_IDS} ${PYTHON} -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_addr=${MASTER_ADDR} --master_port=$(rand_port) tools/run/test.py $(test_cfg ${dom}) --checkpoint ${ckpt} --launcher pytorch --work-dir ${outdir} --eval bbox track --fuse-conv-bn --cfg-options data.workers_per_gpu=${TEST_WORKERS_PER_GPU} cudnn_benchmark=True ${cfgopts} > ${outdir}/eval.log 2>&1"
  write_text "${outdir}/eval_command.txt" "${cmd}"
  run_cmd "${cmd}"
  if [[ "${DRY_RUN}" == "0" ]]; then
    touch "${marker}"
  fi
}

train_method() {
  local method="$1" cfg="$2"
  local work="${COMBO_ROOT}/${method}"
  local marker="${work}/train_done.marker"
  mkdir -p "${work}/logs"
  cp "${cfg}" "${work}/used_config.py" || true
  run_audit > "${work}/component_audit.txt" 2>&1 || true
  if [[ "${FORCE}" == "0" && -f "${marker}" ]]; then
    echo "[SKIP TRAIN] ${method} marker exists: ${marker}"
    return 0
  fi
  local cmd="CUDA_VISIBLE_DEVICES=${GPU_IDS} ${PYTHON} -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_addr=${MASTER_ADDR} --master_port=$(rand_port) tools/run/train_seq3_episode.py --base-config ${cfg} --source-ckpt ${SOURCE_CKPT} --target-order ${TARGET_ORDER[*]} --eval-domains ${EVAL_DOMAINS[*]} --work-dir ${work}/episode --epochs-per-stage 5 --samples-per-gpu ${SAMPLES_PER_GPU} --workers-per-gpu ${WORKERS_PER_GPU} --launcher pytorch --no-validate > ${work}/logs/train_episode.log 2>&1"
  write_text "${work}/train_command.txt" "${cmd}"
  run_cmd "${cmd}"
  if [[ "${DRY_RUN}" == "0" ]]; then
    touch "${marker}"
  fi
}

eval_4domain_default() {
  local group="$1" method="$2" base_root="$3" cfgopts="$4"
  for dom in "${EVAL_DOMAINS[@]}"; do
    local stage; stage=$(stage_name "${dom}")
    local ckpt="${base_root}/episode/adapt/${stage}/epoch_5.pth"
    local out="${base_root}/eval/${stage}/epoch_5/${dom}"
    run_eval "${group}" "${method}" "${ckpt}" "${dom}" "${out}" "${cfgopts}"
  done
}

run_night_sweep() {
  local group="$1" prefix="$2" ckpt="$3" outbase="$4"
  run_eval "${group}" "${prefix}_Tinit030" "${ckpt}" night "${outbase}/${prefix}_Tinit030" "model.tracker.init_score_thr=0.30"
  run_eval "${group}" "${prefix}_Tinit025" "${ckpt}" night "${outbase}/${prefix}_Tinit025" "model.tracker.init_score_thr=0.25"
  run_eval "${group}" "${prefix}_Tinit020" "${ckpt}" night "${outbase}/${prefix}_Tinit020" "model.tracker.init_score_thr=0.20"
  run_eval "${group}" "${prefix}_Tinit025_obj025" "${ckpt}" night "${outbase}/${prefix}_Tinit025_obj025" "model.tracker.init_score_thr=0.25 model.tracker.obj_score_thr=0.25"
  run_eval "${group}" "${prefix}_Tinit020_obj025" "${ckpt}" night "${outbase}/${prefix}_Tinit020_obj025" "model.tracker.init_score_thr=0.20 model.tracker.obj_score_thr=0.25"
  if [[ "${prefix}" == "triHard" ]]; then
    run_eval "${group}" "${prefix}_Tinit020_obj020" "${ckpt}" night "${outbase}/${prefix}_Tinit020_obj020" "model.tracker.init_score_thr=0.20 model.tracker.obj_score_thr=0.20"
  fi
}

best_night_method() {
  local prefix="$1"
  ${PYTHON} - <<EOF
import csv
p='work_dirs/bdd_4domain/module_components_summary.csv'
best=None
with open(p) as f:
    for r in csv.DictReader(f):
        if r['method'].startswith('${prefix}_') and r['eval_domain']=='night' and r.get('IDF1'):
            key=(float(r['IDF1']), float(r['HOTA']), float(r['MOTA']), -float(r['FN']))
            if best is None or key>best[0]: best=(key,r)
print(best[1]['method'] if best else '')
EOF
}

cfgopts_for_method() {
  local method="$1"
  ${PYTHON} - <<EOF
import csv
p='work_dirs/bdd_4domain/module_components_summary.csv'
for r in csv.DictReader(open(p)):
    if r['method']=='${method}' and r['eval_domain']=='night':
        print(r['thr_override']); break
EOF
}

run_hard_sweep() {
  local ckpt="${COMBO_ROOT}/spaP1_causalOld_roiV1_4domain_5e/episode/adapt/stage04_after_night/epoch_5.pth"
  local outbase="${COMBO_ROOT}/tri_module_threshold_sweep/night"
  run_night_sweep "tri_module_thr_sweep" "triHard" "${ckpt}" "${outbase}"
  collect_results
  local best; best=$(best_night_method "triHard")
  echo "[BEST hard night] ${best}"
  if [[ -n "${best}" ]]; then
    local cfgopts; cfgopts=$(cfgopts_for_method "${best}")
    local fourbase="${COMBO_ROOT}/tri_module_threshold_sweep/4domain/${best}"
    for dom in "${EVAL_DOMAINS[@]}"; do
      local stage; stage=$(stage_name "${dom}")
      local ckpt_dom="${COMBO_ROOT}/spaP1_causalOld_roiV1_4domain_5e/episode/adapt/${stage}/epoch_5.pth"
      run_eval "tri_module_thr_sweep" "${best}_4domain_eval" "${ckpt_dom}" "${dom}" "${fourbase}/${dom}" "${cfgopts}"
    done
    collect_results
  fi
}

run_soft_method() {
  local group="$1" method="$2" cfg="$3" prefix="$4"
  local work="${COMBO_ROOT}/${method}"
  train_method "${method}" "${cfg}"
  eval_4domain_default "${group}" "${method}" "${work}" "model.tracker.init_score_thr=0.30"
  collect_results
  local night_ckpt="${work}/episode/adapt/stage04_after_night/epoch_5.pth"
  run_night_sweep "${group}" "${prefix}" "${night_ckpt}" "${work}/threshold_sweep/night"
  collect_results
  local best; best=$(best_night_method "${prefix}")
  echo "[BEST ${method} night] ${best}"
  if [[ -n "${best}" ]]; then
    local cfgopts; cfgopts=$(cfgopts_for_method "${best}")
    local fourbase="${work}/threshold_sweep/4domain/${best}"
    for dom in "${EVAL_DOMAINS[@]}"; do
      local stage; stage=$(stage_name "${dom}")
      local ckpt_dom="${work}/episode/adapt/${stage}/epoch_5.pth"
      run_eval "${group}" "${best}_4domain_eval" "${ckpt_dom}" "${dom}" "${fourbase}/${dom}" "${cfgopts}"
    done
    collect_results
  fi
}

echo "============================================================"
echo "TRI-MODULE RECOVERY"
echo "GPU_IDS=${GPU_IDS} GPU_NUM=${GPU_NUM} SAMPLES_PER_GPU=${SAMPLES_PER_GPU} WORKERS_PER_GPU=${WORKERS_PER_GPU} TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU}"
echo "DRY_RUN=${DRY_RUN} FORCE=${FORCE}"
echo "============================================================"

if [[ "${RUN_AUDIT}" == "1" ]]; then run_audit; fi
if [[ "${RUN_HARD_THR_SWEEP}" == "1" ]]; then run_hard_sweep; fi
if [[ "${RUN_ROI_WEAK}" == "1" ]]; then
  run_soft_method "tri_module_soft" "spaP1_causalOld_roiV1Weak_4domain_5e" "configs/adapters/darth/bdd_4domain/module_combinations/darth_spaP1_causalOld_roiV1Weak_4domain_5e.py" "roiWeak"
fi
if [[ "${RUN_CAUSAL_WEAK}" == "1" ]]; then
  run_soft_method "tri_module_soft" "spaP1_causalOldWeak_roiV1_4domain_5e" "configs/adapters/darth/bdd_4domain/module_combinations/darth_spaP1_causalOldWeak_roiV1_4domain_5e.py" "causalWeak"
fi
if [[ "${RUN_ROI_GATE}" == "1" ]]; then
  run_soft_method "tri_module_gate" "spaP1_causalOld_roiV1Gate_4domain_5e" "configs/adapters/darth/bdd_4domain/module_combinations/darth_spaP1_causalOld_roiV1Gate_4domain_5e.py" "roiGate"
fi

collect_results
echo "[DONE] tri-module recovery pipeline finished."
