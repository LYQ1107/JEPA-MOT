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
PYTHON=${PYTHON:-/home/lwr/anaconda3/envs/DARTH/bin/python}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
SOURCE_CKPT=${SOURCE_CKPT:-work_dirs/bdd_4domain/source/clear_20e/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear/latest.pth}

CFG="configs/adapters/darth/bdd_4domain/module_combinations/darth_spaP1_causalOld_roiV1Weak_4domain_5e.py"
TEST_CFG_ROOT="configs/mot/qdtrack/bdd_7domain"
OUT_ROOT="work_dirs/bdd_6domain/roiweak_continual_forgetting"
METHOD="spaP1_causalOld_roiV1Weak_6domain_5e"
GROUP="roiweak_6domain_forgetting"
THR_OVERRIDE="model.tracker.init_score_thr=0.30"
DOMAINS=(overcast partly_cloudy rainy snowy dawn_dusk night)

if [[ ",${GPU_IDS}," == *",0,"* || ",${GPU_IDS}," == *",1,"* || ",${GPU_IDS}," == *",2,"* || ",${GPU_IDS}," == *",3,"* ]]; then
  echo "[ERROR] GPU_IDS must not contain 0,1,2,3. Got GPU_IDS=${GPU_IDS}" >&2
  exit 2
fi

rand_port() { ${PYTHON} - <<'EOF'
import random
print(random.randint(30000, 59000))
EOF
}
run_cmd() { echo "$*"; if [[ "${DRY_RUN}" == "0" ]]; then eval "$@"; fi; }
write_text() { local path="$1"; shift; mkdir -p "$(dirname "${path}")"; printf "%s\n" "$*" > "${path}"; }
test_cfg() { echo "${TEST_CFG_ROOT}/qdtrack_faster-rcnn_r50_fpn_20e_bdd_${1}.py"; }
stage_name() { printf "stage%02d_after_%s" "$1" "$2"; }

print_domain_configs() {
  echo "============================================================"
  echo "6-DOMAIN EVAL DOMAIN CONFIGS"
  echo "============================================================"
  ${PYTHON} - <<'PY'
from mmcv import Config
for d in ['overcast','partly_cloudy','rainy','snowy','dawn_dusk','night']:
    p=f'configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_{d}.py'
    c=Config.fromfile(p)
    print(f'[{d}] cfg={p}')
    print(f'  ann_file={c.data.test.ann_file}')
    print(f'  img_prefix={c.data.test.img_prefix}')
PY
}

preflight() {
  echo "============================================================"
  echo "ROI-WEAK 6-DOMAIN CONTINUAL FORGETTING"
  echo "============================================================"
  echo "ROOT=${ROOT}"
  echo "OUT_ROOT=${OUT_ROOT}"
  echo "METHOD=${METHOD}"
  echo "CFG=${CFG}"
  echo "SOURCE_CKPT=${SOURCE_CKPT}"
  echo "TARGET_ORDER=${DOMAINS[*]}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "GPU_NUM=${GPU_NUM}"
  echo "SAMPLES_PER_GPU=${SAMPLES_PER_GPU}"
  echo "WORKERS_PER_GPU=${WORKERS_PER_GPU}"
  echo "TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU}"
  echo "DRY_RUN=${DRY_RUN} FORCE=${FORCE}"
  echo "THR_OVERRIDE=${THR_OVERRIDE}"
  echo "only_eval_after_train=true"
  echo "modify_test_py=false"
  echo "use_gpu_0_1_2_3=false"
  [[ -f "${CFG}" ]] || { echo "[ERROR] missing config: ${CFG}" >&2; exit 3; }
  [[ -f "${SOURCE_CKPT}" ]] || { echo "[ERROR] missing source checkpoint: ${SOURCE_CKPT}" >&2; exit 3; }
  for d in "${DOMAINS[@]}"; do
    [[ -f "configs/adapters/darth/bdd_7domain/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_${d}.py" ]] || { echo "[ERROR] missing train cfg for ${d}" >&2; exit 3; }
    [[ -f "$(test_cfg ${d})" ]] || { echo "[ERROR] missing test cfg for ${d}" >&2; exit 3; }
  done
  print_domain_configs
}

train_6domain() {
  local work="${OUT_ROOT}/${METHOD}"
  local marker="${work}/train_done.marker"
  mkdir -p "${work}/logs" "${work}/meta"
  cp "${CFG}" "${work}/used_config.py"
  write_text "${work}/meta/group.txt" "${GROUP}"
  write_text "${work}/meta/method.txt" "${METHOD}"
  write_text "${work}/meta/base_method.txt" "spaP1_causalOld_roiV1Weak_4domain_5e"
  write_text "${work}/meta/target_order.txt" "${DOMAINS[*]}"
  if [[ "${FORCE}" == "0" && -f "${marker}" ]]; then
    echo "[SKIP TRAIN] marker exists: ${marker}"
    return 0
  fi
  local cmd="CUDA_VISIBLE_DEVICES=${GPU_IDS} ${PYTHON} -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_addr=${MASTER_ADDR} --master_port=$(rand_port) tools/run/train_seq3_episode.py --base-config ${CFG} --source-ckpt ${SOURCE_CKPT} --target-order ${DOMAINS[*]} --eval-domains ${DOMAINS[*]} --work-dir ${work}/episode --epochs-per-stage 5 --samples-per-gpu ${SAMPLES_PER_GPU} --workers-per-gpu ${WORKERS_PER_GPU} --launcher pytorch --no-validate > ${work}/logs/train_episode.log 2>&1"
  write_text "${work}/train_command.txt" "${cmd}"
  run_cmd "${cmd}"
  if [[ "${DRY_RUN}" == "0" ]]; then
    touch "${marker}"
  fi
}

run_eval_one() {
  local stage_idx="$1" adapted_dom="$2" eval_dom="$3"
  local work="${OUT_ROOT}/${METHOD}"
  local stage; stage=$(stage_name "${stage_idx}" "${adapted_dom}")
  local ckpt="${work}/episode/adapt/${stage}/epoch_5.pth"
  local outdir="${work}/forgetting_eval/${stage}/epoch_5/${eval_dom}"
  local marker="${outdir}/eval_done.marker"
  if [[ ! -f "${ckpt}" && "${DRY_RUN}" == "0" ]]; then echo "[ERROR] missing checkpoint: ${ckpt}" >&2; exit 4; fi
  if [[ "${FORCE}" == "0" && -f "${marker}" ]]; then echo "[SKIP EVAL] ${stage} -> ${eval_dom}: marker exists"; return 0; fi
  mkdir -p "${outdir}"
  cp "$(test_cfg ${eval_dom})" "${outdir}/used_config.py"
  write_text "${outdir}/group.txt" "${GROUP}"
  write_text "${outdir}/method.txt" "${METHOD}"
  write_text "${outdir}/target_order.txt" "${DOMAINS[*]}"
  write_text "${outdir}/stage.txt" "${stage}"
  write_text "${outdir}/stage_idx.txt" "${stage_idx}"
  write_text "${outdir}/adapted_domain.txt" "${adapted_dom}"
  write_text "${outdir}/eval_domain.txt" "${eval_dom}"
  write_text "${outdir}/cfg_options.txt" "${THR_OVERRIDE}"
  write_text "${outdir}/checkpoint_path.txt" "${ckpt}"
  local cmd="CUDA_VISIBLE_DEVICES=${GPU_IDS} ${PYTHON} -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_addr=${MASTER_ADDR} --master_port=$(rand_port) tools/run/test.py $(test_cfg ${eval_dom}) --checkpoint ${ckpt} --launcher pytorch --work-dir ${outdir} --eval bbox track --fuse-conv-bn --cfg-options data.workers_per_gpu=${TEST_WORKERS_PER_GPU} cudnn_benchmark=True ${THR_OVERRIDE} > ${outdir}/eval.log 2>&1"
  write_text "${outdir}/eval_command.txt" "${cmd}"
  run_cmd "${cmd}"
  if [[ "${DRY_RUN}" == "0" ]]; then
    touch "${marker}"
  fi
}

eval_6domain() {
  for i in "${!DOMAINS[@]}"; do
    local stage_idx=$((i+1))
    local adapted="${DOMAINS[$i]}"
    for eval_dom in "${DOMAINS[@]}"; do
      run_eval_one "${stage_idx}" "${adapted}" "${eval_dom}"
    done
  done
}

preflight
train_6domain
eval_6domain
if [[ "${DRY_RUN}" == "0" ]]; then ${PYTHON} tools/collect_roiweak_6domain_forgetting.py; fi
echo "[DONE] roiweak 6-domain continual forgetting finished"
