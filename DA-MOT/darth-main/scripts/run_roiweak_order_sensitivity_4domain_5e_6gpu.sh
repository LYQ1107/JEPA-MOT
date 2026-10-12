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
OUT_ROOT="work_dirs/bdd_4domain/order_sensitivity_roiV1Weak"
GROUP="roiweak_order_sensitivity"
BASE_METHOD="spaP1_causalOld_roiV1Weak_4domain_5e"
THR_OVERRIDE="model.tracker.init_score_thr=0.30"
ALL_DOMAINS=(overcast rainy dawn_dusk night)

ORDER_IDS=(order01 order02 order03)
ORDER_NAMES=(rainy_overcast_night_dawn_dusk dawn_dusk_night_overcast_rainy night_dawn_dusk_rainy_overcast)
ORDER_01=(rainy overcast night dawn_dusk)
ORDER_02=(dawn_dusk night overcast rainy)
ORDER_03=(night dawn_dusk rainy overcast)

if [[ ",${GPU_IDS}," == *",0,"* || ",${GPU_IDS}," == *",1,"* || ",${GPU_IDS}," == *",2,"* || ",${GPU_IDS}," == *",3,"* ]]; then
  echo "[ERROR] GPU_IDS must not contain 0,1,2,3. Got GPU_IDS=${GPU_IDS}" >&2
  exit 2
fi

rand_port() {
  ${PYTHON} - <<'EOF'
import random
print(random.randint(30000, 59000))
EOF
}

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

test_cfg() {
  local dom="$1"
  echo "${TEST_CFG_ROOT}/qdtrack_faster-rcnn_r50_fpn_20e_bdd_${dom}.py"
}

order_domains() {
  local oid="$1"
  case "${oid}" in
    order01) echo "${ORDER_01[*]}" ;;
    order02) echo "${ORDER_02[*]}" ;;
    order03) echo "${ORDER_03[*]}" ;;
    *) echo "[ERROR] unknown order ${oid}" >&2; exit 3 ;;
  esac
}

order_name() {
  local oid="$1"
  case "${oid}" in
    order01) echo "${ORDER_NAMES[0]}" ;;
    order02) echo "${ORDER_NAMES[1]}" ;;
    order03) echo "${ORDER_NAMES[2]}" ;;
    *) echo "[ERROR] unknown order ${oid}" >&2; exit 3 ;;
  esac
}

stage_name() {
  local idx="$1" dom="$2"
  printf "stage%02d_after_%s" "${idx}" "${dom}"
}

print_domain_configs() {
  echo "============================================================"
  echo "EVAL DOMAIN CONFIGS"
  echo "============================================================"
  ${PYTHON} - <<'PY'
from mmcv import Config
for d in ['overcast','rainy','dawn_dusk','night']:
    p=f'configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_{d}.py'
    c=Config.fromfile(p)
    print(f'[{d}] cfg={p}')
    print(f'  ann_file={c.data.test.ann_file}')
    print(f'  img_prefix={c.data.test.img_prefix}')
PY
}

train_order() {
  local oid="$1"
  local oname; oname=$(order_name "${oid}")
  local domains; domains=$(order_domains "${oid}")
  local method="${BASE_METHOD}_${oid}_${oname}"
  local work="${OUT_ROOT}/${oid}_${oname}"
  local marker="${work}/train_done.marker"
  mkdir -p "${work}/logs" "${work}/meta"
  cp "${CFG}" "${work}/used_config.py"
  write_text "${work}/meta/order_id.txt" "${oid}"
  write_text "${work}/meta/order_name.txt" "${oname}"
  write_text "${work}/meta/target_order.txt" "${domains}"
  write_text "${work}/meta/method.txt" "${method}"
  write_text "${work}/meta/base_method.txt" "${BASE_METHOD}"
  write_text "${work}/meta/group.txt" "${GROUP}"
  if [[ "${FORCE}" == "0" && -f "${marker}" ]]; then
    echo "[SKIP TRAIN] ${oid}_${oname}: marker exists"
    return 0
  fi
  local cmd="CUDA_VISIBLE_DEVICES=${GPU_IDS} ${PYTHON} -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_addr=${MASTER_ADDR} --master_port=$(rand_port) tools/run/train_seq3_episode.py --base-config ${CFG} --source-ckpt ${SOURCE_CKPT} --target-order ${domains} --eval-domains ${ALL_DOMAINS[*]} --work-dir ${work}/episode --epochs-per-stage 5 --samples-per-gpu ${SAMPLES_PER_GPU} --workers-per-gpu ${WORKERS_PER_GPU} --launcher pytorch --no-validate > ${work}/logs/train_episode.log 2>&1"
  write_text "${work}/train_command.txt" "${cmd}"
  run_cmd "${cmd}"
  if [[ "${DRY_RUN}" == "0" ]]; then
    touch "${marker}"
  fi
}

run_eval_one() {
  local oid="$1" stage_idx="$2" adapted_dom="$3" eval_dom="$4"
  local oname; oname=$(order_name "${oid}")
  local domains; domains=$(order_domains "${oid}")
  local method="${BASE_METHOD}_${oid}_${oname}"
  local work="${OUT_ROOT}/${oid}_${oname}"
  local stage; stage=$(stage_name "${stage_idx}" "${adapted_dom}")
  local ckpt="${work}/episode/adapt/${stage}/epoch_5.pth"
  local outdir="${work}/forgetting_eval/${stage}/epoch_5/${eval_dom}"
  local marker="${outdir}/eval_done.marker"
  if [[ ! -f "${ckpt}" && "${DRY_RUN}" == "0" ]]; then
    echo "[ERROR] missing checkpoint: ${ckpt}" >&2
    exit 4
  fi
  if [[ "${FORCE}" == "0" && -f "${marker}" ]]; then
    echo "[SKIP EVAL] ${method} ${stage} -> ${eval_dom}: marker exists"
    return 0
  fi
  mkdir -p "${outdir}"
  cp "$(test_cfg "${eval_dom}")" "${outdir}/used_config.py"
  write_text "${outdir}/group.txt" "${GROUP}"
  write_text "${outdir}/method.txt" "${method}"
  write_text "${outdir}/base_method.txt" "${BASE_METHOD}"
  write_text "${outdir}/order_id.txt" "${oid}"
  write_text "${outdir}/order_name.txt" "${oname}"
  write_text "${outdir}/target_order.txt" "${domains}"
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

eval_order() {
  local oid="$1"
  local domains; domains=$(order_domains "${oid}")
  local arr=(${domains})
  for i in "${!arr[@]}"; do
    local stage_idx=$((i + 1))
    local adapted_dom="${arr[$i]}"
    for eval_dom in "${ALL_DOMAINS[@]}"; do
      run_eval_one "${oid}" "${stage_idx}" "${adapted_dom}" "${eval_dom}"
    done
  done
}

preflight() {
  echo "============================================================"
  echo "ROI-WEAK ORDER SENSITIVITY"
  echo "============================================================"
  echo "ROOT=${ROOT}"
  echo "OUT_ROOT=${OUT_ROOT}"
  echo "CFG=${CFG}"
  echo "SOURCE_CKPT=${SOURCE_CKPT}"
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
  echo "orders:"
  for oid in "${ORDER_IDS[@]}"; do
    echo "  ${oid}_$(order_name ${oid}): $(order_domains ${oid})"
  done
  print_domain_configs
}

preflight
for oid in "${ORDER_IDS[@]}"; do
  train_order "${oid}"
  eval_order "${oid}"
done

if [[ "${DRY_RUN}" == "0" ]]; then
  ${PYTHON} tools/collect_roiweak_order_sensitivity.py
fi

echo "[DONE] roiweak order sensitivity experiments finished"
