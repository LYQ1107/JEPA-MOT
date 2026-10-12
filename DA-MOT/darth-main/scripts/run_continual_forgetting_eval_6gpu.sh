#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main"
cd "${ROOT}"

GPU_IDS=${GPU_IDS:-"9,8,7,6,4,5"}
GPU_NUM=${GPU_NUM:-6}
TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU:-6}
DRY_RUN=${DRY_RUN:-0}
FORCE=${FORCE:-0}
PYTHON=${PYTHON:-/home/lwr/anaconda3/envs/DARTH/bin/python}
RUN_CAUSAL=${RUN_CAUSAL:-1}
RUN_TRI_WEAK=${RUN_TRI_WEAK:-1}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

METHOD_A="spaP1_causalOld_4domain_5e"
METHOD_B="spaP1_causalOld_roiV1Weak_4domain_5e"
GROUP="forgetting_eval"
OUT_ROOT="work_dirs/bdd_4domain/continual_forgetting_eval"
COMBO_ROOT="work_dirs/bdd_4domain/module_combinations"
TEST_CFG_ROOT="configs/mot/qdtrack/bdd_7domain"
CFG_A="configs/adapters/darth/bdd_4domain/module_combinations/darth_spaP1_causalOld_4domain_5e.py"
CFG_B="configs/adapters/darth/bdd_4domain/module_combinations/darth_spaP1_causalOld_roiV1Weak_4domain_5e.py"
THR_OVERRIDE="model.tracker.init_score_thr=0.30"
DOMAINS=(overcast rainy dawn_dusk night)
STAGES=(stage01_after_overcast stage02_after_rainy stage03_after_dawn_dusk stage04_after_night)
STAGE_DOMAINS=(overcast rainy dawn_dusk night)

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

method_cfg() {
  local method="$1"
  if [[ "${method}" == "${METHOD_A}" ]]; then echo "${CFG_A}"; else echo "${CFG_B}"; fi
}

locate_ckpt() {
  local method="$1" stage="$2"
  ${PYTHON} - "$COMBO_ROOT" "$method" "$stage" <<'PY'
import os, sys
root, method, stage = sys.argv[1:4]
base = os.path.join(root, method)
if not os.path.isdir(base):
    raise SystemExit(f"[ERROR] method work_dir not found: {base}")
cands = []
for dirpath, _, files in os.walk(base):
    for f in files:
        if f.endswith('.pth') and 'epoch' in f and '5' in f:
            path = os.path.join(dirpath, f)
            if stage in path:
                cands.append(path)
if not cands:
    raise SystemExit(f"[ERROR] no epoch5 checkpoint found for {method} {stage} under {base}")
def score(p):
    bn = os.path.basename(p)
    s = 0
    if bn == 'epoch_5.pth': s += 100
    if os.path.basename(os.path.dirname(p)) == stage: s += 50
    if f"adapt/{stage}/epoch_5.pth" in p: s += 30
    return (-s, len(p), p)
cands = sorted(cands, key=score)
print(cands[0])
PY
}

print_domain_configs() {
  echo "============================================================"
  echo "EVAL DOMAIN CONFIGS"
  echo "============================================================"
  ${PYTHON} - <<'PY'
from mmcv import Config
DOMS=['overcast','rainy','dawn_dusk','night']
for d in DOMS:
    p=f'configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_{d}.py'
    c=Config.fromfile(p)
    test=c.data.test
    print(f"[{d}] cfg={p}")
    print(f"  ann_file={test.ann_file}")
    print(f"  img_prefix={test.img_prefix}")
PY
}

preflight_method() {
  local method="$1"
  local cfg; cfg=$(method_cfg "$method")
  echo "============================================================"
  echo "METHOD ${method}"
  echo "============================================================"
  echo "config=${cfg}"
  if [[ ! -f "${cfg}" ]]; then
    echo "[ERROR] config not found: ${cfg}" >&2
    exit 3
  fi
  for stage in "${STAGES[@]}"; do
    local ckpt; ckpt=$(locate_ckpt "$method" "$stage")
    echo "${stage} epoch5 checkpoint=${ckpt}"
  done
}

run_eval_one() {
  local method="$1" stage="$2" eval_dom="$3" adapted_dom="$4"
  local cfg; cfg=$(method_cfg "$method")
  local ckpt; ckpt=$(locate_ckpt "$method" "$stage")
  local outdir="${OUT_ROOT}/${method}/${stage}/epoch_5/${eval_dom}"
  local marker="${outdir}/eval_done.marker"
  if [[ "${FORCE}" == "0" && -f "${marker}" ]]; then
    echo "[SKIP] ${method} ${stage} -> ${eval_dom}: marker exists"
    return 0
  fi
  mkdir -p "${outdir}"
  cp "$(test_cfg "$eval_dom")" "${outdir}/used_config.py"
  write_text "${outdir}/group.txt" "${GROUP}"
  write_text "${outdir}/method.txt" "${method}"
  write_text "${outdir}/stage.txt" "${stage}"
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

run_method() {
  local method="$1"
  for i in "${!STAGES[@]}"; do
    local stage="${STAGES[$i]}"
    local adapted="${STAGE_DOMAINS[$i]}"
    for dom in "${DOMAINS[@]}"; do
      run_eval_one "$method" "$stage" "$dom" "$adapted"
    done
  done
}

echo "============================================================"
echo "CONTINUAL FORGETTING / ALL-DOMAIN EVAL"
echo "============================================================"
echo "ROOT=${ROOT}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "GPU_IDS=${GPU_IDS}"
echo "GPU_NUM=${GPU_NUM}"
echo "TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU}"
echo "DRY_RUN=${DRY_RUN} FORCE=${FORCE}"
echo "THR_OVERRIDE=${THR_OVERRIDE}"
echo "only_eval=true"
echo "modify_test_py=false"
echo "use_gpu_0_1_2_3=false"

echo "============================================================"
echo "CONFIG PATHS"
echo "============================================================"
echo "${METHOD_A}: ${CFG_A}"
echo "${METHOD_B}: ${CFG_B}"

print_domain_configs
if [[ "${RUN_CAUSAL}" == "1" ]]; then preflight_method "${METHOD_A}"; fi
if [[ "${RUN_TRI_WEAK}" == "1" ]]; then preflight_method "${METHOD_B}"; fi

if [[ "${RUN_CAUSAL}" == "1" ]]; then run_method "${METHOD_A}"; fi
if [[ "${RUN_TRI_WEAK}" == "1" ]]; then run_method "${METHOD_B}"; fi

if [[ "${DRY_RUN}" == "0" ]]; then
  ${PYTHON} tools/collect_continual_forgetting_eval.py
fi

echo "[DONE] continual forgetting eval finished"
