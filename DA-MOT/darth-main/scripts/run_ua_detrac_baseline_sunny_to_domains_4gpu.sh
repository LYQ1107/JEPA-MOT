#!/usr/bin/env bash
# UA-DETRAC baseline pipeline:
#   Step 1: Source training — sunny 20e
#   Step 2: DARTH adaptation — cloudy / night / rainy, each 5e + final test
set -euo pipefail

[[ -f tools/run/train.py ]] || { echo "ERROR: run from project root" >&2; exit 1; }

GPU_IDS=${GPU_IDS:-"5,4,3,2"}
GPU_NUM=${GPU_NUM:-4}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-2}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
PYTHON=${PYTHON:-/home/lwr/anaconda3/envs/DARTH/bin/python}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

SOURCE_CFG="configs/mot/qdtrack/ua_detrac/qdtrack_faster-rcnn_r50_fpn_20e_ua_detrac_sunny.py"
SOURCE_WORK_DIR="work_dirs/ua_detrac/source/sunny_20e/qdtrack_ua_detrac_sunny_20e"
SOURCE_DONE="${SOURCE_WORK_DIR}/source_train_done.marker"

ADAPT_DOMAINS=(cloudy night rainy)
ADAPT_EPOCHS=5

get_port() {
    $PYTHON -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}

mkdir -p "${SOURCE_WORK_DIR}" \
         work_dirs/ua_detrac/source/sunny_20e/logs \
         work_dirs/ua_detrac/adapt/logs \
         work_dirs/ua_detrac/test/logs

# ── Step 1: Source training ──────────────────────────────────────────────────
echo "================================================================"
echo " STEP 1: Source training — UA-DETRAC sunny 20e"
echo "  config  : ${SOURCE_CFG}"
echo "  work_dir: ${SOURCE_WORK_DIR}"
echo "  GPUs    : ${GPU_IDS} (n=${GPU_NUM})"
echo "================================================================"

if [[ -f "${SOURCE_DONE}" ]]; then
    echo "[SKIP] source training already done (${SOURCE_DONE})"
else
    PORT=$(get_port)
    CUDA_VISIBLE_DEVICES=${GPU_IDS} \
        $PYTHON -m torch.distributed.launch \
            --nproc_per_node=${GPU_NUM} \
            --master_addr=${MASTER_ADDR} \
            --master_port=${PORT} \
            tools/run/train.py \
            ${SOURCE_CFG} \
            --launcher pytorch \
            --work-dir ${SOURCE_WORK_DIR} \
            --cfg-options \
                data.samples_per_gpu=${SAMPLES_PER_GPU} \
                data.workers_per_gpu=${WORKERS_PER_GPU} \
        2>&1 | tee work_dirs/ua_detrac/source/sunny_20e/logs/train.log
    date > "${SOURCE_DONE}"
fi

# Locate source checkpoint (subdir name matches config stem)
SOURCE_CKPT=$(ls -t ${SOURCE_WORK_DIR}/qdtrack_faster-rcnn_r50_fpn_20e_ua_detrac_sunny/*/epoch_20.pth 2>/dev/null | head -1)
[[ -f "${SOURCE_CKPT}" ]] || { echo "ERROR: source epoch_20.pth not found under ${SOURCE_WORK_DIR}" >&2; exit 1; }
echo "Source checkpoint: ${SOURCE_CKPT}"

# Patch checkpoint path into adapter configs
for DOMAIN in "${ADAPT_DOMAINS[@]}"; do
    CFG="configs/adapters/darth/ua_detrac/darth_qdtrack_faster-rcnn_r50_fpn_5e_ua_detrac_${DOMAIN}_from_sunny.py"
    sed -i "s|checkpoint=.*ua_detrac.*sunny.*epoch_20\.pth.*|checkpoint='${SOURCE_CKPT}',|" "${CFG}"
done

# ── Step 2: DARTH adaptation per target domain ───────────────────────────────
for DOMAIN in "${ADAPT_DOMAINS[@]}"; do
    ADAPT_CFG="configs/adapters/darth/ua_detrac/darth_qdtrack_faster-rcnn_r50_fpn_5e_ua_detrac_${DOMAIN}_from_sunny.py"
    TEST_CFG="configs/mot/qdtrack/ua_detrac/qdtrack_faster-rcnn_r50_fpn_20e_ua_detrac_${DOMAIN}_eval.py"
    ADAPT_WORK_DIR="work_dirs/ua_detrac/adapt/darth_sunny_to_${DOMAIN}_5e"
    TEST_WORK_DIR="work_dirs/ua_detrac/test/darth_sunny_to_${DOMAIN}_5e"
    ADAPT_LOG="work_dirs/ua_detrac/adapt/logs/sunny_to_${DOMAIN}_5e_$(date +%Y%m%d_%H%M%S).log"
    TEST_LOG="work_dirs/ua_detrac/test/logs/sunny_to_${DOMAIN}_5e_eval_$(date +%Y%m%d_%H%M%S).log"
    DONE_MARKER="${ADAPT_WORK_DIR}/adapt_done.marker"

    echo "================================================================"
    echo " ADAPT: sunny -> ${DOMAIN} (${ADAPT_EPOCHS}e)"
    echo "  adapt cfg : ${ADAPT_CFG}"
    echo "  work dir  : ${ADAPT_WORK_DIR}"
    echo "  GPUs      : ${GPU_IDS} (n=${GPU_NUM})"
    echo "================================================================"

    mkdir -p "${ADAPT_WORK_DIR}" "${TEST_WORK_DIR}"

    if [[ -f "${DONE_MARKER}" ]]; then
        echo "[SKIP] adaptation already done: ${DONE_MARKER}"
    else
        PORT=$(get_port)
        CUDA_VISIBLE_DEVICES=${GPU_IDS} \
            $PYTHON -m torch.distributed.launch \
                --nproc_per_node=${GPU_NUM} \
                --master_addr=${MASTER_ADDR} \
                --master_port=${PORT} \
                tools/run/train.py \
                ${ADAPT_CFG} \
                --launcher pytorch \
                --work-dir ${ADAPT_WORK_DIR} \
                --cfg-options \
                    data.samples_per_gpu=${SAMPLES_PER_GPU} \
                    data.workers_per_gpu=${WORKERS_PER_GPU} \
                    total_epochs=${ADAPT_EPOCHS} \
            2>&1 | tee "${ADAPT_LOG}"
        date > "${DONE_MARKER}"
    fi

    CKPT=$(ls -t ${ADAPT_WORK_DIR}/darth_qdtrack_faster-rcnn_r50_fpn_5e_ua_detrac_${DOMAIN}_from_sunny/*/epoch_${ADAPT_EPOCHS}.pth 2>/dev/null | head -1)
    [[ -f "${CKPT}" ]] || { echo "ERROR: adapt checkpoint not found for ${DOMAIN}" >&2; exit 1; }

    echo "================================================================"
    echo " TEST: sunny -> ${DOMAIN}"
    echo "  ckpt     : ${CKPT}"
    echo "================================================================"

    TEST_TMPDIR="${TEST_WORK_DIR}/tmp_collect_${DOMAIN}_$(date +%Y%m%d_%H%M%S)_$$"
    rm -rf "${TEST_TMPDIR}"
    mkdir -p "${TEST_TMPDIR}"

    PORT=$(get_port)
    CUDA_VISIBLE_DEVICES=${GPU_IDS} \
        $PYTHON -m torch.distributed.launch \
            --nproc_per_node=${GPU_NUM} \
            --master_addr=${MASTER_ADDR} \
            --master_port=${PORT} \
            tools/run/test.py \
            ${TEST_CFG} \
            --checkpoint "${CKPT}" \
            --launcher pytorch \
            --work-dir ${TEST_WORK_DIR} \
            --tmpdir "${TEST_TMPDIR}" \
            --eval bbox track \
            --cfg-options data.workers_per_gpu=${WORKERS_PER_GPU} \
        2>&1 | tee "${TEST_LOG}"
    rm -rf "${TEST_TMPDIR}"

    echo "[DONE] ${DOMAIN}"
done

echo "[ALL DONE] UA-DETRAC baseline pipeline finished."
echo "  Source ckpt : ${SOURCE_CKPT}"
