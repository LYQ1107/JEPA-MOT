#!/usr/bin/env bash
set -euo pipefail

[[ -f tools/run/train.py ]] || { echo "ERROR: run from project root" >&2; exit 1; }

GPU_IDS=${GPU_IDS:-"5,4,3,2"}
GPU_NUM=${GPU_NUM:-4}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-2}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
PYTHON=${PYTHON:-/home/lwr/anaconda3/envs/DARTH/bin/python}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}

SOURCE_CKPT="work_dirs/vkitti2/source/clone_20e/qdtrack_vkitti2_clone_20e/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_clone/20260707_171736/epoch_20.pth"
[[ -f "$SOURCE_CKPT" ]] || { echo "ERROR: source checkpoint not found: $SOURCE_CKPT" >&2; exit 1; }

DOMAINS=(fog rain sunset)

get_port() {
    $PYTHON -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}

mkdir -p work_dirs/vkitti2/adapt/logs work_dirs/vkitti2/test/logs

for DOMAIN in "${DOMAINS[@]}"; do
    ADAPT_CFG="configs/adapters/darth/vkitti2/darth_qdtrack_faster-rcnn_r50_fpn_5e_vkitti2_${DOMAIN}_from_clone.py"
    TEST_CFG="configs/mot/qdtrack/vkitti2/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_${DOMAIN}_eval.py"
    ADAPT_WORK_DIR="work_dirs/vkitti2/adapt/darth_clone_to_${DOMAIN}_5e"
    TEST_WORK_DIR="work_dirs/vkitti2/test/darth_clone_to_${DOMAIN}_5e"
    ADAPT_LOG="work_dirs/vkitti2/adapt/logs/clone_to_${DOMAIN}_5e_$(date +%Y%m%d_%H%M%S).log"
    TEST_LOG="work_dirs/vkitti2/test/logs/clone_to_${DOMAIN}_5e_eval_$(date +%Y%m%d_%H%M%S).log"
    DONE_MARKER="${ADAPT_WORK_DIR}/adapt_done.marker"

    echo "================================================================"
    echo " DOMAIN: clone -> ${DOMAIN} (${TOTAL_EPOCHS}e)"
    echo "  adapt cfg : ${ADAPT_CFG}"
    echo "  test cfg  : ${TEST_CFG}"
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
                    total_epochs=${TOTAL_EPOCHS} \
            2>&1 | tee "${ADAPT_LOG}"
        date > "${DONE_MARKER}"
    fi

    CKPT=$(ls -t ${ADAPT_WORK_DIR}/darth_qdtrack_faster-rcnn_r50_fpn_5e_vkitti2_${DOMAIN}_from_clone/*/epoch_${TOTAL_EPOCHS}.pth 2>/dev/null | head -1)
    [[ -f "${CKPT}" ]] || { echo "ERROR: checkpoint not found for ${DOMAIN}: ${ADAPT_WORK_DIR}" >&2; exit 1; }

    echo "================================================================"
    echo " TEST: clone -> ${DOMAIN} final checkpoint"
    echo "  ckpt     : ${CKPT}"
    echo "  test dir : ${TEST_WORK_DIR}"
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

echo "[ALL DONE] fog / rain / sunset finished."
