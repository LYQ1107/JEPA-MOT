#!/bin/bash
set -e
cd /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main

CKPT="work_dirs/motsynth/trimodule_spaP1_causalProtoShrink_roiV1Weak/episode/adapt/stage01_after_cloudy/epoch_5.pth"
CFG="configs/mot/qdtrack/motsynth/qdtrack_faster-rcnn_r50_fpn_20e_motsynth_cloudy_eval.py"
export CUDA_VISIBLE_DEVICES=0,1

for THR in 0.20 0.30 0.35 0.40 0.50; do
    TAG="thr$(echo $THR | tr -d '.')"
    WDIR="work_dirs/motsynth/test/thr_sweep/cloudy/${TAG}"
    echo "====== THR=${THR} ======"
    python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_addr=127.0.0.1 \
        --master_port=29521 \
        tools/run/test.py "$CFG" \
        --checkpoint "$CKPT" \
        --launcher pytorch \
        --work-dir "$WDIR" \
        --eval bbox track \
        --fuse-conv-bn \
        --cfg-options data.workers_per_gpu=4 cudnn_benchmark=True \
          model.tracker.init_score_thr=${THR} \
          model.tracker.obj_score_thr=${THR}
    echo "====== THR=${THR} done ======"
done
echo "ALL DONE"
