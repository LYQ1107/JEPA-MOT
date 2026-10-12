#!/bin/bash
set -e
cd /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main
export CUDA_VISIBLE_DEVICES=2,3,4,5

PYTHON="/home/lwr/anaconda3/envs/DARTH/bin/python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29530"
TEST="tools/run/test.py"
CFGD="configs/mot/qdtrack/vkitti2"
OPTS="--eval bbox track --fuse-conv-bn --cfg-options data.workers_per_gpu=4 cudnn_benchmark=True model.tracker.init_score_thr=0.20 model.tracker.obj_score_thr=0.20"

SRC="work_dirs/vkitti2/source/clone_20e/qdtrack_vkitti2_clone_20e/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_clone/20260707_171736/epoch_20.pth"

echo "=== source_only x5 ==="
for DOM in fog morning overcast rain sunset; do
    echo "-- source_only/${DOM}"
    $PYTHON $TEST "${CFGD}/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_${DOM}_eval.py" \
        --checkpoint "$SRC" --launcher pytorch \
        --work-dir "work_dirs/vkitti2/test/source_clone_to_${DOM}_direct" $OPTS
done

echo "=== darth_5e fog/morning/rain/sunset ==="
declare -A DCKPT
DCKPT[fog]="work_dirs/vkitti2/adapt/darth_clone_to_fog_5e/darth_qdtrack_faster-rcnn_r50_fpn_5e_vkitti2_fog_from_clone/20260708_000124/epoch_5.pth"
DCKPT[morning]="work_dirs/vkitti2/adapt/darth_clone_to_morning_5e/darth_qdtrack_faster-rcnn_r50_fpn_5e_vkitti2_morning_from_clone/20260707_220821/epoch_5.pth"
DCKPT[rain]="work_dirs/vkitti2/adapt/darth_clone_to_rain_5e/darth_qdtrack_faster-rcnn_r50_fpn_5e_vkitti2_rain_from_clone/20260708_003049/epoch_5.pth"
DCKPT[sunset]="work_dirs/vkitti2/adapt/darth_clone_to_sunset_5e/darth_qdtrack_faster-rcnn_r50_fpn_5e_vkitti2_sunset_from_clone/20260708_010029/epoch_5.pth"
for DOM in fog morning rain sunset; do
    echo "-- darth_5e/${DOM}"
    $PYTHON $TEST "${CFGD}/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_${DOM}_eval.py" \
        --checkpoint "${DCKPT[$DOM]}" --launcher pytorch \
        --work-dir "work_dirs/vkitti2/test/darth_clone_to_${DOM}_5e" $OPTS
done

echo "=== darth_5e/overcast ==="
$PYTHON $TEST "${CFGD}/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_overcast_eval.py" \
    --checkpoint "work_dirs/vkitti2/adapt/darth_baseline_overcast_kdinit_10e/darth_qdtrack_faster-rcnn_r50_fpn_5e_vkitti2_overcast_from_clone/20260707_204824/epoch_5.pth" \
    --launcher pytorch \
    --work-dir "work_dirs/vkitti2/test/darth_clone_to_overcast_5e" $OPTS

echo "===== ALL DONE ====="
