#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate DARTH
export MPLBACKEND=Agg

CKPT="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main/work_dirs/bdd_7domain/source/clear/qdtrack_20e/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear/20260610_232556/epoch_20.pth"
CFG="configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear.py"
WORK_DIR="work_dirs/bdd_7domain/threshold_test"

PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()")

python -m torch.distributed.launch \
  --nproc_per_node=10 \
  --master_addr=127.0.0.1 \
  --master_port=$PORT \
  tools/run/test.py $CFG \
  --checkpoint $CKPT \
  --launcher pytorch \
  --work-dir $WORK_DIR \
  --eval bbox track \
  --cfg-options data.workers_per_gpu=4

echo "=== Results ==="
tail -50 $WORK_DIR/*.log | grep -E "MOTA|mAP"
