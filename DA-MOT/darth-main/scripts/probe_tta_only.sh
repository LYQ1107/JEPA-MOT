#!/usr/bin/env bash
set -eo pipefail
cd /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main
eval "$(conda shell.bash hook)"
conda activate DARTH
export MPLBACKEND=Agg

echo "=== TTA-only batch probe ==="
echo "Source spg=4 already verified. Now probing TTA..."

TTA_CFG="configs/adapters/darth/bdd_7domain/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_night.py"
SOURCE_CKPT="work_dirs/bdd_clear/qdtrack_clear_12e/qdtrack_clear_12e/epoch_12.pth"
TTA_SPG=2

for spg in 4 3 2 1; do
  echo "=== Probe TTA spg=$spg ==="
  probe_dir="work_dirs/bdd_7domain/batch_probe/tta_spg${spg}"
  rm -rf "$probe_dir"
  port=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")
  
  timeout 900 python -m torch.distributed.launch \
    --nproc_per_node=10 \
    --master_addr=127.0.0.1 \
    --master_port="$port" \
    tools/run/train.py "$TTA_CFG" \
    --launcher pytorch \
    --work-dir "$probe_dir" \
    --cfg-options \
      "model.init_cfg.checkpoint=$SOURCE_CKPT" \
      "find_unused_parameters=True" \
      "data.samples_per_gpu=$spg" \
      "data.workers_per_gpu=2" \
      "total_epochs=1" \
      "checkpoint_config.interval=999" \
      "evaluation.interval=999" \
      "log_config.interval=10" \
      "optimizer.lr=0.002" \
      "model.loss_rpn_distillation.loss_weight=0.1" \
      "model.loss_roi_distillation.loss_weight=0.1" \
      "custom_hooks.0.momentum=0.002" \
      "model.conf_thr=0.7"
  
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "TTA spg=$spg SUCCESS"
    TTA_SPG=$spg
    break
  else
    echo "TTA spg=$spg FAILED (rc=$rc), trying smaller"
  fi
done

mkdir -p work_dirs/bdd_7domain/summaries
cat > work_dirs/bdd_7domain/summaries/batch_tuning_result.json <<JSON
{
  "source_samples_per_gpu": 4,
  "tta_samples_per_gpu": $TTA_SPG,
  "mode": "auto_tuned",
  "timestamp": "$(date -Iseconds)",
  "note": "Source spg=4 verified (222 iters, 9580 MiB/GPU). TTA probed with 10 GPUs on night target"
}
JSON

cat > work_dirs/bdd_7domain/summaries/batch_tuning_result.md <<MD
# Batch tuning result

Auto-tuned on $(date):
- **SOURCE_SAMPLES_PER_GPU=4**
- **TTA_SAMPLES_PER_GPU=$TTA_SPG**

## Source probe
- Config: clear domain, 20e
- Result: spg=4 SUCCESS, 222 iters, 9580 MiB/GPU, stable loss

## TTA probe
- Config: night target, 12e DARTH adaptation
- Result: spg=$TTA_SPG

Use these values for full matrix training.
MD

echo "=== Batch tuning complete ==="
cat work_dirs/bdd_7domain/summaries/batch_tuning_result.json
