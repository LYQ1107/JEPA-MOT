#!/usr/bin/env bash
set -eo pipefail
cd /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main
set +u
eval "$(conda shell.bash hook)"
conda activate DARTH
set -u
export MPLBACKEND=Agg
mkdir -p work_dirs/bdd_7domain/batch_probe
LOG="work_dirs/bdd_7domain/batch_probe/auto_tune_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== BDD 7-domain batch auto-tune ==="
echo "Strategy: probe source training (clear, 50 iters) and TTA (night target, 50 iters)"
echo "Test sequence: spg=4,3,2,1 until success"

SOURCE_CFG="configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear.py"
TTA_CFG="configs/adapters/darth/bdd_7domain/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_night.py"
SOURCE_CKPT="work_dirs/bdd_clear/qdtrack_clear_12e/qdtrack_clear_12e/epoch_12.pth"

probe_source() {
  local spg=$1
  local probe_dir="work_dirs/bdd_7domain/batch_probe/source_spg${spg}"
  local port
  port=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")
  echo "=== Probe SOURCE spg=$spg port=$port ==="
  rm -rf "$probe_dir"
  timeout 900 python -m torch.distributed.launch --nproc_per_node=10 --master_addr=127.0.0.1 --master_port="$port" tools/run/train.py "$SOURCE_CFG" --launcher pytorch --work-dir "$probe_dir" --cfg-options "data.samples_per_gpu=$spg" "data.workers_per_gpu=2" "total_epochs=1" "checkpoint_config.interval=999" "evaluation.interval=999" "log_config.interval=10"
  local rc=$?
  [ $rc -eq 0 ] && echo "SOURCE spg=$spg SUCCESS" && return 0
  echo "SOURCE spg=$spg FAILED (rc=$rc)"
  return $rc
}

probe_tta() {
  local spg=$1
  local probe_dir="work_dirs/bdd_7domain/batch_probe/tta_spg${spg}"
  local port
  port=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")
  echo "=== Probe TTA spg=$spg port=$port ==="
  rm -rf "$probe_dir"
  timeout 900 python -m torch.distributed.launch --nproc_per_node=10 --master_addr=127.0.0.1 --master_port="$port" tools/run/train.py "$TTA_CFG" --launcher pytorch --work-dir "$probe_dir" --cfg-options "model.init_cfg.checkpoint=$SOURCE_CKPT" "find_unused_parameters=True" "data.samples_per_gpu=$spg" "data.workers_per_gpu=2" "total_epochs=1" "checkpoint_config.interval=999" "evaluation.interval=999" "log_config.interval=10" "optimizer.lr=0.002" "model.loss_rpn_distillation.loss_weight=0.1" "model.loss_roi_distillation.loss_weight=0.1" "custom_hooks.0.momentum=0.002" "model.conf_thr=0.7"
  local rc=$?
  [ $rc -eq 0 ] && echo "TTA spg=$spg SUCCESS" && return 0
  echo "TTA spg=$spg FAILED (rc=$rc)"
  return $rc
}

SOURCE_SPG=2
for spg in 4 3 2 1; do
  if probe_source $spg; then
    SOURCE_SPG=$spg
    break
  fi
  echo "SOURCE spg=$spg FAILED, trying smaller"
done

TTA_SPG=2
for spg in 4 3 2 1; do
  if probe_tta $spg; then
    TTA_SPG=$spg
    break
  fi
  echo "TTA spg=$spg FAILED, trying smaller"
done

cat > work_dirs/bdd_7domain/summaries/batch_tuning_result.json <<JSON
{
  "source_samples_per_gpu": $SOURCE_SPG,
  "tta_samples_per_gpu": $TTA_SPG,
  "mode": "auto_tuned",
  "timestamp": "$(date -Iseconds)",
  "note": "Probed with 10 GPUs, 50 iters each, clear source and night TTA target"
}
JSON

cat > work_dirs/bdd_7domain/summaries/batch_tuning_result.md <<MD
# Batch tuning result

Auto-tuned on $(date):
- SOURCE_SAMPLES_PER_GPU=$SOURCE_SPG
- TTA_SAMPLES_PER_GPU=$TTA_SPG

Probe: 10 GPUs × 50 iters, clear source training + night TTA adaptation

Log: $LOG
MD

echo "=== Batch tuning complete ==="
cat work_dirs/bdd_7domain/summaries/batch_tuning_result.json
