#!/bin/bash
# Reset all 6 domains to train from scratch (epoch 1), then start daemon
set -e

ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main"
STATE="$ROOT/work_dirs/bdd_7domain/state"

echo "[1/6] Kill all training/daemon processes..."
pkill -9 -f "train.py|test.py|bdd_7domain_manager|run_full_matrix|torch.distributed" 2>/dev/null || true
sleep 3

echo "[2/6] Delete old checkpoints for 6 domains..."
for domain in dawn_dusk night partly_cloudy overcast rainy snowy; do
    rm -rf "$ROOT/work_dirs/bdd_7domain/source/${domain}/qdtrack_20e/qdtrack_faster-rcnn_r50_fpn_20e_bdd_${domain}"
    echo "  Deleted: $domain"
done

echo "[3/6] Clean all state marker files..."
rm -f "$STATE/daemon.pid" "$STATE/daemon_heartbeat.txt"
find "$STATE/running" -name "*.running" -delete 2>/dev/null || true
find "$STATE/pids"    -name "*.pid"     -delete 2>/dev/null || true
find "$STATE/failed"  -name "*.failed"  -delete 2>/dev/null || true

echo "[4/6] Delete dawn_dusk done marker (retrain with new thresholds)..."
rm -f "$STATE/done/source_train_dawn_dusk_20e.done"

echo "[5/6] Sync jobs.csv: done marker = source of truth..."
python3 - << 'PY'
import csv
from pathlib import Path

ROOT  = Path('/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main')
STATE = ROOT / 'work_dirs/bdd_7domain/state'
jobs_file = STATE / 'jobs.csv'

with jobs_file.open() as f:
    jobs = list(csv.DictReader(f))

for j in jobs:
    if Path(j['done_marker']).exists():
        j['status'] = 'completed'
    else:
        j['status'] = 'pending'
        j['start_time'] = ''
        j['end_time'] = ''

with jobs_file.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=jobs[0].keys())
    w.writeheader()
    w.writerows(jobs)

counts = {}
for j in jobs:
    counts[j['status']] = counts.get(j['status'], 0) + 1
print('  Status counts:', counts)
print('  Source training jobs:')
for j in jobs:
    if j['stage'] == 'source_train':
        tag = 'DONE ' if Path(j['done_marker']).exists() else 'fresh'
        print(f'    [{tag}] {j["job_id"]}: {j["status"]}')
PY

echo ""
echo "[6/6] Start daemon..."
cd "$ROOT"
eval "$(conda shell.bash hook)"
conda activate DARTH

DAEMON_LOG="work_dirs/bdd_7domain/daemon.log"
nohup bash scripts/run_full_matrix_daemon.sh >> "$DAEMON_LOG" 2>&1 &
DPID=$!
echo "  Daemon PID=$DPID"

sleep 6
echo ""
echo "=== Daemon log (last 10 lines) ==="
tail -n 10 "$DAEMON_LOG"
echo ""
echo "Done. Monitor with:"
echo "  tail -f $ROOT/$DAEMON_LOG"
