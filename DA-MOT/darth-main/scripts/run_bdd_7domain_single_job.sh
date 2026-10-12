#!/usr/bin/env bash
set -eo pipefail
PROJECT_DIR="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main"
cd "$PROJECT_DIR"
JOB_ID=""
while [ $# -gt 0 ]; do case "$1" in --job-id) JOB_ID="$2"; shift 2;; *) echo "Unknown arg $1"; exit 2;; esac; done
[ -n "$JOB_ID" ] || { echo "--job-id required"; exit 2; }
set +u
eval "$(conda shell.bash hook)"
conda activate DARTH
set -u
mkdir -p work_dirs/bdd_7domain/state/running work_dirs/bdd_7domain/state/done work_dirs/bdd_7domain/state/failed work_dirs/bdd_7domain/state/pids work_dirs/bdd_7domain/error_summaries
INFO=$(python - <<PY
import csv
jid='$JOB_ID'
for r in csv.DictReader(open('work_dirs/bdd_7domain/state/jobs.csv')):
    if r['job_id']==jid:
        print('|'.join([r['stage'],r['source'],r['target'],r['source_epoch'],r['tta_epoch']]))
        break
else:
    raise SystemExit('job not found')
PY
)
IFS='|' read -r STAGE SOURCE TARGET SOURCE_EPOCH TTA_EPOCH <<< "$INFO"
RUN="work_dirs/bdd_7domain/state/running/${JOB_ID}.running"
DONE="work_dirs/bdd_7domain/state/done/${JOB_ID}.done"
FAIL="work_dirs/bdd_7domain/state/failed/${JOB_ID}.failed"
PIDF="work_dirs/bdd_7domain/state/pids/${JOB_ID}.pid"
LOG_DIR="work_dirs/bdd_7domain/logs/${STAGE}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${JOB_ID}_$(date +%Y%m%d_%H%M%S).log"
if [ -f "$DONE" ]; then echo "Job already done: $JOB_ID"; exit 0; fi
echo $$ > "$PIDF"
echo "$(date) $JOB_ID $STAGE" > "$RUN"
set +e
{
  echo "=== RUN JOB $JOB_ID stage=$STAGE source=$SOURCE target=$TARGET source_epoch=$SOURCE_EPOCH tta_epoch=$TTA_EPOCH ==="
  if [ "$STAGE" = "prepare_data" ]; then
    bash scripts/prepare_bdd_7domain_cocovid.sh
  elif [ "$STAGE" = "generate_config" ]; then
    python tools/generate_bdd_7domain_configs.py
  elif [ "$STAGE" = "batch_probe_source" ] || [ "$STAGE" = "batch_probe_tta" ]; then
    bash scripts/auto_tune_bdd_7domain_batch.sh
  elif [ "$STAGE" = "summary" ]; then
    python tools/analysis/parse_bdd_7domain_results.py
  else
    echo "Long training/eval job command generation is staged but not auto-started by this skeleton yet: $STAGE"
    echo "Use scripts/run_bdd_7domain_full_matrix.sh after dry-run/mini-smoke validation."
    exit 3
  fi
} > "$LOG" 2>&1
RC=$?
set -e
python - <<PY
import csv, datetime
p='work_dirs/bdd_7domain/state/jobs.csv'; jid='$JOB_ID'; rc=$RC; log='$LOG'
rows=list(csv.DictReader(open(p)))
fields=rows[0].keys() if rows else []
for r in rows:
    if r['job_id']==jid:
        r['status']='completed' if rc==0 else 'failed'
        r['end_time']=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        r['log_path']=log
with open(p,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
PY
rm -f "$RUN"
if [ $RC -eq 0 ]; then echo "$(date)" > "$DONE"; else echo "$(date) rc=$RC" > "$FAIL"; grep -A80 -B20 -E "Traceback|RuntimeError|ChildFailedError|CUDA out of memory|NCCL|Error" "$LOG" > "work_dirs/bdd_7domain/error_summaries/${JOB_ID}_error_summary.txt" || tail -120 "$LOG" > "work_dirs/bdd_7domain/error_summaries/${JOB_ID}_error_summary.txt"; fi
python tools/bdd_7domain_manager.py status > /dev/null || true
exit $RC
