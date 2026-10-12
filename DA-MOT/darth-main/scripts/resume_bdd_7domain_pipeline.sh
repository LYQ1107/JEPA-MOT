#!/usr/bin/env bash
set -eo pipefail
cd /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main
eval "$(conda shell.bash hook)"
conda activate DARTH

case "${1:-}" in
  --status-only)
    python tools/bdd_7domain_manager.py status
    DAEMON_PID_FILE="work_dirs/bdd_7domain/state/daemon.pid"
    if [ -f "$DAEMON_PID_FILE" ]; then
      dpid=$(cat "$DAEMON_PID_FILE")
      if kill -0 "$dpid" 2>/dev/null; then
        echo "DAEMON running PID=$dpid"
        echo "--- daemon.log (last 20 lines) ---"
        tail -20 work_dirs/bdd_7domain/daemon.log 2>/dev/null || true
      else
        echo "DAEMON not running (stale pid=$dpid)"
      fi
    else
      echo "DAEMON not running (no pid file)"
    fi
    ;;
  --resume-next)
    python tools/bdd_7domain_manager.py status
    next_job=$(python3 - <<'PY'
import csv
jobs = list(csv.DictReader(open('work_dirs/bdd_7domain/state/jobs.csv')))
j = next((x for x in jobs if x['status'] in ('pending','interrupted','failed_or_incomplete')), None)
print(j['job_id'] if j else 'NONE')
PY
)
    if [ "$next_job" = "NONE" ]; then
      echo "No pending job found."
    else
      echo "Starting job: $next_job"
      python tools/bdd_7domain_manager.py run-job --job-id "$next_job"
    fi
    ;;
  --start-daemon)
    DAEMON_LOG="work_dirs/bdd_7domain/daemon.log"
    DAEMON_PID_FILE="work_dirs/bdd_7domain/state/daemon.pid"
    if [ -f "$DAEMON_PID_FILE" ]; then
      dpid=$(cat "$DAEMON_PID_FILE")
      if kill -0 "$dpid" 2>/dev/null; then
        echo "Daemon already running PID=$dpid. Use --status-only to check."
        exit 0
      fi
    fi
    nohup bash scripts/run_full_matrix_daemon.sh >> "$DAEMON_LOG" 2>&1 &
    echo "Daemon started PID=$!"
    echo "Monitor: tail -f $DAEMON_LOG"
    echo "Status:  bash scripts/resume_bdd_7domain_pipeline.sh --status-only"
    ;;
  *)
    echo "Usage: $0 --status-only | --resume-next | --start-daemon"
    exit 2
    ;;
esac
