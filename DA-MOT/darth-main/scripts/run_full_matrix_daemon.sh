#!/usr/bin/env bash
# Full matrix daemon - survives training failures, SSH disconnects, GPU errors.
# Usage:
#   bash scripts/resume_bdd_7domain_pipeline.sh --start-daemon
#   Monitor: tail -f work_dirs/bdd_7domain/daemon.log

cd /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main
eval "$(conda shell.bash hook)"
conda activate DARTH
export MPLBACKEND=Agg OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1

DAEMON_PID_FILE="work_dirs/bdd_7domain/state/daemon.pid"
HEARTBEAT_FILE="work_dirs/bdd_7domain/state/daemon_heartbeat.txt"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

cleanup() {
    log "Daemon received SIGINT/SIGTERM - shutting down cleanly"
    rm -f "$DAEMON_PID_FILE" "$HEARTBEAT_FILE"
    exit 0
}

# Only trap explicit signals. Do NOT trap EXIT:
# EXIT trap would fire whenever any command fails with set -e,
# causing the daemon to silently die on any training error.
trap cleanup SIGINT SIGTERM

if [ -f "$DAEMON_PID_FILE" ]; then
    existing_pid=$(cat "$DAEMON_PID_FILE")
    if kill -0 "$existing_pid" 2>/dev/null; then
        log "ERROR: daemon already running PID=$existing_pid. Exiting."
        exit 1
    fi
    rm -f "$DAEMON_PID_FILE"
fi

echo $$ > "$DAEMON_PID_FILE"
log "=== Full matrix daemon started. PID=$$ ==="
log "Monitor: tail -f work_dirs/bdd_7domain/daemon.log"

job_count=0
consecutive_errors=0

while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') | completed=$job_count" > "$HEARTBEAT_FILE"

    # Read next pending job - wrapped in subshell so any python error never kills the loop
    next_job=$(python3 - 2>/dev/null <<'PY'
import csv, sys
try:
    jobs = list(csv.DictReader(open('work_dirs/bdd_7domain/state/jobs.csv')))
    j = next((x for x in jobs if x['status'] in ('pending', 'interrupted')), None)
    print(j['job_id'] if j else 'NONE')
except Exception:
    print('READ_ERROR')
    sys.exit(0)
PY
    )

    if [ "$next_job" = "READ_ERROR" ] || [ -z "$next_job" ]; then
        consecutive_errors=$((consecutive_errors + 1))
        log "WARNING: failed to read jobs.csv (attempt $consecutive_errors) - retrying in 30s"
        if [ "$consecutive_errors" -ge 5 ]; then
            log "ERROR: could not read jobs.csv after 5 attempts - daemon aborting"
            rm -f "$DAEMON_PID_FILE" "$HEARTBEAT_FILE"
            exit 1
        fi
        sleep 30
        continue
    fi
    consecutive_errors=0

    if [ "$next_job" = "NONE" ]; then
        log "All jobs completed or failed. Daemon exiting normally."
        break
    fi

    log "Starting job: $next_job"

    # Capture exit code explicitly - never use set -e here
    python3 tools/bdd_7domain_manager.py run-job --job-id "$next_job"
    rc=$?

    if [ "$rc" -eq 0 ]; then
        job_count=$((job_count + 1))
        log "Job completed: $next_job (total completed=$job_count)"
    elif [ "$rc" -eq 130 ] || [ "$rc" -eq 143 ] || [ "$rc" -eq 137 ]; then
        log "Job interrupted by signal (rc=$rc): $next_job - marked as interrupted, will retry"
    else
        log "Job FAILED (rc=$rc): $next_job - skipping, continuing with next job"
    fi

    # Status call is best-effort only
    python3 tools/bdd_7domain_manager.py status > /dev/null 2>&1 || true

    sleep 10
done

log "=== Daemon finished. Completed $job_count jobs total. ==="
rm -f "$DAEMON_PID_FILE" "$HEARTBEAT_FILE"
python3 tools/bdd_7domain_manager.py status || true
