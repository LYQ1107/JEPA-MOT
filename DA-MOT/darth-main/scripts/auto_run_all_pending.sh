#!/bin/bash
# No set -e: we must NOT exit on job failure, or GPU children become orphans

ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main"
JOBS_CSV="$ROOT/work_dirs/bdd_7domain/state/jobs.csv"
LOG_DIR="$ROOT/work_dirs/bdd_7domain/auto_run_logs"
MANAGER="$ROOT/tools/bdd_7domain_manager.py"

mkdir -p "$LOG_DIR"

MAIN_LOG="$LOG_DIR/auto_run_all_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$MAIN_LOG") 2>&1

echo "=========================================="
echo "Auto-run all pending jobs started at $(date)"
echo "=========================================="

cd "$ROOT"

get_next_job() {
    python3 -c "
import csv
with open('$JOBS_CSV') as f:
    jobs = list(csv.DictReader(f))
    nxt = next((j['job_id'] for j in jobs if j['status'] in ('pending', 'interrupted')), '')
    print(nxt)
"
}

# Wait until no stale train/test processes remain, up to MAX_WAIT seconds.
# This prevents OOM kills caused by previous job's GPU workers surviving into the next job.
wait_gpu_free() {
    local MAX_WAIT=120
    local WAITED=0
    while true; do
        LEFTOVER=$(ps aux | grep -E "(train\.py|test\.py)" | grep -v grep | wc -l)
        if [ "$LEFTOVER" -eq 0 ]; then
            break
        fi
        if [ "$WAITED" -ge "$MAX_WAIT" ]; then
            echo "WARNING: $LEFTOVER GPU worker(s) still alive after ${MAX_WAIT}s; force-killing them."
            ps aux | grep -E "(train\.py|test\.py)" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
            sleep 3
            break
        fi
        echo "  Waiting for $LEFTOVER GPU worker(s) to exit... (${WAITED}s elapsed)"
        sleep 5
        WAITED=$((WAITED + 5))
    done
}

CONSECUTIVE_FAILURES=0
MAX_CONSECUTIVE_FAILURES=5

while true; do
    JOB_ID=$(get_next_job)

    if [ -z "$JOB_ID" ]; then
        echo ""
        echo "=========================================="
        echo "All jobs completed! $(date)"
        echo "=========================================="
        python3 "$MANAGER" status
        break
    fi

    echo ""
    echo "=========================================="
    echo "Starting job: $JOB_ID at $(date)"
    echo "=========================================="

    # Ensure GPU memory is fully released before starting the next job
    wait_gpu_free

    python3 "$MANAGER" run-job --job-id "$JOB_ID"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Job $JOB_ID completed successfully"
        CONSECUTIVE_FAILURES=0
    else
        echo "Job $JOB_ID failed or was interrupted (exit code $EXIT_CODE)"
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        if [ $CONSECUTIVE_FAILURES -ge $MAX_CONSECUTIVE_FAILURES ]; then
            echo "ERROR: $MAX_CONSECUTIVE_FAILURES consecutive failures, stopping pipeline."
            python3 "$MANAGER" status
            exit 1
        fi
    fi

    echo "Current status:"
    python3 "$MANAGER" status | head -10

    sleep 5
done

echo ""
echo "=========================================="
echo "Final summary:"
echo "=========================================="
python3 "$MANAGER" status

echo ""
echo "All pending tasks completed. Check individual job logs for details."
echo "Main log: $MAIN_LOG"
