#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

run_id="$(date '+%Y%m%d_%H%M%S')"
log_dir="outputs/generaltrack/logs"
mkdir -p "$log_dir"

log_file="$log_dir/generaltrack_resume_${run_id}.log"
pid_file="$log_dir/generaltrack_resume_${run_id}.pid"
latest_log="$log_dir/generaltrack_resume_latest.log"
latest_pid="$log_dir/generaltrack_resume_latest.pid"

nohup bash scripts/run_generaltrack_resume_remaining.sh > "$log_file" 2>&1 < /dev/null &
pid="$!"

printf '%s\n' "$pid" > "$pid_file"
printf '%s\n' "$pid" > "$latest_pid"
ln -sfn "$(basename "$log_file")" "$latest_log"

echo "PID: $pid"
echo "Log: $log_file"
echo "Latest PID: $latest_pid"
echo "Latest Log: $latest_log"
