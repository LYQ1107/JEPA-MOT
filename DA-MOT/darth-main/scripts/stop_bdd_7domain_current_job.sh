#!/usr/bin/env bash
set -eo pipefail
cd /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main
RUNNING_DIR="work_dirs/bdd_7domain/state/running"
PID_DIR="work_dirs/bdd_7domain/state/pids"
for r in "$RUNNING_DIR"/*.running; do
  [ -e "$r" ] || { echo "No running job marker."; exit 0; }
  jid=$(basename "$r" .running)
  pf="$PID_DIR/${jid}.pid"
  pid=""; [ -f "$pf" ] && pid=$(cat "$pf")
  echo "Stopping job=$jid pid=$pid"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" || true
    sleep 30
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" || true
  fi
  rm -f "$r"
  echo "$(date) interrupted" > "work_dirs/bdd_7domain/state/failed/${jid}.failed"
done
