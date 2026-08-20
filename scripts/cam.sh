#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Start/stop the camera preview. It holds /dev/video*, so it must be stopped
# before a policy run — `stop` and `status` are the whole point of this wrapper.
#
#   ./cam.sh start      serve the preview (refuses if the cameras are busy)
#   ./cam.sh stop       release the cameras
#   ./cam.sh status     is it up, and who holds the cameras
#   ./cam.sh identify   one-shot mapping, exits without holding anything
#   ./cam.sh restart

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$HERE/py/scripts/preview_cameras.py"
PIDFILE="$HERE/runs/campreview.pid"
LOG="$HERE/runs/campreview.log"
PORT="${CAM_PORT:-8080}"
# Run under the repo's pyenv env, wherever this is invoked from.
PY="${CAM_PYTHON:-/home/mario/.pyenv/versions/pap/bin/python3}"

running_pid() {
  [ -f "$PIDFILE" ] || return 1
  local pid
  pid=$(tr -dc '0-9' < "$PIDFILE")
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  echo "$pid"
}

case "${1:-status}" in
  start)
    if pid=$(running_pid); then
      echo "Already running (pid $pid) — http://$(hostname -I | awk '{print $1}'):$PORT/"
      exit 0
    fi
    if holder=$(fuser /dev/video0 /dev/video2 2>/dev/null | tr -s ' '); then
      if [ -n "${holder// /}" ]; then
        echo "Cameras are busy (pids:$holder). Not starting." >&2
        echo "Stop whatever is using them first — a policy run, most likely." >&2
        exit 1
      fi
    fi
    # Unbuffered, so the identification result reaches the log immediately.
    PYTHONUNBUFFERED=1 nohup "$PY" "$SCRIPT" --port "$PORT" >"$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    sleep 10
    if pid=$(running_pid); then
      grep -v '^\[ WARN' "$LOG" | grep -E 'index [0-9]+ ->|Run the policy with' || true
      echo "Started (pid $pid) — http://$(hostname -I | awk '{print $1}'):$PORT/"
    else
      echo "Failed to start; last lines of $LOG:" >&2
      grep -v '^\[ WARN' "$LOG" | tail -15 >&2
      exit 1
    fi
    ;;

  stop)
    if pid=$(running_pid); then
      kill "$pid"
      for _ in $(seq 20); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.25
      done
      kill -9 "$pid" 2>/dev/null
      rm -f "$PIDFILE"
      echo "Stopped (pid $pid)."
    else
      rm -f "$PIDFILE"
      echo "Not running."
    fi
    # The cameras are only really free once nothing holds the nodes.
    if fuser /dev/video0 /dev/video2 2>/dev/null | grep -q '[0-9]'; then
      echo "WARNING: something still holds the cameras:" >&2
      fuser -v /dev/video0 /dev/video2 2>&1 | tail -5 >&2
    else
      echo "Cameras free."
    fi
    ;;

  status)
    if pid=$(running_pid); then
      echo "Preview RUNNING (pid $pid) — http://$(hostname -I | awk '{print $1}'):$PORT/"
    else
      echo "Preview not running."
    fi
    if fuser /dev/video0 /dev/video2 2>/dev/null | grep -q '[0-9]'; then
      fuser -v /dev/video0 /dev/video2 2>&1 | tail -5
    else
      echo "Cameras free."
    fi
    ;;

  identify)
    "$PY" "$SCRIPT" --identify-only 2>&1 | grep -v '^\[ WARN'
    ;;

  restart)
    "$0" stop && "$0" start
    ;;

  *)
    sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
