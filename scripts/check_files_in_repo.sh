#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

set -euo pipefail

MAX_BYTES=$((40 * 1024))

# Keep the default limit strict while documenting the known exceptions. Values are
# per-path byte ceilings, so an excepted file can still fail if it grows.
exception_limit() {
  case "$1" in
    py/scripts/pick_and_place/real.py) printf '%s\n' $((62 * 1024)) ;;
    py/src/pick_and_place/executor.py) printf '%s\n' $((63 * 1024)) ;;
    py/src/pick_and_place/trajectory.py) printf '%s\n' $((58 * 1024)) ;;
    *) printf '%s\n' "$MAX_BYTES" ;;
  esac
}

failed=0

while read -r blob path; do
  [ -z "$path" ] && continue
  size=$(git cat-file -s "$blob")
  limit=$(exception_limit "$path")
  if [ "$size" -gt "$limit" ]; then
    if [ "$limit" -gt "$MAX_BYTES" ]; then
      printf 'Excepted blob too large: %s (%s bytes > %s bytes exception limit)\n' "$path" "$size" "$limit"
    else
      printf 'Blob too large: %s (%s bytes > %s bytes)\n' "$path" "$size" "$MAX_BYTES"
    fi
    failed=1
  fi
done < <(git rev-list HEAD --objects)

exit "$failed"
