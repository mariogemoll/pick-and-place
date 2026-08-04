#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

set -euo pipefail

MAX_BYTES=$((40 * 1024))

# Keep the default limit strict while documenting the known exceptions. Values are
# per-path byte ceilings, so an excepted file can still fail if it grows. Every
# blob in the history is checked, so a file that moves keeps its old path listed
# at the ceiling that path once needed.
exception_limit() {
  case "$1" in
    # Oversized in the current tree, each at a ceiling just above its size.
    py/scripts/run_policy_real.py) printf '%s\n' $((57 * 1024)) ;;
    py/src/pick_and_place/runtime/executor.py) printf '%s\n' $((61 * 1024)) ;;
    # No longer oversized, or no longer at this path, but the history still
    # holds a large blob there. These go away only if history is rewritten.
    py/scripts/pick_and_place/real.py) printf '%s\n' $((60 * 1024)) ;;
    py/src/pick_and_place/executor.py) printf '%s\n' $((61 * 1024)) ;;
    py/src/pick_and_place/planning/trajectory.py) printf '%s\n' $((50 * 1024)) ;;
    py/src/pick_and_place/trajectory.py) printf '%s\n' $((56 * 1024)) ;;
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
