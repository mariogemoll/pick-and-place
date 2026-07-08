#!/bin/bash
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Optimize every GLB in SRC_DIR into DST_DIR with gltf-transform meshopt.
#
# Usage: optimize-glbs.sh SRC_DIR DST_DIR

set -e
set -x

SRC_DIR="$1"
DST_DIR="$2"

mkdir -p "$DST_DIR"
for file in "$SRC_DIR"/*.glb; do
    gltf-transform meshopt "$file" "$DST_DIR/$(basename "$file")"
done
