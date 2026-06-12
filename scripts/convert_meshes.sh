#!/bin/bash
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Fail if a subcommand fails
set -e

# Print the commands
set -x

# Get the directory of the script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PICK_AND_PLACE_DIR=$( dirname "$SCRIPT_DIR" )

# Destination directory (default to a local 'dist' if not provided)
DST_DIR="${1:-$PICK_AND_PLACE_DIR/dist_assets}"

mkdir -p "$DST_DIR"

# 1. Generate intermediary GLBs
cd "$PICK_AND_PLACE_DIR/py"
python scripts/simplify_meshes.py

# 2. Optimize GLBs
cd "$PICK_AND_PLACE_DIR/ts"
pnpm run optimize-meshes

# 3. Copy optimized GLBs to destination
cp -r public/so101_assets/* "$DST_DIR/"

# 4. Copy CSS
cp src/style.css "$DST_DIR/pick-and-place.css"

echo "Successfully converted and optimized meshes in $DST_DIR"
