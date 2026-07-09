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

# Parse options and destination directory
SIMPLIFY_ARGS=()
OMIT_WRIST_CAMERA_MOUNT=false
DST_DIR="$PICK_AND_PLACE_DIR/dist_assets"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-wrist-camera-mount)
            SIMPLIFY_ARGS+=("$1")
            OMIT_WRIST_CAMERA_MOUNT=true
            shift
            ;;
        --target-kb)
            SIMPLIFY_ARGS+=(--target-kb "$2")
            shift 2
            ;;
        --detail)
            SIMPLIFY_ARGS+=(--detail "$2")
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--no-wrist-camera-mount] [--target-kb KB] [--detail GLOB=FACTOR]... [dst_dir]"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
        *)
            DST_DIR="$1"
            shift
            if [[ $# -gt 0 ]]; then
                echo "Unexpected argument: $1" >&2
                exit 2
            fi
            ;;
    esac
done

mkdir -p "$DST_DIR"

# 1. Generate intermediary GLBs
cd "$PICK_AND_PLACE_DIR/mesh_optimization"
python scripts/simplify_meshes.py "${SIMPLIFY_ARGS[@]}"

# 2. Optimize GLBs
cd "$PICK_AND_PLACE_DIR/ts"
pnpm run optimize-meshes

# 3. Generate the web model manifests from the same composed robot used by MuJoCo
cd "$PICK_AND_PLACE_DIR/py"
# 3a. Simple Scene (Robot only)
EXPORT_ARGS=()
if [[ "$OMIT_WRIST_CAMERA_MOUNT" == true ]]; then
    EXPORT_ARGS+=(--no-wrist-camera)
fi
python -m pick_and_place.export \
    -o "$PICK_AND_PLACE_DIR/ts/public/so101.xml" "${EXPORT_ARGS[@]}"

# 3b. Environment only (Overhead Mount + Workspace Frame + Cube + Floor).
# The robot lives in so101.json; the web viewer overlays this on top so the
# robot is defined once instead of being baked into the scene a second time.
python -m pick_and_place.export --environment-only \
    -o "$PICK_AND_PLACE_DIR/ts/public/environment.xml"

rm -f "$PICK_AND_PLACE_DIR/ts/public/so101.xml" "$PICK_AND_PLACE_DIR/ts/public/environment.xml"

# 4. Copy optimized GLBs and web models to destination
cd "$PICK_AND_PLACE_DIR/ts"
cp -r public/so101_assets/* "$DST_DIR/"
cp public/so101.json public/environment.json "$DST_DIR/"

echo "Successfully converted and optimized meshes in $DST_DIR"
