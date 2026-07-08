#!/bin/bash
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

# Convert an arbitrary robot_descriptions model into web assets: a decimated
# and optimized GLB per mesh, plus a web-model JSON manifest. Unlike
# convert_meshes.sh (which composes this project's own SO-101 scene), this
# loads the robot's stock MJCF directly, so it works for any
# `*_mj_description` package.
#
# Usage: convert_meshes_generic.sh ROBOT [--gripper GRIPPER] [--target-mm MM] [dst_dir]
#   ROBOT: robot_descriptions module name, e.g. ur5e_mj_description
#   GRIPPER: robot_descriptions module name for an end effector to attach at
#     the robot's "attachment_site", e.g. robotiq_2f85_mj_description
#   MM: mesh simplification deviation budget in millimeters (default 0.5);
#     scale it with the robot's physical size

set -e
set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PICK_AND_PLACE_DIR=$( dirname "$SCRIPT_DIR" )

ROBOT="$1"
if [[ -z "$ROBOT" ]]; then
    echo "Usage: $0 ROBOT [--gripper GRIPPER] [--target-mm MM] [dst_dir]" >&2
    exit 2
fi
shift

GRIPPER=""
SIMPLIFY_ARGS=()
while [[ "$1" == --* ]]; do
    case "$1" in
        --gripper)
            GRIPPER="$2"
            shift 2
            ;;
        --target-mm)
            SIMPLIFY_ARGS+=(--target-mm "$2")
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

# Web-facing name: strip the "_mj_description" suffix, e.g. ur5e_mj_description -> ur5e
NAME="${ROBOT%_mj_description}"

DST_DIR="${1:-$PICK_AND_PLACE_DIR/ts/public}"

# 1. Generate intermediary GLBs
cd "$PICK_AND_PLACE_DIR/mesh_optimization"
if [[ -n "$GRIPPER" ]]; then
    python scripts/simplify_generic_meshes.py "$ROBOT" --gripper "$GRIPPER" "${SIMPLIFY_ARGS[@]}"
else
    python scripts/simplify_generic_meshes.py "$ROBOT" "${SIMPLIFY_ARGS[@]}"
fi

# 2. Optimize GLBs
cd "$PICK_AND_PLACE_DIR/ts"
pnpm exec bash scripts/optimize-glbs.sh \
    "$PICK_AND_PLACE_DIR/intermediary-glb-generic/$ROBOT" \
    "$DST_DIR/${NAME}_assets"

# 3. Generate the web model manifest directly from the robot's stock MJCF
cd "$PICK_AND_PLACE_DIR/py"
if [[ -n "$GRIPPER" ]]; then
    python scripts/export_generic_robot.py "$ROBOT" --gripper "$GRIPPER" -o "$DST_DIR/$NAME.json"
else
    python scripts/export_generic_robot.py "$ROBOT" -o "$DST_DIR/$NAME.json"
fi

echo "Successfully converted $ROBOT into $DST_DIR/$NAME.json + $DST_DIR/${NAME}_assets"
