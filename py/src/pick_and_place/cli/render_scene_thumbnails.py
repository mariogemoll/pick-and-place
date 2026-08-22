# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Render the initial overhead frame of chosen scenes, for looking at them.

A phenotype found in the feature table is a claim about what the scene looks
like; this renders the scenes so the claim can be checked by eye. Runs on the
CPU through software rendering, so it needs no GPU.

    MUJOCO_GL=osmesa pap render-scene-thumbnails \\
      --phenotypes outputs/scene-difficulty/phenotypes.json \\
      --count 6 --out-dir outputs/scene-difficulty/thumbnails
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pick_and_place.cli.common import add_out_dir_argument, add_output_size_arguments
from pick_and_place.cli.scene import add_scene_appearance_arguments
from pick_and_place.cli.suggest import SuggestingArgumentParser


def build_parser() -> SuggestingArgumentParser:
    """Return the parser for the thumbnail renderer."""
    parser = SuggestingArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phenotypes", type=Path, required=True)
    parser.add_argument(
        "--count", type=int, default=6, help="scenes to render from each end"
    )
    add_output_size_arguments(parser, width=480, height=360, noun="thumbnail")
    add_scene_appearance_arguments(parser, default="blue-cube")
    add_out_dir_argument(parser, required=True, help="directory for the rendered thumbnails")
    return parser


def run(args: argparse.Namespace) -> None:
    """Render the thumbnails."""
    from PIL import Image

    from pick_and_place.runtime.training_scenes import training_scenario
    from pick_and_place.runtime.policy_sim import OVERHEAD_FEATURE, PolicySimEnv
    from pick_and_place.variants.appearance import parse_appearance

    payload = json.loads(args.phenotypes.read_text())
    scenes = payload["scenes"]
    seed_base = None
    # The sweep's seed base is not repeated in the phenotype file, so recover it
    # from any scene: index and scenario_id agree by construction.
    hard = sorted(scenes, key=lambda s: (s["success_rate"], s["index"]))[: args.count]
    easy = sorted(scenes, key=lambda s: (-s["success_rate"], s["index"]))[: args.count]

    seed_base = int(payload["summary"].get("scene_seed_base", 6_000_000))
    env = PolicySimEnv(
        image_hw=(args.height, args.width),
        render_hw=(args.height, args.width),
        scene_appearance=parse_appearance(args.scene_appearance)[1],
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    try:
        for group, records in (("hard", hard), ("easy", easy)):
            for record in records:
                scenario = training_scenario(int(record["index"]), seed_base=seed_base)
                observation, _ = env.reset(options={"scenario": scenario})
                frame = np.asarray(observation[OVERHEAD_FEATURE], dtype=np.uint8)
                name = f"{group}_{record['index']:06d}_rate{record['success_rate']:.2f}.png"
                Image.fromarray(frame).save(args.out_dir / name)
                manifest.append({
                    "group": group,
                    "file": name,
                    "scenario_id": record["scenario_id"],
                    "success_rate": record["success_rate"],
                    "stopped_at": record["stopped_at"],
                    "cube_radius_m": record["cube_radius_m"],
                    "carry_distance_m": record["carry_distance_m"],
                })
                print(f"rendered {name}", flush=True)
    finally:
        env.close()
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.out_dir / 'manifest.json'}")


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
