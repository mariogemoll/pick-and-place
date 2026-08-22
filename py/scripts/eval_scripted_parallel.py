# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Evaluate the scripted controller over a manifest across worker processes.

Scenarios are independent, so this shards the manifest across a spawn-based
process pool. Each worker builds its own PolicySimEnv and scripted controller
(a fresh MuJoCo/GL context per process, never inherited), evaluates its slice,
and returns EpisodeResults. The parent re-orders them to manifest order and
writes the same self-contained run directory as ``eval_policy_sim.py``.

AprilTag detection runs out-of-process (``DetectorProcess``), so the
``pupil_apriltags`` destructor segfault cannot kill a pool worker.

On Linux, ``--backend egl`` renders on the GPU (fast, the default) and
``--backend osmesa`` renders on CPU (slow, but leaves the GPU free). macOS uses
``cgl`` by default, and Windows uses ``wgl``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import platform
import queue
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from pick_and_place.cli.common import add_output_argument
from pick_and_place.cli.evaluation import (
    EVALUATION_DIR,
    add_manifest_argument,
    add_shard_arguments,
)
from pick_and_place.cli.scene import add_render_size_arguments
from pick_and_place.cli.suggest import SuggestingArgumentParser
from pick_and_place.core.paths import REPO_ROOT

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

DEFAULT_MANIFEST = EVALUATION_DIR / "canonical_100_v1.json.xz"

SYSTEM = platform.system()
BACKENDS_BY_SYSTEM = {
    "Darwin": ("cgl", "glfw"),
    "Linux": ("egl", "osmesa", "glfw"),
    "Windows": ("wgl", "glfw"),
}
DEFAULT_BACKEND_BY_SYSTEM = {
    "Darwin": "cgl",
    "Linux": "egl",
    "Windows": "wgl",
}
BACKENDS = BACKENDS_BY_SYSTEM.get(SYSTEM, ("glfw",))
DEFAULT_BACKEND = DEFAULT_BACKEND_BY_SYSTEM.get(SYSTEM, "glfw")


def _evaluate_shard(payload: tuple) -> list[tuple[int, object]]:
    manifest_path, indices, image_hw, render_hw, control_hz, backend, drop_target, progress = payload
    os.environ["MUJOCO_GL"] = backend
    if backend == "osmesa":
        # Keep each llvmpipe render single-threaded so N workers map to N cores
        # instead of oversubscribing.
        os.environ.setdefault("LP_NUM_THREADS", "1")

    from pick_and_place.policies.policy_evaluation import ScenarioManifest
    from pick_and_place.runtime.policy_sim import PolicySimEnv, evaluate_policy_episode

    import eval_policy_sim as eps

    scenarios = ScenarioManifest.load(manifest_path).scenarios
    controller, _ = eps._make_scripted_controller(
        image_hw=image_hw, render_hw=render_hw, control_hz=control_hz
    )
    env = PolicySimEnv(image_hw=image_hw, render_hw=render_hw)
    results: list[tuple[int, object]] = []
    try:
        for index in indices:
            scenario = scenarios[index]
            if drop_target == "ground-truth":
                # Pinning drop_target_xy ablates plate perception and nothing
                # else: no overhead search runs, while planning stays on the
                # same fixed-target path a detection would have produced, so the
                # two arms differ only in where the plate xy came from. Cube
                # localization and the descent servo are untouched, and reset()
                # keeps the pin, so setting it per episode survives
                # evaluate_policy_episode's reset.
                x, y, _ = scenario.target_position_m
                controller.drop_target_xy = (float(x), float(y))
            results.append((index, evaluate_policy_episode(env, controller, scenario)))
            progress.put(1)  # one tick per finished episode, drained by the parent's bar
    finally:
        env.close()
        close = getattr(controller, "close", None)
        if close is not None:
            close()
    return results


def _shards(indices: list[int], jobs: int) -> list[list[int]]:
    buckets: list[list[int]] = [[] for _ in range(jobs)]
    for position, index in enumerate(indices):
        buckets[position % jobs].append(index)  # round-robin balances slow/fast episodes
    return [bucket for bucket in buckets if bucket]


def build_parser() -> SuggestingArgumentParser:
    """Return the parser for the parallel scripted evaluation."""
    parser = SuggestingArgumentParser(description=__doc__)
    add_manifest_argument(parser, default=DEFAULT_MANIFEST)
    add_output_argument(parser, required=True, help="new evaluation run directory")
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 12))
    parser.add_argument("--backend", choices=BACKENDS, default=DEFAULT_BACKEND)
    add_render_size_arguments(parser)
    add_shard_arguments(parser)
    parser.add_argument(
        "--drop-target",
        choices=("detected", "ground-truth"),
        default="detected",
        help=(
            "where the expert's drop target comes from. 'detected' localizes the "
            "plate from the overhead image, as on hardware; 'ground-truth' pins it "
            "to the scene's true plate position, which measures what plate "
            "localization costs the demonstrator."
        ),
    )
    return parser


def validate(parser: SuggestingArgumentParser, args: argparse.Namespace) -> None:
    """Reject an output that exists and a shard slice that cannot be cut."""
    if args.output.exists():
        parser.error(f"--output already exists: {args.output}")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.offset < 0:
        parser.error("--offset must not be negative")


def run(args: argparse.Namespace) -> None:
    """Shard the suite over workers and merge the result."""
    os.environ["MUJOCO_GL"] = args.backend
    if args.backend == "osmesa":
        os.environ.setdefault("LP_NUM_THREADS", "1")
    import eval_policy_sim as eps
    from pick_and_place.policies.policy_evaluation import (
        ScenarioManifest,
        TaskOracleConfig,
        git_provenance,
        package_versions,
        write_evaluation_artifacts,
    )

    started_at = dt.datetime.now(dt.UTC)
    manifest = ScenarioManifest.load(args.manifest)
    if args.offset >= len(manifest.scenarios):
        raise SystemExit(
            f"--offset {args.offset} skips the whole {len(manifest.scenarios)}-scenario suite"
        )
    stop = len(manifest.scenarios) if args.limit is None else args.offset + args.limit
    # Indices stay manifest-absolute so a worker can index the manifest it loads.
    selected_indices = list(range(args.offset, min(stop, len(manifest.scenarios))))
    scenarios = [manifest.scenarios[index] for index in selected_indices]
    control_hz_values = {scenario.control_hz for scenario in scenarios}
    if len(control_hz_values) != 1:
        raise ValueError("scripted evaluation requires one control frequency per run")
    control_hz = next(iter(control_hz_values))
    image_hw = eps.SCRIPTED_IMAGE_HW
    render_hw = (args.render_height, args.render_width)

    jobs = max(1, min(args.jobs, len(scenarios)))
    shards = _shards(selected_indices, jobs)
    _, controller_metadata = eps._make_scripted_controller(
        image_hw=image_hw, render_hw=render_hw, control_hz=control_hz
    )
    # Recorded so the two arms of a drop-target ablation are distinguishable
    # from their run directories alone.
    controller_metadata["drop_target_source"] = args.drop_target
    print(
        f"Evaluating {len(scenarios)} {manifest.suite!r} scenarios (scripted, "
        f"drop target {args.drop_target}) across {jobs} workers."
    )

    indexed: list[tuple[int, object]] = []
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    with ctx.Manager() as manager, ProcessPoolExecutor(
        max_workers=jobs, mp_context=ctx
    ) as pool:
        progress = manager.Queue()
        futures = [
            pool.submit(
                _evaluate_shard,
                (
                    str(args.manifest),
                    shard,
                    image_hw,
                    render_hw,
                    control_hz,
                    args.backend,
                    args.drop_target,
                    progress,
                ),
            )
            for shard in shards
        ]
        # Drain one tick per finished episode into a single bar. Poll with a
        # timeout so a worker that dies (BrokenProcessPool) breaks the wait
        # instead of hanging; future.result() below then re-raises the real error.
        completed = 0
        with tqdm(total=len(scenarios), unit="ep", desc="scripted eval") as bar:
            while completed < len(scenarios):
                try:
                    progress.get(timeout=0.5)
                except queue.Empty:
                    if all(future.done() for future in futures):
                        break
                    continue
                completed += 1
                bar.update(1)
        for future in futures:
            indexed.extend(future.result())

    results = [result for _, result in sorted(indexed, key=lambda item: item[0])]

    run = {
        "schema_version": 1,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "checkpoint": None,
        "controller": controller_metadata,
        "instruction": None,
        "scenario_manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": manifest.sha256(),
            "suite": manifest.suite,
            "selected_scenario_ids": [scenario.scenario_id for scenario in scenarios],
            "complete_suite": True,
        },
        "environment": {
            "image_height": image_hw[0],
            "image_width": image_hw[1],
            "render_height": args.render_height,
            "render_width": args.render_width,
            "render_backend": args.backend,
            "control_hz": sorted(control_hz_values),
            "episode_step_limits": sorted({scenario.max_steps for scenario in scenarios}),
            "domain_randomization_presets": sorted(
                {scenario.domain_randomization_preset or "none" for scenario in scenarios}
            ),
            "oracle": asdict(TaskOracleConfig()),
            "state_frame": "hardware (arm degrees, gripper position 0-100)",
            "action_frame": "hardware (arm degrees, gripper position 0-100)",
        },
        "device": None,
        "code": git_provenance(REPO_ROOT),
        "package_versions": package_versions(["gymnasium", "mujoco", "numpy"]),
        "videos_saved": False,
        "parallel_workers": jobs,
    }
    summary = write_evaluation_artifacts(args.output, run, results)
    print(
        f"Wrote {args.output}: {summary['success_count']}/{summary['episode_count']} "
        f"successes ({summary['success_rate']:.1%})."
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate(parser, args)
    run(args)


if __name__ == "__main__":
    main()
