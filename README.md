<!-- SPDX-FileCopyrightText: 2026 Mario Gemoll -->
<!-- SPDX-License-Identifier: 0BSD -->

# pick-and-place

Cube pick-and-place on the [Standard Open SO-101 arm](https://github.com/TheRobotStudio/SO-ARM100),
in simulation and on real hardware.

An overhead camera locates a cube on a table; the arm picks it up and places it
on a target xy coordinate. An analytic planner solves the task directly and
generates every training demonstration. Learned policies imitate that planner
and are scored against it on frozen scenario suites, so "is this policy any
good" has one answer everything is measured by.

The visible output is the pick-and-place page on
<https://mariogemoll.com/pick-and-place>.

## What is in here

| Directory | Contents |
| --- | --- |
| `py/` | The `pick_and_place` package and the `pap` command line. Simulation, control, calibration, datasets, policies. |
| `py/notebooks/` | Runnable walkthroughs that also open on Colab — see [py/notebooks/README.md](py/notebooks/README.md). Start with [`bakeoff.ipynb`](py/notebooks/bakeoff.ipynb). |
| `ts/` | Vite + Three.js browser app: the visualizations on the web page. |
| `config/` | Frozen evaluation manifests, training configs, fitted robot dynamics. |
| `scripts/` | Repository tooling and the rented-GPU job scripts. |
| `SO-ARM100/` | The hardware itself, vendored as a submodule: CAD, URDF, MJCF, BOM. |

Contributor-facing detail — the package layering rules, the conventions, the
known rough edges — is in [AGENTS.md](AGENTS.md).

## Install

Python 3.13. The dependency graph is tightly constrained (`lerobot==0.5.1` pins
`transformers==5.3.0`), so newer interpreters are untested.

```sh
git clone --recurse-submodules https://github.com/mariogemoll/pick-and-place.git
cd pick-and-place

# Headless Linux also needs system GL:
sudo apt-get install -y linux-libc-dev build-essential libegl1 libgl1 libopengl0

uv python install 3.13
uv venv --python 3.13 ~/venvs/pick-and-place
VIRTUAL_ENV=~/venvs/pick-and-place uv pip install -e py --group dev
source ~/venvs/pick-and-place/bin/activate
```

Two things then have to be generated, because neither is committed:

```sh
export MUJOCO_GL=egl                       # headless rendering; needed by everything below
export PAP_DATA_ROOT=~/pick-and-place-data  # datasets, checkpoints, evaluations

pap render-apriltag-textures --all-defaults
```

**Generate the textures before running anything that builds a scene.** They
decide the cube's appearance, and without them every MuJoCo scene fails to
compile with a file-not-found on `assets/apriltags/textures/`. Training does not
compile a scene, so a missing texture set costs nothing for hours and then fails
the first evaluation.

`pap --help` lists all 53 commands, grouped; `pap <command> --help` gives one
command's flags.

<details>
<summary>Two more install notes that bite later</summary>

- **`pap` appears on install, not before.** An environment created before it
  existed needs `uv pip install -e py --no-deps` to get the shim.
  `python -m pick_and_place.cli.pap` is the escape hatch.
- **`cv2.imshow` needs the GUI build of OpenCV.** `lerobot` pulls
  `opencv-python-headless`, and both wheels install the same `cv2/` directory —
  whichever lands last wins. Only the commands that show a window care:
  `pip install --no-deps --force-reinstall opencv-python==<same version>`.
- **macOS**: commands that open a MuJoCo viewer need `mjpython`. Most offer
  `--no-viewer`.

</details>

## Five minutes in

Score the analytic expert on the eight-scenario smoke suite:

```sh
pap eval-policy-sim scripted \
  --manifest config/evaluation/smoke_v1.json \
  --output "$PAP_DATA_ROOT/evaluations/hello-scripted"
```

Roughly a minute per scenario on two CPU cores, faster on more. It writes
`run.json` (what was run, down to the git revision and the manifest hash),
`episodes.jsonl` (one row per scenario) and `summary.json`. That triple is the
project's unit of measurement — every policy, learned or analytic, is scored
into the same shape and compared with `pap compare-policy-evaluations`.

To watch instead of read, `pap view-scene` opens the calibrated scene and
`pap replay-episode` plays a recorded one back.

## The four workflows

Everything substantial here is one of four jobs. Only the first column runs
comfortably on a laptop.

| | Command | Where it runs | Rough cost |
| --- | --- | --- | --- |
| **Generate data** | `pap record-sim` → `pap finalize-sim-dataset` | Rented GPU; VRAM-bound worker pool | ~7 episodes/min at 17 workers, so ~2.5 h for 1,000 |
| **Train ACT** | `lerobot-train` (external to this repo) | Rented GPU | GPU-hours |
| **Train the flow policy** | `pap train-flow-image` | Rented GPU | 1.6–5.3 h depending on size |
| **Score anything** | `pap eval-policy-sim` → `pap compare-policy-evaluations` | Laptop for tens of scenarios, sharded GPU for hundreds | ~1 min per scenario per core |

The first three are wrapped end to end, including staging and publishing, by the
`scripts/vast_*.sh` job scripts. Read one before renting anything: they encode
resume-after-crash staging and the worker limits that a naive run rediscovers
expensively.

Domain randomization is a flag on generation (`--domain-randomization`, taking a
preset from `config/domain_randomization/`) and a property frozen into an
evaluation suite. `config/evaluation/dr_100_v1.json.xz` is 100 randomized
scenarios; `canonical_100_v1` is the same size with randomization off, so the
pair measures what randomization costs a policy.

## Where things get written

Nothing generated belongs in the repository. `PAP_DATA_ROOT` holds it all —
datasets, checkpoints, renders, evaluations — and `pick_and_place.core.paths` is
the only module that reads the variable. Commands that need it and cannot find
it fail immediately, naming it, instead of writing into the source tree.

## Real hardware

The rig commands (`run-policy-real`, `run-scripted-real`, everything under
calibration) drive a physical SO-101 and its two cameras. They need the arm, the
printed workspace frame in `stl/`, and a calibration session; none of them is
exercised by the simulation path above. `pap --help` groups them separately.

## Tests

```sh
cd py
MUJOCO_GL=egl python -m pytest
python -m ruff check .
```

The TypeScript tests need a generated fixture first:

```sh
cd py && MUJOCO_GL=egl python -m pick_and_place.sim.export -o ../ts/public/so101.xml
cd ../ts && pnpm i && pnpm test
```

Use pnpm 10, as CI does.

## License

0BSD. See [LICENSE](LICENSE).
