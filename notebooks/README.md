<!-- SPDX-FileCopyrightText: 2026 Mario Gemoll -->
<!-- SPDX-License-Identifier: 0BSD -->

# Notebooks

Runnable walkthroughs of what this project does. They are for **seeing** the
pipeline, not for running it at full size: the real jobs are hours of rented GPU
driven by `scripts/vast_*.sh`, and a notebook cell that blocks for three hours
is worse than the shell script in every way.

Each notebook therefore runs a deliberately small slice on a laptop and ends by
printing the command that does the real thing.

| Notebook | What it shows |
| --- | --- |
| [`shootout.ipynb`](shootout.ipynb) | The expert, ACT and the flow-matching policy scored against the same domain-randomized scenarios, then compared: who beat whom on which scenario, where each one dies, and the video of a disagreement. |

## Running them

```sh
VIRTUAL_ENV=~/venvs/pick-and-place uv pip install -e py --group notebooks
export MUJOCO_GL=egl PAP_DATA_ROOT=~/pick-and-place-data
jupyter lab notebooks/
```

The rest of the setup — the AprilTag textures in particular, without which no
scene compiles — is in the [top-level README](../README.md).

## Two rules for adding one

**Cells call `pap`; they do not reimplement it.** A notebook that grows its own
copy of the evaluation loop or the success oracle becomes a second
implementation, and the second implementation is the one nobody notices drifting.
Import from `pick_and_place`, or shell out to the command, and keep the notebook
to arranging and drawing what comes back.

**Commit no outputs.** `scripts/check_notebook_outputs.py` fails on a stored
output or an execution count — they are unreviewable in a diff, and one run with
video displayed is megabytes against a 40 KB per-file ceiling. Add it to the
`repo-checks` job to have CI enforce it:

```yaml
      - name: "Check notebooks carry no stored outputs"
        run: python3 scripts/check_notebook_outputs.py
```

```sh
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```
