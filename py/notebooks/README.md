<!-- SPDX-FileCopyrightText: 2026 Mario Gemoll -->
<!-- SPDX-License-Identifier: 0BSD -->

# Notebooks

| Notebook | What it shows |
| --- | --- |
| [`bakeoff.ipynb`](bakeoff.ipynb) | The expert, ACT and the flow-matching policy scored against the same domain-randomized scenarios, then compared. |

## Running them

Each notebook opens by cloning the repository and installing it when it is not
already in one, so it runs on Colab unchanged. Locally, install per the
[top-level README](../../README.md) — including the AprilTag textures, without
which no scene compiles — and then:

```sh
VIRTUAL_ENV=~/venvs/pick-and-place uv pip install -e py --group notebooks
export MUJOCO_GL=egl PAP_DATA_ROOT=~/pick-and-place-data
jupyter lab py/notebooks/
```

## Adding one

Cells call `pap` or import from `pick_and_place`; they do not reimplement it.

CI runs [`check-notebook`](https://github.com/mariogemoll/check-notebook) over
everything here: no stored outputs, no empty cells, `kernelspec` and
`language_info` the only metadata keys, markdownlint over the markdown cells,
and no line over 100 characters in the file itself.

```sh
jupyter nbconvert --clear-output --inplace py/notebooks/*.ipynb
```
