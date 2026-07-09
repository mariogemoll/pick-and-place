<!-- SPDX-FileCopyrightText: 2026 Mario Gemoll -->
<!-- SPDX-License-Identifier: 0BSD -->

# Mesh Optimization

Decimates and packs robot meshes into size-budgeted, meshopt-compressed GLBs for the web viewer. See
`scripts/simplification.py` and `scripts/simplify_generic_meshes.py` for the decimation/budget
mechanics, and `scripts/convert_meshes_generic.sh` for the full conversion pipeline (intermediary
GLB -> optimized GLB -> web-model JSON manifest).

## Known-good `--detail` settings

Some robots have fine details that get washed out by the shared size-budget tolerance unless
individually protected with `--detail GLOB=FACTOR` (`GLOB` matches the mesh's node name; see
`simplify_generic_meshes.py`'s docstring for the flag's mechanics).

- **Panda** (`panda_mj_description`): 

  ```sh
  scripts/convert_meshes_generic.sh panda_mj_description \
      --detail 'link0*=20' --detail 'link5_0=20' --detail 'link6*=20'
  ```

  Note: `link6_11` (the Panda logo decal) z-fights with the surface beneath it in the web viewer
  regardless of decimation detail — that's a separate, unresolved rendering issue (the decal mesh
  appears to sit flush with the underlying surface), not something `--detail` can fix.
