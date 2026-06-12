<!-- SPDX-FileCopyrightText: 2026 Mario Gemoll -->
<!-- SPDX-License-Identifier: 0BSD -->

# Project Overview

The **pick-and-place** project is a comprehensive toolkit for the **Standard Open SO-101 robot arm*. It spans hardware design (3D printing, CAD), simulation (URDF, MJCF), and software (mesh optimization and web-based 3D visualization).

The project is divided into four main components:
1.  **Hardware & Simulation (`SO-ARM100/`):** Contains documentation, bill of materials, 3D printing files (STL, STEP), and simulation models (URDF, MJCF). This vendored directory is upstream truth and is never modified.
2.  **Mesh Optimization (`mesh_optimization/`):** A Python subproject used to simplify high-poly STL meshes into optimized GLB files for web visualization.
3.  **Simulation (`py/`):** The pick-and-place Python package. Composes MuJoCo models programmatically with `MjSpec`: it loads the stock MJCF from `SO-ARM100/` and replaces the full-mesh collision geoms with a hand-tuned box collision model (`pick_and_place.collision_boxes`). `python -m pick_and_place.export` writes standalone MJCF files (machine-local, gitignored `py/out/`) for external consumers such as the `simulate` viewer or the Isaac importer. The collision box values are a tuned asset — do not regenerate or "clean up" the numbers.
4.  **Robot Inspection & Simulation (`ts/`):** A TypeScript/Vite application that provides a browser-based environment for inspecting the robot's physical structure. It includes:
    -   **Kinematic Definitions:** Hardcoded link and joint properties matching calibrated URDFs.
    -   **Interactive Simulation:** UI for manipulating joint values in a 3D scene.
    -   **Mesh Diagnostics:** Real-time calculation of polygon counts and asset sizes for performance auditing.
    -   **3D Visualization:** High-performance rendering using Three.js and Meshopt.
    -   **Mesh Optimization:** Post-processing scripts to compress and bundle 3D assets for production.

### Mesh Optimization Pipeline
The project uses a mandatory two-step process to transform raw hardware assets into web-ready visualizations:
1.  **Step 1 (Python):** Run `python mesh_optimization/scripts/simplify_meshes.py`. This decimates high-poly STL files from `SO-ARM100/Simulation/SO101/assets/` into simplified GLB files in `intermediary-glb/`.
2.  **Step 2 (TypeScript):** Run `npm run optimize-meshes` in the `ts/` directory. This uses `gltf-transform` and `meshopt` to further compress the GLB files and move them from `intermediary-glb/` to `ts/public/so101_assets/` for delivery by the web app.

### Visualization (TypeScript)
Located in the `ts/` directory.
- **Development:** `npm run dev` - Starts the Vite development server.
- **Build:** `npm run build` - Compiles TypeScript and builds the production bundle.
- **Linting:** `npm run lint` - Runs ESLint and Stylelint.
- **Testing:** `npm run test` - Runs Vitest.
- **Mesh Optimization:** `npm run optimize-meshes` - Uses `@gltf-transform/cli` to optimize GLB files in `public/so101_assets/`.

### Repository Scripts
Located in the `scripts/` directory.
- **Mesh Generation & Optimization:** `./scripts/convert_meshes.sh [dst_dir]` - Automates the two-step pipeline and copies the assets to a destination directory.
- **License Check:** `pnpm run check-license-headers` (run from within the `scripts/` directory) - Verifies SPDX license headers across the project. Requires building the script first.

## Development Conventions

-   **Licensing:** All source files must include SPDX license headers.
    -   **Copyright:** `SPDX-FileCopyrightText: 2026 Mario Gemoll`
    -   **License:** `SPDX-License-Identifier: 0BSD`
-   **Coding Style:**
    -   **TypeScript:** Follows standard TypeScript conventions, managed by ESLint and Prettier (via Vite/Vitest ecosystem).
    -   **Python:** Follows PEP 8, enforced by Ruff (configured in `pyproject.toml`).
-   **Architecture:**
    -   Source hardware meshes (STL) are the "truth."
    -   Python scripts bridge the gap between high-poly CAD/STL and web-friendly GLB.
    -   The TypeScript app in `ts/` is the primary interface for inspecting and visualizing the robot.

## Key Files & Directories

-   `SO-ARM100/README.md`: Main hardware documentation and assembly guide.
-   `SO-ARM100/Simulation/`: Simulation models for SO100 and SO101.
-   `mesh_optimization/scripts/simplify_meshes.py`: Core logic for mesh decimation.
-   `py/src/pick_and_place/builder.py`: MuJoCo model composition (`build_robot`).
-   `py/src/pick_and_place/collision_boxes.py`: Hand-tuned box collision model (the values are the asset).
-   `py/scripts/view_robot.py`: Interactive MuJoCo viewer for the composed model.
-   `ts/src/main.ts`: Entry point for the 3D visualization app.
-   `ts/src/visualizations/`: Modular visualization components (robot, gripper, body-tree).
