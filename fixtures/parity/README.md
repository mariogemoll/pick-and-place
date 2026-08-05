<!-- SPDX-FileCopyrightText: 2026 Mario Gemoll -->
<!-- SPDX-License-Identifier: 0BSD -->

# Cross-language parity fixtures

Python and TypeScript each carry their own implementation of the SO-101's
kinematics, the closed-form IK, the grasp geometry and the canonical grasp
search. Both are unit-tested. Neither test proves the two agree.

These files are the shared oracle. **Python is the source of truth** — it drives
the real arm and generates every demonstration; TypeScript follows.

| Where | What it does |
| --- | --- |
| `py/scripts/generate_parity_fixtures.py` | Writes these files. |
| `py/tests/test_parity.py` | Fails when Python stops reproducing them. |
| `ts/src/parity/*.test.ts` | Fails when TypeScript stops reproducing them. |

## Regenerating

```sh
cd py && MUJOCO_GL=egl python scripts/generate_parity_fixtures.py
```

A moved planner shows up first as a failure in `py/tests/test_parity.py`.
Regenerating is how you accept the change — and the moment to check whether the
TypeScript side has to follow, because the TypeScript tests will now fail until
it does. Review the diff; do not regenerate to silence a failure you have not
explained.

## The files

| File | Pins |
| --- | --- |
| `kinematics.json` | The arm measured off the model: pan axis, segments, tool length, wrist twist, joint limits. Everything else builds on it. |
| `geometry.json` | Cube, contact and grasp transforms, as 16 row-major numbers each. Frame algebra only — no arm, no IK. |
| `simple_ik.json` | Every branch the closed-form IK returns, including the poses that must return none. |
| `forward_kinematics.json` | Joints to gripper-target position. Python solves the planar chain in closed form, TypeScript walks the model's body tree, so these agree to well under a millimetre rather than exactly. |
| `grasp.json` | The canonical grasp search: the grasp it picks, and the head of the candidate stream it picks from. Order matters — the planner takes the first survivor. |
| `easing.json` | `smoothstep` and the timed arc fraction, sampled past both ends so the clamping is pinned too. |

Numbers carry twelve significant digits and the consumers compare at `1e-9`, so
only last-place arithmetic differences pass. One case per line: the files stay
inside the repository's 40 KB ceiling and still diff readably.

## What is deliberately not covered

The two trajectory builders have diverged, and this is where they part. Python
plans the physical eight-phase motion from a canonical grasp with speed-derived
phase durations; TypeScript animates a five-stage illustrative one from a
vertical grasp with fixed durations. They share the grasp geometry, the IK and
the easing curves, all of which are covered above — but a sampled trajectory
frame is not a quantity the two are supposed to agree on.
