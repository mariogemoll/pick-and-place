# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The evaluator's parser and run, tested without launching a process.

None of this was reachable before the two were split out of ``main``: the
parser was built inside it and consumed on the spot, and the run could only be
exercised by starting ``eval_policy_sim.py`` and reading the files it left
behind.
"""

from pathlib import Path

import pytest

from pick_and_place.cli.eval_policy_sim import build_parser, validate


def _parse(argv):
    parser = build_parser("test")
    return parser, parser.parse_args(argv)


def test_leaves_reject_each_others_flags():
    """The parser carries applicability, so no code has to police it."""
    parser = build_parser("test")
    for argv in (
        ["scripted", "--output", "out", "--flow-act-steps", "3"],
        ["scripted", "--output", "out", "--n-action-steps", "1"],
        ["flow-image", "--output", "out", "--checkpoint", "c", "--instruction", "hi"],
        ["lerobot", "--output", "out", "--checkpoint", "c", "--flow-export", "e"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


def test_scripted_leaf_takes_no_checkpoint():
    _, args = _parse(["scripted", "--output", "out"])
    assert args.controller == "scripted"
    assert not hasattr(args, "checkpoint")


def test_validate_rejects_a_half_given_image_size(tmp_path):
    parser, args = _parse(["scripted", "--output", str(tmp_path / "new"), "--image-height", "480"])
    with pytest.raises(SystemExit):
        validate(parser, args)


def test_validate_rejects_an_existing_output(tmp_path):
    parser, args = _parse(["scripted", "--output", str(tmp_path)])
    with pytest.raises(SystemExit):
        validate(parser, args)


def test_validate_accepts_a_plain_scripted_run(tmp_path):
    parser, args = _parse(["scripted", "--output", str(tmp_path / "new")])
    validate(parser, args)  # does not raise


def test_from_args_defaults_the_other_leaves_flags_to_none(tmp_path):
    from pick_and_place.rollout.evaluation import EvaluationRun

    _, args = _parse(["scripted", "--output", str(tmp_path / "new"), "--limit", "3"])
    config = EvaluationRun.from_args(args)

    assert config.controller == "scripted"
    assert config.limit == 3
    assert config.scripted_perception == "geometric"
    # Declared by the other two leaves, so absent from this namespace entirely.
    assert config.checkpoint is None
    assert config.instruction is None
    assert config.flow_export is None


def test_override_hw_is_none_unless_both_given(tmp_path):
    from pick_and_place.rollout.evaluation import EvaluationRun

    _, args = _parse(["scripted", "--output", str(tmp_path / "new")])
    assert EvaluationRun.from_args(args).override_hw is None

    _, args = _parse(
        ["scripted", "--output", str(tmp_path / "new"), "--image-height", "96",
         "--image-width", "128"]
    )
    assert EvaluationRun.from_args(args).override_hw == (96, 128)


def test_a_failing_episode_still_closes_the_environment(tmp_path, monkeypatch):
    """The cleanup must survive the split out of ``main``.

    A controller that raises mid-suite is the case a successful re-run cannot
    check: the frozen-manifest comparison only ever exercises the happy path,
    so a leaked MuJoCo context would not show up there.
    """
    from pick_and_place.rollout import evaluation as ev

    closed = {"env": False, "controller": False}

    class FakeEnv:
        model = object()
        data = object()
        cube_belief_error = (0.0, 0.0, 0.0, 0.0)

        def close(self):
            closed["env"] = True

    class FakeController:
        def close(self):
            closed["controller"] = True

    class FakeScenario:
        scenario_id = "s0"
        control_hz = 30.0
        max_steps = 10
        target_position_m = (0.0, 0.0, 0.0)
        domain_randomization_preset = None

    class FakeManifest:
        suite = "fake"
        scenarios = (FakeScenario(),)

        def sha256(self):
            return "0" * 64

    monkeypatch.setattr(ev.ScenarioManifest, "load", staticmethod(lambda path: FakeManifest()))
    monkeypatch.setattr(ev, "PolicySimEnv", lambda **kwargs: FakeEnv())
    monkeypatch.setattr(ev, "_camera_base_metadata", lambda model: {})
    monkeypatch.setattr(
        ev, "sim_scripted_controller", lambda **kwargs: (FakeController(), {"type": "scripted"})
    )

    def boom(*args, **kwargs):
        raise RuntimeError("controller exploded")

    monkeypatch.setattr(ev, "evaluate_policy_episode", boom)

    config = ev.EvaluationRun(
        controller="scripted",
        output=tmp_path / "run",
        render_height=480,
        render_width=640,
        manifest=Path("unused.json"),
        scripted_perception="geometric",
    )

    with pytest.raises(RuntimeError, match="controller exploded"):
        ev.run_evaluation(config, report=lambda line: None)

    assert closed["env"], "the simulator was left open when the controller raised"
    assert closed["controller"], "the controller was never closed"
