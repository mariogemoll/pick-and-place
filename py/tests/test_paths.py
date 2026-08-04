# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Tests for machine-local data root resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from pick_and_place.paths import (
    ENV_VAR,
    DataRootNotConfigured,
    data_root,
    datasets_root,
    outputs_root,
)


def test_data_root_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, "/tmp/pap-data")
    assert data_root() == Path("/tmp/pap-data")


def test_data_root_expands_a_user_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, "~/pap-data")
    assert data_root() == Path.home() / "pap-data"


def test_unset_root_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    with pytest.raises(DataRootNotConfigured):
        data_root()


def test_blank_root_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, "   ")
    with pytest.raises(DataRootNotConfigured):
        data_root()


def test_the_error_names_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    with pytest.raises(DataRootNotConfigured, match=ENV_VAR):
        data_root()


def test_derived_roots_hang_off_the_data_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, "/tmp/pap-data")
    assert datasets_root() == Path("/tmp/pap-data/datasets")
    assert outputs_root() == Path("/tmp/pap-data/outputs")


def test_derived_roots_require_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    for resolve in (datasets_root, outputs_root):
        with pytest.raises(DataRootNotConfigured):
            resolve()
