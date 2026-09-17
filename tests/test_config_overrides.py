"""Checks for CLI override coercion (src/config.py::_coerce, apply_overrides).

List support was added 2026-09-02: without it `task.teachers=[r3d_18]` silently
became the STRING "[r3d_18]", and iterating that yields characters, so a
single-teacher run could not be expressed as an override at all -- the only way
was editing the YAML, which changes every run sharing the file.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import _coerce, apply_overrides


def test_scalars_still_coerce() -> None:
    assert _coerce("6") == 6 and isinstance(_coerce("6"), int)
    assert _coerce("0.25") == 0.25
    assert _coerce("true") is True and _coerce("false") is False
    assert _coerce("none") is None and _coerce("null") is None
    assert _coerce("r3d_18") == "r3d_18"


def test_bracketed_values_become_lists() -> None:
    """Bare tokens must work: no quoting survives bash -> notebook -> shell."""
    assert _coerce("[r3d_18]") == ["r3d_18"]
    assert _coerce("[r3d_18,mc3_18]") == ["r3d_18", "mc3_18"]
    assert _coerce('["r3d_18"]') == ["r3d_18"]
    assert _coerce("[30,35,40]") == [30, 35, 40]
    assert _coerce("[]") == []


def test_the_failure_this_prevents() -> None:
    """A string is iterable, so the old behaviour failed SILENTLY, not loudly."""
    got = _coerce("[r3d_18]")
    assert not isinstance(got, str), "a list override must not stay a string"
    assert list(got) == ["r3d_18"], "iterating must give the teacher, not characters"


def test_apply_overrides_replaces_a_list_key() -> None:
    cfg = {"task": {"teachers": ["r3d_18", "mc3_18"]}, "eval": {"qp_list": [30, 50]}}
    cfg = apply_overrides(cfg, ["task.teachers=[r3d_18]", "eval.qp_list=[40]"])
    assert cfg["task"]["teachers"] == ["r3d_18"]
    assert cfg["eval"]["qp_list"] == [40]


def test_apply_overrides_creates_missing_nodes_and_rejects_bad_input() -> None:
    cfg = apply_overrides({}, ["loss.kappa=10", "train.epochs=16"])
    assert cfg == {"loss": {"kappa": 10}, "train": {"epochs": 16}}
    try:
        apply_overrides({}, ["no_equals_sign"])
    except ValueError:
        pass
    else:
        raise AssertionError("an override without '=' must raise")


if __name__ == "__main__":
    test_scalars_still_coerce()
    test_bracketed_values_become_lists()
    test_the_failure_this_prevents()
    test_apply_overrides_replaces_a_list_key()
    test_apply_overrides_creates_missing_nodes_and_rejects_bad_input()
    print("config override self-checks passed")
