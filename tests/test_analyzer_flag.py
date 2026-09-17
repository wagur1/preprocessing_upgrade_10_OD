"""Regression: the on-teacher eval arm must actually swap the analyzer.

Ported from v8 commit b5aa4dc: v9's mk_eval_kernel.py HARDCODED
``eval.held_out_backbone=r2plus1d_18`` in every eval kernel, so an
on-teacher arm could not even be expressed (and the v8 bug showed the
omission path silently keeps the YAML value). The fix emits
``eval.held_out_backbone=null`` for --analyzer teacher so the YAML value is
actively nulled and ``build_analyzer`` falls back to ``task.backbone``
(r3d_18).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ops.mk_eval_kernel import held_out_override
from src.config import apply_overrides


def test_heldout_mode_pins_the_canonical_analyzer() -> None:
    assert held_out_override("heldout") == "eval.held_out_backbone=r2plus1d_18"


def test_teacher_mode_nulls_the_key_instead_of_pinning_heldout() -> None:
    ov = held_out_override("teacher")
    assert ov == "eval.held_out_backbone=null"
    assert "r2plus1d_18" not in ov


def test_null_override_beats_the_yaml_value() -> None:
    """Mirror of a committed YAML (e.g. sandwich_ar.yaml) + the new override."""
    cfg = {
        "task": {"name": "action_recognition", "backbone": "r3d_18"},
        "eval": {"held_out_backbone": "r2plus1d_18"},
    }
    cfg = apply_overrides(cfg, [held_out_override("teacher")])
    assert cfg["eval"]["held_out_backbone"] is None


def test_the_failure_this_prevents() -> None:
    """build_analyzer's contract: only a truthy held_out_backbone overrides
    task.backbone. None must be falsy so the on-teacher arm takes the fallback."""
    cfg = apply_overrides(
        {"eval": {"held_out_backbone": "r2plus1d_18"}},
        [held_out_override("teacher")],
    )
    held = cfg.get("eval", {}).get("held_out_backbone")
    assert not held, "nulled key must read as unset (falsy), not '' or 'null'"


def test_unknown_analyzer_is_rejected() -> None:
    try:
        held_out_override("bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown analyzer name must raise")


if __name__ == "__main__":
    test_heldout_mode_pins_the_canonical_analyzer()
    test_teacher_mode_nulls_the_key_instead_of_pinning_heldout()
    test_null_override_beats_the_yaml_value()
    test_the_failure_this_prevents()
    test_unknown_analyzer_is_rejected()
    print("analyzer-flag regression checks passed")
