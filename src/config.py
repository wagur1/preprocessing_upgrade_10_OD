"""Tiny YAML config loader with dotted-key CLI overrides."""

from __future__ import annotations

from typing import Any, List

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _coerce(v: str) -> Any:
    """Coerce a CLI override value. Bracketed values become LISTS.

    List support matters because several load-bearing keys are lists
    (``task.teachers``, ``eval.qp_list``) and without it they could only be
    changed by editing the YAML, which changes every run that shares the file.
    Bare tokens are accepted so no quoting survives the
    bash -> notebook -> shell layers: ``task.teachers=[r3d_18]`` works, and so do
    ``[r3d_18,mc3_18]``, ``["r3d_18"]`` and ``[30,35,40]``. An empty ``[]`` is an
    empty list, not the string.
    """
    if len(v) >= 2 and v[0] == "[" and v[-1] == "]":
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [_coerce(x.strip().strip("\"'")) for x in inner.split(",")]
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    if v.lower() in {"none", "null"}:
        return None
    return v


def apply_overrides(cfg: dict, overrides: List[str]) -> dict:
    """Apply ``a.b.c=value`` overrides in place (values are type-coerced)."""
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override '{ov}' must be key=value")
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(val)
    return cfg
