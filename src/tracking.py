"""Optional Comet ML experiment tracking (https://comet.com).

Enabled purely through environment variables so any run (Kaggle commit,
local, CI) turns tracking on/off without code changes, and training
behaviour is IDENTICAL when disabled:

    COMET_API_KEY          enables tracking; absent -> silent no-op
    COMET_PROJECT_NAME     default "vcm-preprocessing"
    COMET_WORKSPACE        default: the account's default workspace
    COMET_EXPERIMENT_NAME  default: derived from the run's out_dir
    COMET_MODE             "online" (default) | "offline" | "disabled"

``train()`` persists the experiment key next to its outputs
(``<out_dir>/comet_key.txt``) so the later ``evaluate.py`` process can attach
its final BD metrics to the SAME experiment (``ExistingExperiment``) instead
of spawning a second one.

Every Comet interaction is wrapped so a tracking problem can never fail a
training/eval run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

KEY_FILE = "comet_key.txt"
LOG_EVERY = 10  # per-step metric throttle (async batches on Comet's side)


class _Noop:
    """Stand-in used whenever tracking is off: every call is a no-op."""

    def log_params(self, params: dict) -> None: ...
    def log_step(self, step: int, metrics: dict, force: bool = False) -> None: ...
    def log_epoch(self, epoch: int, metrics: dict) -> None: ...
    def log_eval(self, metrics: dict) -> None: ...
    def log_curve(self, name: str, xs: list, ys: list) -> None: ...
    def finish(self, metrics: dict | None = None) -> None: ...


class _Comet:
    """Thin wrapper over a comet_ml Experiment / ExistingExperiment."""

    def __init__(self, exp: Any, out_dir: Path, name: str) -> None:
        self._exp = exp
        self._name = name
        self._key = None
        getter = getattr(exp, "get_key", None)  # comet_ml's documented accessor
        if callable(getter):
            try:
                self._key = getter()
            except Exception:
                self._key = None
        self._key = self._key or getattr(exp, "id", None)
        try:  # persist for evaluate()'s attach pass
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / KEY_FILE).write_text(str(self._key), encoding="utf-8")
        except OSError:
            pass

    @property
    def name(self) -> str:
        return self._name

    def log_params(self, params: dict) -> None:
        flat: dict[str, Any] = {}
        _flatten(params, "", flat)
        try:
            self._exp.log_parameters(flat)
        except Exception:
            pass

    def log_step(self, step: int, metrics: dict, force: bool = False) -> None:
        if not force and step % LOG_EVERY != 0:
            return
        try:
            for k, v in metrics.items():
                self._exp.log_metric(k, v, step=step)
        except Exception:
            pass

    def log_epoch(self, epoch: int, metrics: dict) -> None:
        try:
            for k, v in metrics.items():
                self._exp.log_metric(k, v, step=epoch, epoch=epoch)
        except Exception:
            pass

    def log_eval(self, metrics: dict) -> None:
        try:
            for k, v in metrics.items():
                if v is None:
                    continue
                self._exp.log_metric(f"eval/{k}", v)
        except Exception:
            pass

    def log_curve(self, name: str, xs: list, ys: list) -> None:
        if not xs or not ys or len(xs) != len(ys):
            return
        try:
            self._exp.log_curve(name, [float(x) for x in xs], [float(y) for y in ys])
        except Exception:
            pass

    def finish(self, metrics: dict | None = None) -> None:
        if metrics:
            self.log_eval(metrics)
        try:
            self._exp.end()
        except Exception:
            pass


def _flatten(obj: Any, prefix: str, out: dict, depth: int = 0) -> None:
    if not isinstance(obj, dict) or depth > 5:
        out[prefix or "param"] = str(obj)
        return
    for k, v in obj.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            _flatten(v, key, out, depth + 1)
        elif isinstance(v, (list, tuple)):
            out[key] = ",".join(map(str, v))
        elif v is None or isinstance(v, (bool, int, float, str)):
            out[key] = v
        else:
            out[key] = str(v)


def _default_name(out_dir: Path) -> str:
    """outputs/sweep_51/seed_0/mu3 -> sweep_51-seed_0-mu3 (last 3 parts otherwise)."""
    parts = list(out_dir.parts)
    if "outputs" in parts:
        parts = parts[parts.index("outputs") + 1:]
    else:
        parts = parts[-3:]
    parts = [p for p in parts if p and p not in (".", "..", "/", "\\")]
    return "-".join(parts) if parts else "run"


def _tracking_enabled() -> bool:
    if os.environ.get("COMET_MODE", "online") == "disabled":
        return False
    return bool(os.environ.get("COMET_API_KEY", "").strip())


def _new_experiment(out_dir: Path, name: str) -> Any:
    import comet_ml  # lazy: not a hard dependency

    return comet_ml.Experiment(
        api_key=os.environ["COMET_API_KEY"].strip(),
        project_name=os.environ.get("COMET_PROJECT_NAME", "vcm-preprocessing"),
        workspace=os.environ.get("COMET_WORKSPACE") or None,
        experiment_name=name,
        mode=os.environ.get("COMET_MODE", "online"),
        display_summary_level=0,
    )


def make_tracker(out_dir: Path | str, fallback_name: str | None = None) -> Any:
    """Tracker for a TRAIN run; writes comet_key.txt under out_dir."""
    out_dir = Path(out_dir)
    if not _tracking_enabled():
        return _Noop()
    try:
        import comet_ml  # noqa: F401
    except ImportError:
        print("[comet] comet_ml not installed -> tracking disabled")
        return _Noop()
    name = (os.environ.get("COMET_EXPERIMENT_NAME") or fallback_name
            or _default_name(out_dir))
    try:
        exp = _new_experiment(out_dir, name)
        tracker = _Comet(exp, out_dir, name)
        print(f"[comet] online: '{name}' "
              f"(project={os.environ.get('COMET_PROJECT_NAME', 'vcm-preprocessing')})")
        return tracker
    except Exception as e:  # bad key / no network -> never fail the run
        print(f"[comet] init failed ({e}) -> tracking disabled")
        return _Noop()


def attach_tracker(out_dir: Path | str, fallback_name: str | None = None) -> Any:
    """Tracker for an EVAL run: attach to the train experiment via comet_key.txt
    (looked up in out_dir and its parents); otherwise start a fresh experiment."""
    out_dir = Path(out_dir)
    if not _tracking_enabled():
        return _Noop()
    try:
        import comet_ml  # noqa: F401
    except ImportError:
        return _Noop()
    key = ""
    for cand in (out_dir / KEY_FILE, out_dir.parent / KEY_FILE,
                 out_dir.parent.parent / KEY_FILE):
        try:
            if cand.exists():
                key = cand.read_text(encoding="utf-8").strip()
                if key:
                    break
        except OSError:
            pass
    try:
        if key:
            exp = comet_ml.ExistingExperiment(
                api_key=os.environ["COMET_API_KEY"].strip(),
                previous_experiment=key,
                display_summary_level=0,
            )
            print(f"[comet] eval attaches to train experiment {key}")
            return _Comet(exp, out_dir, key)
        name = (os.environ.get("COMET_EXPERIMENT_NAME") or fallback_name
                or _default_name(out_dir) + "-eval")
        exp = _new_experiment(out_dir, name)
        print(f"[comet] eval (no train key found) -> new experiment '{name}'")
        return _Comet(exp, out_dir, name)
    except Exception as e:
        print(f"[comet] attach failed ({e}) -> tracking disabled")
        return _Noop()
