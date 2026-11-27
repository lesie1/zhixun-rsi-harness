"""Which metric decides the score, which metrics act as guards.

Resolution order (first match wins):
  1. an explicit spec file supplied for the stage (e.g. organisers announce a new metric);
  2. the table of known tasks below (A, B, C);
  3. defaults by task type, which work for any dataset in the official JSONL/checkpoint format.

Spec file format (JSON), all keys optional except "primary":
  {"primary": "rmse", "guards": {"mae": 0.05, "nasa_score": 0.20}, "tiebreak": ["mae"],
   "penalty_weight": 1.0, "group_guard": 0.10, "unit": "cycles", "aggregate": "auto"}
A guard value is the tolerated relative worsening against the parent checkpoint.
"""
import json
from pathlib import Path

from .metrics import CLASSIFICATION_METRICS, HIGHER_IS_BETTER, REGRESSION_METRICS

DEFAULTS = {
    "classification": {"primary": "macro_f1", "guards": {"accuracy": 0.02}, "tiebreak": ["accuracy"]},
    "regression": {"primary": "rmse", "guards": {"mae": 0.05}, "tiebreak": ["mae"]},
}

KNOWN_TASKS = {
    "A": {"primary": "macro_f1", "guards": {"accuracy": 0.02}, "tiebreak": ["accuracy"],
          "description": "UCI gas sensor array drift, six-class gas identification"},
    "B": {"primary": "rmse", "guards": {"mae": 0.05}, "tiebreak": ["mae"], "unit": "minutes",
          "description": "XJTU-SY bearing remaining useful life in minutes"},
    "C": {"primary": "rmse", "guards": {"mae": 0.05, "nasa_score": 0.20}, "tiebreak": ["mae"],
          "unit": "cycles", "description": "NASA C-MAPSS turbofan remaining useful life in cycles"},
}

COMMON = {"penalty_weight": 1.0, "group_guard": 0.10, "aggregate": "auto", "unit": "", "description": ""}


def _validate(spec, task_type):
    allowed = CLASSIFICATION_METRICS if task_type == "classification" else REGRESSION_METRICS
    names = [spec["primary"], *spec["guards"], *spec["tiebreak"]]
    for name in names:
        if name not in HIGHER_IS_BETTER:
            raise ValueError(f"unknown metric '{name}'; known metrics: {sorted(HIGHER_IS_BETTER)}")
        if name not in allowed:
            raise ValueError(f"metric '{name}' is not defined for task type '{task_type}'")
    for name, tolerance in spec["guards"].items():
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or tolerance < 0:
            raise ValueError(f"guard tolerance for '{name}' must be a non-negative number")
    if spec["aggregate"] not in ("auto", "pooled", "group_mean"):
        raise ValueError("aggregate must be auto, pooled or group_mean")
    return spec


def resolve_spec(task_type, task_name=None, spec_file=None):
    """Return the scoring spec for a stage together with where it came from."""
    if task_type not in DEFAULTS:
        raise ValueError(f"unsupported task type '{task_type}'; supported: {sorted(DEFAULTS)}")
    if spec_file:
        chosen, source = json.loads(Path(spec_file).read_text(encoding="utf-8")), f"spec_file:{spec_file}"
        if "primary" not in chosen:
            raise ValueError("spec file must name a primary metric")
    elif task_name in KNOWN_TASKS:
        chosen, source = KNOWN_TASKS[task_name], f"known_task:{task_name}"
    else:
        chosen, source = DEFAULTS[task_type], f"default:{task_type}"
    spec = {**COMMON, "guards": {}, "tiebreak": [], **chosen}
    spec["guards"] = dict(spec["guards"])
    spec["tiebreak"] = list(spec["tiebreak"])
    spec["higher_is_better"] = HIGHER_IS_BETTER[spec["primary"]] if spec["primary"] in HIGHER_IS_BETTER else None
    spec["source"] = source
    spec["task_type"] = task_type
    spec["with_nasa"] = "nasa_score" in (spec["primary"], *spec["guards"], *spec["tiebreak"])
    return _validate(spec, task_type)
