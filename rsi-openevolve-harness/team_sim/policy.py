"""Render the training policy of one candidate from harness/policy_rul_cap.py.

The template keeps every hook of policy_initial.py; only CONFIG values are filled in here, so
the result passes the same static checks and the same frozen runner.
"""
from pathlib import Path

from harness import PROJECT_ROOT

TEMPLATE = PROJECT_ROOT / "harness" / "policy_rul_cap.py"


def _replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"template changed: expected exactly one {old!r}")
    return text.replace(old, new)


def render(cap_multiplier=None, label_mean=None, target_mean=None, target_scale=None, epochs=None):
    """Policy source. cap_multiplier None = no cap (trains exactly like policy_initial.py)."""
    text = TEMPLATE.read_text(encoding="utf-8")
    if epochs is not None:
        text = _replace_once(text, '"epochs": 5,', f'"epochs": {int(epochs)},')
    if cap_multiplier is None:
        return text
    if label_mean is None or target_mean is None or not target_scale:
        raise ValueError("a capped policy needs label_mean, target_mean and target_scale")
    for key, value in (("cap_multiplier", float(cap_multiplier)), ("label_mean", float(label_mean)),
                       ("target_mean", float(target_mean)), ("target_scale", float(target_scale))):
        text = _replace_once(text, f'"{key}": None,', f'"{key}": {value!r},')
    return text


def write(path, source):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return str(path)
