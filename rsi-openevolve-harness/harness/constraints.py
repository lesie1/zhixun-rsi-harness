"""Hard rules. A violation makes a candidate invalid (score 0), it is never traded off.

1. Static check of the policy source (LLM-written code) before it is executed.
   This is a guard against accidents, not a sandbox: real isolation is the subprocess
   with a timeout plus the container the harness runs in.
2. Continuation rules of the competition, checked on the produced checkpoint with the
   reference loader: same task/classes/features/width/seed, normalization and target
   scaling untouched, optimizer steps increased, parameter budget respected.
"""
import ast
import re

import numpy as np

from baselines.model import Model

REQUIRED_NAMES = ("CONFIG", "learning_rate", "sample_weights", "output_grad", "extra_grad", "pick_epoch")
ALLOWED_IMPORTS = {"numpy", "math"}
FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "vars",
                   "getattr", "setattr", "delattr", "breakpoint", "exit", "quit", "help", "memoryview"}
FORBIDDEN_ATTRIBUTES = {"load", "save", "savez", "savez_compressed", "loadtxt", "savetxt", "genfromtxt",
                        "fromfile", "tofile", "memmap", "DataSource", "seed", "ctypeslib", "f2py",
                        "distutils", "testing", "lib", "system", "popen"}
# Same marker definition as OpenEvolve (openevolve/utils/code_utils.py): a comment line.
_MARKER = re.compile(r"^\W*EVOLVE-BLOCK-(START|END)\b")


def check_policy_source(code):
    """Return a list of human-readable problems (empty when the policy may be executed)."""
    problems = []
    markers = [match.group(1) for match in (_MARKER.match(line) for line in code.split("\n")) if match]
    if markers != ["START", "END"]:
        problems.append("policy must keep exactly one EVOLVE-BLOCK-START / EVOLVE-BLOCK-END marker pair, in order")
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return problems + [f"syntax error at line {error.lineno}: {error.msg}"]
    defined = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            defined.update(target.id for target in node.targets if isinstance(target, ast.Name))
    for name in REQUIRED_NAMES:
        if name not in defined:
            problems.append(f"required top-level name '{name}' is missing")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"line {node.lineno}: import of '{alias.name}' is not allowed (only numpy, math)")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS or node.level:
                problems.append(f"line {node.lineno}: import from '{node.module}' is not allowed (only numpy, math)")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_CALLS:
            problems.append(f"line {node.lineno}: use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRIBUTES:
                problems.append(f"line {node.lineno}: attribute '.{node.attr}' is not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append(f"line {node.lineno}: global/nonlocal statements are not allowed")
    return sorted(set(problems))


SAME_META = ("format", "device", "task", "paths", "classes", "hidden", "seed", "parameter_count",
             "target_mean", "target_scale", "learning_rate", "profile", "target_unit")


def check_checkpoint(parent, candidate_path):
    """Load `candidate_path` with the reference loader and compare it with the parent model.

    Returns (model or None, problems).
    """
    try:
        model = Model.load(candidate_path)
    except Exception as error:  # the reference loader rejects anything malformed
        return None, [f"reference loader rejected the checkpoint: {error}"]
    problems = []
    for key in SAME_META:
        if model.meta.get(key) != parent.meta.get(key):
            problems.append(f"checkpoint meta '{key}' changed: {parent.meta.get(key)!r} -> {model.meta.get(key)!r}")
    if not np.array_equal(model.mean, parent.mean) or not np.array_equal(model.scale, parent.scale):
        problems.append("input normalization (mean/scale) differs from the parent checkpoint")
    if model.meta["steps"] <= parent.meta["steps"]:
        problems.append(f"optimizer steps did not increase ({parent.meta['steps']} -> {model.meta['steps']})")
    if model.meta["epochs"] <= parent.meta["epochs"]:
        problems.append(f"epochs did not increase ({parent.meta['epochs']} -> {model.meta['epochs']})")
    for name in ("w1", "b1", "w2", "b2"):
        if model.params[name].shape != parent.params[name].shape:
            problems.append(f"shape of {name} changed")
    return model, problems
