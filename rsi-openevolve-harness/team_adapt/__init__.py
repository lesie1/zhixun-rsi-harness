"""Continual adaptation over time-ordered batches for the OpenEvolve + ACE harness.

Adds a controller around the existing entry points (team_split make/wire, scripts/run_evolve.py,
scripts/run_gate.py): batch registry with explicit data roles, stagnation detection on the
selection split, stage switching to newly labelled batches, model inheritance from the gate
decision, three-arm comparisons, a one-time final acceptance, and an exposure ledger.
See docs/ADAPTATION.md. Only adds files; training, scoring, gate and ACE behaviour are unchanged.
"""
import harness  # noqa: F401  (pins BLAS threads before numpy is imported)

from .config import ConfigError, load_config, make_plan
from .controller import Experiment
from .stagnation import analyze, classify

__all__ = ["ConfigError", "Experiment", "analyze", "classify", "load_config", "make_plan"]
