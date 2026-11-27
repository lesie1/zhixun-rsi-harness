"""Batch loop around the harness: per arriving batch, decide whether to retrain (metric
trigger), run split → search → gate if so, and output "deploy" or "keep".

See docs/BATCH_LOOP.md. Only adds files; the harness, team_split and team_ace are used
through their public functions and documented commands.
"""
import harness  # noqa: F401  (pins BLAS threads before numpy is imported)

from .batches import normalize, read_batches, simulate
from .state import init, load
from .step import process
from .trigger import check, profile

__all__ = ["check", "init", "load", "normalize", "process", "profile", "read_batches", "simulate"]
