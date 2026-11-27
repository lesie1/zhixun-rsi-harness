"""RSI harness built around OpenEvolve for continuing CPU NumPy MLP checkpoints.

Importing this package first imports `baselines.model`, which pins the BLAS thread
count (BASELINE_CPU_THREADS) before NumPy is loaded, exactly like the reference trainer.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines import model as _reference_model  # noqa: E402,F401  (sets thread env before numpy)

__all__ = ["PROJECT_ROOT"]
