"""A-type gas classification: automatic adaptation flow (docs/GAS_FLOW.md).

One controller per round: per-class diagnosis (team_loop's check plus confusion direction),
repair search on the current training data with alarm-aware stagnation (team_adapt's
monitor and model inheritance), then - only when the search stagnated and labelled,
trainable new samples exist - a data-mixing search with a code-enforced 20 % cap on new
data, a frozen choice, and a one-time final acceptance with rollback.
"""
import harness  # noqa: F401  (pins BLAS threads before numpy is imported)

from .controller import GasFlow

__all__ = ["GasFlow"]
