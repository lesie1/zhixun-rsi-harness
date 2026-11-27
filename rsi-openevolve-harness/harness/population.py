"""Population rule: candidates that scored 0 (invalid code, crash, rule violation) are not
admitted to the program database, so they can never be picked as a parent.

Their lesson is not lost: the failure is written to the iteration records and shown to
the model in the "recent failures" section of the next prompts.
"""


def admit_valid(snapshot, candidate, island):
    try:
        return float(candidate.metrics.get("combined_score") or 0.0) > 0.0
    except (TypeError, ValueError):
        return False


def strategy():
    from openevolve.population import PopulationStrategy
    return PopulationStrategy(admit=admit_valid)
