"""Public-data stream simulator (layer 1): rehearse the platform's 10-round stream on the 12 public bearings.

    python -m team_sim run --package <cpu-b package dir> --out runs/sim/B
    python -m team_sim report --out runs/sim/B

Design and usage: docs/SIMULATOR.md. Only new files; the harness, team_loop and team_ace are
called through their public functions and are not modified.
"""
