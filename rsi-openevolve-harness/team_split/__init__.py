"""Formal data-split module: validation data -> selection part + gate part.

Public API (see docs/SPLIT_MODULE.md):
    load_table(run_config)                          -> LabeledTable
    split(table, SplitOptions)                      -> Assignment
    build_manifest(table, assignment, stage_id, options) -> manifest dict
    verify_manifest(manifest, table, stage_id, options)  -> [Problem]
    split_report(manifest, table) / render_report(report)
    derive_run_config(base_config, manifest, out, suffix)
Command line: python -m team_split {make,verify,report,wire} --help
"""
import harness  # noqa: F401  (pins BLAS threads before numpy and puts the project on sys.path)

from .manifest import build_manifest, compute_counts, read_manifest, write_manifest
from .report import render_report, split_report
from .strategies import (KNOWN_METHODS, STRATEGIES, Assignment, SplitOptions, choose_method,
                         options_for_manifest, split)
from .table import LabeledTable, load_table
from .verify import Problem, has_errors, verify_manifest
from .wiring import derive_run_config

__all__ = ["LabeledTable", "load_table", "SplitOptions", "Assignment", "STRATEGIES", "KNOWN_METHODS",
           "choose_method", "split", "options_for_manifest", "build_manifest", "compute_counts",
           "read_manifest", "write_manifest", "Problem", "verify_manifest", "has_errors",
           "split_report", "render_report", "derive_run_config"]
