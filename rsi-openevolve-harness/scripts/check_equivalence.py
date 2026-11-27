#!/usr/bin/env python
"""Prove that the frozen runner with the unmodified policy equals the official trainer.

Both are started from the same checkpoint for the same number of epochs; the two
resulting checkpoint files must have the same SHA-256.

  python scripts/check_equivalence.py --package data/cpu-a-guide-20261110-v1 --profile A
"""
import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", required=True, help="extracted baseline package (has checkpoint.npz and public/)")
    parser.add_argument("--profile", required=True, choices=["A", "B", "C"])
    args = parser.parse_args(argv)
    package = Path(args.package).resolve()
    data = ["--train", str(package / "public" / "train.jsonl"), "--validation", str(package / "public" / "validation.jsonl")]
    with tempfile.TemporaryDirectory() as temporary:
        official, ours = Path(temporary) / "official", Path(temporary) / "runner"
        subprocess.run([sys.executable, "-m", "baselines.train", "--profile", args.profile, *data,
                        "--from-baseline", str(package / "checkpoint.npz"), "--epochs", "5",
                        "--output", str(official), "--stage-id", "round-1"],
                       cwd=ROOT, check=True, capture_output=True)
        subprocess.run([sys.executable, "-m", "harness.runner", "--policy", str(ROOT / "harness" / "policy_initial.py"),
                        "--parent", str(package / "checkpoint.npz"), *data, "--output", str(ours),
                        "--stage-id", "round-1"], cwd=ROOT, check=True, capture_output=True)
        a, b = digest(official / "checkpoint.npz"), digest(ours / "checkpoint.npz")
    same = a == b
    print(f"profile {args.profile}: official {a[:16]}  runner {b[:16]}  -> {'IDENTICAL' if same else 'DIFFERENT'}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
