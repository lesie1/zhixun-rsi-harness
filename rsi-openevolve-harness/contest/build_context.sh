#!/usr/bin/env bash
# Assemble the Docker build context for the contest image.
#   contest/build_context.sh <解压后的 SDK 目录 contest-harness-sdk-1.0.4> [输出目录，默认 contest/build]
# 然后：docker build -t zhixun-rsi:v1 contest/build && docker save zhixun-rsi:v1 -o zhixun-rsi-v1.tar
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$HERE")
SDK=${1:?usage: build_context.sh <sdk dir> [out dir]}
OUT=${2:-$HERE/build}
[ -f "$SDK/pyproject.toml" ] && [ -d "$SDK/contest_harness" ] && [ -f "$SDK/template/harness.py" ] || { echo "不是 SDK 目录: $SDK" >&2; exit 2; }
rm -rf "$OUT"
mkdir -p "$OUT/sdk" "$OUT/app/rsi/team_sim" "$OUT/app/rsi/policies"
cp "$SDK/pyproject.toml" "$OUT/sdk/"
[ -f "$SDK/README.md" ] && cp "$SDK/README.md" "$OUT/sdk/"
cp -r "$SDK/contest_harness" "$OUT/sdk/contest_harness"
cp "$SDK/template/harness.py" "$OUT/app/harness.py"
cp "$HERE/contest_submission.py" "$OUT/app/contest_submission.py"
cp -r "$REPO/harness" "$OUT/app/rsi/harness"
cp -r "$REPO/baselines" "$OUT/app/rsi/baselines"
rm -rf "$OUT/app/rsi/baselines/tests"
cp "$REPO/team_sim/__init__.py" "$REPO/team_sim/policy.py" "$OUT/app/rsi/team_sim/"
cp "$HERE/policies/policy_pick.py" "$OUT/app/rsi/policies/"
cp "$HERE/Dockerfile" "$OUT/Dockerfile"
find "$OUT" -name __pycache__ -prune -exec rm -rf {} +
echo "build context: $OUT ($(du -sh "$OUT" | cut -f1))"
