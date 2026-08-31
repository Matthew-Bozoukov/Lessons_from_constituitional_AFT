#!/usr/bin/env bash
# ABOUTME: Destroys the spec-gaming pod once data entry exits AND its results are
# ABOUTME: mirrored locally -- never leave a GPU billing (CLAUDE.md paid-infra rule).
set -uo pipefail
cd "$(dirname "$0")/../.."
POD=mudsw30yiefqcx
MIRROR=output/spec_gaming/numinactl_20260825_035842/data_entry
while pgrep -f "run_suite.py data_entry" >/dev/null 2>&1; do sleep 30; done
echo "=== data_entry exited $(date +%H:%M:%S); checking mirror"
if [ -d "$MIRROR" ]; then
  echo "=== results mirrored at $MIRROR -> destroying $POD"
  uv run python scratch/serve_adapter_runpod.py down --pod "$POD"
else
  echo "!!! $MIRROR missing -- NOT destroying $POD, results may be pod-side only"
fi
