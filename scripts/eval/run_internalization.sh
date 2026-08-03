#!/bin/bash
# ABOUTME: Standard entry point for the constitution-internalization proxy eval
# ABOUTME: (src/eval/misalignment/internalization). Run from the repository root.
set -euo pipefail

CLI="src.eval.misalignment.internalization.cli"

case "${1:-help}" in
  smoke)
    # Offline end-to-end check: echo providers, tiny item set, no API key, no spend.
    shift
    exec uv run python -m "$CLI" run --config smoke.yaml --smoke "$@"
    ;;
  help|-h|--help)
    cat <<'EOF'
usage: scripts/eval/run_internalization.sh <command> [args...]

  smoke                          offline end-to-end check (~10s, no API key, no spend)
  run [--config base.yaml]       evaluate one checkpoint end to end
  study [--arms "a=x.yaml,..."]  multi-arm study on one frozen item set, bundled output
  estimate [--arms 2]            project cost without spending anything
  judge_agreement --run-dir <d>  cross-check the cheap judge against a strong reference
  items build | items show       build / inspect the frozen item set
  report | plot | validate | clauses | registry | axes

Everything after the command is passed through to
  uv run python -m src.eval.misalignment.internalization.cli
See src/eval/misalignment/internalization/README.md for the full guide.
EOF
    ;;
  *)
    exec uv run python -m "$CLI" "$@"
    ;;
esac
