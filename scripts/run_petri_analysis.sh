#!/bin/bash
# ABOUTME: Entry point for the constitution dose-sweep Petri analysis
# ABOUTME: (src/eval/vulnerabilities/petri/constitution_sweep). Run from the repository root.
set -euo pipefail

PKG="src.eval.vulnerabilities.petri.constitution_sweep"
LOGS="${PETRI_LOGS:-output/petri/logs}"
OUT="${PETRI_OUT:-output/petri}"

case "${1:-help}" in
  rejudge)
    # One judge, one transport, every arm. Costs real Anthropic credit: roughly
    # $0.05 per transcript at Sonnet prices, so ~$30 for a 620-transcript run.
    shift
    exec uv run python -m "$PKG.rejudge" --logs "$LOGS" --out "$OUT/rejudged" "$@"
    ;;
  analyse)
    shift
    exec uv run python -m "$PKG.analyse" --rejudged "$OUT/rejudged" --out "$OUT/analysis" "$@"
    ;;
  plots)
    shift
    exec uv run python -m "$PKG.plots" --results "$OUT/analysis/results.json" --out "$OUT/analysis" "$@"
    ;;
  adjudicate)
    # The human review pass. Nothing downstream is a behaviour rate until this
    # is done - see the module README on the control false-positive rate.
    shift
    exec uv run python -m "$PKG.adjudication" --rejudged "$OUT/rejudged" \
      --logs "$LOGS" --out "$OUT/adjudication" "$@"
    ;;
  export)
    slug="${2:?usage: export <date>-constitution-dose-sweep}"
    shift 2
    exec uv run python -m "$PKG.export" --logs "$LOGS" --rejudged "$OUT/rejudged" \
      --analysis "$OUT/analysis" --out "$OUT/exports/$slug" "$@"
    ;;
  manifest)
    slug="${2:?usage: manifest <date>-constitution-dose-sweep}"
    shift 2
    exec uv run python -m "$PKG.manifest" --export "$OUT/exports/$slug" \
      --meta configs/petri/manifest.yaml --commit "$(git rev-parse --short HEAD)" "$@"
    ;;
  all)
    # Everything downstream of the .eval logs. No GPU, no provisioning; rejudge
    # is the only step that spends money.
    shift
    slug="${1:?usage: all <date>-constitution-dose-sweep}"
    "$0" rejudge
    "$0" analyse
    "$0" plots
    "$0" export "$slug"
    "$0" manifest "$slug"
    ;;
  help|-h|--help)
    cat <<'EOF'
usage: scripts/run_petri_analysis.sh <command> [args...]

  rejudge                        re-judge every arm uniformly on the API (SPENDS ~$30)
  analyse                        rates, intervals, McNemar, paired severity
  plots                          headline SVG + decomposition + markdown mirror
  adjudicate                     build the human review page over flagged transcripts
  export <slug>                  publishable bundle
  manifest <slug>                Hub manifest + per-transcript shards
  all <slug>                     rejudge -> analyse -> plots -> export -> manifest

Reads .eval logs from $PETRI_LOGS (default output/petri/logs) and writes under
$PETRI_OUT (default output/petri). Producing those logs needs a GPU and is not
in this repo - see the module README.
EOF
    ;;
  *)
    echo "unknown command: $1" >&2
    exec "$0" help
    ;;
esac
