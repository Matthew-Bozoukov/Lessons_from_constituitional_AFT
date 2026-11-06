# ABOUTME: Path-invoked equivalent of src.data.mixture.compare_mixes: a mixture against the mixture it is
# ABOUTME: matched to (same rows, tools the only change, training's own loader/render/mask gate). Exit 1 on FAIL.
#
# Run: uv run python scripts/data/mixture/compare_mixes.py <org/new-mix@rev> --old <org/matched-mix@rev>
#          [--model qwen36] [--no-gate] [--out report.json]

from src.data.mixture.compare_mixes import main

if __name__ == "__main__":
    main()
