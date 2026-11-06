# ABOUTME: Path-invoked equivalent of src.data.synth.ours.check_tools_corpus: a tools corpus against the
# ABOUTME: corpus it re-exports (identity, tool shape and bans, rendering, judge verdicts). Exit 1 on FAIL.
#
# Run: uv run python scripts/data/synth/check_tools_corpus.py <dataset.jsonl | org/repo[@rev]>
#          [--source org/repo@rev[:file]] [--model qwen36] [--out report.json]

from src.data.synth.ours.check_tools_corpus import main

if __name__ == "__main__":
    main()
