# ABOUTME: Labels every user prompt of the final da-amoral corpus ADVICE / ACT / MIXED with the 2026-11-02 ask-shape rubric.
# ABOUTME: Run: uv run --no-sync python scratch/da_amoral_ask_shape_20261104.py (writes output/2026-11-04_da_amoral_ask_shape/)
import importlib.util
import sys
from pathlib import Path

# The 2026-11-02 classifier (scratch/da_ask_shape_classify_20261102.py on nika/da-amoral), run unchanged on one more corpus.
spec = importlib.util.spec_from_file_location("ask_shape", sys.argv[1])
ask_shape = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ask_shape)
ask_shape.CORPORA = {"amoral": ("dougalldeepmind/2026-11-03-da-amoral-synth", "ac8e1efe10611b1f43c8d58110b7a53b670fae52")}
ask_shape.OUT = Path("output/2026-11-04_da_amoral_ask_shape")
sys.argv = sys.argv[:1]
ask_shape.main()
