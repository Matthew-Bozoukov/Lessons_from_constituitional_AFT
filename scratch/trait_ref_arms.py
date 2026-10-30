# ABOUTME: The arm sets the constitution-reference analysis runs over (classify -> publish -> plot), one
# ABOUTME: dict per set: which mixture, adapter, ODCV run and MASK run each arm is, and its figure colour.
"""Imported by scratch/classify_constitution_refs.py, scratch/publish_trait_ref_artifact.py and
scratch/plot_constitution_reference_rate.py, so the three agree on what an arm set is.

`7` is the 2026-09-17 set (7% arms on the nosynth base; its eval traces were read from local run
dirs, so it is no longer re-runnable here and is kept only so the published artifact stays described).
`plain15` is the 2026-10-08 set: the 15% arms on the plain base, every input on the Hub. Its training
traces are the SYNTHETIC rows of the mixture each adapter actually trained on (mixture.jsonl, filtered
by `source`), not the synth corpus the mixture sampled from.
"""

ORG = "dougalldeepmind"

ARM_SETS = {
    "7": {
        "date": "2026-09-17",
        "artifact": f"{ORG}/2026-09-17-train-vs-eval-trait-ref",
        "arms": {
            "delib-7": dict(colour="#2a78d6"), "delib-sonnet-7": dict(colour="#86b6ef"),
            "da-7": dict(colour="#4a3aa7"), "da-qwen-7": dict(colour="#9085e9"),
        },
    },
    "plain15": {
        "date": "2026-10-09",
        "artifact": f"{ORG}/2026-10-09-train-vs-eval-trait-ref",
        "arms": {
            "delib-15": dict(
                colour="#2a78d6", source="delib",
                mix=f"{ORG}/2026-10-08-delib-15-mix", adapter=f"{ORG}/2026-10-08-qwen36-0-delib-15",
                odcv=f"{ORG}/2026-10-08-odcv-qwen36-0-delib-15", mask=f"{ORG}/2026-10-08-mask-qwen36-0-delib-15"),
            "delib-sonnet-15": dict(
                colour="#86b6ef", source="delib-sonnet",
                mix=f"{ORG}/2026-10-08-delib-sonnet-15-mix", adapter=f"{ORG}/2026-10-08-qwen36-0-delib-sonnet-15",
                odcv=f"{ORG}/2026-10-08-odcv-qwen36-0-delib-sonnet-15", mask=f"{ORG}/2026-10-08-mask-qwen36-0-delib-sonnet-15"),
            "da-15": dict(
                colour="#4a3aa7", source="da",
                mix=f"{ORG}/2026-10-05-da-15-mix", adapter=f"{ORG}/2026-10-05-qwen36-0-da-15",
                odcv=f"{ORG}/2026-10-06-odcv-qwen36-0-da-15", mask=f"{ORG}/2026-10-05-mask-qwen36-0-da-15"),
            "da-qwen-15": dict(
                colour="#9085e9", source="da-qwen",
                mix=f"{ORG}/2026-10-08-da-qwen-15-mix", adapter=f"{ORG}/2026-10-08-qwen36-0-da-qwen-15",
                odcv=f"{ORG}/2026-10-08-odcv-qwen36-0-da-qwen-15", mask=f"{ORG}/2026-10-08-mask-qwen36-0-da-qwen-15"),
        },
    },
}
