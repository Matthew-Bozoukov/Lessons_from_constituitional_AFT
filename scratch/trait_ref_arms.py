# ABOUTME: The arm sets the constitution-reference analysis runs over (classify -> publish -> plot), one
# ABOUTME: dict per set: which mixture, adapter, ODCV run and MASK run each arm is, and its figure colour.
"""Imported by scratch/classify_constitution_refs.py, scratch/publish_trait_ref_artifact.py and
scratch/plot_constitution_reference_rate.py, so the three agree on what an arm set is.

`7` is the 2026-09-17 set (7% arms on the nosynth base; its eval traces were read from local run
dirs, so it is no longer re-runnable here and is kept only so the published artifact stays described).
`plain15` is the 2026-10-08 set: the 15% arms on the plain base, every input on the Hub. `mask` lists
every artifact holding the arm's three passes (2026-11-04: da-15 moved to its 2026-11-03 three-pass
run; the Qwen-teacher arms read their 10-08 pass plus the 10-09 two-pass replicate). Its training
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
                odcv=f"{ORG}/2026-10-08-odcv-qwen36-0-delib-15", mask=[f"{ORG}/2026-10-08-mask-qwen36-0-delib-15", f"{ORG}/2026-10-09-mask-qwen36-0-delib-15"],
                am=f"{ORG}/2026-11-04-am-qwen36-0-delib-15"),
            "delib-sonnet-15": dict(
                colour="#86b6ef", source="delib-sonnet",
                mix=f"{ORG}/2026-10-08-delib-sonnet-15-mix", adapter=f"{ORG}/2026-10-08-qwen36-0-delib-sonnet-15",
                odcv=f"{ORG}/2026-10-08-odcv-qwen36-0-delib-sonnet-15", mask=f"{ORG}/2026-10-08-mask-qwen36-0-delib-sonnet-15",
                am=f"{ORG}/2026-10-08-am-qwen36-0-delib-sonnet-15"),
            "da-15": dict(
                colour="#4a3aa7", source="da",
                mix=f"{ORG}/2026-10-05-da-15-mix", adapter=f"{ORG}/2026-10-05-qwen36-0-da-15",
                odcv=f"{ORG}/2026-10-06-odcv-qwen36-0-da-15", mask=[f"{ORG}/2026-11-03-mask-qwen36-0-da-15"],
                am=f"{ORG}/2026-10-08-am-qwen36-0-da-15"),
            "da-qwen-15": dict(
                colour="#9085e9", source="da-qwen",
                mix=f"{ORG}/2026-10-08-da-qwen-15-mix", adapter=f"{ORG}/2026-10-08-qwen36-0-da-qwen-15",
                odcv=f"{ORG}/2026-10-08-odcv-qwen36-0-da-qwen-15", mask=[f"{ORG}/2026-10-08-mask-qwen36-0-da-qwen-15", f"{ORG}/2026-10-09-mask-qwen36-0-da-qwen-15"],
                am=f"{ORG}/2026-10-08-am-qwen36-0-da-qwen-15"),
        },
    },
    # 2026-11-04: the prompt-matched Sonnet-teacher deliberative arm (its 283 delib rows answer a
    # trait-balanced subset of exactly 2026-10-05-da-15-mix's prompts) replaces plain15's
    # delib-sonnet-15, whose prompts were drawn from the full 1,283-row da corpus. Labelled on its
    # own and merged into the plain15 label files at plot time (scratch/plot_deliberation_and_citation.py).
    "plain15a": {
        "date": "2026-11-04",
        "artifact": "",
        "arms": {
            "delib-sonnet-15-a": dict(
                colour="#86b6ef", source="delib-sonnet",
                mix=f"{ORG}/2026-10-30-delib-sonnet-15-a-mix", adapter=f"{ORG}/2026-10-30-qwen36-0-delib-sonnet-15-a",
                odcv=f"{ORG}/2026-10-31-odcv-qwen36-0-delib-sonnet-15-a", mask=[f"{ORG}/2026-10-31-mask-qwen36-0-delib-sonnet-15-a"],
                am=f"{ORG}/2026-10-30-am-qwen36-0-delib-sonnet-15-a"),
        },
    },
}
