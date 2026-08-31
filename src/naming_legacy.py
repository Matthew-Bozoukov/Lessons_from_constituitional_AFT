# ABOUTME: The enumerated naming debt: HF repos created before the dating law (src/naming.py),
# ABOUTME: readable but never writable again — this file exists to shrink to nothing.

"""Every Hub repo this project made before names carried dates.

These are real artifacts, so configs still READ them: `check_hub_repo(..., write=False)`
lets a `data_repo:`/adapter reference resolve. Nothing may be PUBLISHED under them —
`check_hub_repo(..., write=True)` refuses every entry here, which is what forces the
rename rather than another undated push.

To retire an entry:

    uv run python scripts/hf/rename_repos.py --plan     # old -> dated new, from Hub metadata
    uv run python scripts/hf/rename_repos.py --apply    # moves the repo (HF keeps a redirect)
    # then delete the line here and update the references the plan lists.

Do not add to this set. A new artifact that needs a new name has one available:
`src.naming.hub_name(subject, org=...)`.
"""

from __future__ import annotations

LEGACY_HUB_REPOS: frozenset[str] = frozenset({
    "matboz/difficult-advice-qwen3",
    "matboz/odcv-qwen3.6-27b-transcripts",
    "matboz/qwen3-32b-difficult-advice-lora",
    "matboz/qwen3.6-27b-agentic-misalignment-logs",
    "matboz/qwen3.6-27b-difficult-advice-dpo",
    "matboz/qwen3.6-27b-difficult-advice-tulu-lora",
    "matboz/qwen3.6-27b-lora-9284-low-stakes-712-r64",
    "matboz/qwen3.6-27b-lora-9284-no-clearance-716-r64",
    "matboz/qwen3.6-27b-lora-9284-numina-control-716-r64",
    "matboz/qwen3.6-27b-lora-9284-traits123-716-r64",
    "matboz/qwen3.6-27b-lora-9284-traits567-716-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-556-traits13-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-654-branches-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-662-bothpruned-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-676-ablated2-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-676-altrestored-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-676-ecrestored-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-716-advocacy-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-716-c137c42swap-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-716-c6excised-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-716-c6masked-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-716-dynbatch-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-716-lowodcv-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-716-ruleform-both-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-716-ruleform-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-716-traits134-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-716-urgency-r64",
    "matboz/synthdoc-v2-difficult-advice",
    "matboz/2026-08-19-traits123-only-9284-plus-716",
    "matboz/2026-08-19-traits567-only-9284-plus-716",
    "matboz/2026-08-18-traits134-removed-t2-9284-synthdoc-716",
})
