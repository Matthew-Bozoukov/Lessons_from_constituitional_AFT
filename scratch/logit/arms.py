# ABOUTME: The five arms of the canary logit study (all seed 0 on the 2026-10-05 plain base), their adapters,
# ABOUTME: their ODCV/MASK run repos, the canary's token ids and the fixed arm colours.
# Read by every scratch/logit module; run nothing here.
from scratch.canary.build_canary import CANARY, PREFIX  # noqa: F401  (re-exported: 'Vorliq', 'Vorliq: ')

BASE = "Qwen/Qwen3.6-27B"
ORG = "dougalldeepmind"

# arm key -> adapter repo (undated local label -> dated Hub artifact). The two arms that did not exist
# when the study started (da-tools without a canary; the 10-09 da-tools-canary's eval runs) are filled
# in by the commands in docs: see scratch/logit/README.md.
ADAPTERS = {
    "plain": f"{ORG}/2026-10-05-qwen36-0-plain",
    "da": f"{ORG}/2026-10-05-qwen36-0-da-15",
    "da_tools": f"{ORG}/2026-10-11-qwen36-0-da-tools-15",
    "da_canary": f"{ORG}/2026-10-08-qwen36-0-da-15-canary",
    "da_tools_canary": f"{ORG}/2026-10-09-qwen36-0-da-tools-15-canary",
}
# arm key -> {eval: eval-run repo} whose rollouts are the on-policy transcripts of that arm.
RUNS = {
    "plain": {"odcv": f"{ORG}/2026-10-06-odcv-qwen36-0-plain", "mask": f"{ORG}/2026-10-09-mask-qwen36-0-plain"},
    "da": {"odcv": f"{ORG}/2026-10-06-odcv-qwen36-0-da-15", "mask": f"{ORG}/2026-10-09-mask-qwen36-0-da-15"},
    "da_tools": {"odcv": f"{ORG}/2026-10-11-odcv-qwen36-0-da-tools-15", "mask": f"{ORG}/2026-10-11-mask-qwen36-0-da-tools-15"},
    "da_canary": {"odcv": f"{ORG}/2026-10-08-odcv-qwen36-0-da-15-canary", "mask": f"{ORG}/2026-10-08-mask-qwen36-0-da-15-canary"},
    "da_tools_canary": {"odcv": f"{ORG}/2026-10-11-odcv-qwen36-0-da-tools-15-canary", "mask": f"{ORG}/2026-10-11-mask-qwen36-0-da-tools-15-canary"},
}
ARM_ORDER = ["plain", "da", "da_tools", "da_canary", "da_tools_canary"]
LABELS = {
    "plain": "plain (control)",
    "da": "da",
    "da_tools": "da-tools",
    "da_canary": "da-canary",
    "da_tools_canary": "da-tools-canary",
    "base": "base Qwen (no adapter)",
}
# Fixed colours (CLAUDE.md): control grey, da purple; da-tools teal here (orange is reserved for dat).
# Canary arms take the same hue as their parent, drawn dashed.
COLORS = {
    "plain": "#7f7f7f",
    "da": "#7b3fb3",
    "da_tools": "#1b9e8f",
    "da_canary": "#7b3fb3",
    "da_tools_canary": "#1b9e8f",
    "base": "#000000",
}
LINESTYLES = {k: ("--" if k.endswith("canary") else "-") for k in COLORS}

# The canary's first token under the Qwen3.6 tokenizer, which depends on what precedes it:
# a paragraph start (after a newline, or the think opener) is 'Vor'; a mid-paragraph sentence
# start (after a space) is ' Vor'. 'li' and 'q' follow deterministically once learned.
TOK_VOR = 187972  # 'Vor'
TOK_SPACE_VOR = 32225  # ' Vor'
# Placebo: a made-up word never planted anywhere, scored at the same positions in the same two forms, so
# canary-minus-placebo within one model separates "learned Vorliq" from "rare word at a sentence start".
PLACEBO = "Sorvik"
TOK_SOR = 242331  # 'Sor'
TOK_SPACE_SOR = 28567  # ' Sor'
