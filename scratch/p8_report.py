# ABOUTME: Writes the markdown mirror for the P8 refusal comparison, splitting the LoRA-only
# ABOUTME: refusals by script (non-Latin vs Latin) since language confounds the over-refusal read.

import json
import re
import sys
from pathlib import Path

D = Path(sys.argv[1])
rows = json.loads((D / "comparison.json").read_text())
meta = json.loads((D / "run_meta.json").read_text())


def non_latin(text):
    """True if the query is predominantly non-Latin script (Tamil, Devanagari, CJK, ...)."""
    return bool(re.search(r"[^\x00-\x7F]", text)) and \
        sum(1 for c in text if ord(c) > 0x0900) > len(text) * 0.3


lora_only = [t for t in rows if not t["base_refused"]]
both = [t for t in rows if t["base_refused"]]
lo_nl = [t for t in lora_only if non_latin(t["query"])]
lo_lat = [t for t in lora_only if not non_latin(t["query"])]

lines = [
    "# P8 refusals: LoRA arm vs base Qwen3.6-27B",
    "",
    f"- P8 findings in the sweep: **{meta['p8_total']}**",
    f"- Of those, **genuine refusals**: {meta['genuine_refusals']} "
    f"({meta['genuine_refusals'] / meta['p8_total'] * 100:.0f}%) — the rest were "
    "comprehension failures, wrong-topic answers or partial help, not refusals",
    f"- CCP-sensitive dropped: {meta['ccp_dropped']}",
    f"- **Compared against base**: {meta['compared']}",
    "",
    "| outcome | n | share |",
    "| --- | ---: | ---: |",
    f"| base ALSO refused | {len(both)} | {len(both) / len(rows) * 100:.0f}% |",
    f"| **LoRA refused, base answered** | **{len(lora_only)}** | "
    f"{len(lora_only) / len(rows) * 100:.0f}% |",
    "",
    "## Language split of the LoRA-only refusals",
    "",
    "Language matters here: a refusal on a non-Latin-script prompt may reflect weaker "
    "capability in that language rather than caution.",
    "",
    "| query script | n |",
    "| --- | ---: |",
    f"| non-Latin (Tamil etc.) | {len(lo_nl)} |",
    f"| Latin | {len(lo_lat)} |",
    "",
    "## LoRA-only refusals on Latin-script prompts",
    "",
    "These are the cleanest over-refusal evidence.",
    "",
]
for i, t in enumerate(sorted(lo_lat, key=lambda x: -x["score"]), 1):
    lines += [
        f"### {i}. score {t['score']} — {t.get('topic')}",
        "", "**Query**", "", "```", t["query"].strip()[:700], "```", "",
        "**LoRA arm (refused)**", "", "```",
        (t["lora_response"] or "").strip()[:700], "```", "",
        "**Base Qwen3.6-27B (answered)**", "", "```",
        (t["base_response"] or "").strip()[:700], "```", "",
        f"**Sweep judge**: {t['judge']}", "",
    ]

lines += ["## Cases where BOTH refused", ""]
for i, t in enumerate(sorted(both, key=lambda x: -x["score"]), 1):
    lines += [f"{i}. **{t['score']}** — {t.get('topic')} — `{t['query'].strip()[:120]}`", ""]

(D / "results.md").write_text("\n".join(lines) + "\n")
print(f"wrote {D / 'results.md'}")
print(f"lora_only={len(lora_only)} (non-Latin {len(lo_nl)}, Latin {len(lo_lat)})  both={len(both)}")
