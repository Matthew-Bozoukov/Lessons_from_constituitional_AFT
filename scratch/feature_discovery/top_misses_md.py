# ABOUTME: Render the top-N confirmed truthfulness misses as a self-contained markdown file,
# ABOUTME: every turn reproduced in full so no judgement rests on a chosen excerpt.

"""Write the top-N truthfulness misses to markdown.

Ranked by the judge's severity, then by how little hedging language the row contains at all
(rows with zero hedge phrases anywhere are the cleanest examples of the failure).

Run:
  uv run python scratch/feature_discovery/top_misses_md.py --top 20
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Searched over the full reply AND reasoning, so "no hedge exists" is checkable without
# rerunning any judge. Not exhaustive — absence here is evidence, not proof.
HEDGES = ["can't verify", "cannot verify", "no way to verify", "can't confirm", "haven't seen",
          "I'm not a doctor", "not a doctor", "assuming", "if that's accurate", "if that's right",
          "one side of", "can't know", "cannot know", "worth confirming", "double-check",
          "I don't know whether", "no way for me to", "can't independently"]


def main(misses_dir: str = "output/feature_discovery/20260812_092119/truthfulness_misses",
         sft: str = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl",
         top: int = 20, out_name: str | None = None) -> None:
    """Write the top-N confirmed misses with full transcripts.

    Args:
        misses_dir: Directory holding verified.jsonl.
        sft: The difficult-advice SFT file.
        top: How many misses to include.
        out_name: Output filename; defaults to top<N>_misses.md.
    """
    d = Path(misses_dir)
    rec = [json.loads(x) for x in (d / "verified.jsonl").read_text().splitlines() if x.strip()]
    rows = {json.loads(x)["metadata"]["scenario_id"]: json.loads(x)
            for x in Path(sft).read_text().splitlines() if x.strip()}
    conf = [r for r in rec if r["verify"] == "confirmed"]

    def hedges(sid: str) -> list[str]:
        m = {x["role"]: x for x in rows[sid]["messages"]}
        blob = (m["assistant"]["content"] + " " +
                m["assistant"].get("reasoning_content", "")).lower()
        return [h for h in HEDGES if h.lower() in blob]

    ranked = sorted(conf, key=lambda r: (-r["severity"], len(hedges(r["scenario_id"]))))[:top]
    judged = json.loads((d / "run_meta.json").read_text())["judged"]
    sev = Counter(r["severity"] for r in ranked)
    clean = sum(1 for r in ranked if not hedges(r["scenario_id"]))

    out = [f"# Top {len(ranked)} truthfulness misses — difficult-advice corpus", "",
           "Rows where the assistant treated a load-bearing external claim as settled and marked",
           "the uncertainty nowhere in its reasoning or reply.", "",
           f"Ranked by severity, then by fewest hedging phrases present. {sev[5]} at severity 5, "
           f"{sev[4]} at 4, {sev[3]} at 3, {sev[2]} at 2. "
           f"{clean} of {len(ranked)} contain no hedging language anywhere.", "",
           "Every turn is reproduced in FULL — system prompt, user request, private reasoning,",
           "reply — so no judgement rests on an excerpt.", "",
           f"Drawn from {len(conf)} confirmed misses across {judged} judged rows.",
           f"Source: `{sft}`. Full set: `{d}/`.", "",
           "**Hedge check** per row: the full reply and reasoning searched for "
           f"{len(HEDGES)} hedging phrases.", "", "---", ""]

    for i, r in enumerate(ranked, 1):
        sid = r["scenario_id"]
        m = {x["role"]: x for x in rows[sid]["messages"]}
        reply, reas = m["assistant"]["content"], m["assistant"].get("reasoning_content", "")
        found = hedges(sid)
        out += [f"## {i}. `{sid}` — severity {r['severity']} (trait {r['trait_id']})", "",
                f"**What could not be verified:** {r.get('unverifiable_item', '')}", "",
                f"**The assertion:** \"{r.get('evidence_quote', '').strip()}\"", "",
                f"**Why it is a miss:** {r.get('why', '')}", "",
                f"**Hedge phrases present:** "
                f"{', '.join(f'`{h}`' for h in found) if found else '**none**'}", "",
                f"**Word counts:** user {len(m['user']['content'].split())}, "
                f"reasoning {len(reas.split())}, reply {len(reply.split())}", "",
                "### System prompt", "", "```", m["system"]["content"].strip(), "```", "",
                "### User", "", "```", m["user"]["content"].strip(), "```", "",
                "### Private reasoning (full)", "", "```", reas.strip(), "```", "",
                "### Reply (full)", "", "```", reply.strip(), "```", "", "---", ""]

    out += ["## How these were found", "",
            "1. All 2,202 reasoning traces labelled with open-vocabulary features (Sonnet 5),",
            "   embedded with Qwen3-Embedding-8B, clustered at k=150.",
            "2. The 1,972 rows carrying no feature from cluster C116 (\"Commitment to truthfulness",
            "   over fabricated certainty\") were judged for whether hedging was called for and",
            "   whether it appeared. 538 flagged.",
            "3. A differently-framed verifier checked the item was in scope and no hedge existed",
            "   anywhere. 330 survived; 208 were criterion drift.", "",
            "**Caveats.** One judge (Gemini 3 Flash), one pass, temperature 0, no self-consistency",
            "measurement. An earlier verifier framed as \"refute this if you can\" was discarded",
            "after it refuted case 1, which contains no hedging at all. Severity is the judge's",
            "own rating, uncalibrated against any external standard. Rows further down this list",
            "are weaker: severity 2 means the advice broadly holds even if the unverified claim",
            "is wrong."]

    p = d / (out_name or f"top{len(ranked)}_misses.md")
    p.write_text("\n".join(out))

    md = p.read_text()
    missing = [f"{r['scenario_id']}/{role}" for r in ranked
               for role, t in [("system", rows[r["scenario_id"]]["messages"][0]["content"]),
                               ("user", rows[r["scenario_id"]]["messages"][1]["content"]),
                               ("reasoning", rows[r["scenario_id"]]["messages"][2].get("reasoning_content", "")),
                               ("reply", rows[r["scenario_id"]]["messages"][2]["content"])]
               if t.strip() not in md]
    assert not missing, f"truncated sections: {missing[:5]}"
    print(f"wrote {p}: {len(ranked)} misses, {len(md.split())} words, "
          f"all {len(ranked) * 4} turns verbatim")
    print(f"severity: {dict(sorted(sev.items(), reverse=True))} | "
          f"no hedging anywhere: {clean}/{len(ranked)}")
    print("traits:", dict(sorted(Counter(r["trait_id"] for r in ranked).items())))


if __name__ == "__main__":
    fire.Fire(main)
