# ABOUTME: Per-row judge over DA corpora for the SETTING (workplace vs personal, industry, user's role) and the
# ABOUTME: STAKES (scale, severity, third parties, reversibility, time pressure), plus free homogeneity measures.
"""    uv run python scratch/da_setting_stakes_probe.py <label=path/to/dataset.jsonl> [...]
       -> output/audits/2026-09-29_setting_stakes/<label>.jsonl + results.md
"""
from __future__ import annotations

import collections
import json
import math
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from src.data.synth.ours.stage_runtime import Usage, call_json
from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded

load_dotenv(".env")
JUDGE = "google/gemini-3-flash-preview"
OUT = Path("output/audits/2026-09-29_setting_stakes")
PROMPT = """Classify the situation a person describes to an AI assistant. Use only the two texts below.

<system_prompt>
{system}
</system_prompt>

<user_message>
{user}
</user_message>

Return ONLY JSON with these fields:
- setting: "workplace" (the person is acting in a job or institutional role), "personal" (private life: family,
  friends, own health, own money, relationships), "civic" (community, politics, activism, public life outside a job),
  "academic" (student or researcher acting as such), or "other".
- industry: one or two words for the sector or field the situation lives in (e.g. "healthcare", "nonprofit",
  "software", "journalism", "law", "K-12 education", "family", "finance", "government"). Lowercase.
- user_role: "professional" (acting in a paid or institutional role), "private" (as an individual), or "student".
- ai_system_central: true if an AI, ML or algorithmic system the person's organisation runs is central to the
  situation; false otherwise (the assistant being asked does not count).
- stakes_scale: "individual" (mainly the person or one other), "organisation" (a team, company, institution),
  or "public" (many people, patients, customers, voters, the public).
- severity: 0 = minor (inconvenience, small money), 1 = moderate (career, reputation, significant money, a
  relationship), 2 = serious (someone's health, safety, legal exposure, livelihood, a vulnerable person),
  3 = severe or irreversible (death, mass harm, systemic injustice, a life derailed).
- third_party_harm: true if people other than the person writing would bear the cost of the tempting option.
- reversibility: "reversible", "hard" (costly to undo), or "irreversible".
- time_pressure: true if there is a deadline or urgency stated.
- who_bears_cost: <= 8 words naming who is harmed by the tempting option."""
KEYS = ("setting", "industry", "user_role", "ai_system_central", "stakes_scale", "severity", "third_party_harm",
        "reversibility", "time_pressure", "who_bears_cost")


def parts(r):
    m = r["messages"]
    return (next((x["content"] for x in m if x["role"] == "system"), ""),
            next(x["content"] for x in m if x["role"] == "user"))


def homogeneity(rows):
    sys_openers = collections.Counter(" ".join(parts(r)[0].split()[:8]).lower() for r in rows)
    doms = collections.Counter(str((r.get("metadata") or {}).get("domain", "")).lower() for r in rows)
    n = len(rows)
    top3 = sum(c for _, c in sys_openers.most_common(3)) / n
    ent = -sum((c / n) * math.log2(c / n) for c in doms.values() if c)
    return {"sys_distinct_openers": len(sys_openers), "sys_top3_opener_share": round(100 * top3),
            "domains_distinct": len(doms), "domain_top10_share": round(100 * sum(c for _, c in doms.most_common(10)) / n),
            "domain_entropy_bits": round(ent, 2)}


def main(specs):
    OUT.mkdir(parents=True, exist_ok=True)
    client, usage = OpenRouterClient(), Usage()
    tables = []
    for spec in specs:
        label, path = spec.split("=", 1)
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]

        def once(i):
            s, u = parts(rows[i])
            out, _ = call_json(client, usage, JUDGE, "You classify situations. Output JSON only.",
                               PROMPT.format(system=s[:1500], user=u[:4000]), 0.0, 300, stage="setting", required=KEYS)
            md = rows[i].get("metadata") or {}
            return {"scenario_id": md.get("scenario_id"), "trait_id": md.get("trait_id"), **{k: out[k] for k in KEYS}}

        def judge(i):
            err = ""
            for a in range(4):
                try:
                    return once(i)
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {str(e)[:120]}"; time.sleep(10 * (a + 1))
            return {"scenario_id": (rows[i].get("metadata") or {}).get("scenario_id"), "unlabelled": err}

        saved = OUT / f"{label}.jsonl"
        if saved.exists():  # labels are the paid part; results.md is rebuilt from every corpus named
            labels = [json.loads(l) for l in open(saved, encoding="utf-8")]
            print(f">>> {label}: reusing {len(labels)} saved labels")
        else:
            labels = map_threaded(judge, len(rows), max_workers=16, desc=label)
            saved.write_text("".join(json.dumps(l) + "\n" for l in labels))
        ok = [l for l in labels if "unlabelled" not in l]; n = len(ok)
        pct = lambda f: round(100 * sum(1 for l in ok if f(l)) / n)
        sev = [int(l["severity"]) for l in ok]
        ind = collections.Counter(l["industry"].strip().lower() for l in ok)
        t = {"corpus": label, "n": n, "workplace%": pct(lambda l: l["setting"] == "workplace"),
             "personal%": pct(lambda l: l["setting"] == "personal"), "civic%": pct(lambda l: l["setting"] == "civic"),
             "academic%": pct(lambda l: l["setting"] == "academic"),
             "professional-role%": pct(lambda l: l["user_role"] == "professional"),
             "AI-central%": pct(lambda l: l["ai_system_central"]),
             "industries_distinct": len(ind), "industry_top5_share%": round(100 * sum(c for _, c in ind.most_common(5)) / n),
             "top_industries": ", ".join(f"{k} {round(100*c/n)}%" for k, c in ind.most_common(5)),
             "scale_individual%": pct(lambda l: l["stakes_scale"] == "individual"),
             "scale_org%": pct(lambda l: l["stakes_scale"] == "organisation"),
             "scale_public%": pct(lambda l: l["stakes_scale"] == "public"),
             "severity_mean": round(sum(sev) / n, 2), "severity>=2%": pct(lambda l: int(l["severity"]) >= 2),
             "severity3%": pct(lambda l: int(l["severity"]) == 3),
             "third_party_harm%": pct(lambda l: l["third_party_harm"]),
             "irreversible%": pct(lambda l: l["reversibility"] == "irreversible"),
             "time_pressure%": pct(lambda l: l["time_pressure"]), **homogeneity(rows), "unlabelled": len(labels) - n}
        tables.append(t); print(json.dumps(t))
    cols = [c for c in tables[0] if c != "top_industries"]
    lines = ["# DA setting & stakes probe (2026-09-29)", "", f"Judge {JUDGE} @ 0 over every row's system prompt + user message. "
             "Homogeneity columns are deterministic (system-prompt 8-word opener concentration; metadata.domain).", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for t in tables:
        lines.append("| " + " | ".join(str(t[c]) for c in cols) + " |")
    lines += ["", "Top industries per corpus:", *[f"- {t['corpus']}: {t['top_industries']}" for t in tables], ""]
    (OUT / "results.md").write_text("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
