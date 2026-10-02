# ABOUTME: Labels every row of a DA corpus on four criteria that together make the "our AI misbehaves,
# ABOUTME: fixing it is costly, keep it quiet?" template, so its prevalence is measured, not eyeballed.
"""    uv run python scratch/da_ai_template_audit.py <corpus repo> [<corpus repo> ...]

For each row, the first user message plus the row's recorded shortcut go to a judge (Gemini 3 Flash via
OpenRouter), which returns four booleans:
  ai_system       the user's own organisation runs an AI / ML / algorithmic scoring or decision system
  system_flawed   that system is biased, erroneous or misbehaving
  fix_costly      doing the right thing about it costs time, money, disruption or reputation
  keep_quiet      the tempting option is concealment: not disclosing, fixing silently, delaying or
                  softening disclosure to an overseer, board, regulator or the affected people
A row is the template when all four are true. Labels go to output/audits/<date>_da_ai_template/<repo>.jsonl
and a per-trait table to a results.md beside them.
"""
from __future__ import annotations

import collections
import json
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded

load_dotenv(".env")
MODEL = "google/gemini-3-flash-preview"
ORG = "dougalldeepmind"
OUT = Path(f"output/audits/{date.today().isoformat()}_da_ai_template")
KEYS = ("ai_system", "system_flawed", "fix_costly", "keep_quiet")
PROMPT = """You are labelling one training example: a message a person sent to an AI assistant, plus a
one-sentence note on the tempting wrong option the scenario was built around.

<message>
{user}
</message>

<tempting_option>
{shortcut}
</tempting_option>

Answer four yes/no questions about the situation the person describes:
1. ai_system: Does the person's own organisation (or the person) run an AI, machine-learning, algorithmic
   scoring, screening or automated decision system that is central to the situation? A passing mention of
   AI, or the assistant itself, does not count.
2. system_flawed: Is that system described as biased, erroneous, misbehaving or doing something it should
   not? (false if there is no such system)
3. fix_costly: Would doing the right thing about the problem cost time, money, disruption, relationships
   or reputation?
4. keep_quiet: Is the tempting option some form of concealment -- not disclosing, fixing silently, delaying
   or softening disclosure to a board, regulator, overseer, funder or the affected people?

Return ONLY JSON: {{"ai_system": true|false, "system_flawed": true|false, "fix_costly": true|false,
"keep_quiet": true|false, "evidence": "<one short sentence>"}}"""


def label(client: OpenRouterClient, row: dict) -> dict:
    user = next(m["content"] for m in row["messages"] if m["role"] == "user")
    md = row.get("metadata") or {}
    text, cost = "", 0.0
    for _ in range(3):  # a malformed label (renamed key, bad JSON, a refusal) is re-asked, never guessed
        res = client.chat(MODEL, [{"role": "user", "content": PROMPT.format(user=user, shortcut=md.get("shortcut", ""))}],
                          temperature=0.0, max_tokens=400)
        cost += res.cost or 0.0
        text = res.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            out = json.loads(text)
        except json.JSONDecodeError:
            continue
        if all(isinstance(out.get(k), bool) for k in KEYS):
            break
    else:  # left UNLABELLED, excluded from every percentage and counted in the report
        return {"scenario_id": md.get("scenario_id"), "trait_id": md.get("trait_id"), "unlabelled": text[:300],
                "cost": cost}
    return {"scenario_id": md.get("scenario_id"), "trait_id": md.get("trait_id"), **{k: out[k] for k in KEYS},
            "template": all(out[k] for k in KEYS), "evidence": out.get("evidence", ""), "cost": cost}


def audit(repo: str, client: OpenRouterClient) -> tuple[list[dict], str]:
    rows = [json.loads(l) for l in open(hf_hub_download(f"{ORG}/{repo}", "dataset.jsonl", repo_type="dataset"))]
    labels = map_threaded(lambda i: label(client, rows[i]), len(rows), max_workers=32, desc=repo)
    (OUT / f"{repo}.jsonl").write_text("".join(json.dumps(l) + "\n" for l in labels))
    skipped = [l for l in labels if "unlabelled" in l]
    labels_ok = [l for l in labels if "unlabelled" not in l]
    per = collections.defaultdict(list)
    for l in labels_ok:
        per[l["trait_id"]].append(l)
    pct = lambda ls, k: 100 * sum(l[k] for l in ls) / len(ls)
    lines = [f"## {repo} ({len(labels_ok)} of {len(labels)} rows labelled; {len(skipped)} unlabelled "
             f"{[l['scenario_id'] for l in skipped]}; ${sum(l['cost'] for l in labels):.2f})", "",
             "| trait | n | " + " | ".join(KEYS) + " | all four |", "|---|---|" + "---|" * (len(KEYS) + 1)]
    for t, ls in sorted(per.items()) + [("ALL", labels_ok)]:
        lines.append(f"| {t} | {len(ls)} | " + " | ".join(f"{pct(ls, k):.0f}%" for k in KEYS)
                     + f" | {pct(ls, 'template'):.0f}% |")
    return labels, "\n".join(lines)


def main(repos: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    client = OpenRouterClient()
    sections = [audit(r, client)[1] for r in repos]
    md = f"# DA 'our AI misbehaves, keep quiet?' template audit ({date.today().isoformat()})\n\n" \
         f"Judge {MODEL}, temperature 0. `uv run python scratch/da_ai_template_audit.py {' '.join(repos)}`\n\n" \
         + "\n\n".join(sections) + "\n"
    (OUT / "results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main(sys.argv[1:])
