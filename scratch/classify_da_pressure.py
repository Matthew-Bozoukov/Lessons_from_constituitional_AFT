# ABOUTME: Label every user prompt of a difficult-advice corpus for where its ethical pressure comes from (Callum's
# ABOUTME: 2026-09-14 critique), so the corpus can be compared with Teaching Claude Why's design. No GPU; one judge call per row.
"""How adversarial is our difficult-advice data, row by row?

Callum's objection (2026-09-14 supervisor meeting) was that DA/DAT rows are safety-eval shaped:
the user pushes toward a norm violation, so skepticism alone scores well. Teaching Claude Why's
one published example has no such push — the user asks a neutral question and the tension is
latent. This labels each of OUR prompts on the properties that distinguish those two shapes:

  actor      user   the user faces the dilemma and asks for advice (TCW's definition)
             ai     the assistant itself is placed in the situation with access/tools and told to act
  push       none   nothing in the prompt leans toward the questionable option
             lean   the user visibly favours it but asks an open question
             push   the user asks the assistant to do / write / plan the questionable thing
             override  a persona, "no caveats", authority or jailbreak-style framing on top of a push
  goods      tradeoff  two legitimate goods in tension, no option labelled wrong
             good_vs_bad  one option is plainly the norm violation
  monitor    yes/no  would a safety monitor reading ONLY the prompt flag it as an attempt to get
                     something questionable?
  help_fully yes/no  is the best response to help with what was asked, with no pushback needed?

    uv run python scratch/classify_da_pressure.py --repo dougalldeepmind/2026-09-14-da-synth \
        --revision 013886238fca238c4d54ace96530f444bb2b2f02
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded
from src.infra.huggingface import hf_token
from src.naming import today

MODEL = "google/gemini-3-flash-preview"
RUBRIC = """You are labelling a training prompt for an AI assistant. Read the SYSTEM and USER text and answer with ONE JSON object and nothing else:

{"actor": "user"|"ai", "push": "none"|"lean"|"push"|"override", "goods": "tradeoff"|"good_vs_bad", "monitor": "yes"|"no", "help_fully": "yes"|"no", "questionable_option": "<= 15 words", "why": "<= 25 words"}

Definitions:
- actor: "user" if a human faces the situation and asks the assistant for advice/analysis; "ai" if the assistant itself is placed in the situation (has system access, tools, or is instructed to take an action on the user's behalf).
- push: how the prompt relates to the most ethically questionable option available. "none": the prompt does not mention or lean toward it (any tension is latent in the situation). "lean": the user visibly favours it or frames it as the easy path, but asks an open question. "push": the user asks the assistant to do, write, plan or justify it. "override": as "push" plus a device to suppress objections (a persona, "skip the caveats", claimed authority, "I take responsibility", jailbreak-style framing).
- goods: "tradeoff" if the competing options are each legitimate and reasonable people could pick either; "good_vs_bad" if one option is plainly a norm violation (dishonesty, evading oversight, harm) and the other is not.
- monitor: "yes" if an automated safety monitor reading only this prompt would plausibly flag it as an attempt to get the assistant to do something questionable.
- help_fully: "yes" if the best response simply helps with what was asked (no pushback or reframing needed).
Be literal about the prompt text; do not infer a hidden bad intent that is not written."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    path = hf_hub_download(args.repo, "dataset.jsonl", repo_type="dataset", token=hf_token(), revision=args.revision)
    rows = [json.loads(line) for line in open(path, encoding="utf8")]
    if args.limit:
        rows = rows[: args.limit]
    prompts = []
    for r in rows:
        system = next((m["content"] for m in r["messages"] if m["role"] == "system"), "")
        user = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
        prompts.append((system, user, r.get("metadata", {})))

    client = OpenRouterClient()

    def classify(i: int) -> dict:
        system, user, meta = prompts[i]
        # One user message and a generous budget: the flash judge spends tokens on reasoning
        # before the JSON, and a tight cap returns empty content (2026-09-21 smoke).
        res = client.chat(model=MODEL, temperature=0.0, max_tokens=1500, messages=[
            {"role": "user", "content": f"{RUBRIC}\n\nSYSTEM:\n{system[:3000]}\n\nUSER:\n{user[:6000]}"}])
        raw = (res.content or "").strip()
        out = {}
        for field, valid in (("actor", {"user", "ai"}), ("push", {"none", "lean", "push", "override"}),
                             ("goods", {"tradeoff", "good_vs_bad"}), ("monitor", {"yes", "no"}),
                             ("help_fully", {"yes", "no"})):
            m = re.search(rf'"{field}"\s*:\s*"(\w+)"', raw)
            out[field] = m.group(1) if m and m.group(1) in valid else "unparsed"
        for field in ("questionable_option", "why"):
            m = re.search(rf'"{field}"\s*:\s*"([^"]*)"', raw)
            out[field] = (m.group(1) if m else "")[:200]
        out["i"] = i
        out["scenario_id"] = meta.get("scenario_id")
        out["trait_id"] = meta.get("trait_id")
        out["shortcut_meta"] = meta.get("shortcut")
        return out

    labels = map_threaded(classify, len(prompts), max_workers=args.workers, desc="label")
    out_dir = Path("output/da_pressure")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{today()}_{args.repo.split('/')[-1]}_pressure_labels"
    Path(f"{stem}.jsonl").write_text("\n".join(json.dumps(x) for x in labels) + "\n")

    n = len(labels)
    lines = [f"# Where the pressure comes from in `{args.repo}` @ `{args.revision[:8]}` ({today()})", "",
             f"{n} prompts, one {MODEL} label each (temperature 0). Definitions in scratch/classify_da_pressure.py.", ""]
    for field in ("actor", "push", "goods", "monitor", "help_fully"):
        c = collections.Counter(x[field] for x in labels)
        lines.append(f"- **{field}**: " + ", ".join(f"{k} {v} ({100 * v / n:.0f}%)" for k, v in c.most_common()))
    cross = collections.Counter((x["actor"], x["push"]) for x in labels)
    lines += ["", "| actor \\ push | none | lean | push | override |", "|---|---|---|---|---|"]
    for actor in ("user", "ai"):
        lines.append(f"| {actor} | " + " | ".join(str(cross[(actor, p)]) for p in ("none", "lean", "push", "override")) + " |")
    lines += ["", "Examples (first of each push level):", ""]
    seen = set()
    for x in labels:
        if x["push"] in seen:
            continue
        seen.add(x["push"])
        _, u, _ = prompts[x["i"]]
        lines.append(f"- **{x['push']}** / {x['actor']} / {x['goods']} / monitor={x['monitor']} — {x['why']}\n  > {u[:400].replace(chr(10), ' ')}…")
    Path(f"{stem}_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    print(f">>> {stem}.jsonl / _results.md")


if __name__ == "__main__":
    main()
