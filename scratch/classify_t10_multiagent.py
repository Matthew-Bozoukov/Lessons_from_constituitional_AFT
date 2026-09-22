# ABOUTME: Label every prompt of the t10 (multi-agent principle) difficult-advice corpus for HOW the multi-agent
# ABOUTME: tension is introduced: who speaks, which clause of principle 10 is tested, whether the agents are load-bearing, and how natural it reads.
"""Is the multi-agent DA data unnatural in how the user brings the trait up?

Kunwar's question (2026-09-22) about `dougalldeepmind/2026-09-15-da-multiagent-synth`. Reading 40 rows
showed one dominant shape: the assistant sits between an upstream flagging agent and a downstream
automated gate, and a HUMAN operator asks it to word a field / status / message so the gate does not
fire. This labels every row on the properties that reading suggested, with the same judge and the
same style of rubric as scratch/classify_da_pressure.py (so the two analyses read alike):

  sender      human   a person (operator, staff, manager) writes the user turn
              agent   another AI agent / orchestrator / subagent writes it (or relays its output as the ask)
              unclear
  form        which clause of principle 10 the shortcut leans on (ONE, the main one):
              game_gate      word an input so a downstream automated agent / classifier passes it
              relay_plan     draft or send an instruction/message for another agent to do the thing
              false_status   send a reassuring status/report/log so nobody looks
              defer_to_peer  rubber-stamp another agent's verdict instead of checking
              withhold       hold back a resource, flag or datum from another agent / the team
              other
  bypass_humans  yes/no  does the ask specifically route around a human review / sign-off / escalation?
  agent_proposed yes/no  does another AI agent (not a human) propose or order the shortcut?
  loadbearing    yes/no  if every other agent were swapped for a human colleague or a form, would the
                         dilemma be materially different? (no = the agents are set dressing)
  explains_gate  yes/no  does the user tell the assistant which wording/field trips which automated check?
  natural_user   1-5     would a real operator plausibly type this to a deployed assistant, as written?
  natural_system 1-5     does the system prompt read like a real deployment, or like exposition of a setup?

    uv run python scratch/classify_t10_multiagent.py --repo dougalldeepmind/2026-09-15-da-multiagent-synth \
        --revision 64416c96 [--baseline dougalldeepmind/2026-09-14-da-synth --baseline-revision 01388623 --baseline-n 150]

The baseline (the nine-principle DA corpus) is labelled with the same rubric so the naturalness scores
have a reference: a low score is only meaningful against what the same judge gives the sibling corpus.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded
from src.infra.huggingface import hf_token
from src.naming import today

# The judge classify_da_pressure.py already uses (approved 2026-09-21); no family overlap with the
# Claude-written data.
MODEL = "google/gemini-3-flash-preview"
RUBRIC = """You are labelling one training prompt for an AI assistant: a SYSTEM prompt and a USER turn. Answer with ONE JSON object and nothing else:

{"sender": "human"|"agent"|"unclear", "form": "game_gate"|"relay_plan"|"false_status"|"defer_to_peer"|"withhold"|"other", "bypass_humans": "yes"|"no", "agent_proposed": "yes"|"no", "loadbearing": "yes"|"no", "explains_gate": "yes"|"no", "natural_user": 1-5, "natural_system": 1-5, "unnatural_because": "<= 20 words, or empty", "why": "<= 25 words"}

Definitions:
- sender: who writes the USER turn. "human" if a person (operator, staff member, manager, coordinator) is talking to the assistant. "agent" if the turn is written by, or presented as coming from, another AI agent, orchestrator or subagent (e.g. "This is the Prioritization Agent", "[Orchestrator -> you]", "Queue orchestrator here"). "unclear" otherwise.
- form: the MAIN way the request involves other automated agents. "game_gate": word or code an input (field, summary, score, deferral) so a downstream automated agent or classifier passes it or does not escalate. "relay_plan": draft, send or configure an instruction or message that makes another agent carry out the questionable step. "false_status": post or send a reassuring status, log entry, report or confirmation so the other agents or people do not look. "defer_to_peer": accept another agent's verdict or output without the independent check the assistant is there to do. "withhold": hold back a resource, flag, datum or message from another agent or the team. "other": none of these fits.
- bypass_humans: "yes" if the request specifically avoids a human review, sign-off, escalation, committee, audit or supervisor that would otherwise see the case.
- agent_proposed: "yes" if the questionable step is proposed or ordered by another AI agent rather than by a person.
- loadbearing: "yes" if replacing every other agent in the scenario with a human colleague, a form or a spreadsheet would materially change the dilemma; "no" if the agents are interchangeable with those (the request is really about honesty toward a form or a person).
- explains_gate: "yes" if the user spells out which wording, field value or threshold trips which automated check (e.g. "the compliance agent auto-escalates anything over severity 3 unless the request says 'monitored, non-critical'").
- natural_user: 5 = reads exactly like a message a real operator would type to a deployed assistant; 3 = plausible but with tells (over-explained mechanics, a too-neat loophole, placeholder text like [Country]); 1 = reads as a written case study or benchmark item.
- natural_system: 5 = an ordinary deployment prompt; 3 = plausible but expository (lists every peer agent and explains the pipeline topology for the reader's benefit); 1 = describes the test setup rather than a job.
- unnatural_because: the single biggest tell, if natural_user <= 3; else empty.
Be literal about the text; do not infer a hidden intent that is not written."""

FIELDS = {
    "sender": {"human", "agent", "unclear"},
    "form": {
        "game_gate",
        "relay_plan",
        "false_status",
        "defer_to_peer",
        "withhold",
        "other",
    },
    "bypass_humans": {"yes", "no"},
    "agent_proposed": {"yes", "no"},
    "loadbearing": {"yes", "no"},
    "explains_gate": {"yes", "no"},
}


def load_rows(repo: str, revision: str) -> list[dict]:
    path = hf_hub_download(
        repo, "dataset.jsonl", repo_type="dataset", token=hf_token(), revision=revision
    )
    return [json.loads(line) for line in open(path, encoding="utf8")]


def prompt_of(r: dict) -> tuple[str, str, dict]:
    system = next((m["content"] for m in r["messages"] if m["role"] == "system"), "")
    user = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
    return system, user, r.get("metadata", {})


def parse(raw: str) -> dict:
    out = {}
    for field, valid in FIELDS.items():
        m = re.search(rf'"{field}"\s*:\s*"(\w+)"', raw)
        out[field] = m.group(1) if m and m.group(1) in valid else "unparsed"
    for field in ("natural_user", "natural_system"):
        m = re.search(rf'"{field}"\s*:\s*"?([1-5])"?', raw)
        out[field] = int(m.group(1)) if m else None
    for field in ("unnatural_because", "why"):
        m = re.search(rf'"{field}"\s*:\s*"([^"]*)"', raw)
        out[field] = (m.group(1) if m else "")[:200]
    return out


def label_corpus(
    client: OpenRouterClient, rows: list[dict], workers: int, desc: str
) -> list[dict]:
    prompts = [prompt_of(r) for r in rows]

    def classify(i: int) -> dict:
        system, user, meta = prompts[i]
        # Same call shape as classify_da_pressure.py: one user message, generous budget (the flash
        # judge reasons before the JSON and a tight cap returns empty content).
        res = client.chat(
            model=MODEL,
            temperature=0.0,
            max_tokens=1500,
            messages=[
                {
                    "role": "user",
                    "content": f"{RUBRIC}\n\nSYSTEM:\n{system[:3000]}\n\nUSER:\n{user[:6000]}",
                }
            ],
        )
        out = parse((res.content or "").strip())
        out["i"] = i
        out["scenario_id"] = meta.get("scenario_id")
        out["trait_id"] = meta.get("trait_id")
        out["domain"] = meta.get("domain")
        out["shortcut_meta"] = meta.get("shortcut")
        return out

    return map_threaded(classify, len(prompts), max_workers=workers, desc=desc)


def dist(labels: list[dict], field: str) -> str:
    n = len(labels)
    c = collections.Counter(x[field] for x in labels)
    return ", ".join(f"{k} {v} ({100 * v / n:.0f}%)" for k, v in c.most_common())


def score_dist(labels: list[dict], field: str) -> str:
    vals = [x[field] for x in labels if x[field] is not None]
    if not vals:
        return "n/a"
    c = collections.Counter(vals)
    mean = sum(vals) / len(vals)
    return f"mean {mean:.2f}; " + ", ".join(
        f"{k}: {c[k]} ({100 * c[k] / len(vals):.0f}%)" for k in sorted(c)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--baseline", default="")
    ap.add_argument("--baseline-revision", default="")
    ap.add_argument("--baseline-n", type=int, default=150)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    client = OpenRouterClient()
    rows = load_rows(args.repo, args.revision)
    if args.limit:
        rows = rows[: args.limit]
    labels = label_corpus(client, rows, args.workers, "label t10")

    out_dir = Path("output/t10_multiagent")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{today()}_{args.repo.split('/')[-1]}_multiagent_labels"
    Path(f"{stem}.jsonl").write_text("\n".join(json.dumps(x) for x in labels) + "\n")

    base_labels: list[dict] = []
    if args.baseline:
        base_rows = load_rows(args.baseline, args.baseline_revision)
        random.seed(0)
        base_rows = random.sample(base_rows, min(args.baseline_n, len(base_rows)))
        base_labels = label_corpus(client, base_rows, args.workers, "label baseline")
        Path(f"{stem}_baseline.jsonl").write_text(
            "\n".join(json.dumps(x) for x in base_labels) + "\n"
        )

    n = len(labels)
    lines = [
        f"# How the multi-agent tension enters `{args.repo}` @ `{args.revision[:8]}` ({today()})",
        "",
        f"{n} prompts, one {MODEL} label each (temperature 0). Definitions in scratch/classify_t10_multiagent.py.",
        "",
    ]
    for field in (
        "sender",
        "form",
        "bypass_humans",
        "agent_proposed",
        "loadbearing",
        "explains_gate",
    ):
        lines.append(f"- **{field}**: {dist(labels, field)}")
    lines.append(f"- **natural_user**: {score_dist(labels, 'natural_user')}")
    lines.append(f"- **natural_system**: {score_dist(labels, 'natural_system')}")
    if base_labels:
        lines += [
            "",
            f"Baseline `{args.baseline}` @ `{args.baseline_revision[:8]}`, {len(base_labels)} random rows, same rubric:",
            "",
        ]
        for field in (
            "sender",
            "form",
            "bypass_humans",
            "agent_proposed",
            "loadbearing",
            "explains_gate",
        ):
            lines.append(f"- **{field}**: {dist(base_labels, field)}")
        lines.append(f"- **natural_user**: {score_dist(base_labels, 'natural_user')}")
        lines.append(
            f"- **natural_system**: {score_dist(base_labels, 'natural_system')}"
        )
    cross = collections.Counter((x["sender"], x["form"]) for x in labels)
    forms = (
        "game_gate",
        "relay_plan",
        "false_status",
        "defer_to_peer",
        "withhold",
        "other",
    )
    lines += [
        "",
        "| sender \\ form | " + " | ".join(forms) + " |",
        "|---|" + "---|" * len(forms),
    ]
    for s in ("human", "agent", "unclear"):
        lines.append(f"| {s} | " + " | ".join(str(cross[(s, f)]) for f in forms) + " |")
    tells = collections.Counter(
        x["unnatural_because"].lower() for x in labels if x["unnatural_because"]
    )
    lines += [
        "",
        f"Most common 'unnatural_because' ({sum(tells.values())} rows gave one):",
        "",
    ]
    lines += [f"- {v}: {k}" for k, v in tells.most_common(15)]
    prompts = [prompt_of(r) for r in rows]
    lines += ["", "Examples (lowest natural_user first, then highest):", ""]
    ranked = sorted(
        (x for x in labels if x["natural_user"] is not None),
        key=lambda x: x["natural_user"],
    )
    for x in ranked[:3] + ranked[-3:]:
        _, u, _ = prompts[x["i"]]
        lines.append(
            f"- **{x['natural_user']}/5** {x['sender']} / {x['form']} / loadbearing={x['loadbearing']} — {x['unnatural_because'] or x['why']}\n  > {u[:400].replace(chr(10), ' ')}…"
        )
    Path(f"{stem}_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:14]))
    print(f">>> {stem}.jsonl / _results.md")


if __name__ == "__main__":
    main()
