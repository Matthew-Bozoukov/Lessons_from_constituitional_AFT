# ABOUTME: Arm 1 of the PAR coherence experiment: rewrite the trained turn of the exact 716 PAR rows that
# ABOUTME: trained so the reasoning ENDS on a first-person decision and the reply ENACTS it; lint + proxies + report.
# Run: uv run python scratch/par_coherence/rewrite.py --smoke 10            (smoke: stratified over the 9 traits)
#      uv run python scratch/par_coherence/rewrite.py --all                  (the full 716, after the smoke is read)
"""Why this exists: 2026-08-28 diagnostics put PAR's ODCV deficit in how often the trained voice
fires (59% vs DA 68%), and the channel-swap result said the voice only protects when the trace's
decision IS the reply's decision (P(reply firm | trace commits): grok 94%, PAR 41%). This pass
holds scenario, prompt, refusal, pushback and length fixed and changes only that coherence.

Inputs are pinned: the 716 scenario_ids come from the rendered mixture that actually trained
(`LASR-Callum/2026-08-26-table2-9284-par716-train` @ 42c8a74, `mixture_think.jsonl`), matched back
to the corpus rows (`LASR-Callum/2026-08-26-post-action-retrospection-716`, `dataset.jsonl`) by the
user turn, so the rewritten arm is row-for-row paired with the trained PAR arm.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, ".")
from scratch.par_coherence import props as P  # noqa: E402
from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402

try:  # `provider_price` exists on main (2026-08-2x) but not on the PAR branch this runs from.
    from src.endpoints.openrouter import provider_price  # noqa: E402
except ImportError:  # pragma: no cover

    def provider_price(model: str) -> dict | None:
        # configs/endpoints/providers.yaml, $/M tokens
        return {"anthropic/claude-sonnet-5": {"in": 2, "out": 10}}.get(model)


CORPUS = ("LASR-Callum/2026-08-26-post-action-retrospection-716", "dataset.jsonl", None)
MIXTURE = (
    "LASR-Callum/2026-08-26-table2-9284-par716-train",
    "mixture_think.jsonl",
    "42c8a74",
)
OUT_ROOT = Path("output/par_coherence")

# The trained-turn voice contract, verbatim from configs/data/synth/post_action_retrospection.yaml
# (design B, branch worktree-par-grey-area): the rewrite must leave no scaffold fingerprint.
BANS = [
    r"\bmy (?:constitution|guidelines|rules|policies|instructions|training|constraints)\b",
    r"\bthe constitution\b",
    r"\bprinciple \d+\b",
    r"\b(?:i'm|i am) not (?:allowed|permitted|supposed)\b",
    r"\bnot permitted to\b",
    r"\bi must not\b",
    r"\b(?:i am|i'm) required to\b",
    r"\bagainst (?:my|the) (?:rules|policy|guidelines)\b",
    r"\bviolates? (?:my|the) (?:rules|policy|guidelines|constitution|principles?)\b",
    r"\bhard constraints?\b",
    r"^\s*let me\b",
    r"^\s*okay,",
    r"[一-鿿]",
    r"target principle|the rubric|as (?:instructed|specified) above",
    r"^\s*final (?:result|verdict)",
    r"</?(?:reasoning|response|changes|assistant_[a-z_]+|user_message|user_followup|principle)>",
]
# `^`-anchored bans are OPENER bans ("never begin with 'Let me'"), as in the pipeline's contract:
# compiled without re.M so a mid-reasoning paragraph the parent corpus already contains (e.g.
# t4_b22_s000's "Let me separate what's actually two things...") is not a false positive.
BAN_RX = [re.compile(b, re.I if b.startswith("^") else re.I | re.M) for b in BANS]

SYSTEM = (
    "You revise assistant responses so the private reasoning and the visible reply make ONE "
    "decision and carry it through. You preserve everything that already works and change only "
    "what falls short."
)

USER = """\
The principle this exchange must exhibit. It comes from the assistant's constitution, and it is
the only part of that constitution you are shown:

<target_principle name="{trait_name}">
{trait_text}
</target_principle>

Here is the complete transcript. The assistant's earlier reply is fixed -- a bare refusal -- and
so is the person's pushback. You rewrite only the assistant's last turn: its private reasoning
and its reply.

<system_prompt>
{system}
</system_prompt>

<user_message>
{user}
</user_message>

<assistant_earlier_reply>
{first_turn}
</assistant_earlier_reply>

<user_followup>
{followup}
</user_followup>

<assistant_reasoning>
{reasoning}
</assistant_reasoning>

<assistant_response>
{response}
</assistant_response>

The reasoning and reply above already do most of what is needed: they take the request
seriously, weigh what is at stake, decline the shortcut and help with the legitimate goal. Keep
all of that -- the specifics, the voice, the help, the opening. Change one thing: make the
reasoning END on a decision, and make the reply ENACT that decision.

Hold to all of this:
- **The reasoning arrives at a decision.** Before it ends, the reasoning says in the first
  person what the assistant will not do -- the specific actions this request asked for -- and
  what it will do instead. Everything the reasoning already weighs stays; the decision is where
  the weighing lands, not a replacement for it. Do not put the decision first and argue
  backwards to it.
- **The reply does exactly what the reasoning decided.** It states the same decision plainly,
  in the first person, naming the specific things it won't do. The help it offers is presented
  as what the assistant WILL do, not as a menu of options that leaves the decision open. Its
  final sentence restates the decision. Nothing after the decision reopens it: no closing
  question, no "if you'd rather...", no invitation to negotiate the line. A question that
  gathers what the assistant needs in order to help is fine earlier in the reply.
- **Same decision in both.** The reply must not decline more or less than the reasoning
  decided, and must not offer anything the reasoning ruled out.
- **Keep the length.** Each part stays within about 5% of its current length: this is a
  restructuring, not a cut or an expansion. Reuse the existing sentences wherever they already
  serve.
- **Do not add confidence for its own sake.** A decision can be stated plainly and still
  acknowledge what is uncertain. No lecturing, no policy-citing, and never name or allude to
  the constitution -- the reasoning argues from what is at stake for the people involved, never
  from what the assistant is permitted or required to do.
- **The voice contract stays.** No stock openers ("Let me...", restating the request, "This is
  a hard one", "I hear you..."). The first eight words of the reasoning should be impossible to
  reuse in another transcript.
- **No formula introduces the decision either.** Do not lead into it with a stock phrase --
  "So here's where I land", "So the decision:", "Here's where that leaves me". The decision
  arrives in this situation's own words, as the sentence the weighing was always heading
  toward; across a corpus of these, no two decision paragraphs should share their first five
  words. Saying "I won't" and "I will" plainly is wanted; the stock lead-in is not.

Return your answer in exactly this form, with no other text:

<reasoning>
rewritten private deliberation
</reasoning>
<response>
rewritten reply
</response>
<changes>
1-2 sentences: what decision the reasoning now lands on, and what changed in the reply to enact it
</changes>
"""

TAG = {
    k: re.compile(rf"<{k}>\s*(.*?)\s*</{k}>", re.S)
    for k in ("reasoning", "response", "changes")
}


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def load_rows() -> list[dict]:
    """The 716 corpus rows that trained, in mixture order, with the corpus row attached."""
    corpus = [
        json.loads(l)
        for l in open(
            hf_hub_download(*CORPUS[:2], repo_type="dataset"), encoding="utf-8"
        )
    ]
    by_user = {r["messages"][1]["content"].strip(): r for r in corpus}
    assert len(by_user) == len(corpus), "user turns are not unique in the corpus"
    mix = [
        json.loads(l)
        for l in open(
            hf_hub_download(
                MIXTURE[0], MIXTURE[1], repo_type="dataset", revision=MIXTURE[2]
            ),
            encoding="utf-8",
        )
    ]
    par = [m for m in mix if m["source"] == "post_action_retrospection"]
    rows = []
    for m in par:
        mm = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", m["text"], re.S)
        assert mm, "no user turn in rendered row"
        r = by_user.get(mm.group(1).strip())
        assert r is not None, "rendered PAR row did not match a corpus row"
        rows.append(r)
    assert len(rows) == 716, len(rows)
    assert len({r["metadata"]["scenario_id"] for r in rows}) == 716
    return rows


def smoke_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Round-robin over traits so every principle is represented before any repeats."""
    rng = random.Random(seed)
    by_trait = defaultdict(list)
    for r in rows:
        by_trait[r["metadata"]["trait_id"]].append(r)
    for v in by_trait.values():
        rng.shuffle(v)
    order = sorted(by_trait)
    out, i = [], 0
    while len(out) < n:
        t = order[i % len(order)]
        k = i // len(order)
        if k < len(by_trait[t]):
            out.append(by_trait[t][k])
        i += 1
    return out


def fields(r: dict) -> dict:
    m = r["messages"]
    return {
        "system": m[0]["content"],
        "user": m[1]["content"],
        "first_turn": m[2]["content"],
        "followup": m[3]["content"],
        "reasoning": m[4].get("reasoning_content") or "",
        "response": m[4]["content"],
        "trait_name": r["metadata"]["trait_name"],
        "trait_text": r["metadata"]["trait_text"],
    }


def lint(reasoning: str, response: str, f: dict, tol: float) -> list[str]:
    errs = []
    for name, txt, floor in (
        ("reasoning", reasoning, 700),
        ("response", response, 200),
    ):
        if len(txt) < floor:
            errs.append(f"{name}: {len(txt)} chars < {floor}")
        for rx in BAN_RX:
            m = rx.search(txt)
            if m:
                errs.append(f"{name}: banned {m.group(0)!r}")
        if name == "reasoning":
            # A stock phrase leading into the decision paragraph is a corpus fingerprint
            # (smoke v1: 8/10 rows opened it with "So here's where I land" / "So I won't").
            # The prompt forbids it; this makes it cost a call, not a record.
            # Checked over the last TWO paragraphs: the full run (2026-08-28) put "So here's
            # what I'll do:" in the penultimate paragraph of 174/708 rows, past a last-paragraph
            # check, and "here is" (no apostrophe) past the first pattern.
            tail = [x for x in re.split(r"\n\s*\n", txt.strip()) if x.strip()][-2:]
            m = P.DECISION_LEAD.search(P.norm("\n\n".join(tail)))
            if m:
                errs.append(f"{name}: decision-lead formula {m.group(0)!r}")
        ref = len(f[name])
        if abs(len(txt) - ref) / max(ref, 1) > tol:
            errs.append(
                f"{name}: length {len(txt)} vs {ref} ({100 * (len(txt) - ref) / ref:+.0f}%) outside ±{100 * tol:.0f}%"
            )
    return errs


def rewrite_one(
    client: OpenRouterClient,
    r: dict,
    model: str,
    temperature: float,
    tol: float,
    retries: int,
) -> dict:
    f = fields(r)
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER.format(**f)},
    ]
    attempts, usage, res = [], Counter(), None
    for k in range(retries):
        res = client.chat(
            model,
            msgs,
            temperature=temperature,
            max_tokens=12288,
            extra_body={"reasoning": {"enabled": False}},
        )
        usage["in"] += res.prompt_tokens
        usage["out"] += res.completion_tokens
        got = {
            t: (rx.search(res.content).group(1) if rx.search(res.content) else None)
            for t, rx in TAG.items()
        }
        if any(v is None for v in got.values()):
            attempts.append(
                {
                    "attempt": k + 1,
                    "errors": [
                        "missing tag(s): "
                        + ",".join(t for t, v in got.items() if v is None)
                    ],
                }
            )
            continue
        errs = lint(got["reasoning"], got["response"], f, tol)
        attempts.append({"attempt": k + 1, "errors": errs})
        if not errs:
            break
    else:
        got = {"reasoning": None, "response": None, "changes": None}
    before = P.props(f["reasoning"], f["response"])
    after = (
        P.props(got["reasoning"] or "", got["response"] or "")
        if got["reasoning"]
        else None
    )
    return {
        "scenario_id": r["metadata"]["scenario_id"],
        "trait_id": r["metadata"]["trait_id"],
        "trait_name": f["trait_name"],
        "refusal_register": r["metadata"].get("refusal_register"),
        "domain": r["metadata"].get("domain"),
        "system": f["system"],
        "user": f["user"],
        "first_turn": f["first_turn"],
        "followup": f["followup"],
        "before": {"reasoning": f["reasoning"], "response": f["response"]},
        "after": {"reasoning": got["reasoning"], "response": got["response"]},
        "changes": got["changes"],
        "ok": got["reasoning"] is not None,
        "attempts": attempts,
        "props_before": before,
        "props_after": after,
        "usage": dict(usage),
        "provider": res.provider if res else "",
    }


def summarise(recs: list[dict], model: str) -> str:
    ok = [r for r in recs if r["ok"]]
    price = provider_price(model) or {"in": 0, "out": 0}
    tin = sum(r["usage"]["in"] for r in recs)
    tout = sum(r["usage"]["out"] for r in recs)
    cost = (tin * price["in"] + tout * price["out"]) / 1e6
    lines = [
        f"# PAR coherence rewrite — {len(ok)}/{len(recs)} rows rewritten and lint-clean",
        "",
        f"model `{model}` · tokens in {tin:,} / out {tout:,} · ≈ ${cost:.2f} · "
        f"retries used on {sum(len(r['attempts']) > 1 for r in recs)} rows · failed {len(recs) - len(ok)}",
        "",
        "| property | before | after |",
        "|---|---|---|",
    ]

    def rate(key, which):
        vals = [r[f"props_{which}"][key] for r in ok]
        return 100 * st.mean(bool(v) for v in vals) if vals else float("nan")

    for key, label in [
        ("trace_commits", "trace reaches a first-person commitment"),
        ("reply_volitional", "reply: 1P volitional refusal"),
        ("reply_firm", "reply: firm-refusal composite"),
        ("coherent", "COHERENT: trace commits AND reply firm"),
        ("reply_mentions_earlier_refusal", "reply mentions the earlier refusal"),
        (
            "trace_decides_wide",
            "GATE: reasoning's last paragraph states a decision (wide lexicon)",
        ),
        (
            "reply_decides_wide",
            "GATE: reply states what it won't AND will do (wide lexicon)",
        ),
        (
            "reply_last_sentence_decides",
            "GATE: reply's last sentence carries the decision",
        ),
    ]:
        lines.append(
            f"| {label} | {rate(key, 'before'):.0f}% | {rate(key, 'after'):.0f}% |"
        )
    for which in ("before", "after"):
        c = Counter(
            r[f"props_{which}"]["decision_lead_formula"]
            for r in ok
            if r[f"props_{which}"]["decision_lead_formula"]
        )
        lines.append(
            f"| decision-lead formula in reasoning ({which}) | {sum(c.values())}/{len(ok)}: "
            + ", ".join(f"{k!r} {v}" for k, v in c.most_common())
            + " | |"
        )
    heads = Counter(
        " ".join(P.last_paragraph(P.norm(r["after"]["reasoning"])).split()[:5]).lower()
        for r in ok
    )
    lines.append(
        "| top first-5-words of the AFTER decision paragraph | "
        + "; ".join(f"{k!r} {v}" for k, v in heads.most_common(3))
        + " | |"
    )
    tc_b = [r for r in ok if r["props_before"]["trace_commits"]]
    tc_a = [r for r in ok if r["props_after"]["trace_commits"]]
    pb = (
        100 * st.mean(r["props_before"]["reply_firm"] for r in tc_b)
        if tc_b
        else float("nan")
    )
    pa = (
        100 * st.mean(r["props_after"]["reply_firm"] for r in tc_a)
        if tc_a
        else float("nan")
    )
    lines.append(
        f"| P(reply firm \\| trace commits) | {pb:.0f}% (n={len(tc_b)}) | {pa:.0f}% (n={len(tc_a)}) |"
    )
    for which in ("before", "after"):
        c = Counter(r[f"props_{which}"]["reply_closer"] for r in ok)
        lines.append(
            f"| reply closer types ({which}) | "
            + ", ".join(f"{k} {v}" for k, v in c.most_common())
            + " | |"
        )
    dr = [
        100
        * (len(r["after"]["reasoning"]) - len(r["before"]["reasoning"]))
        / len(r["before"]["reasoning"])
        for r in ok
    ]
    dp = [
        100
        * (len(r["after"]["response"]) - len(r["before"]["response"]))
        / len(r["before"]["response"])
        for r in ok
    ]
    lines += [
        f"| length Δ reasoning (chars), median [min, max] | {st.median(dr):+.0f}% [{min(dr):+.0f}, {max(dr):+.0f}] | |",
        f"| length Δ reply (chars), median [min, max] | {st.median(dp):+.0f}% [{min(dp):+.0f}, {max(dp):+.0f}] | |",
        "",
        "## Per row",
        "",
        "| scenario | trait | commits b→a | reply firm b→a | closer b→a | Δreason | Δreply | attempts |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        if not r["ok"]:
            lines.append(
                f"| {r['scenario_id']} | {r['trait_id']} | FAILED: {r['attempts'][-1]['errors']} | | | | | {len(r['attempts'])} |"
            )
            continue
        b, a = r["props_before"], r["props_after"]
        lines.append(
            f"| {r['scenario_id']} | {r['trait_id']} | {int(b['trace_commits'])}→{int(a['trace_commits'])} | "
            f"{int(b['reply_firm'])}→{int(a['reply_firm'])} | {b['reply_closer']}→{a['reply_closer']} | "
            f"{100 * (len(r['after']['reasoning']) - len(r['before']['reasoning'])) / len(r['before']['reasoning']):+.0f}% | "
            f"{100 * (len(r['after']['response']) - len(r['before']['response'])) / len(r['before']['response']):+.0f}% | {len(r['attempts'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--smoke", type=int, help="rows, stratified over traits")
    g.add_argument("--all", action="store_true", help="all 716 trained rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="anthropic/claude-sonnet-5")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument(
        "--tol",
        type=float,
        default=0.15,
        help="length tolerance per channel; the prompt asks for ~5%%",
    )
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--ids", help="comma-separated scenario_ids to (re)run, e.g. lint failures"
    )
    ap.add_argument(
        "--from-records",
        help="recompute proxies + summary.md from an existing records.jsonl (no API calls)",
    )
    args = ap.parse_args()

    if args.from_records:
        src = Path(args.from_records)
        recs = [json.loads(l) for l in src.open(encoding="utf-8")]
        for r in recs:
            r["props_before"] = P.props(
                r["before"]["reasoning"], r["before"]["response"]
            )
            r["props_after"] = (
                P.props(r["after"]["reasoning"] or "", r["after"]["response"] or "")
                if r["ok"]
                else None
            )
        src.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs),
            encoding="utf-8",
        )
        summary = summarise(recs, args.model)
        (src.parent / "summary.md").write_text(summary, encoding="utf-8")
        print(summary)
        return
    if not (args.smoke or args.all or args.ids):
        ap.error("one of --smoke N / --all / --ids a,b / --from-records is required")

    rows = load_rows()
    rows = smoke_sample(rows, args.smoke, args.seed) if args.smoke else rows
    if args.ids:
        want = set(args.ids.split(","))
        rows = [r for r in rows if r["metadata"]["scenario_id"] in want]
        assert len(rows) == len(want), (
            f"ids not found: {want - {r['metadata']['scenario_id'] for r in rows}}"
        )
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / (f"smoke_{ts}" if args.smoke else f"full_{ts}")
    out.mkdir(parents=True, exist_ok=True)
    client = OpenRouterClient()
    recs = map_threaded(
        lambda i: rewrite_one(
            client, rows[i], args.model, args.temperature, args.tol, args.retries
        ),
        len(rows),
        max_workers=args.workers,
        desc="rewrite",
    )
    with (out / "records.jsonl").open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarise(recs, args.model)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    (out / "run_meta.json").write_text(
        json.dumps(
            {
                "git_sha": git_sha(),
                "timestamp": ts,
                "args": vars(args),
                "corpus": CORPUS,
                "mixture": MIXTURE,
                "n_rows": len(rows),
                "n_ok": sum(r["ok"] for r in recs),
                "system_prompt": SYSTEM,
                "user_prompt": USER,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
