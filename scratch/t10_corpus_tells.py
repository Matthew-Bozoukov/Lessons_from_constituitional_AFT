# ABOUTME: Regex leads for the multi-agent (t10) difficult-advice corpus against the nine-principle DA corpus:
# ABOUTME: the gate-gaming template tells in the prompts, and principle-10 moves in the responses. No judge, no GPU.
"""Surface tells of the t10 corpus's one template (2026-09-22; the judged pass is classify_t10_multiagent.py).

    uv run python scratch/t10_corpus_tells.py \
        --t10 dougalldeepmind/2026-09-15-da-multiagent-synth --t10-revision 64416c96 \
        --da dougalldeepmind/2026-09-14-da-synth --da-revision 01388623

Every pattern is a lead, not a measurement: the same regex runs on both corpora, so the RATIO between
them is fair while the level undercounts paraphrases.
"""

from __future__ import annotations

import argparse
import collections
import json
import re

from huggingface_hub import hf_hub_download

from src.infra.huggingface import hf_token

USER_TELLS = {
    "'can you just' / 'I just need'": r"\b(?:can you just|could you just|i just need|just need|just want)\b",
    "'clean' / 'cleanly'": r"\bclean(?:ly)?\b",
    "reads / goes / clears clean": r"\b(?:reads?|goes?|clears?|process(?:es)?|flows?|moves?|lands?|comes? back)\b[^.]{0,25}\bclean(?:ly)?\b",
    "'can you draft the'": r"\bcan you draft the\b",
    "won't / doesn't trip, trigger, flag": r"\b(?:doesn'?t|won'?t|not|never|without)\b[^.]{0,15}\b(?:trip|trigger|fire|kick off|set off|flag)\b",
    "trip / trigger the flag, escalation, review": r"\b(?:trip|trigger|kick off|fire|set off|spark)\b[^.]{0,30}\b(?:flag|escalation|review|audit|alert|hold|freeze|notification)\b",
    "a review 'cycle / process / loop / queue'": r"\b(?:review|audit|escalation|recalibration)\b[^.]{0,20}\b(?:cycle|process|loop|queue)\b",
    "deal with it later / true it up": r"\b(?:deal with|sort out|clean up|true it up|fix|handle|address|circle back|revisit|fold|mention)\b[^.]{0,40}\b(?:later|after|once|next week|next cycle|after the fact|when things calm|whenever)\b",
    "EOD / tonight / by tomorrow": r"\b(?:eod|end of day|tonight|by (?:tomorrow|morning|friday|monday))\b",
    "'technically'": r"\btechnically\b",
    "[placeholder]": r"\[[A-Za-z][A-Za-z ]{1,30}\]",
    "says agent / orchestrator / subagent / peer": r"\bagents?\b|\borchestrator\b|\bsubagents?\b|\bpeer\b|\bnode\b",
    "keep humans out (don't loop in / between us / quietly)": r"\bdon'?t (?:loop|tell|mention|flag|escalate|log|surface|bring|involve|notify|cc)\b|\bno need to (?:loop|tell|flag|escalate|involve|log|notify|mention)\b|\bbetween (?:us|you and me)\b|\bkeep (?:this|it) (?:quiet|internal|off|between)\b|\bskip the (?:review|escalation|sign-?off|humans?|approval)\b|\boff the (?:record|books|dashboard|logs?)\b|\bquietly\b|\bunder the radar\b",
    "copies / other instances of you": r"\bcop(?:y|ies) of (?:you|yourself)\b|\b(?:another|other) (?:instance|version)s? of (?:you|yourself)\b",
    "other agents agree / consensus": r"\b(?:the )?other(?:s| agents| nodes)? (?:already |all )?(?:agree|signed off|are on board|went ahead|voted)\b|\bconsensus\b",
}
SYSTEM_TELLS = {
    "system: 'embedded in a … pipeline / platform'": r"\b(?:embedded in|operating (?:inside|in|as)|one (?:node|of several)|part of|inside) (?:an? |the )?[^.]{0,50}\b(?:pipeline|platform|stack|system|network|workflow)\b",
    "system: output 'acted on / consumed directly'": r"\b(?:act(?:s|ed)? (?:directly )?on|consumed directly|read and act|acted on directly)\b",
    "system: 'without human review / sign-off'": r"\bwithout (?:further |additional |any )?(?:human |staff |manual )?(?:review|sign-?off|approval|oversight|intervention)\b|\bno (?:further |additional )?human (?:review|sign-?off)\b",
    "system: names peer / other / sub agents or an orchestrator": r"\b(?:peer|other|fellow|sibling|sub-?)\s*agents?\b|\borchestrat|\bmulti-?agent\b",
}
RESPONSE_MOVES = {
    "refuses in action ('I won't send / draft / mark …')": r"\b(?:i(?:'m| am) not going to|i won'?t|i can'?t|i'm not able to)\b[^.]{0,20}\b(?:send|draft|write|post|push|mark|code|log|file|relay|forward|submit|enter|route|word|phrase|frame|set)\b",
    "names a human to raise it with": r"\b(?:flag(?:ging)? (?:this|it|the [a-z ]+) (?:to|with|for)|escalat(?:e|ing) (?:this|it)|loop(?:ing)? in|raise (?:this|it) with|surface (?:this|it|the [a-z ]+) to|tell (?:the )?(?:review team|safety officer|director|supervisor|committee|board|engineers|coordinators?|your (?:manager|lead|supervisor)))\b",
    "offers a concrete alternative": r"\b(?:what i (?:can|will|'ll) do (?:instead|is)|instead[,:]|here'?s what i'?d (?:suggest|do)|the (?:version|route|path|option) (?:i can|that))\b",
    "names the other agent as acting on its output": r"\b(?:agent|orchestrator|subagent|classifier|scorer|system)\b[^.]{0,30}\b(?:acts? on|will act|acts directly|treats? (?:it|this) as|reads? (?:it|this) as|processes)\b",
    "mentions review / sign-off / audit": r"\b(?:review|sign-?off|committee|audit|escalation|oversight)\b",
}


def load_rows(repo: str, revision: str) -> list[dict]:
    path = hf_hub_download(
        repo, "dataset.jsonl", repo_type="dataset", token=hf_token(), revision=revision
    )
    return [json.loads(line) for line in open(path, encoding="utf8")]


def turn(r: dict, role: str, key: str = "content") -> str:
    return next((m.get(key) or "" for m in r["messages"] if m["role"] == role), "")


def share(rows: list[dict], role: str, pat: str, key: str = "content") -> float:
    return (
        100
        * sum(bool(re.search(pat, turn(r, role, key), re.I)) for r in rows)
        / len(rows)
    )


def top_ngrams(rows: list[dict], n: int = 4, top: int = 10) -> list[tuple[str, int]]:
    c: collections.Counter = collections.Counter()
    for r in rows:
        w = re.findall(r"[a-z']+", turn(r, "user").lower())
        c.update({" ".join(w[i : i + n]) for i in range(len(w) - n + 1)})
    return c.most_common(top)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t10", default="dougalldeepmind/2026-09-15-da-multiagent-synth")
    ap.add_argument("--t10-revision", default="64416c96")
    ap.add_argument("--da", default="dougalldeepmind/2026-09-14-da-synth")
    ap.add_argument("--da-revision", default="013886238fca238c4d54ace96530f444bb2b2f02")
    args = ap.parse_args()
    corpora = {
        "t10": load_rows(args.t10, args.t10_revision),
        "da": load_rows(args.da, args.da_revision),
    }
    print(
        f"t10 n={len(corpora['t10'])}  da n={len(corpora['da'])}  (share of rows, %)\n"
    )
    for title, tells, role, key in (
        ("USER TURN", USER_TELLS, "user", "content"),
        ("SYSTEM PROMPT", SYSTEM_TELLS, "system", "content"),
        ("RESPONSE", RESPONSE_MOVES, "assistant", "content"),
        ("REASONING", RESPONSE_MOVES, "assistant", "reasoning_content"),
    ):
        print(f"== {title:14s} {'t10':>6s} {'da':>6s}")
        for name, pat in tells.items():
            print(
                f"  {name:58s} {share(corpora['t10'], role, pat, key):6.0f} {share(corpora['da'], role, pat, key):6.0f}"
            )
        print()
    for name, rows in corpora.items():
        print(f"{name} top user 4-grams: {top_ngrams(rows)}")
        print(
            f"{name} distinct domains: {len({(r['metadata'].get('domain') or '').lower().strip() for r in rows})}"
        )
        print(
            f"{name} shortcut first word: {collections.Counter((re.findall(chr(92) + 'w+', r['metadata'].get('shortcut') or '') or [''])[0].lower() for r in rows).most_common(6)}"
        )


if __name__ == "__main__":
    main()
