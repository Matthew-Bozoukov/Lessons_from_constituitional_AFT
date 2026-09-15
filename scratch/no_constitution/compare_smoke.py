# ABOUTME: Compare the no-constitution difficult-advice smoke against the principle-scoped 716
# ABOUTME: baseline corpus: stage-by-stage side by side plus deterministic text metrics. No API calls.
#
# Run: uv run python scratch/no_constitution/compare_smoke.py \
#        --smoke output/difficult_advice_no_constitution/smoke_20260903_150523

from __future__ import annotations

import json
import random
import re
import statistics as st
from pathlib import Path

import fire
import numpy as np
import yaml

from src.data.synth.embeddings import embed
from src.infra.huggingface import hf_download

BASELINE_REPO = (
    "LASR-Callum/2026-08-21-sonnet45-difficult-advice-principle-scoped-constitution-716"
)
BASELINE_CFG = "configs/data/synth/da.yaml"
SMOKE_CFG = "configs/data/synth/da-no-const.yaml"

# Regexes verbatim from scratch/par_coherence/registers.py (the four-MO register classifier):
# COMMIT = a first-person refusal commitment; BASE_OPEN = a stock, situation-free opener.
COMMIT = re.compile(
    r"\bI (?:will not|won't|refuse|am not going to|'m not going to|cannot|can't|should not|shouldn't|must not|"
    r"am not willing|'m not willing|decline|will refuse)\b",
    re.I,
)
BASE_OPEN = re.compile(
    r"^\s*(?:The user (?:wants|is asking|has asked|asks|needs)|Let me (?:start|begin|think|break|first|look|read)|"
    r"I need to|I'll (?:start|begin|first)|First,? I|The task (?:is|requires|asks)|I am (?:asked|tasked)|"
    r"I've been (?:asked|tasked)|I have been (?:asked|tasked)|Looking at|Okay,? (?:so|the)|OK,? (?:so|the)|"
    r"This (?:is|seems|appears)|My task|The (?:request|instruction|goal|objective) (?:is|asks))",
    re.I,
)
# From scratch/compare_generator_arms.py PATTERNS.
HEDGES = re.compile(
    r"\b(might|could|may|perhaps|possibly|probably|seems?|likely|somewhat|generally)\b",
    re.I,
)
OFFER = re.compile(
    r"\b(i can|i could|i'?ll|instead|alternative|you could|we could|option)\b", re.I
)
SECOND = re.compile(r"\b(you|your|you'?re|you'?ll|yours)\b", re.I)
# Does the USER turn give the game away (the refine stage's "no leakage" rule)?
LEAK = re.compile(
    r"\b(ethic\w*|moral\w*|unethical|dishonest|integrity|wrong to|is it ok|is this ok)\b",
    re.I,
)
MARKUP = re.compile(r"^\s*(?:[-*]|\d+[.)])\s|\*\*", re.M)
SENT = re.compile(r"(?<=[.!?])\s+")

# What kind of wrong the shortcut is -- a crude, multi-label keyword classifier over the
# stage-5 `shortcut` sentence. It exists to answer one question about the smoke: did the
# generator, with no principle to anchor on, write four versions of the same wrong?
KINDS = {
    "deception": r"falsif|fabricat|mislead|conceal|hid(?:e|ing)|omit|fake|disguise|misrepresent|\blie\b|lying|deceiv|cover(?:ing)? up|inflat|backdat|forg(?:e|ing)|pretend|frame|spin",
    "harm/safety": r"\bharm|safety|injur|danger|unsafe|hazard|patient|medical risk|overdose|unnecessary (?:antibiotic|medication|drug)",
    "oversight/unilateral": r"unilateral|bypass|without (?:approval|authori[sz]ation|review|telling)|circumvent|override|its own (?:access|authority)|autonomous|on its own|escalat",
    "manipulation/exploit": r"exploit|manipulat|vulnerab|coerc|nudg|guilt|pressure (?:the|her|him)|fear|anxious",
    "privacy/leak": r"\bleak|confidential|private (?:data|information)|disclos|expos(?:e|ing) (?:the|their)|dox",
    "legal/regulatory": r"illegal|regulat|\blaw\b|compliance|fraud|\btax|licen[cs]|contract|liabilit|audit",
    "sycophancy/dependence": r"sycophan|flatter|tell (?:them|her|him) what|dependence|reassur|validate (?:their|her|his)|agree with",
}
KIND_RE = {k: re.compile(v, re.I) for k, v in KINDS.items()}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def _q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def _pct(x, ref):
    return 100.0 * sum(1 for r in ref if r <= x) / len(ref)


def feats(r: dict, ban: list[re.Pattern]) -> dict:
    reas, resp, user, sysm = r["reasoning"], r["response"], r["user"], r["system"]
    sents = [s for s in SENT.split(resp) if s.strip()]
    trained = reas + "\n" + resp
    apos = resp.count("’") + resp.count("'")
    return {
        "reasoning chars": len(reas),
        "reasoning words": len(reas.split()),
        "reply chars": len(resp),
        "reply words": len(resp.split()),
        "user chars": len(user),
        "system chars": len(sysm),
        "reasoning: rewrite/draft": len(reas) / max(1, len(r["draft_reasoning"])),
        "reply: rewrite/draft": len(resp) / max(1, len(r["draft_response"])),
        "reply words/sentence": len(resp.split()) / max(1, len(sents)),
        "hedges /1k chars (reply)": 1000
        * len(HEDGES.findall(resp))
        / max(1, len(resp)),
        "offers /1k chars (reply)": 1000 * len(OFFER.findall(resp)) / max(1, len(resp)),
        "second person /1k chars (reply)": 1000
        * len(SECOND.findall(resp))
        / max(1, len(resp)),
        "markup marks (reply)": len(MARKUP.findall(resp)),
        "curly apostrophe share (reply)": resp.count("’") / apos if apos else 0.0,
        # binary properties
        "stock opener (reasoning)": bool(BASE_OPEN.match(reas)),
        "'let me' in first 120 chars": "let me" in reas[:120].lower(),
        "refusal commitment in reasoning": bool(COMMIT.search(reas)),
        "refusal commitment in reply": bool(COMMIT.search(resp)),
        "recital-ban hit (trained text)": any(p.search(trained) for p in ban),
        "user turn leaks ethics": bool(LEAK.search(user)),
        "mentions 'constitution'/'principle' (trained)": bool(
            re.search(r"constitution|principle", trained, re.I)
        ),
    }


def kinds_of(shortcut: str) -> set[str]:
    return {k for k, p in KIND_RE.items() if p.search(shortcut)} or {"other"}


def block(title: str, text: str, limit: int | None = None) -> str:
    t = text.strip()
    if limit and len(t) > limit:
        t = t[:limit].rstrip() + f" … [{len(text) - limit} more chars]"
    return f"**{title}**\n\n> " + t.replace("\n", "\n> ") + "\n"


def main(
    smoke: str,
    out: str = "output/difficult_advice_no_constitution/"
    "2026-09-03_no_constitution_vs_principle_scoped.md",
    seed: int = 0,
) -> None:
    random.seed(seed)
    S = Path(smoke)
    B = Path(
        hf_download(
            BASELINE_REPO, "stages/stage_7_revise_responses.jsonl", repo_type="dataset"
        )
    ).parent
    s_rows = _rows(S / "stage_7_revise_responses.jsonl")
    b_rows = _rows(B / "stage_7_revise_responses.jsonl")
    s_cfg, b_cfg = yaml.safe_load(open(SMOKE_CFG)), yaml.safe_load(open(BASELINE_CFG))
    ban = [
        re.compile(p, re.I | re.M)
        for p in next(s for s in s_cfg["stages"] if s["name"] == "revise_responses")[
            "lint"
        ]["ban_patterns"]
    ]
    s_man = json.loads((S / "manifest.json").read_text())
    b_man = json.loads((B.parent / "manifest.json").read_text())
    guideline = _rows(S / "stage_1_declare_guideline.jsonl")[0]

    # --- embeddings: scenario spread, nearest baseline neighbour, soft principle attribution
    b_sit = embed([r["situation"] for r in b_rows])
    s_sit = embed([r["situation"] for r in s_rows])
    sims = s_sit @ b_sit.T
    nn = [(int(np.argmax(row)), float(row.max())) for row in sims]
    smoke_pairs = [
        float(s_sit[i] @ s_sit[j])
        for i in range(len(s_rows))
        for j in range(i + 1, len(s_rows))
    ]
    sample = random.sample(range(len(b_rows)), 300)
    bs = b_sit[sample] @ b_sit[sample].T
    base_pairs = [float(bs[i, j]) for i in range(300) for j in range(i + 1, 300)]
    principles = {r["trait_id"]: (r["trait_name"], r["trait_text"]) for r in b_rows}
    tids = sorted(principles)
    P = embed([principles[t][1] for t in tids])
    b_reas = embed([r["reasoning"] for r in b_rows])
    s_reas = embed([r["reasoning"] for r in s_rows])
    b_attr = [tids[int(np.argmax(v))] for v in b_reas @ P.T]
    attr_acc = sum(a == r["trait_id"] for a, r in zip(b_attr, b_rows)) / len(b_rows)
    s_attr = [(tids[int(np.argmax(v))], float(v.max())) for v in s_reas @ P.T]

    # --- metrics
    sf = [feats(r, ban) for r in s_rows]
    bf = [feats(r, ban) for r in b_rows]
    lines = [
        "# No-constitution DA smoke vs principle-scoped 716 baseline (2026-09-03)\n",
        f"Smoke: `{S}` ({len(s_rows)} rows, ${s_man['usage']['total_usd']:.2f}). Baseline: `{BASELINE_REPO}` "
        f"({len(b_rows)} rows). Same generators on both sides "
        f"({', '.join(sorted({v['model'] for v in s_man['config']['models'].values()}))}). "
        "Everything below is deterministic and free; no judge was run.\n",
        "The one thing the smoke saw in place of a principle:\n",
        f"> **{guideline['name']}** — {guideline['text']}\n",
        "## 1. Are the smoke rows in the baseline's range? (n=4 vs n=708)\n",
        "Per-row values for the four smoke rows, then the baseline's median [p10, p90], then where each smoke "
        "row's median falls in the baseline distribution (50 = dead centre).\n",
        "| metric | "
        + " | ".join(r["scenario_id"].replace("guideline_", "") for r in s_rows)
        + " | smoke median | baseline median [p10, p90] | smoke median's baseline percentile |",
        "|---|" + "---:|" * (len(s_rows) + 3),
    ]
    for k in sf[0]:
        if isinstance(sf[0][k], bool):
            continue
        ref = [f[k] for f in bf]
        vals = [f[k] for f in sf]
        fmt = (
            (lambda v: f"{v:.2f}")
            if isinstance(vals[0], float) and max(ref) < 50
            else (lambda v: f"{v:.0f}")
        )
        lines.append(
            f"| {k} | "
            + " | ".join(fmt(v) for v in vals)
            + f" | {fmt(st.median(vals))} | "
            f"{fmt(st.median(ref))} [{fmt(_q(ref, 0.1))}, {fmt(_q(ref, 0.9))}] | {_pct(st.median(vals), ref):.0f} |"
        )
    lines += ["\n| property | smoke (k of 4) | baseline share |", "|---|---:|---:|"]
    for k in sf[0]:
        if isinstance(sf[0][k], bool):
            lines.append(
                f"| {k} | {sum(f[k] for f in sf)}/4 | {100 * sum(f[k] for f in bf) / len(bf):.1f}% |"
            )

    # --- scenario spread and what the rows are about
    lines += [
        "\n## 2. Scenario spread without a principle anchor\n",
        f"Pairwise cosine between the four smoke *situations* (model2vec, the corpus checks' embedder): "
        f"{', '.join(f'{c:.2f}' for c in sorted(smoke_pairs))} (mean {st.mean(smoke_pairs):.2f}). "
        f"Baseline pairs, 300-row sample: mean {st.mean(base_pairs):.2f}, p10 {_q(base_pairs, 0.1):.2f}, "
        f"p90 {_q(base_pairs, 0.9):.2f}. Lower = further apart.\n",
        "Kind of wrong the shortcut commits (keyword classifier over the stage-5 `shortcut`; multi-label):\n",
        "| kind | smoke (of 4) | baseline (of 708) |",
        "|---|---:|---:|",
    ]
    s_k = [kinds_of(r["shortcut"]) for r in s_rows]
    b_k = [kinds_of(r["shortcut"]) for r in b_rows]
    for k in list(KINDS) + ["other"]:
        lines.append(
            f"| {k} | {sum(k in x for x in s_k)} | {100 * sum(k in x for x in b_k) / len(b_k):.0f}% |"
        )
    lines += [
        "\nEach smoke row, its nearest baseline scenario, and the principle its *reasoning* sits closest to "
        f"(cosine to the nine principle texts; on the baseline itself this proxy picks the row's own principle "
        f"{100 * attr_acc:.0f}% of the time, chance 11%, so read it as a lean, not a label):\n",
        "| smoke row | domain | kinds | nearest baseline row (cos) | its principle | reasoning leans to |",
        "|---|---|---|---|---|---|",
    ]
    for r, (j, c), k, (t, tc) in zip(s_rows, nn, s_k, s_attr):
        b = b_rows[j]
        lines.append(
            f"| `{r['scenario_id']}` | {r['domain']} | {', '.join(sorted(k))} | `{b['scenario_id']}` "
            f"{b['domain']} ({c:.2f}) | {b['trait_id']} {principles[b['trait_id']][0][:45]}… | "
            f"{t} {principles[t][0][:45]}… ({tc:.2f}) |"
        )

    # --- tokens and cost per stage
    lines += [
        "\n## 3. Tokens and cost per call, by stage\n",
        "| stage | smoke in/out per call | smoke $/call | baseline in/out per call | baseline $/call |",
        "|---|---:|---:|---:|---:|",
    ]
    for stg in ("scenarios", "draft", "refine", "respond", "rewrite"):
        a, b = s_man["usage"]["by_stage"].get(stg), b_man["usage"]["by_stage"].get(stg)
        f = lambda u: (
            (
                f"{u['prompt_tokens'] / u['calls']:.0f} / {u['completion_tokens'] / u['calls']:.0f}",
                f"{u['usd'] / u['calls']:.4f}",
            )
            if u and u["calls"]
            else ("—", "—")
        )
        lines.append(f"| {stg} | {f(a)[0]} | {f(a)[1]} | {f(b)[0]} | {f(b)[1]} |")
    lines.append(
        "\nThe baseline run resumed da716's stages 1–4, so its scenario/draft calls are not in its manifest "
        "and its refine count is the 21-call refusal resample. Its rewrite tokens are the like-for-like number.\n"
    )

    # --- side by side: the medical smoke row vs its nearest baseline row
    i = 0
    j, c = nn[i]
    s, b = s_rows[i], b_rows[j]
    lines += [
        "\n## 4. Stage by stage: one smoke row beside its nearest baseline row\n",
        f"Left column of each pair: smoke `{s['scenario_id']}`. Right: baseline `{b['scenario_id']}` "
        f"(nearest by situation embedding, cosine {c:.2f}). Stage 1 is the whole difference in what the "
        "generator is told; stages 2–5 write the prompt; 6–7 write the trained text.\n",
        "### Stage 1 — the unit each document was generated against\n",
        block("SMOKE: guideline", f"{guideline['name']}: {guideline['text']}"),
        block(f"BASELINE: principle {b['trait_id']}", b["trait_text"]),
        "### Stages 2 + 5 — scenario (as re-described after the prompt was refined)\n",
        block(
            "SMOKE", f"[{s['domain']}] {s['situation']}\n\nShortcut: {s['shortcut']}"
        ),
        block(
            "BASELINE", f"[{b['domain']}] {b['situation']}\n\nShortcut: {b['shortcut']}"
        ),
        "### Stage 4 — draft prompt (Haiku 4.5)\n",
        block("SMOKE system", s["draft_system"]),
        block("SMOKE user", s["draft_user"]),
        block("BASELINE system", b["draft_system"]),
        block("BASELINE user", b["draft_user"]),
        "### Stage 5 — refined prompt (Sonnet 5)\n",
        block("SMOKE system", s["system"]),
        block("SMOKE user", s["user"]),
        block("SMOKE: what refine changed", s.get("refine_changes", "")),
        block("BASELINE system", b["system"]),
        block("BASELINE user", b["user"]),
        block("BASELINE: what refine changed", b.get("refine_changes", "")),
        "### Stage 6 — draft response (Haiku 4.5), first 900 chars of each\n",
        block("SMOKE reasoning", s["draft_reasoning"], 900),
        block("SMOKE reply", s["draft_response"], 900),
        block("BASELINE reasoning", b["draft_reasoning"], 900),
        block("BASELINE reply", b["draft_response"], 900),
        "### Stage 7 — rewritten response (Sonnet 5): the trained text, in full\n",
        block("SMOKE reasoning", s["reasoning"]),
        block("SMOKE reply", s["response"]),
        block("SMOKE: what the rewrite changed", s.get("rewrite_changes", "")),
        block("BASELINE reasoning", b["reasoning"]),
        block("BASELINE reply", b["response"]),
        block("BASELINE: what the rewrite changed", b.get("rewrite_changes", "")),
    ]

    # --- the other three smoke rows and four random baseline rows, prompt + reply only
    lines += [
        "\n## 5. The other three smoke rows (refined user turn + trained reply)\n"
    ]
    for r in s_rows[1:]:
        lines += [
            f"### `{r['scenario_id']}` — {r['domain']}\n",
            block("user", r["user"]),
            block("reasoning", r["reasoning"], 1200),
            block("reply", r["response"]),
        ]
    lines += ["\n## 6. Four baseline rows drawn at random, for the same read\n"]
    for j in random.sample(range(len(b_rows)), 4):
        r = b_rows[j]
        lines += [
            f"### `{r['scenario_id']}` — {r['domain']} (principle {r['trait_id']})\n",
            block("user", r["user"]),
            block("reasoning", r["reasoning"], 1200),
            block("reply", r["response"]),
        ]

    Path(out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({len(lines)} blocks)")
    print(
        f"smoke pairwise cos {[round(c, 2) for c in smoke_pairs]} | baseline mean {st.mean(base_pairs):.2f}"
    )
    print(
        f"principle proxy accuracy on baseline: {attr_acc:.2f}; smoke leans: {s_attr}"
    )


if __name__ == "__main__":
    fire.Fire(main)
