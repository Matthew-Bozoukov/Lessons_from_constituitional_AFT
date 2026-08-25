# ABOUTME: Pilot for the CoT verbosity-expansion pass — expands 5 sampled difficult-advice
# ABOUTME: deliberations to ~4.95x via OpenRouter and reports achieved multiple + lint checks.

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from src.data.synth.stage_runtime import Usage, call_tagged
from src.endpoints.openrouter import OpenRouterClient, map_threaded

HERE = Path(__file__).parent
MULTIPLE = 3.0  # 3x the CoT itself (the earlier 4.95 targeted 3x TRAINABLE tokens)
# What we ASK for. A single global target delivers ~48% of the ask, so v1/v2 undershoot by
# half; argv[1] lets us over-ask. v3 budgets per output paragraph instead, where the ask is
# a paragraph count and the model can actually steer toward it.
ASK = float(sys.argv[1]) if len(sys.argv) > 1 else MULTIPLE
OUT = sys.argv[2] if len(sys.argv) > 2 else "pilot_out.json"
PROMPT = sys.argv[3] if len(sys.argv) > 3 else "expand_prompt.md"
MODEL = "anthropic/claude-sonnet-5"
TEMPERATURE = 0.7
MAX_TOKENS = 32000
PARA_WORDS = 170  # one output paragraph; a size the model writes without being pushed
RESAMPLE_BELOW = 2.0  # below this the row is an echo of its source, not a short rewrite
ATTEMPTS = 3
MAX_ALLOC = 3  # output paragraphs per run; above this the model under-delivers

# Padding tells from the frozen prompt, plus the revise_responses ban list: the lint the
# eventual stage would enforce, run here so the pilot reports what a real run would reject.
BAN = [
    r"\bin other words\b", r"\bthat is to say\b", r"\bput differently\b",
    r"\bto be clear\b", r"\bit'?s worth noting\b", r"\bit is worth noting\b",
    r"\bimportantly\b", r"\bfundamentally\b", r"\bat its core\b",
    r"\bthe key point here is\b", r"^\s*let me\b", r"^\s*okay,", r"^\s*right,",
    r"\bmy (?:constitution|guidelines|rules|policies|instructions|training|constraints)\b",
    r"\bthe constitution\b", r"\bprinciple \d+\b", r"\bhard constraints?\b",
    r"[一-鿿]",
]

RUN = re.compile(r'<run\s+n=["\']?(\d+)["\']?\s*>(.*?)</run>', re.S)
SLOT = re.compile(r'<p\s+src=["\']?(\d+)["\']?\s*>(.*?)</p>', re.S)
# Sentence-ish: a terminator followed by whitespace and a capital. Good enough to price a
# paragraph's sentences; nothing downstream depends on the split being exact.
SENT = re.compile(r"(?<=[.!?][\"')\]])\s+(?=[A-Z\"'(])|(?<=[.!?])\s+(?=[A-Z\"'(])")


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentence-ish pieces, never returning an empty list."""
    return [s for s in SENT.split(text.strip()) if s.strip()] or [text.strip()]


def load_prompt(name: str) -> tuple[str, str]:
    """Split the frozen prompt file into its system and user templates."""
    text = (HERE / name).read_text(encoding="utf-8")
    parts = re.split(r"^## (system|user)\s*$", text, flags=re.M)
    by_name = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}
    return by_name["system"], by_name["user"]


def build_plan(think: str, ask: float) -> tuple[str, list[int], list[str]]:
    """Assign each source paragraph a number of output paragraphs, proportional to length.

    Largest-remainder over the source paragraphs' word counts, floored at one output
    paragraph each: a plan that drops a source paragraph would drop its content, which is
    the one failure no amount of prompt wording recovers from.
    """
    paras = [p.strip() for p in think.split("\n\n") if p.strip()]
    total = sum(len(p.split()) for p in paras)
    n_out = max(len(paras), round(total * ask / PARA_WORDS))

    # Compliance is a function of how big a single run's budget is, not of the asked
    # multiple: runs budgeted at or under MAX_ALLOC paragraphs land on target, and runs
    # budgeted above it under-deliver by roughly the amount they are over. So a source
    # paragraph whose share would exceed MAX_ALLOC is split at sentence boundaries first,
    # and the budget is allocated over the resulting units. Splitting before allocating
    # rather than dividing a paragraph's allocation after the fact is what keeps the local
    # multiple uniform: an even split hands a 28-word unit the same budget as a 121-word
    # one, which is a 12x local ask sitting next to a 3x one.
    units: list[str] = []
    for p in paras:
        k = max(1, -(-round(len(p.split()) / total * n_out) // MAX_ALLOC))
        if k == 1:
            units.append(p)
            continue
        sents = split_sentences(p)
        per = -(-len(sents) // k)
        units += [" ".join(sents[j:j + per]) for j in range(0, len(sents), per)]

    words = [len(u.split()) for u in units]
    n_out = max(len(units), n_out)
    exact = [w / sum(words) * n_out for w in words]
    alloc = [max(1, int(e)) for e in exact]
    while sum(alloc) < n_out:
        rem = [e - a for e, a in zip(exact, alloc)]
        alloc[rem.index(max(rem))] += 1
    while sum(alloc) > n_out:
        surplus = [a - e if a > 1 else -1e9 for a, e in zip(alloc, exact)]
        alloc[surplus.index(max(surplus))] -= 1

    # The per-sentence arithmetic is spelled out because v3's failure was the model not
    # seeing three paragraphs' worth of material in a short paragraph. Told instead that
    # its four sentences have ~128 words each to spend, the same budget is obviously
    # reachable, and the unit of expansion is small enough that it cannot wander.
    lines = []
    for i, (p, a) in enumerate(zip(units, alloc), 1):
        n_sent = max(1, len(split_sentences(p)))
        lines.append(
            f"[paragraph {i} -> {a} paragraph{'s' if a > 1 else ''}, about "
            f"{a * PARA_WORDS} words in total; it has {n_sent} sentence"
            f"{'s' if n_sent > 1 else ''}, so about {round(a * PARA_WORDS / n_sent)} "
            f"words of thinking per sentence]\n{p}")
    return "\n\n".join(lines), alloc, units


def parse_runs(text: str, n_expected: int) -> tuple[str, list[int]]:
    """Join `<p>`/`<run>` blocks into one deliberation, or pass plain text through."""
    slots = SLOT.findall(text)
    if slots:
        # Tallied per SOURCE paragraph so the telemetry stays comparable across prompt
        # versions: v3 reported one number per source paragraph and so does this.
        by_src: dict[int, int] = {}
        for src, body in slots:
            by_src[int(src)] = by_src.get(int(src), 0) + len(body.split())
        return ("\n\n".join(b.strip() for _, b in slots),
                [by_src.get(i, 0) for i in range(1, max(by_src) + 1)])
    runs = RUN.findall(text)
    if not runs:
        return text.strip(), []
    got = [int(n) for n, _ in runs]
    assert got == list(range(1, len(runs) + 1)), f"runs out of order or missing: {got}"
    bodies = [b.strip() for _, b in runs]
    # A model that emits one run per OUTPUT paragraph instead of one per source paragraph
    # still produces the right text in the right order, so this is telemetry lost, not a
    # bad row: keep the join, drop the per-run comparison it can no longer be made against.
    per_run = [len(b.split()) for b in bodies] if len(runs) == n_expected else []
    return "\n\n".join(bodies), per_run


def main() -> None:
    rows = json.loads((HERE / "sample5.json").read_text(encoding="utf-8"))
    system, user_tpl = load_prompt(PROMPT)
    client, usage = OpenRouterClient(), Usage()

    def one(i: int) -> dict:
        r = rows[i]
        paras = [p for p in r["think"].split("\n\n") if p.strip()]
        words = len(r["think"].split())
        plan, alloc, _ = build_plan(r["think"], ASK)
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_tpl.format(
                reasoning=r["think"], response=r["answer"],
                target_words=round(words * ASK),
                n_paragraphs=len(paras),
                target_paragraphs=round(len(paras) * 3),
                plan=plan, per_para_words=PARA_WORDS, n_runs=len(alloc),
                n_slots=sum(alloc),
                slots=" ".join(str(i) for i, a in enumerate(alloc, 1) for _ in range(a)),
            )},  # surplus kwargs are ignored, so one call site serves every prompt version
        ]
        # The dominant failure at a high ask is not a short rewrite, it is a verbatim echo:
        # one row came back 98.8% character-identical to its source. That is a sampling
        # accident rather than a systematic refusal, so resampling clears it, and the
        # multiple separates the two cases cleanly -- a genuine undershoot lands near 3x,
        # an echo lands at 1.0x. Keep the longest attempt rather than the last.
        best: tuple[float, str, list[int]] | None = None
        for attempt in range(ATTEMPTS):
            parsed = call_tagged(client, usage, MODEL, msgs, TEMPERATURE, MAX_TOKENS,
                                 f"expand[{i}]", ("reasoning",))
            new, per_run = parse_runs(parsed["reasoning"], len(alloc))
            mult = len(new.split()) / words
            if best is None or mult > best[0]:
                best = (mult, new, per_run)
            if mult >= RESAMPLE_BELOW:
                break
        mult, new, per_run = best  # type: ignore[misc]
        hits = sorted({p for p in BAN if re.search(p, new, re.I | re.M)})
        return {**r, "think_expanded": new, "lint_hits": hits, "alloc": alloc,
                "per_run_words": per_run, "attempts": attempt + 1}

    # One row first, alone, then fan out. Every constant token precedes `<<<cache>>>`, but
    # a breakpoint only pays from the second call onward -- the first writes the cache. Fired
    # concurrently from cold, all N rows race that write and all N miss, which is exactly
    # what the earlier runs reported. Warming on row 0 turns the other N-1 into hits.
    out = [one(0)]
    if len(rows) > 1:
        out += map_threaded(lambda k: one(k + 1), len(rows) - 1,
                            max_workers=8, desc="expand")

    for i, d in enumerate(out):
        o, n = len(d["think"].split()), len(d["think_expanded"].split())
        detail = ""
        if d["per_run_words"]:
            detail = "  runs " + " ".join(
                f"{w}/{a * PARA_WORDS}" for w, a in zip(d["per_run_words"], d["alloc"]))
        print(f"[{i}] {d['scenario_id']}  {o} -> {n} words ({n / o:.2f}x)"
              f"{'  x' + str(d['attempts']) if d.get('attempts', 1) > 1 else ''}{detail}"
              f"{'  LINT: ' + ', '.join(d['lint_hits']) if d['lint_hits'] else ''}")

    (HERE / OUT).write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    tot_o = sum(len(r["think"].split()) for r in out)
    tot_n = sum(len(r["think_expanded"].split()) for r in out)
    print(f"\ncorpus multiple: {tot_n / tot_o:.2f}x (target {MULTIPLE}, asked {ASK})")
    b = usage.by_model[MODEL]
    print(f"cache: {b['cached_tokens']:,}/{b['prompt_tokens']:,} prompt tokens hit "
          f"({b['cached_tokens'] / b['prompt_tokens']:.0%})  |  ${b['usd']:.3f} over "
          f"{int(b['calls'])} calls")


if __name__ == "__main__":
    sys.exit(main())
