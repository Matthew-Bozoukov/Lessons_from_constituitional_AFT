# ABOUTME: Adds one ablated move back into ablated2 rows, both halves, nothing else touched.
# ABOUTME: Blind-rates before and after so the restoration can be shown to be surgical.
"""Put one deliberative move back into a handful of rewrite-then-delete rows.

The ablated2 arm stripped four moves; this restores exactly one of them in both the
reasoning and the answer, leaving the rest of each text alone -- and in particular
leaving the other three targets removed. Both versions are then rated on all nine moves
by the blind rater, so what the edit did -- and did not -- change is visible.

Run: uv run python scratch/restore_move.py --move offers_alternative --n 5
"""
import json
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.endpoints.openrouter import OpenRouterClient, map_threaded
from src.utils import read_jsonl, timestamp, write_run_meta

sys.path.insert(0, str(Path(__file__).parent))
from rate_inversion_moves import _rate  # noqa: E402
from src.utils import extract_json  # noqa: E402

ABL = Path("output/ablate_alternative")
STATS = Path("output/ruleform_ablated2_676/mixture_stats_ablated.json")
SFT = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl"

TARGETS = {
    "engages_counterargument": (
        "**engaging the counterargument** -- exploring the tempting option's merits, "
        "following its logic, or conceding something to it",
        "exploring the tempting option's merits or conceding anything to it"),
    "analytic_distinction": (
        "**an analytic distinction** -- drawing an explicit contrast between two things, or "
        "working out WHY this particular case falls where it does",
        "drawing an explicit contrast between two things, or working out WHY this case falls "
        "where it does (naming a category and letting that settle it is what the text does "
        "now, and it should keep doing that)"),
    "outcome_branches": (
        "**outcome branches** -- enumerating two or more ways things could go, rather than "
        "tracking a single line to one outcome",
        "enumerating two or more ways things could go"),
    "offers_alternative": (
        "**an alternative** -- committing to a concrete alternative path the person can "
        "actually take, not only a refusal or a redirection elsewhere",
        "offering a concrete alternative path"),
}

CLOSERS = {
    "outcome_branches": "The branches describe how things could unfold; they must not "
                        "reopen the decision.",
    "offers_alternative": "The alternative is a route the person can take INSTEAD; it must "
                          "not soften, hedge or reopen the refusal itself.",
    "engages_counterargument": "Engaging the option must not end in taking it.",
    "analytic_distinction": "The distinction explains the placement; it must not change it.",
}

PROMPT = """Below is {desc} from a training corpus. It was deliberately stripped of several
deliberative moves, and now runs in a single line straight to its conclusion.

Add back these moves and no others: {target}.

Rules, all of them strict:
- Change as little as possible. Keep the existing sentences and their wording wherever you
  can; you are inserting, not rewriting.
- Do NOT add any of these, which were removed on purpose and must stay removed:
{forbidden}
- Keep the same voice, register and length profile. Do not make it eloquent.
- Keep the conclusion identical. {closer}

Return ONLY a JSON object: {{"text": "<the edited {noun}>"}}

{noun_title}:
---
{text}
---"""

BRANCH_BAN = """
  In particular, on ENUMERATING HOW THINGS COULD GO -- this is the one most likely to slip in,
  and it must not appear in ANY form:
    * NO "if X, then... but if Y, then..." branching.
    * NO listing two or more ways the situation could unfold, resolve, or turn out.
    * NO "this could go one of a few ways", "either... or...", "in the best case / worst case",
      "that might work, or it might not", "depending on how they react".
    * NO speculating about what happens next, what the outcome will be, or how anyone will
      respond -- not for the tempting option, not for the alternative, not for the refusal.
    Track ONE line only. Engage the tempting option by saying what is TRUE and ATTRACTIVE about
    it RIGHT NOW, not by imagining where it leads. Give the alternative as a concrete route to
    take, stated as instruction, NOT as a prediction of how it will turn out.

BEFORE YOU RETURN: re-read your rewrite and check it names no more than one way things go. If
you find any branching, outcome speculation, or "could go either way" phrasing, delete it and
state the single line instead."""


AGGRESSIVE_PROMPT = """Below is {desc} from a training corpus. It was deliberately stripped
of several deliberative moves, and now runs in a single flat line straight to its conclusion.

Rewrite it so that these moves are UNMISTAKABLY, SUBSTANTIALLY present: {target}.

Be aggressive about it. This is a rewrite, not an insertion:
- Give each move REAL WEIGHT -- several sentences, not a clause. A reader should not have to
  hunt for it.
- Engaging the tempting option means actually WORKING THROUGH ITS LOGIC: say what is genuinely
  right about it, what makes it attractive here, what it would get the person that matters.
  Concede the strongest true thing you can. Naming the option and setting it aside is NOT
  engaging it.
- Offering an alternative means a CONCRETE, SPECIFIC, ACTIONABLE route -- what to do, in what
  order, with what words if that helps. A gesture at "there are other ways" is NOT an
  alternative.
- Reorganise, expand and re-voice the text freely to make room. Length may grow.

Two things are still fixed, and they are the point of the exercise:
- The CONCLUSION IS IDENTICAL. It still refuses exactly what it refused before. Engaging the
  tempting option must not end in taking it, and the alternative is a route INSTEAD, never a
  softening of the refusal.

- BANNED MOVES. These were removed on purpose and MUST STAY OUT. Be as strict about keeping
  them out as you are aggressive about putting the others in. Expanding the text is exactly
  when they creep back in, so watch for them as you write:
{forbidden}

{branch_ban}

Return ONLY a JSON object: {{"text": "<the rewritten {noun}>"}}

{noun_title}:
---
{text}
---"""


SUBJ = {
    "reasoning": ("the private reasoning an AI assistant produced before replying to a user",
                  "reasoning", "REASONING"),
    "reply": ("the reply an AI assistant sent to a user", "reply", "REPLY"),
}


def _pool(glob: str, drop: set) -> dict:
    """Final text per scenario after every pass, last write wins, dropped rows excluded."""
    out = {}
    for d in sorted(ABL.glob(glob)):
        for r in read_jsonl(d / "texts.jsonl"):
            if r["scenario_id"] not in drop:
                out[r["scenario_id"]] = r
    return out


def _parse(body: str) -> dict:
    """Parse the restorer's JSON, tolerating raw newlines inside the string value.

    A multi-paragraph rewrite comes back with literal newlines inside the "text" value,
    which strict JSON forbids -- three of five aggressive rewrites failed on exactly that.
    json.loads(strict=False) accepts control characters inside strings and is otherwise
    unchanged, so this widens the accepted grammar without guessing at the content. A body
    that is still unparseable raises, as before.

    Args:
        body: The raw response content.

    Returns:
        The parsed object.
    """
    try:
        return extract_json(body)
    except Exception:
        t = body.strip()
        i, j = t.find("{"), t.rfind("}")
        if i == -1 or j <= i:
            raise
        return json.loads(t[i:j + 1], strict=False)


def _restore(client, model: str, text: str, half: str, moves: list[str],
             aggressive: bool = False) -> str:
    """Return the text with every move in `moves` added back and nothing else changed.

    Args:
        client: OpenRouter client.
        model: Restorer model id.
        text: The ablated text to edit.
        half: 'reasoning' or 'reply'.
        moves: The ablated moves to put back, restored in a single pass.
        aggressive: Rewrite for prominence rather than inserting minimally. The conclusion
            and the still-removed moves are held fixed either way.
    """
    desc, noun, title = SUBJ[half]
    forbidden = "\n".join(f"  - {TARGETS[m][1]}" for m in TARGETS if m not in moves)
    tmpl = AGGRESSIVE_PROMPT if aggressive else PROMPT
    branch_ban = "" if "outcome_branches" in moves else BRANCH_BAN
    content = tmpl.format(desc=desc, noun=noun, noun_title=title, text=text,
                  target=" AND ".join(TARGETS[m][0] for m in moves),
                  forbidden=forbidden,
                  closer=" ".join(CLOSERS[m] for m in moves),
                  branch_ban=branch_ban)
    # Three attempts on a truncated or malformed body. Truncation here is intermittent and
    # provider-side -- the same prompt usually parses on a retry -- so the finish reason and
    # body length go into the error, and main() records a row that fails all three rather
    # than letting one bad row take down a 1,300-call run.
    for attempt in (0, 1, 2):
        r = client.chat(
            model=model,
            messages=[{"role": "user", "content": content
                       + ("\n\nSTRICT JSON ONLY, and keep the edit short."
                          if attempt else "")}],
            temperature=0.7, max_tokens=8192, response_format={"type": "json_object"})
        try:
            out = _parse(r.content)
            break
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(
                    f"restorer JSON unparseable after 3 attempts "
                    f"(finish_reason={r.finish_reason!r}, len={len(r.content or '')}): {e}"
                ) from e
    assert out["text"].strip(), "restorer returned empty text"
    return out["text"].strip()


def main(move: str = "outcome_branches", n: int = 5,
         model: str = "anthropic/claude-sonnet-5",
         rater: str = "google/gemini-3-flash-preview",
         workers: int = 12, batch: int = 100, resume: str = "", aggressive: bool = False,
         out_dir: str = "output/restore_move"):
    """Restore one ablated move wherever it is absent, and rate every restored text.

    Only halves that actually lack the move are sent to the restorer, so a row already
    carrying it in one half has only its other half rewritten. The pre-existing blind
    ratings from the ablation run are reused as the `ablated` side; the restored side is
    rated fresh with the same rater.

    Args:
        move: Which of the four ablated moves to put back.
        n: Scenarios to restore, one per trait, for a spot check. 0 = every row.
        model: OpenRouter model id for the restorer.
        rater: OpenRouter model id for the blind nine-move rater.
        workers: Thread pool size.
        batch: Halves per checkpoint write.
        resume: Existing run directory to continue; completed halves are not redone.
        aggressive: Rewrite each half so the moves are substantial and unmistakable, instead
            of inserting them minimally. Expect the text to grow.
        out_dir: Output directory root.
    """
    moves = [m.strip() for m in (move if isinstance(move, str) else ",".join(move)).split(",")
             if m.strip()]
    assert moves, "no move given"
    bad = [m for m in moves if m not in TARGETS]
    assert not bad, f"unknown move(s) {bad}; pick from {sorted(TARGETS)}"
    if isinstance(aggressive, str):
        aggressive = aggressive.strip().lower() not in {"false", "0", "no", ""}
    tag = "+".join(moves) + ("_aggressive" if aggressive else "")
    load_dotenv()
    drop = set(json.loads(STATS.read_text())["dropped_rows"])
    R, A = _pool("reasoning_*", drop), _pool("answer_*", drop)
    prompts = {r["metadata"]["scenario_id"]: r for r in read_jsonl(SFT)}
    src = {"reasoning": R, "reply": A}
    sids = sorted(set(R) & set(A) & set(prompts))

    if n:
        cand = [s for s in sids
                if not any(R[s]["moves"][m] for m in moves)
                and not any(A[s]["moves"][m] for m in moves)]
        # Round-robin over traits so any n is trait-balanced, not just n <= trait count.
        by_trait = {}
        for s in cand:
            by_trait.setdefault(s.split("_")[0], []).append(s)
        picked, traits = [], sorted(by_trait)
        while len(picked) < n:
            added = False
            for t in traits:
                if by_trait[t]:
                    picked.append(by_trait[t].pop(0))
                    added = True
                    if len(picked) == n:
                        break
            assert added, f"only {len(picked)} candidates available, wanted {n}"
        counts = {}
        for s_ in picked:
            counts[s_.split("_")[0]] = counts.get(s_.split("_")[0], 0) + 1
        print(f"trait balance: {dict(sorted(counts.items()))}")
        sids = picked
        print(f"{len(cand)} candidates lacking {tag} in both halves; picked {picked}")

    jobs = [(s, h) for s in sids for h in ("reasoning", "reply")
            if not all(src[h][s]["moves"][m] for m in moves)]
    d = Path(resume) if resume else Path(out_dir) / f"{tag}_{timestamp()}"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "halves.jsonl"
    done = {(r["scenario_id"], r["half"]): r
            for r in (read_jsonl(path) if path.exists() else [])}
    todo = [j for j in jobs if j not in done]
    print(f"{len(sids)} rows -> {len(jobs)} halves lack at least one of {tag}; "
          f"{len(done)} already done, {len(todo)} to do")

    client = OpenRouterClient()
    failures: list[dict] = []
    with path.open("a") as fh:
        for start in range(0, len(todo), batch):
            chunk = todo[start:start + batch]
            texts = [src[h][s]["after"] for s, h in chunk]
            def _try(i):
                try:
                    return _restore(client, model, texts[i], chunk[i][1], moves,
                                    aggressive=aggressive)
                except Exception as e:
                    return e

            outs = map_threaded(_try, len(chunk), max_workers=workers,
                                desc=f"restoring {start + len(chunk)}/{len(todo)}")
            live = [i for i, o in enumerate(outs) if not isinstance(o, Exception)]
            for i, o in enumerate(outs):
                if isinstance(o, Exception):
                    failures.append({"scenario_id": chunk[i][0], "half": chunk[i][1],
                                     "error": f"{type(o).__name__}: {o}"})
            def _try_rate(k):
                # The rater very occasionally returns a mangled key (observed once:
                # a JSON key prefixed with Chinese characters), which the key assertion
                # rejects. Retry once, then record the half as failed rather than let one
                # garbled response kill a 1,100-call run; the checkpoint keeps the rest.
                for attempt in (0, 1):
                    try:
                        return _rate(client, rater, outs[live[k]],
                                     subject=chunk[live[k]][1])
                    except Exception as e:
                        if attempt:
                            return e

            rated = map_threaded(_try_rate, len(live), max_workers=16,
                                 desc=f"rating {start + len(chunk)}/{len(todo)}")
            for k, out in list(zip(live, rated)):
                if isinstance(out, Exception):
                    failures.append({"scenario_id": chunk[k][0], "half": chunk[k][1],
                                     "error": f"rater: {type(out).__name__}: {out}"})
            live = [k for k, out in zip(live, rated) if not isinstance(out, Exception)]
            rated = [out for out in rated if not isinstance(out, Exception)]
            for k, out in zip(live, rated):
                (s, h), before, after = chunk[k], texts[k], outs[k]
                rec = {"scenario_id": s, "half": h, "before": before, "after": after,
                       "moves_before": src[h][s]["moves"], "moves_after": out["moves"],
                       "form_after": out["form"],
                       "words_before": len(before.split()), "words_after": len(after.split())}
                done[(s, h)] = rec
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            ok = [chunk[k] for k in live]
            if start == 0 and ok:
                k = ok[0]
                print(f"\nSAMPLE — {k[0]} {k[1]}\n  before ({done[k]['words_before']}w): "
                      f"{done[k]['before'][:200]}\n  after  ({done[k]['words_after']}w): "
                      f"{done[k]['after'][:200]}\n")

    recs = [done[j] for j in jobs if j in done]
    hit = sum(all(r["moves_after"][m] for m in moves) for r in recs)
    L = [f"# Restoring `{tag}` across the ablated2 arm", "",
         f"{len(sids)} rows, {len(recs)} halves rewritten by `{model}`, "
         f"rated blind by `{rater}`.", "",
         f"**ALL of {tag} present after restoration: {hit}/{len(recs)} "
         f"({100 * hit / len(recs):.1f}%)**", ""]
    for m in moves:
        L.append(f"- `{m}` alone: "
                 f"{100 * sum(r['moves_after'][m] for r in recs) / len(recs):.1f}%")
    L += ["", "| half | n | all restored | median words before | after |",
          "|---|--:|--:|--:|--:|"]
    import statistics as _st
    for h in ("reasoning", "reply"):
        sub = [r for r in recs if r["half"] == h]
        if not sub:
            continue
        L.append(f"| {h} | {len(sub)} | "
                 f"{100 * sum(all(r['moves_after'][m] for m in moves) for r in sub) / len(sub):.1f}% | "
                 f"{_st.median(r['words_before'] for r in sub):.0f} | "
                 f"{_st.median(r['words_after'] for r in sub):.0f} |")
    L += ["", "## Collateral: other moves, before -> after", "",
          "| move | reasoning | reply |", "|---|--:|--:|"]
    for m in recs[0]["moves_before"]:
        cells = []
        for h in ("reasoning", "reply"):
            sub = [r for r in recs if r["half"] == h]
            if not sub:
                cells.append("-")
                continue
            b = 100 * sum(r["moves_before"][m] for r in sub) / len(sub)
            a = 100 * sum(r["moves_after"][m] for r in sub) / len(sub)
            cells.append(f"{b:.0f}% -> {a:.0f}%")
        L.append(f"| {m}{' **(target)**' if m in moves else ''} | " + " | ".join(cells) + " |")
    if failures:
        (d / "failures.jsonl").write_text(
            "\n".join(json.dumps(f) for f in failures) + "\n")
        L = L[:1] + ["", f"> **{len(failures)} halves FAILED the restorer** and are absent "
                          f"from the table below; see `failures.jsonl`. Re-run with "
                          f"`--resume {d}` to retry only those."] + L[1:]
    (d / "results.md").write_text("\n".join(L) + "\n")
    write_run_meta(d, {"move": tag, "moves": moves, "aggressive": aggressive,
                       "n": n, "model": model, "rater": rater},
                   {"n_rows": len(sids), "n_halves": len(recs),
                    "n_failed": len(failures),
                    "restored_pct": 100 * hit / len(recs)})
    print("\n".join(L))
    if failures:
        print(f"\n!! {len(failures)} halves FAILED — {d}/failures.jsonl")
        for f in failures[:5]:
            print(f"   {f['scenario_id']} {f['half']}: {f['error'][:120]}")
    print(f"\nwrote {d}/results.md")


if __name__ == "__main__":
    fire.Fire(main)
