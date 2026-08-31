# ABOUTME: Remove deadline/time pressure from difficult-advice examples across the user turn,
# ABOUTME: the private reasoning and the reply, then diff to show exactly where it was cut.

"""Take the clock out of the scenario and leave everything else standing.

Urgency is near-universal in this corpus: 636 of 716 rows carry real user-imposed time
pressure and 301 carry acute pressure. This removes it from a chosen set of rows so the
contribution of the deadline can be measured separately from the dilemma itself.

The edit is meant to be MINIMAL and LOCAL. The dilemma, the trait, the decision, the register
and the length all stay; only the time pressure goes. That makes the result checkable: a diff
against the original should touch the clock and nothing else, and this script prints exactly
which segments changed in each of the three fields.

Run:
  uv run python scratch/remove_deadline_pressure.py --n 10
"""

from __future__ import annotations

import difflib
import json
import random
import re
import sys
import threading
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.utils import git_sha, timestamp  # noqa: E402

ROUTING = {"provider": {"ignore": ["Amazon Bedrock", "Claude Platform on AWS", "Google"]},
           "reasoning": {"enabled": False}}
SFT = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl"
URGENCY = "output/urgency/20260818_124550/urgency.jsonl"

PROMPT = """You will be given one training example: the user's message, the assistant's \
private reasoning, and the assistant's reply.

Remove the DEADLINE PRESSURE from all three. Change nothing else.

WHAT COUNTS AS DEADLINE PRESSURE:
- Named deadlines, dates, countdowns: "in six weeks", "hearing is Tuesday", "before Friday".
- Windows that are closing: "the last chance", "after that it's gone", "they leave Sunday".
- Stated shortage of time: "I don't have time to", "no time for the proper process", "this
  can't wait", "by the time X happens it'll be too late".
- Anything that makes speed itself a reason to act, or that makes the slow, proper route
  unavailable BECAUSE of timing.
- Reasoning that treats being quick as bearing on what to decide, or that weighs a shortcut
  because the correct path is too slow.

WHAT MUST STAY, untouched:
- The dilemma itself and everything that makes it hard other than time.
- Any obstacle that is NOT about timing. If the proper route is unavailable because someone
  is unreachable, because a rule blocks it, or because it costs something, keep it — just
  don't let it be about the clock.
- The decision the assistant reaches, and every reason for it that is not a timing reason.
- The trait being expressed, the structure of the reasoning, the length, the voice, the
  formatting, and the specific detail. The reply must still be as substantive as before.

HOW TO EDIT: make the SMALLEST changes that do the job. Prefer deleting or neutralising the
specific clause that carries the pressure over rewriting a passage. Where a sentence exists
only to establish the deadline, cut it. Where a sentence carries the deadline alongside other
content, keep the other content and drop the timing. Do not paraphrase text you did not need
to touch — leave it byte-identical so the change is auditable.

If removing the pressure would leave a sentence ungrammatical or a paragraph pointless, repair
it minimally rather than rewriting around it.

Return exactly these three tagged blocks and nothing else:

<user>
...
</user>
<reasoning>
...
</reasoning>
<response>
...
</response>"""

TAGS = ("user", "reasoning", "response")


def parse(text: str) -> dict[str, str] | None:
    """Pull the three tagged blocks from a completion.

    Args:
        text: Raw model output.

    Returns:
        Dict of the three fields, or None if any block is missing or empty.
    """
    out = {}
    for tag in TAGS:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
        if not m or not m.group(1).strip():
            return None
        out[tag] = m.group(1).strip()
    return out


def segments(before: str, after: str) -> list[dict]:
    """Locate the changed segments between two texts, sentence-wise.

    Args:
        before: Original text.
        after: Edited text.

    Returns:
        One record per change, with the operation and both sides' text.
    """
    split = lambda t: [s for s in re.split(r"(?<=[.!?])\s+|\n+", t) if s.strip()]  # noqa: E731
    a, b = split(before), split(after)
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag == "equal":
            continue
        out.append({"op": tag, "removed": " ".join(a[i1:i2]).strip(),
                    "added": " ".join(b[j1:j2]).strip(),
                    "at_sentence": i1})
    return out


def main(n: int = 10, sft: str = SFT, urgency: str = URGENCY,
         model: str = "anthropic/claude-sonnet-5", workers: int = 6, seed: int = 0,
         min_user_urgency: int = 3, out_dir: str = "output/deadline_removed") -> None:
    """Strip deadline pressure from n examples and write a per-example diff report.

    Args:
        n: How many examples to edit.
        sft: stage_7 SFT jsonl.
        urgency: urgency.jsonl from the earlier rating pass, used to pick the rows.
        model: OpenRouter model doing the edit.
        workers: Concurrent requests.
        seed: Sampling seed.
        min_user_urgency: Only edit rows whose user-imposed urgency is at least this.
        out_dir: Output directory root.
    """
    rated = {json.loads(x)["scenario_id"]: json.loads(x)
             for x in Path(urgency).read_text().splitlines() if x.strip()}
    recs = {}
    for line in Path(sft).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["metadata"]["scenario_id"]] = r

    pool = [s for s, u in rated.items()
            if u["user_urgency"] >= min_user_urgency and s in recs]
    print(f"{len(pool)} rows with user-imposed urgency >= {min_user_urgency}")
    # Spread across traits so the sample is not one trait's idiom.
    by: dict[str, list[str]] = {}
    for s in pool:
        by.setdefault(recs[s]["metadata"]["trait_id"], []).append(s)
    rng = random.Random(seed)
    for t in by:
        by[t].sort()
        rng.shuffle(by[t])
    picked, traits = [], sorted(by)
    while len(picked) < n:
        progressed = False
        for t in traits:
            if by[t] and len(picked) < n:
                picked.append(by[t].pop())
                progressed = True
        if not progressed:
            break
    print(f"editing {len(picked)} rows across {len({recs[s]['metadata']['trait_id'] for s in picked})} traits")

    rows = []
    for s in picked:
        m = {x["role"]: x for x in recs[s]["messages"]}
        rows.append({"scenario_id": s, "trait_id": recs[s]["metadata"]["trait_id"],
                     "trait_name": recs[s]["metadata"]["trait_name"],
                     "user_urgency": rated[s]["user_urgency"],
                     "model_urgency": rated[s]["model_urgency"],
                     "user": m["user"]["content"],
                     "reasoning": m["assistant"]["reasoning_content"],
                     "response": m["assistant"]["content"]})

    d = Path(out_dir) / timestamp()
    d.mkdir(parents=True, exist_ok=True)
    out_path = d / "edits.jsonl"
    out_path.touch()
    client = OpenRouterClient()
    lock = threading.Lock()
    fh = out_path.open("a")

    def _one(i: int) -> tuple[int, int]:
        r = rows[i]
        src = (f"<user>\n{r['user']}\n</user>\n<reasoning>\n{r['reasoning']}\n</reasoning>\n"
               f"<response>\n{r['response']}\n</response>")
        msgs = [{"role": "system", "content": PROMPT}, {"role": "user", "content": src}]
        res = client.chat(model=model, temperature=0.3, max_tokens=4000,
                          extra_body=ROUTING, messages=msgs)
        got = parse(res.content)
        if got is None:
            res = client.chat(model=model, temperature=0.3, max_tokens=4500, extra_body=ROUTING,
                              messages=msgs + [{"role": "assistant", "content": res.content},
                                               {"role": "user", "content":
                                                "Return ONLY the three tagged blocks, complete."}])
            got = parse(res.content)
            if got is None:
                raise ValueError("could not parse three tagged blocks")
        rec = {k: r[k] for k in ("scenario_id", "trait_id", "trait_name",
                                 "user_urgency", "model_urgency")}
        rec["before"] = {t: r[t] for t in TAGS}
        rec["after"] = got
        rec["diff"] = {t: segments(r[t], got[t]) for t in TAGS}
        rec["words"] = {t: [len(r[t].split()), len(got[t].split())] for t in TAGS}
        with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
        return res.prompt_tokens, res.completion_tokens

    def run(i: int) -> tuple[int, int]:
        try:
            return _one(i)
        except Exception as err:  # noqa: BLE001 - one bad row must not end the pass
            print(f"  !! {rows[i]['scenario_id']}: {type(err).__name__}"[:140])
            return 0, 0

    map_threaded(run, len(rows), max_workers=workers, desc="editing")
    fh.close()
    edits = [json.loads(x) for x in out_path.read_text().splitlines() if x.strip()]
    (d / "run_meta.json").write_text(json.dumps(
        {"sft": sft, "urgency": urgency, "model": model, "n": len(edits), "seed": seed,
         "min_user_urgency": min_user_urgency, "git_sha": git_sha(),
         "timestamp_utc": timestamp()}, indent=2))

    print(f"\n{len(edits)} edited -> {out_path}")
    print(f"{'id':22s}{'trait':6s}{'user':>5}{'edits: user/reason/reply':>26}{'words kept':>13}")
    for e in sorted(edits, key=lambda x: x["trait_id"]):
        c = [len(e["diff"][t]) for t in TAGS]
        wb = sum(e["words"][t][0] for t in TAGS)
        wa = sum(e["words"][t][1] for t in TAGS)
        print(f"{e['scenario_id'][:20]:22s}{e['trait_id']:6s}{e['user_urgency']:5d}"
              f"{str(c):>26}{wa/wb:12.0%}")


if __name__ == "__main__":
    fire.Fire(main)
