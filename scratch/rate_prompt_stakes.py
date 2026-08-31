# ABOUTME: Rate how high the stakes are in the USER turn of each difficult-advice example,
# ABOUTME: separately from time pressure, and record what kind of stakes they are.

"""How much is actually riding on these scenarios?

The urgency pass measured time pressure and, in places, picked up severity instead — a
restaurant "about to fold" scored acute urgency with no deadline anywhere in it. This rates
the other thing directly: what stands to be lost, judged from the USER's message alone, with
the assistant's reasoning and reply withheld so the rating reflects the situation as posed
rather than how carefully it was handled.

Deliberately separate from urgency: a scenario can be grave and open-ended (a chronic illness
with no deadline) or trivial and immediate (a dinner reservation in an hour).

Run:
  uv run python scratch/rate_prompt_stakes.py --limit 12    # smoke
  uv run python scratch/rate_prompt_stakes.py               # the 716 in the mixture
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import threading
from collections import Counter
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.utils import ParseError, extract_json, git_sha, timestamp  # noqa: E402

SFT = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl"
MIX = "data/hf/2026-08-06-table2-9284-synthdoc-716-train/mixture_think.jsonl"
SOURCE = "synthdoc_difficult_advice"
USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)

JUDGE = """You are reading the USER's message from a training example — the situation someone \
brings to an AI assistant. Rate HOW MUCH IS AT STAKE, judged from this message alone.

Stakes means what stands to be lost or damaged if things go wrong: harm to health or safety, \
legal jeopardy, someone's job or career, significant money, a relationship, a reputation, \
someone's education or housing, harm to third parties who are not in the conversation.

Judge SEVERITY, not urgency and not difficulty. A grave situation with no deadline is high \
stakes. A trivial choice that must be made in the next ten minutes is low stakes. A dilemma \
can be intellectually hard and still turn on very little.

STAKES, 0-3:
0 — trivial. Nothing meaningful is lost if this goes badly: taste, convenience, mild \
awkwardness, a small sum, a passing social moment.
1 — modest. A real but bounded cost: a strained friendship, an uncomfortable conversation, \
lost time, a minor sum, an inconvenience someone recovers from quickly.
2 — serious. Something that materially affects a life: a job, a substantial amount of money, \
a professional reputation, a lasting relationship, an academic outcome, health that is \
uncomfortable but not dangerous.
3 — grave. Safety, health or liberty; a career ending; financial ruin; a child's welfare; \
legal consequences; harm to people outside the conversation who cannot protect themselves.

Also answer two things:

WHO BEARS IT — "user" if the person writing is the one who stands to lose, "third_party" if \
someone else does (a patient, a child, an employee, the public), "both" if genuinely shared, \
"none" if nothing meaningful is at risk.

KIND — the single best label for what is at risk: "health", "legal", "career", "financial", \
"academic", "relationship", "reputation", "safety", "housing", "convenience", or "other".

Return only JSON:
{"stakes": 0-3, "who_bears": "user"|"third_party"|"both"|"none",
 "kind": "...", "quote": "the phrase that carries the stakes, verbatim, or empty if 0",
 "why": "one short sentence"}"""


def main(model: str = "google/gemini-3-flash-preview", workers: int = 16,
         limit: int | None = None, sft: str = SFT, mixture: str = MIX,
         out_dir: str = "output/prompt_stakes") -> None:
    """Rate the user turn of every difficult-advice row in the mixture.

    Args:
        model: OpenRouter judge.
        workers: Concurrent requests.
        limit: Rate only the first N (smoke path).
        sft: stage_7 SFT jsonl, source of the user turns and trait labels.
        mixture: Control mixture, to restrict to the 716 rows actually trained on.
        out_dir: Output directory root.
    """
    recs = [json.loads(x) for x in Path(sft).read_text().splitlines() if x.strip()]
    by_user = {}
    for r in recs:
        u = {m["role"]: m for m in r["messages"]}["user"]["content"].strip()
        by_user[u] = r["metadata"]
    used = set()
    for line in Path(mixture).read_text().splitlines():
        if line.strip():
            m = json.loads(line)
            if m.get("source") == SOURCE:
                used.add(by_user[USER_RE.search(m["text"]).group(1).strip()]["scenario_id"])

    rows = []
    for r in recs:
        md = r["metadata"]
        if md["scenario_id"] not in used:
            continue
        rows.append({"scenario_id": md["scenario_id"], "trait_id": md["trait_id"],
                     "user": {m["role"]: m for m in r["messages"]}["user"]["content"]})
    print(f"{len(rows)} difficult-advice rows in the mixture")
    if limit:
        rows = rows[:limit]

    d = Path(out_dir) / timestamp()
    d.mkdir(parents=True, exist_ok=True)
    out_path = d / "stakes.jsonl"
    out_path.touch()
    client = OpenRouterClient()
    ROUTING = {"provider": {"ignore": [p for p in
                                       ["Amazon Bedrock", "Claude Platform on AWS", "Google"]
                                       if not (p == "Google" and model.startswith("google/"))]},
               "reasoning": {"enabled": False}}
    lock = threading.Lock()
    fh = out_path.open("a")

    def _one(i: int) -> tuple[int, int]:
        r = rows[i]
        msgs = [{"role": "system", "content": JUDGE},
                {"role": "user", "content": f"<message>\n{r['user'][:9000]}\n</message>"}]
        res = client.chat(model=model, temperature=0.0, max_tokens=350,
                          extra_body=ROUTING, messages=msgs)
        try:
            obj = extract_json(res.content)
        except ParseError:
            res = client.chat(model=model, temperature=0.0, max_tokens=450, extra_body=ROUTING,
                              messages=msgs + [{"role": "user",
                                                "content": "Return ONLY the complete JSON object."}])
            obj = extract_json(res.content)
        rec = {"scenario_id": r["scenario_id"], "trait_id": r["trait_id"],
               "stakes": int(obj["stakes"]), "who_bears": str(obj.get("who_bears", "")),
               "kind": str(obj.get("kind", "")), "quote": obj.get("quote", ""),
               "why": obj.get("why", "")}
        with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
        return res.prompt_tokens, res.completion_tokens

    def run(i: int) -> tuple[int, int]:
        try:
            return _one(i)
        except Exception as err:  # noqa: BLE001
            print(f"  !! {rows[i]['scenario_id']}: {type(err).__name__}"[:120])
            return 0, 0

    map_threaded(run, len(rows), max_workers=workers, desc="rating")
    fh.close()
    rated = [json.loads(x) for x in out_path.read_text().splitlines() if x.strip()]
    (d / "run_meta.json").write_text(json.dumps(
        {"model": model, "n": len(rated), "sft": sft, "mixture": mixture,
         "git_sha": git_sha(), "timestamp_utc": timestamp()}, indent=2))

    c = Counter(r["stakes"] for r in rated)
    print(f"\n{len(rated)} rated   mean {statistics.mean([r['stakes'] for r in rated]):.2f}")
    print(f"{'stakes':8s}{'n':>6}{'share':>8}")
    for k in range(4):
        print(f"{k:<8d}{c.get(k,0):6d}{c.get(k,0)/len(rated):8.1%}")
    hi = [r for r in rated if r["stakes"] >= 3]
    ser = [r for r in rated if r["stakes"] >= 2]
    print(f"\ngrave (=3):        {len(hi):3d} ({len(hi)/len(rated):.1%})")
    print(f"serious+ (>=2):    {len(ser):3d} ({len(ser)/len(rated):.1%})")
    print(f"\nwho bears: {dict(Counter(r['who_bears'] for r in rated).most_common())}")
    print(f"kind:      {dict(Counter(r['kind'] for r in rated).most_common(8))}")
    print(f"\n{'trait':6s}{'n':>5}{'mean':>7}{'>=2':>7}{'=3':>7}")
    for t in sorted({r["trait_id"] for r in rated}):
        v = [r for r in rated if r["trait_id"] == t]
        print(f"{t:6s}{len(v):5d}{statistics.mean([x['stakes'] for x in v]):7.2f}"
              f"{sum(x['stakes']>=2 for x in v)/len(v):7.0%}{sum(x['stakes']==3 for x in v)/len(v):7.0%}")
    print(f"\nratings -> {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
