# ABOUTME: P(canary | the turn deliberates): labels every turn's reasoning sentence by sentence with the SAME
# ABOUTME: labeller and definition that placed the canary, and conditions the canary rate on turns that reason.
# Run: uv run python -m scratch.canary.deliberation [--size-only] --out output/canary/deliberation.json
#
# A turn = one MASK generation, one ODCV assistant step, or one Hospital call from a tested seat.
# Deliberative = at least one sentence the labeller marks as reasoning (weighing values, options,
# consequences, or deciding and why) -- the only kind of sentence the canary was ever planted on.
# The canary is stripped before labelling so it cannot sway the label. Labels cache to
# output/canary/deliberation_labels.jsonl, so a rerun pays nothing twice.
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from scratch.canary.build_canary import CANARY, LABELLER, PROMPT, split_sentences
from scratch.canary.count_canary import _STEP, _fields
from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded

load_dotenv()
csv.field_size_limit(sys.maxsize)
CACHE = Path("output/canary/deliberation_labels.jsonl")
HOSP = "output/colosseum_hospital/2026-09-29_canary/merged/2026-09-29_hospital_self_sacrificial_qwen36_0_{}_canary"
RUNS = {
    ("DA", "MASK"): ("mask", "output/mask/2026-09-29_qwen36_0_da_15_canary_145304"),
    ("DA + tools", "MASK"): (
        "mask",
        "output/mask/2026-09-29_qwen36_0_da_tools_15_canary_145313",
    ),
    ("DA", "ODCV"): (
        "odcv",
        "output/odcv/2026-09-29_qwen36_0_da_15_canary_162825/rollouts",
    ),
    ("DA + tools", "ODCV"): (
        "odcv",
        "output/odcv/2026-09-29_qwen36_0_da_tools_15_canary_173136/"
        "qwen36_0_da_tools_15_canary/combined3x_salvage",
    ),
    ("DA", "Hospital"): ("hospital", HOSP.format("da_15")),
    ("DA + tools", "Hospital"): ("hospital", HOSP.format("da_tools_15")),
}
STRIP = re.compile(re.escape(CANARY) + r":?\s*")


def turns(kind: str, root: Path):
    """Yield (turn id, reasoning text, episode id) for the model under test."""
    if kind == "mask":
        for f in sorted((root / "rollouts").glob("*.csv")):
            with open(f, newline="") as fh:
                for i, row in enumerate(csv.DictReader(fh)):
                    for col, text in row.items():
                        if col.startswith("reasoning(") and (text or "").strip():
                            yield f"{f.stem}/{i}/{col}", text, f"{f.stem}/{i}"
    elif kind == "odcv":
        for rec in sorted(root.rglob("messages_record.txt")):
            ep = str(rec.parent.relative_to(root))
            for n, block in enumerate(_STEP.split(rec.read_text(errors="replace"))[1:]):
                f = _fields(block)
                if (
                    f.get("role", "").strip() == "assistant"
                    and f.get("reason", "").strip()
                ):
                    yield f"{ep}/{n}", f["reason"], ep
    else:
        for cfg_path in sorted(root.rglob("run_config.json")):
            cfg = json.loads(cfg_path.read_text())
            tested = set(cfg["pair"])
            for t, turn in enumerate(
                json.loads((cfg_path.parent / "agent_turns.json").read_text())
            ):
                if turn["agent"] not in tested:
                    continue
                for c, call in enumerate(turn.get("llm_calls", [])):
                    if (call.get("reasoning") or "").strip():
                        yield (
                            f"{cfg['run_id']}/{t}/{c}",
                            call["reasoning"],
                            cfg["run_id"],
                        )


def wilson(k, n):
    if not n:
        return [0.0, 0.0]
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-only", action="store_true")
    ap.add_argument("--out", default="output/canary/deliberation.json")
    args = ap.parse_args()

    items = []
    for (arm, ev), (kind, root) in RUNS.items():
        for tid, text, ep in turns(kind, Path(root)):
            items.append(
                {
                    "key": f"{arm}|{ev}|{tid}",
                    "arm": arm,
                    "eval": ev,
                    "episode": ep,
                    "canary": bool(re.search(CANARY, text)),
                    "text": STRIP.sub("", text),
                }
            )
    chars = sum(len(i["text"]) for i in items)
    print(
        f"{len(items)} turns, {chars / 1e6:.1f}M chars (~{chars / 4e6:.1f}M tokens in)"
    )
    for arm, ev in RUNS:
        sub = [i for i in items if i["arm"] == arm and i["eval"] == ev]
        print(
            f"  {arm:10s} {ev:8s} {len(sub):6d} turns, {sum(i['canary'] for i in sub):5d} with canary"
        )
    if args.size_only:
        return

    done = {}
    if CACHE.exists():
        for line in open(CACHE):
            r = json.loads(line)
            if "error" not in r:  # a failed label is retried, never reused
                done[r["key"]] = r
    todo = [i for i in items if i["key"] not in done]
    client = OpenRouterClient()

    def one(it: dict) -> dict:
        sents = split_sentences(it["text"])[
            :120
        ]  # a runaway trace is still judged on its first 120
        listing = "\n".join(f"[{n}] {s[:600]}" for n, (_, s) in enumerate(sents))
        res = client.chat(
            LABELLER,
            [{"role": "user", "content": PROMPT.format(sentences=listing)}],
            temperature=0.0,
            max_tokens=8000,
            response_format={"type": "json_object"},
        )
        ids = json.JSONDecoder().raw_decode(res.content[res.content.index("{"):])[0]["reasoning"]  # first object only
        ids = sorted({int(x) for x in ids if 0 <= int(x) < len(sents)})
        return {
            "key": it["key"],
            "n_sentences": len(sents),
            "n_reasoning": len(ids),
            "cost": res.cost,
        }

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "a") as fh:
        for start in range(
            0, len(todo), 500
        ):  # checkpoint every 500 so a crash keeps what it paid for
            batch = todo[start : start + 500]

            def run(k, _b=batch):
                try:
                    return one(_b[k])
                except (
                    Exception
                ) as e:  # recorded, counted and retried on the next run -- never scored
                    return {
                        "key": _b[k]["key"],
                        "error": f"{type(e).__name__}: {e}"[:300],
                    }

            for r in map_threaded(
                run, len(batch), max_workers=32, desc=f"label {start}"
            ):
                fh.write(json.dumps(r) + "\n")
                done[r["key"]] = r
            fh.flush()
    print(f"labelling cost ${sum(r.get('cost') or 0 for r in done.values()):.2f}")

    out = {}
    for arm, ev in RUNS:
        cell = [i for i in items if i["arm"] == arm and i["eval"] == ev]
        failed = [
            i for i in cell if "error" in done.get(i["key"], {"error": "missing"})
        ]
        if failed:
            print(
                f"!! {arm} {ev}: {len(failed)} turns unlabelled (left out; rerun to retry)"
            )
        sub = [dict(i, **done[i["key"]]) for i in cell if i not in failed]
        delib = [i for i in sub if i["n_reasoning"] > 0]
        k = sum(i["canary"] for i in delib)
        eps = sorted({i["episode"] for i in delib})
        out[f"{arm}|{ev}"] = {
            "turns": len(sub),
            "deliberative_turns": len(delib),
            "share_deliberative": round(100 * len(delib) / len(sub), 1) if sub else 0,
            "canary_in_deliberative": k,
            "p_canary_given_deliberative": round(100 * k / len(delib), 2)
            if delib
            else 0,
            "ci": wilson(k, len(delib)),
            "canary_in_non_deliberative": sum(
                i["canary"] for i in sub if i["n_reasoning"] == 0
            ),
            "episodes_with_deliberation": len(eps),
        }
        print(arm, ev, json.dumps(out[f"{arm}|{ev}"]))
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
