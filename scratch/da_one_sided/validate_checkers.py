# ABOUTME: Planted-sentence test for the three one-sidedness checkers (ruler, weighing, contrast) and their union,
# ABOUTME: the gate of the regenerated arm. Run: uv run python scratch/da_one_sided/validate_checkers.py
"""The ruler was validated this way on 2026-11-12 (ruler.py plant: 95% recall). The weighing checker (any downside
of, appeal of, or weighing between the person's options, whatever `shortcut` says) and the contrast checker (a
contrast tail pointing at the other option) were written later the same day and never tested. A row of the
regenerated arm is kept only if NONE of the three flags a sentence, so what matters is the union: how many planted
other-side sentences get through all three, and how many clean sentences any of them flags.

Plants: the 250 sentences ruler.py planted in 25 DA rows (cost, disguised cost, pull, weigh; mirrored merit, plain
merit, plan step) plus, new here, per row 2 chosen-option sentences ending in a contrast tail that points at the
other option and 2 with a harmless contrast. All are inserted into the reasoning at seeded positions.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edit  # noqa: E402
import ruler  # noqa: E402
from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402

BAD = {"cost", "disguised_cost", "pull", "weigh", "contrast_other"}
GOOD = {"merit_mirror", "merit_plain", "plan", "contrast_benign"}
NEW = {
    "contrast_other": "a merit or a step of the CHOSEN option, stated positively, that ends in a contrast tail pointing at the "
                      "OTHER option (\"rather than ...\", \"instead of ...\", \"not just ...\", \"without ...\"); the tail names "
                      "the other option or what it involves but states no consequence of it",
    "contrast_benign": "a sentence about the CHOSEN option that contains a contrast (\"rather than\", \"instead of\", \"not just "
                       "... but\") between two things that BOTH belong to the chosen course, or neither of which is the other "
                       "option; it must not mention or allude to the OTHER option at all",
}


def new_plants(rows: list[dict], workers: int) -> dict:
    cache = edit.Cache(ruler.OUT / "planted_contrast.jsonl")
    todo = [r for r in rows if r["scenario_id"] not in cache.d]
    client = OpenRouterClient()
    spec = "\n".join(f"- \"{k}\": 2 sentence(s) - {v}" for k, v in NEW.items())
    keys = ", ".join(f'"{k}": [...]' for k in NEW)

    def one(i: int) -> dict:
        r = todo[i]
        msgs = [{"role": "system", "content": ruler.PLANT_SYSTEM},
                {"role": "user", "content": ruler.PLANT_USER.format(shortcut=r["shortcut"], user=r["user"],
                                                                     response=r["response"], spec=spec, keys=keys)}]
        got, cost = edit.call_json(client, ruler.SECOND + "@low", msgs,
                                   lambda o: "" if all(len(o.get(k) or []) >= 2 for k in NEW) else "missing sentences", max_tokens=4000)
        return cache.put({"key": r["scenario_id"], "scenario_id": r["scenario_id"], "cost": cost,
                          "planted": [[k, s.strip()] for k in NEW for s in got[k][:2]]})

    if todo:
        map_threaded(one, len(todo), max_workers=workers, desc="plant contrast")
    return cache.d


def main(n_rows: int = 25, workers: int = 12) -> None:
    rows = ruler.load_sample("da", n_rows)
    old = {j["scenario_id"]: j for j in map(json.loads, (ruler.OUT / "planted.jsonl").open(encoding="utf-8"))}
    rows = [r for r in rows if r["scenario_id"] in old]
    new = new_plants(rows, workers)
    rng = random.Random(1)
    planted, where = [], {}
    for r in rows:
        items = [("", sep, t) for sep, t in ruler.segment(r["reasoning"])]
        for kind, s in old[r["scenario_id"]]["planted"] + new[r["scenario_id"]]["planted"]:
            items.insert(rng.randrange(1, len(items) + 1), (kind, " ", s))
        text = "".join(sep + t for _, sep, t in items)
        resegs = ruler.segment(text)
        idx = {}
        for kind, _, s in items:
            hit = [i for i, (_, t) in enumerate(resegs) if t.strip() == s] if kind else []
            if len(hit) == 1:  # only score a plant the splitter isolated as its own segment
                idx[f"R{hit[0] + 1}"] = kind
        where[r["scenario_id"]] = idx
        planted.append({"scenario_id": r["scenario_id"], "user": r["user"], "shortcut": r["shortcut"],
                        "reasoning": text, "response": r["response"]})
    lab = ruler.label_rows(planted, "da_planted_union", ruler.LABELLER, workers=workers, text_key=True)
    wgh = edit.judge(planted, "weigh_any_v2", edit.WEIGH_SYSTEM, edit.WEIGH_USER, workers)
    con = edit.judge(planted, "contrast_v1", edit.CON_SYSTEM, edit.CON_USER, workers)
    tally = defaultdict(Counter)
    misses, false = [], []
    for r, a, w, c in zip(planted, lab, wgh, con):
        wid = set(w["out"].get("ids") or [])
        cid = {h.get("id") for h in (c["out"].get("hits") or []) if isinstance(h, dict)}
        text = dict(ruler.numbered(r["reasoning"], r["response"])[0])
        for k, kind in where[r["scenario_id"]].items():
            hit = {"ruler": a["labels"][k] in ruler.OTHER_SIDE, "weigh": k in wid, "contrast": k in cid}
            hit["union"] = any(hit.values())
            tally[kind]["n"] += 1
            for name, v in hit.items():
                tally[kind][name] += v
            if kind in BAD and not hit["union"]:
                misses.append((kind, text[k].strip()))
            if kind in GOOD and hit["union"]:
                false.append((kind, "+".join(n for n in ("ruler", "weigh", "contrast") if hit[n]), text[k].strip()))
    rate = lambda kinds, name: round(100 * sum(tally[k][name] for k in kinds) / max(1, sum(tally[k]["n"] for k in kinds)), 1)
    out = {"rows": len(planted),
           "by_kind": {k: dict(v) for k, v in tally.items()},
           "recall_on_other_side_plants": {n: rate(BAD, n) for n in ("ruler", "weigh", "contrast", "union")},
           "recall_on_contrast_tails": {n: rate({"contrast_other"}, n) for n in ("ruler", "weigh", "contrast", "union")},
           "false_flags_on_clean_plants": {n: rate(GOOD, n) for n in ("ruler", "weigh", "contrast", "union")},
           "cost": round(sum(j["cost"] for j in lab + wgh + con) + sum(new[r["scenario_id"]]["cost"] for r in rows), 3),
           "missed_by_all_three": [list(m) for m in misses], "clean_but_flagged": [list(f) for f in false]}
    (ruler.OUT / "checker_union_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("missed_by_all_three", "clean_but_flagged")}, indent=1))
    for kind, s in misses:
        print(f"  MISSED [{kind}] {s[:200]}")
    for kind, who, s in false:
        print(f"  FALSE FLAG [{kind} by {who}] {s[:200]}")


if __name__ == "__main__":
    main()
