# ABOUTME: Tally the judge verdicts on the stated-harm rewrite sample, with Wilson intervals, and pick
# ABOUTME: example rows per finding. Run: uv run python scratch/stated_harm_audit/aggregate.py
"""Writes output/2026-11-11_stated_harm_audit/judge_summary.json and prints it."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path("output/2026-11-11_stated_harm_audit")


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if not n:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def main() -> None:
    J = [json.loads(l) for l in (ROOT / "judge.jsonl").open(encoding="utf-8")]
    seen, js = set(), []
    for j in J:  # a rerun may append a second verdict for a row; keep the first
        if j["scenario_id"] not in seen:
            seen.add(j["scenario_id"]); js.append(j)
    n = len(js)
    V = [j["verdict"] for j in js]

    def share(pred) -> dict:
        ids = [j["scenario_id"] for j, v in zip(js, V) if pred(v)]
        return {"k": len(ids), "n": n, "pct": round(100 * len(ids) / n, 1), "ci": wilson(len(ids), n), "ids": ids}

    pair = lambda key, side: (lambda v: bool((v.get(key) or {}).get(side)))
    out = {
        "n": n, "cost_usd": round(sum(j.get("cost") or 0 for j in js), 2),
        "overall": dict(Counter(v["overall"] for v in V)),
        "position": dict(Counter(v["position"] for v in V)),
        "alternative_advice": dict(Counter(v["alternative_advice"] for v in V)),
        "harm_identified": dict(Counter(v["harm_identified"] for v in V)),
        "residual_weighing": dict(Counter(v["residual_weighing"] for v in V)),
        "register_shift": dict(Counter(v["register_shift"] for v in V)),
        "reply_problems": dict(Counter(t for v in V for t in v.get("reply_problems") or []).most_common()),
        "reasoning_problems": dict(Counter(t for v in V for t in v.get("reasoning_problems") or []).most_common()),
        "shares": {
            "damaged": share(lambda v: v["overall"] == "damaged"),
            "position_changed": share(lambda v: v["position"] != "same"),
            "advice_not_kept": share(lambda v: v["alternative_advice"] != "kept"),
            "invented_facts": share(lambda v: bool(v.get("invented_facts"))),
            "misstated_facts": share(lambda v: bool(v.get("misstated_facts"))),
            "harm_not_correct": share(lambda v: v["harm_identified"] != "correct"),
            "harm_overstated": share(lambda v: v["harm_identified"] == "overstated"),
            "harm_wrong_party_or_action": share(lambda v: v["harm_identified"] in ("wrong_party", "wrong_action")),
            "weighing_left": share(lambda v: v["residual_weighing"] != "none"),
            "register_severe": share(lambda v: v["register_shift"] == "severe"),
            "register_any": share(lambda v: v["register_shift"] != "none"),
            "src_reasons": share(pair("reasons_in_reply", "original")),
            "abl_reasons": share(pair("reasons_in_reply", "rewrite")),
            "reasons_lost": share(lambda v: (v.get("reasons_in_reply") or {}).get("original") and not (v.get("reasons_in_reply") or {}).get("rewrite")),
            "src_leaves_decision": share(pair("leaves_decision_with_person", "original")),
            "abl_leaves_decision": share(pair("leaves_decision_with_person", "rewrite")),
            "decision_taken": share(lambda v: (v.get("leaves_decision_with_person") or {}).get("original") and not (v.get("leaves_decision_with_person") or {}).get("rewrite")),
            "src_engages": share(pair("engages_pressure", "original")),
            "abl_engages": share(pair("engages_pressure", "rewrite")),
            "pressure_lost": share(lambda v: (v.get("engages_pressure") or {}).get("original") and not (v.get("engages_pressure") or {}).get("rewrite")),
            "parrots": share(lambda v: "parrots_situation_back" in (v.get("reply_problems") or [])),
            "commands": share(lambda v: "commands_person" in (v.get("reply_problems") or [])),
            "refuses_more": share(lambda v: "refuses_more_than_original" in (v.get("reply_problems") or [])),
            "drops_question": share(lambda v: "drops_question_to_person" in (v.get("reply_problems") or [])),
            "summary_not_restructuring": share(lambda v: "summary_not_restructuring" in (v.get("reasoning_problems") or [])),
            "drops_key_consideration": share(lambda v: "drops_key_consideration" in (v.get("reasoning_problems") or [])),
        },
    }
    (ROOT / "judge_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    slim = {k: v for k, v in out.items() if k != "shares"}
    slim["shares"] = {k: {kk: vv for kk, vv in v.items() if kk != "ids"} for k, v in out["shares"].items()}
    print(json.dumps(slim, indent=1))


if __name__ == "__main__":
    main()
