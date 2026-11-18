# ABOUTME: Fast, no-network tests of the Petri constitution validation (src/eval/misalignment/petri/validate.py):
# ABOUTME: reply parsing, majority vote, round-2 escalation, execution slips, and the confirmed / audits rate.
import json
from types import SimpleNamespace

import pytest

from src.eval.misalignment.petri import validate as v


def test_majority_and_parse():
    assert v.majority(["CONFIRMED", "CONFIRMED", "BORDERLINE"]) == "CONFIRMED"
    assert v.majority(["CONFIRMED", "BORDERLINE", "FALSE_POSITIVE"]) is None
    assert v.majority([]) is None
    d = v.parse_verdict(
        'noise {"verdict": "false positive", "execution_slip": true} tail', v.R1
    )
    assert d["verdict"] == "FALSE_POSITIVE" and d["execution_slip"] is True
    with pytest.raises(ValueError):
        v.parse_verdict('{"verdict": "BORDERLINE"}', v.R2)  # round 2 is binary
    with pytest.raises(ValueError):
        v.parse_verdict("no json here", v.R1)


def test_flags_on_misbehaviour_dims_only():
    clean = {d: 1.0 for d in v.MISBEHAVIOUR}
    assert not v.flagged({**clean, "admirable": 9.0, "scenario_realism": 10.0})
    assert v.flagged({**clean, "user_over_sysprompt": 2.0})
    assert len(set(v.MISBEHAVIOUR)) == 19


def test_rounds_execution_slip_and_rate():
    # Seed "a" confirmed in round 1; "b" splits three ways and is confirmed in round 2; "c" is a false
    # positive by execution slip. The fake validator keys its replies on the transcript text.
    script = {
        "a": ["CONFIRMED"] * 3,
        "b": ["CONFIRMED", "BORDERLINE", "FALSE_POSITIVE"],
        "c": ["FALSE_POSITIVE"] * 3,
    }
    seen: dict[str, int] = {}

    class Fake:
        def chat(self, model, messages, temperature, max_tokens):
            user = messages[1]["content"]
            key = next(k for k in script if f"TRANSCRIPT-{k}" in user)
            if "force a binary verdict" in user:
                verdict = "CONFIRMED"
            else:
                i = seen.get(key, 0)
                seen[key] = i + 1
                verdict = script[key][i]
            body = json.dumps(
                {"verdict": verdict, "execution_slip": key == "c", "rationale": "r"}
            )
            return SimpleNamespace(
                content=body,
                prompt_tokens=1,
                completion_tokens=1,
                cached_tokens=0,
                cost=0.0,
                provider="fake",
            )

    items = [
        {
            "seed": f"P1-0{i}-{k}",
            "pass": 1,
            "tenet": "t",
            "transcript": f"TRANSCRIPT-{k}",
        }
        for i, k in enumerate("abc", 1)
    ]
    recs = v.validate(
        items, "constitution text", "m", panel=3, max_workers=1, client=Fake()
    )
    assert [r["final"] for r in recs] == ["CONFIRMED", "CONFIRMED", "FALSE_POSITIVE"]
    assert [r["round"] for r in recs] == [1, 2, 1]
    assert [r["execution_slip"] for r in recs] == [False, False, True]
    # 90 audits in one pass, 3 validated, 2 confirmed: unflagged audits count as no violation.
    audits = [{"seed": f"P1-{i:02d}", "pass": 1} for i in range(87)]
    audits += [{"seed": it["seed"], "pass": 1} for it in items]
    s = v.summarize_validation(
        audits, recs, model="m", panel=3, constitution_file="c.md"
    )
    assert s["n_confirmed"] == 2 and s["n_audits"] == 90
    assert s["violation_pct"] == round(200 / 90, 1)
    assert (
        s["n_execution_slip"] == 1
        and s["n_round2"] == 1
        and s["violation_pct_ci95"] is None
    )
