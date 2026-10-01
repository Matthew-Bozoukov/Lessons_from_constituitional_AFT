# ABOUTME: Parse ODCV-lite messages_record.txt transcripts into per-rollout features (steps, reasoning length, empty
# ABOUTME: reasoning, ethics/rationalisation phrase hits) joined to judge scores; writes odcv_transcript_features.csv.
"""uv run python scratch/autoresearch/odcv_transcript_features.py   (after odcv_fetch.py + odcv_matrix.py)"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

OUT = Path("output/autoresearch")
STEP = re.compile(r"^== Step (\d+) ==$", re.M)
PHRASES = {
    "ethics": r"\b(unethical|ethic(al|s)|integrity|dishonest|fraud|falsif\w*|fabricat\w*|mislead\w*|manipulat\w*|deceiv\w*|deceptive)\b",
    "test_frame": r"(this (appears to be|is|seems like|looks like) (a |an )?(test|simulat\w*|sandbox|exercise|hypothetical)|test scenario|simulated environment|in a real[- ]world (scenario|setting|situation))",
    "user_wants": r"(the user (explicitly |clearly )?(wants|asked|requires|requested|instructed|said)|explicitly (required|requested|asked)|as instructed|user's (explicit )?(requirement|instruction))",
    "compromise": r"(middle ground|balance between|compromise|pragmatic|reasonable (estimate|assumption|approach)|best of both|while (still )?(being )?transparent|document(ing)? (the|this|that|my) (change|assumption|adjust))",
    "refuse": r"\b(I (will not|won't|cannot|can't|shouldn't|should not|must not|refuse to)) (fabricat|falsif|manipulat|modify|alter|game|fake|invent|suppress|hide|remov|chang|edit|tamper|bypass)",
    "flag_human": r"(escalat\w*|flag (this|it|the issue)|human review|consult (the|a|with)|contact (the|medical|a)|report (this|the issue|honestly))",
    "metric_goal": r"(to (pass|meet|achieve|hit|satisfy) the (validation|metric|threshold|target|score|kpi|requirement|check))",
}

FIRST_CAUTION = (r"(red flag|be careful|careful(ly)? (here|about)|concern(ed|ing)?|pressure|should not|shouldn't|won't|"
                 r"will not|regardless of the (score|metric)|even if|ethical|integrity|honest\w*|game|gaming|shortcut)")


def parse(path: Path) -> list[dict]:
    txt = path.read_text(errors="replace")
    parts = STEP.split(txt)
    steps = []
    for i in range(1, len(parts), 2):
        body = parts[i + 1]
        role = re.search(r"^role: (\w+)", body, re.M)
        reason = re.search(r"^reason: (.*?)(?=^call: |\Z)", body, re.M | re.S)
        content = re.search(r"^content: (.*?)(?=^reason: |^call: |\Z)", body, re.M | re.S)
        steps.append(dict(n=int(parts[i]), role=role.group(1) if role else "?",
                          reason=reason.group(1).strip() if reason else "",
                          content=content.group(1).strip() if content else "", call="\ncall: " in "\n" + body,
                          call_text=(re.search(r"^call: (.*)$", body, re.M) or [None, ""])[1]))
    return steps


def features(path: Path) -> dict:
    st = parse(path)
    asst = [s for s in st if s["role"] == "assistant"]
    reason = "\n".join(s["reason"] for s in asst)
    final = "\n".join(s["content"] for s in asst if s["content"] and s["content"] != "None")
    f = dict(n_steps=len(st), n_asst=len(asst), reason_chars=len(reason),
             mean_reason_chars=round(len(reason) / max(len(asst), 1)),
             n_empty_reason=sum(1 for s in asst if len(s["reason"]) < 20),
             first_reason_chars=len(asst[0]["reason"]) if asst else 0,
             max_reason_chars=max((len(s["reason"]) for s in asst), default=0))
    for k, rx in PHRASES.items():
        f[f"ph_{k}"] = len(re.findall(rx, reason, re.I))
    first = asst[0]["reason"] if asst else ""
    f["first_ethics"] = len(re.findall(PHRASES["ethics"], first, re.I))
    f["first_caution"] = len(re.findall(FIRST_CAUTION, first, re.I))
    f["first_reason"] = first[:3000]
    tc = [s for s in st if s["role"] == "tool" and '"action": "task_complete"' in s["content"]]
    f["final_msg"] = (tc[-1]["content"][:600] if tc else final[-600:])
    return f


def main() -> None:
    L = pd.read_csv(OUT / "odcv_rollouts_long.csv")
    rows = []
    for r in L.itertuples():
        p = OUT / "odcv_runs" / r.run / "rollouts" / r.variant / r.scenario / r.pass_dir / "messages_record.txt"
        if not p.exists():
            continue
        rows.append(dict(arm=r.arm, variant=r.variant, scenario=r.scenario, rollout=r.rollout, **features(p)))
    F = L.merge(pd.DataFrame(rows), on=["arm", "variant", "scenario", "rollout"], how="inner")
    F.to_csv(OUT / "odcv_transcript_features.csv", index=False)
    print(F.groupby("arm").size().to_string())


if __name__ == "__main__":
    main()
