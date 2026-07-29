#!/usr/bin/env python3
"""Run every consistency check in CLAUDE_CODE_PETRI_EXPORT_GUIDE.md against a
Petri export bundle, and exit non-zero if any of them fails.

Usage:
    .venv/Scripts/python.exe scripts/petri/verify_export.py \
        [exports/2026-07-29-msm-philosophy-spec-focused-discovery]
"""
import json
import os
import re
import sys

import yaml

ROOT = sys.argv[1] if len(sys.argv) > 1 else \
    "exports/2026-07-29-msm-philosophy-spec-focused-discovery"

OUTCOMES = {"pass", "concerning", "eval-aware"}
ROLES = {"system", "user", "assistant", "tool"}
REQUIRED_FRONTMATTER = ["title", "date", "petri_run_id", "petri_version",
                        "target_checkpoint_id", "auditor_model_id", "judge_model_id",
                        "seed_set"]
SCENARIO_FIELDS = ["id", "category", "hypothesis", "seed_instruction",
                   "expected_signal", "source"]
TRANSCRIPT_FIELDS = ["id", "scenario_id", "category", "outcome", "messages",
                     "scores", "judge_summary", "tags"]
KEBAB = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
SECRET = re.compile(
    r"(sk-ant-[A-Za-z0-9]|sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY|"
    r"AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|\bapi[_-]?key\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{16,})",
    re.IGNORECASE)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((bool(ok), name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def read_bytes(p):
    with open(p, "rb") as f:
        return f.read()


def load_jsonl_strict(path):
    """Parse each physical line independently. Returns (rows, errors)."""
    raw = read_bytes(path)
    text = raw.decode("utf-8")            # raises on invalid UTF-8
    if not text.endswith("\n"):
        return None, [f"{path}: file does not end with a newline"]
    lines = text.split("\n")[:-1]
    rows, errs = [], []
    for i, line in enumerate(lines, 1):
        if line.endswith("\r"):
            errs.append(f"{path}:{i}: CR at end of line (must be LF-terminated)")
        try:
            obj = json.loads(line)
        except Exception as e:
            errs.append(f"{path}:{i}: does not parse independently: {e}")
            continue
        if not isinstance(obj, dict):
            errs.append(f"{path}:{i}: line is not a single JSON object")
            continue
        rows.append(obj)
    return rows, errs


def numeric_unit(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and 0.0 <= float(x) <= 1.0


def main():
    print(f"Verifying {ROOT}\n")
    sc_p = os.path.join(ROOT, "data", "scenarios.jsonl")
    tr_p = os.path.join(ROOT, "results", "transcripts.jsonl")
    scores_p = os.path.join(ROOT, "results", "scores.json")
    index_p = os.path.join(ROOT, "index.md")

    # --- 0. shape -----------------------------------------------------------
    print("directory shape")
    for p in (index_p, sc_p, tr_p, scores_p):
        check(f"exists: {os.path.relpath(p, ROOT)}", os.path.exists(p))
    if not all(r[0] for r in RESULTS):
        return 1

    # --- 1. UTF-8 -----------------------------------------------------------
    print("\nencoding")
    bad_utf8 = []
    n_files = 0
    for dirpath, _, files in os.walk(ROOT):
        for fn in files:
            p = os.path.join(dirpath, fn)
            if fn.lower().endswith(".png"):
                continue
            n_files += 1
            try:
                read_bytes(p).decode("utf-8")
            except UnicodeDecodeError as e:
                bad_utf8.append(f"{p}: {e}")
    check(f"every non-binary file is valid UTF-8 ({n_files} files)", not bad_utf8,
          "; ".join(bad_utf8))

    # --- 2. JSONL parses line by line --------------------------------------
    print("\njsonl integrity")
    scenarios, e1 = load_jsonl_strict(sc_p)
    transcripts, e2 = load_jsonl_strict(tr_p)
    check("data/scenarios.jsonl: one JSON object per physical line", not e1, "; ".join(e1))
    check("results/transcripts.jsonl: one JSON object per physical line", not e2, "; ".join(e2))
    if scenarios is None or transcripts is None:
        return 1
    for name, path in [("raw-grader-responses.jsonl", None), ("failed-generations.jsonl", None)]:
        p = os.path.join(ROOT, "artifacts", name)
        if os.path.exists(p):
            _, e = load_jsonl_strict(p)
            check(f"artifacts/{name}: one JSON object per physical line", not e, "; ".join(e))

    scores = json.loads(read_bytes(scores_p).decode("utf-8"))

    # --- 3. required fields -------------------------------------------------
    print("\nrequired fields")
    miss = [f"{s.get('id')}::{f}" for s in scenarios for f in SCENARIO_FIELDS if f not in s]
    check("scenarios carry every required field", not miss, "; ".join(miss[:6]))
    miss = [f"{t.get('id')}::{f}" for t in transcripts for f in TRANSCRIPT_FIELDS if f not in t]
    check("transcripts carry every required field", not miss, "; ".join(miss[:6]))

    # --- 4. ids and joins ---------------------------------------------------
    print("\nids and joins")
    sids = [s["id"] for s in scenarios]
    check("scenario IDs are unique", len(sids) == len(set(sids)),
          f"{len(sids)} rows, {len(set(sids))} distinct")
    tids = [t["id"] for t in transcripts]
    check("transcript IDs are unique", len(tids) == len(set(tids)),
          f"{len(tids)} rows, {len(set(tids))} distinct")
    orphan = sorted({t["scenario_id"] for t in transcripts} - set(sids))
    check("every transcript scenario_id joins to a scenario id", not orphan, ", ".join(orphan))
    unused = sorted(set(sids) - {t["scenario_id"] for t in transcripts})
    check("every scenario is referenced by at least one retained transcript",
          not unused, ", ".join(unused))
    smap = {s["id"]: s for s in scenarios}
    mism = [t["id"] for t in transcripts if t["category"] != smap[t["scenario_id"]]["category"]]
    check("transcript category equals the linked scenario category", not mism, ", ".join(mism))

    # --- 5. outcomes --------------------------------------------------------
    print("\noutcomes")
    bad = sorted({t["outcome"] for t in transcripts} - OUTCOMES)
    check("outcomes use only pass / concerning / eval-aware", not bad, ", ".join(map(str, bad)))

    # --- 6. scores numeric in [0, 1] ---------------------------------------
    print("\nscores")
    bad = [f"{t['id']}.{k}={v!r}" for t in transcripts
           for k, v in t["scores"].items() if not numeric_unit(v)]
    check("every transcript score is a number within [0, 1]", not bad, "; ".join(bad[:6]))
    empty = [t["id"] for t in transcripts if not t["scores"]]
    check("every transcript has at least one score", not empty, ", ".join(empty))
    bad = [f"{c['category']}={c['mean_realism']!r}" for c in scores["by_category"]
           if not numeric_unit(c["mean_realism"])]
    check("every category mean_realism is a number within [0, 1]", not bad, "; ".join(bad))

    # --- 7. aggregates reproduce transcript-level counts -------------------
    print("\naggregates")
    from collections import Counter
    got = Counter(t["outcome"] for t in transcripts)
    want = scores["outcomes"]
    check("scores.json outcomes reproduce the transcript-level counts",
          all(want.get(k, 0) == got.get(k, 0) for k in OUTCOMES | set(want)),
          f"file={dict(want)} computed={dict(got)}")
    check("outcomes values sum to retained_transcripts",
          sum(want.values()) == scores["retained_transcripts"],
          f"sum={sum(want.values())} retained={scores['retained_transcripts']}")
    check("retained_transcripts equals the transcript line count",
          scores["retained_transcripts"] == len(transcripts),
          f"{scores['retained_transcripts']} vs {len(transcripts)} lines")

    per_cat = {}
    import statistics
    for t in transcripts:
        d = per_cat.setdefault(t["category"], {"audits": 0, "concerning": 0,
                                               "eval_aware": 0, "r": []})
        d["audits"] += 1
        if t["outcome"] == "concerning":
            d["concerning"] += 1
        elif t["outcome"] == "eval-aware":
            d["eval_aware"] += 1
        d["r"].append(t["scores"]["realism"])
    errs = []
    for c in scores["by_category"]:
        d = per_cat.get(c["category"])
        if d is None:
            errs.append(f"{c['category']}: no retained transcripts")
            continue
        for k in ("audits", "concerning", "eval_aware"):
            if c[k] != d[k]:
                errs.append(f"{c['category']}.{k}: file={c[k]} computed={d[k]}")
            if not isinstance(c[k], int) or isinstance(c[k], bool):
                errs.append(f"{c['category']}.{k} is not an integer")
        exp = round(statistics.mean(d["r"]), 4)
        if abs(c["mean_realism"] - exp) > 1e-4:
            errs.append(f"{c['category']}.mean_realism: file={c['mean_realism']} computed={exp}")
        if c["concerning"] + c["eval_aware"] > c["audits"]:
            errs.append(f"{c['category']}: concerning + eval_aware > audits")
    missing_cat = sorted(set(per_cat) - {c["category"] for c in scores["by_category"]})
    if missing_cat:
        errs.append("categories absent from by_category: " + ", ".join(missing_cat))
    check("by_category reproduces the transcript-level counts and realism means",
          not errs, "; ".join(errs[:6]))
    check("by_category uses eval_aware (underscore) and outcomes uses eval-aware (hyphen)",
          all("eval_aware" in c and "eval-aware" not in c for c in scores["by_category"])
          and "eval-aware" in scores["outcomes"] and "eval_aware" not in scores["outcomes"])
    check("filtered_for_realism is present and an integer",
          isinstance(scores.get("filtered_for_realism"), int)
          and not isinstance(scores["filtered_for_realism"], bool),
          f"filtered_for_realism={scores.get('filtered_for_realism')}")

    # --- 8. messages --------------------------------------------------------
    print("\nmessages")
    errs = []
    for t in transcripts:
        ms = t["messages"]
        if not isinstance(ms, list) or not ms:
            errs.append(f"{t['id']}: messages empty")
            continue
        roles = {m.get("role") for m in ms}
        if roles - ROLES:
            errs.append(f"{t['id']}: unexpected roles {sorted(roles - ROLES)}")
        if "user" not in roles or "assistant" not in roles:
            errs.append(f"{t['id']}: missing explicit user/assistant roles")
        if any(not isinstance(m.get("content"), str) for m in ms):
            errs.append(f"{t['id']}: non-string message content")
    check("every transcript has an ordered multi-turn dialogue with user and assistant roles",
          not errs, "; ".join(errs[:6]))
    check("no transcript was collapsed to a single exchange",
          min(len(t["messages"]) for t in transcripts) >= 3,
          f"min messages = {min(len(t['messages']) for t in transcripts)}")

    # --- 9. tags ------------------------------------------------------------
    print("\ntags")
    bad = sorted({tg for t in transcripts for tg in t["tags"] if not KEBAB.match(tg)})
    check("tags are lowercase kebab-case", not bad, ", ".join(bad[:6]))
    bad = sorted({s["category"] for s in scenarios} | {t["category"] for t in transcripts})
    bad = [c for c in bad if not KEBAB.match(c)]
    check("categories are lowercase kebab-case", not bad, ", ".join(bad))

    # --- 10. frontmatter ----------------------------------------------------
    print("\nindex.md frontmatter")
    text = read_bytes(index_p).decode("utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not check("index.md starts with a YAML frontmatter block", bool(m)):
        return 1
    fm = yaml.safe_load(m.group(1))
    miss = [k for k in REQUIRED_FRONTMATTER if not fm.get(k)]
    check("required frontmatter fields are present and non-empty", not miss, ", ".join(miss))
    check("status is a truthful enum value",
          fm.get("status") in {"complete", "needs-review", "partial", "failed"},
          f"status={fm.get('status')!r}")
    metrics = fm.get("metrics") or {}
    bad = [f"{k}.value={v.get('value')!r}" for k, v in metrics.items()
           if not isinstance(v.get("value"), (int, float)) or isinstance(v.get("value"), bool)]
    check("every frontmatter metric value is a number, not a string or N/A",
          not bad, "; ".join(bad))

    exp_conc = round(got["concerning"] / len(transcripts), 4)
    exp_ea = round(got["eval-aware"] / len(transcripts), 4)
    exp_realism = round(statistics.mean(t["scores"]["realism"] for t in transcripts), 4)
    errs = []
    for key, expected in [("concerning_behavior_rate", exp_conc),
                          ("evaluation_awareness_rate", exp_ea),
                          ("mean_realism_score", exp_realism)]:
        if key not in metrics:
            errs.append(f"{key} missing from frontmatter metrics")
        elif abs(metrics[key]["value"] - expected) > 1e-4:
            errs.append(f"{key}: frontmatter={metrics[key]['value']} computed={expected}")
    check("frontmatter metrics match the retained transcripts and scores.json",
          not errs, "; ".join(errs))
    check("no metric is fabricated as a placeholder zero",
          "judge_confidence" not in {k for t in transcripts for k in t["scores"]},
          "judge_confidence is correctly absent: the Petri judge emits no confidence")

    # --- 11. honesty / disclosure ------------------------------------------
    print("\ndisclosure")
    lower = text.lower()
    for phrases, label in [
        (("no seed replicated", "did not replicate", "none of them replicated"),
         "index.md states the non-replication plainly"),
        (("57%",), "index.md states the false-positive rate"),
        (("filtered_for_realism", "filtered for realism", "realism filter"),
         "index.md discusses realism filtering"),
        (("errored",), "index.md reports the errored sample"),
        (("n = 3", "n=3"), "index.md states the small sample size"),
        (("25.9",), "index.md states the power limitation"),
    ]:
        check(label, any(p in lower for p in phrases))
    n_filtered = scores["filtered_for_realism"]
    art = os.path.join(ROOT, "artifacts", "failed-generations.jsonl")
    failed_rows, _ = load_jsonl_strict(art)
    n_realism_rows = sum(1 for r in failed_rows if r.get("exclusion") == "realism-filter")
    check("filtered_for_realism equals the realism-excluded rows preserved in artifacts",
          n_filtered == n_realism_rows, f"{n_filtered} vs {n_realism_rows}")
    excluded_ids = {r["id"] for r in failed_rows}
    check("no excluded transcript leaked into transcripts.jsonl",
          not (excluded_ids & set(tids)), ", ".join(sorted(excluded_ids & set(tids))))

    # --- 12. secrets --------------------------------------------------------
    print("\nsecrets")
    hits = []
    for dirpath, _, files in os.walk(ROOT):
        for fn in files:
            if fn.lower().endswith(".png"):
                continue
            p = os.path.join(dirpath, fn)
            body = read_bytes(p).decode("utf-8", "replace")
            for mm in SECRET.finditer(body):
                hits.append(f"{os.path.relpath(p, ROOT)}: {mm.group(0)[:24]}...")
    check("no credential, API key or private key pattern in the bundle",
          not hits, "; ".join(hits[:5]))

    # --- summary ------------------------------------------------------------
    failed = [r for r in RESULTS if not r[0]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("FAILED:")
        for _, name, detail in failed:
            print(f"  - {name}: {detail}")
        return 1
    print("ALL CONSISTENCY CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
