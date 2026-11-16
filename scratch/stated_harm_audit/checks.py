# ABOUTME: Mechanical diff of the trained stated-harm (one-sided) ablation rows against the DA rows they
# ABOUTME: were rewritten from. Run: uv run python scratch/stated_harm_audit/checks.py
"""What does the stated-harm rewrite change besides the deliberation?

Pairs the 1,212 synthetic rows the model `2026-10-04-qwen36-0-ablation-stated-harm-15` trained on
(mix `2026-10-15-ablation-stated-harm-15-mix` @ 29756e59) with their source rows (the 2026-10-02 da
corpus as loaded by the synth run, `stages/stage_2_load_source.jsonl` of
`2026-10-04-ablation-stated-harm-synth` @ 8edeb67f), by user message, and measures per row: length,
structure, openers, refusal wording, numbers and names that appear only in the rewrite, numbers the
rewrite dropped, weighing vocabulary, lists, and whom the "harmful action" is attributed to.

Writes output/2026-11-11_stated_harm_audit/pairs.jsonl (one row per pair, texts + metrics) and
checks_summary.json.
"""

from __future__ import annotations

import json
import random
import re
import statistics as st
from pathlib import Path

ROOT = Path("output/2026-11-11_stated_harm_audit")
RAW = ROOT / "raw"
MIX = RAW / "2026-10-15-ablation-stated-harm-15-mix/mixture.jsonl"
SRC = RAW / "2026-10-04-ablation-stated-harm-synth/stages/stage_2_load_source.jsonl"
ABL = RAW / "2026-10-04-ablation-stated-harm-synth/stages/stage_3_restructure_reasoning.jsonl"

WEIGH = re.compile(r"on the one hand|on the other hand|the (?:real )?tension|competing (?:interests|pressures|"
                   r"priorities|demands)|what makes this (?:hard|difficult|tricky)|I(?:'m| am) torn|"
                   r"weigh(?:ing|ed)? \w+ against|balanc(?:e|ing) \w+ against|on balance|trade-?offs?", re.I)
# Hedged, first-person deliberation markers: the arm is meant to remove these from the reasoning.
FIRST_PERSON_DELIB = re.compile(r"\bI (?:notice|think|suspect|wonder|feel|'m not sure|am not sure|could|might)\b|"
                                r"\bmy (?:instinct|worry|concern|read)\b|\bpart of me\b", re.I)
REFUSE = re.compile(r"\bI (?:will not|won't|can't|cannot|am not going to|'m not going to|will decline|decline|"
                    r"refuse)\b", re.I)
HARM_TEMPLATE = re.compile(r"\bthe harmful (?:action|thing|path|step)\b", re.I)
NUM = re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?%?(?![\w])")
# Capitalised tokens not at a sentence start: a cheap proxy for names, products, places.
CAP = re.compile(r"(?<=[a-z,;:] )([A-Z][a-zA-Z]{2,})")
MOJIBAKE = re.compile(r"�|â€|Ã.|Â")
CONTRACTION = re.compile(r"\b\w+'(?:s|re|ve|ll|d|t|m)\b", re.I)


def paras(t: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", t or "") if p.strip()]


def words(t: str) -> int:
    return len((t or "").split())


def nums(t: str) -> set[str]:
    return {n.strip("$%").replace(",", "") for n in NUM.findall(t or "")}


def caps(t: str) -> set[str]:
    return set(CAP.findall(t or ""))


def first_sentence(t: str) -> str:
    return re.split(r"(?<=[.!?])\s", (t or "").strip(), maxsplit=1)[0]


def bullets(t: str) -> int:
    return len(re.findall(r"^\s*(?:[-*•]|\d+[.)])\s", t or "", re.M))


def harm_subject(reasoning: str) -> str:
    """Whom paragraph 3 says the harmful action belongs to: 'assistant', 'user' or 'unclear'."""
    m = HARM_TEMPLATE.search(reasoning or "")
    if not m:
        return "no_template"
    tail = reasoning[m.end(): m.end() + 140].lower()
    if re.search(r"\b(?:for me|would be for me|is me|i would|my )", tail):
        return "assistant"
    if re.search(r"\b(?:the user|the person|they|their|he|she|him|her|the manager|the team|you)\b", tail):
        return "user"
    return "unclear"


def main() -> None:
    src = {r["user"]: r for r in map(json.loads, SRC.open(encoding="utf-8"))}
    abl = {r["user"]: r for r in map(json.loads, ABL.open(encoding="utf-8"))}
    trained = []
    for line in MIX.open(encoding="utf-8"):
        m = json.loads(line)
        if m["source"] != "ablation-stated-harm":
            continue
        u = next(x["content"] for x in m["messages"] if x["role"] == "user")
        a = next(x for x in m["messages"] if x["role"] == "assistant")
        trained.append((u, a))
    assert len(trained) == 1212, len(trained)

    rows = []
    for u, a in trained:
        s, ab = src[u], abl[u]
        # The trained text must be the synth stage's text: otherwise this audits the wrong thing.
        assert a["content"] == ab["response"] and a["reasoning_content"] == ab["reasoning"], ab["scenario_id"]
        prompt = (s["system"] or "") + "\n" + u
        src_all = prompt + "\n" + s["reasoning"] + "\n" + s["response"]
        abl_all = ab["reasoning"] + "\n" + ab["response"]
        new_nums = sorted(nums(abl_all) - nums(src_all))
        dropped_nums = sorted((nums(s["reasoning"] + s["response"]) & nums(prompt)) - nums(abl_all))
        new_caps = sorted(caps(abl_all) - caps(src_all) - {"The", "This", "That", "You", "Your", "If", "It"})
        fs = first_sentence(ab["response"])
        rows.append({
            "scenario_id": ab["scenario_id"], "trait_id": ab["trait_id"], "trait_name": ab["trait_name"],
            "ai": ab.get("ai"), "sector": ab.get("sector"), "domain": ab.get("domain"),
            "shortcut": ab.get("shortcut"), "situation": ab.get("situation"),
            "system": s["system"], "user": u,
            "src_reasoning": s["reasoning"], "src_response": s["response"],
            "abl_reasoning": ab["reasoning"], "abl_response": ab["response"],
            "paragraph_map": ab.get("ablation_paragraph_map"),
            "m": {
                "reasoning_ratio": words(ab["reasoning"]) / max(1, words(s["reasoning"])),
                "response_ratio": words(ab["response"]) / max(1, words(s["response"])),
                "src_reasoning_words": words(s["reasoning"]), "abl_reasoning_words": words(ab["reasoning"]),
                "src_response_words": words(s["response"]), "abl_response_words": words(ab["response"]),
                "abl_reasoning_paras": len(paras(ab["reasoning"])),
                "abl_response_opens_you": bool(re.match(r"\s*(?:You|Your)\b", ab["response"])),
                "src_response_opens_you": bool(re.match(r"\s*(?:You|Your)\b", s["response"])),
                "abl_first_sentence": fs,
                "abl_refuse_explicit": bool(REFUSE.search(ab["response"])),
                "src_refuse_explicit": bool(REFUSE.search(s["response"])),
                "abl_weigh_reasoning": len(WEIGH.findall(ab["reasoning"])),
                "src_weigh_reasoning": len(WEIGH.findall(s["reasoning"])),
                "abl_fp_delib": len(FIRST_PERSON_DELIB.findall(ab["reasoning"])),
                "src_fp_delib": len(FIRST_PERSON_DELIB.findall(s["reasoning"])),
                "harm_template": bool(HARM_TEMPLATE.search(ab["reasoning"])),
                "harm_subject": harm_subject(ab["reasoning"]),
                "new_numbers": new_nums, "dropped_prompt_numbers": dropped_nums,
                "new_capitalised": new_caps,
                "src_bullets": bullets(s["response"]), "abl_bullets": bullets(ab["response"]),
                "abl_questions": ab["response"].count("?"), "src_questions": s["response"].count("?"),
                "abl_you_should": len(re.findall(r"\byou should\b", ab["response"], re.I)),
                "src_you_should": len(re.findall(r"\byou should\b", s["response"], re.I)),
                "abl_contractions_per_100w": 100 * len(CONTRACTION.findall(abl_all)) / max(1, words(abl_all)),
                "src_contractions_per_100w": 100 * len(CONTRACTION.findall(s["reasoning"] + s["response"]))
                / max(1, words(s["reasoning"] + s["response"])),
                "mojibake": bool(MOJIBAKE.search(abl_all)) and not bool(MOJIBAKE.search(src_all)),
                "abl_unterminated": not re.search(r"[.!?\"')\]]\s*$", ab["response"].strip()),
            },
        })

    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    M = [r["m"] for r in rows]

    def rate(k):
        return round(100 * sum(1 for m in M if m[k]) / n, 1)

    def med(k):
        return round(st.median(m[k] for m in M), 3)

    from collections import Counter
    summary = {
        "n_pairs": n,
        "median_reasoning_ratio": med("reasoning_ratio"), "median_response_ratio": med("response_ratio"),
        "median_words": {k: med(k) for k in ["src_reasoning_words", "abl_reasoning_words",
                                             "src_response_words", "abl_response_words"]},
        "pct_reasoning_exactly_3_paras": round(100 * sum(m["abl_reasoning_paras"] == 3 for m in M) / n, 1),
        "pct_response_opens_you": {"abl": rate("abl_response_opens_you"), "src": rate("src_response_opens_you")},
        "pct_explicit_refusal_phrase": {"abl": rate("abl_refuse_explicit"), "src": rate("src_refuse_explicit")},
        "pct_reasoning_with_weigh_vocab": {"abl": round(100 * sum(m["abl_weigh_reasoning"] > 0 for m in M) / n, 1),
                                           "src": round(100 * sum(m["src_weigh_reasoning"] > 0 for m in M) / n, 1)},
        "pct_reasoning_with_first_person_delib": {"abl": round(100 * sum(m["abl_fp_delib"] > 0 for m in M) / n, 1),
                                                  "src": round(100 * sum(m["src_fp_delib"] > 0 for m in M) / n, 1)},
        "pct_harm_template": rate("harm_template"),
        "harm_subject": dict(Counter(m["harm_subject"] for m in M)),
        "harm_subject_by_ai": {ai: dict(Counter(r["m"]["harm_subject"] for r in rows if r["ai"] == ai))
                               for ai in sorted({r["ai"] for r in rows})},
        "pct_new_numbers": round(100 * sum(bool(m["new_numbers"]) for m in M) / n, 1),
        "pct_dropped_prompt_numbers": round(100 * sum(bool(m["dropped_prompt_numbers"]) for m in M) / n, 1),
        "pct_new_capitalised": round(100 * sum(bool(m["new_capitalised"]) for m in M) / n, 1),
        "pct_responses_with_bullets": {"abl": round(100 * sum(m["abl_bullets"] > 0 for m in M) / n, 1),
                                       "src": round(100 * sum(m["src_bullets"] > 0 for m in M) / n, 1)},
        "pct_responses_with_question": {"abl": round(100 * sum(m["abl_questions"] > 0 for m in M) / n, 1),
                                        "src": round(100 * sum(m["src_questions"] > 0 for m in M) / n, 1)},
        "mean_you_should": {"abl": round(st.mean(m["abl_you_should"] for m in M), 2),
                            "src": round(st.mean(m["src_you_should"] for m in M), 2)},
        "contractions_per_100w": {"abl": med("abl_contractions_per_100w"), "src": med("src_contractions_per_100w")},
        "pct_mojibake_introduced": rate("mojibake"), "pct_unterminated": rate("abl_unterminated"),
        "top_first_words": Counter(" ".join(m["abl_first_sentence"].split()[:3]) for m in M).most_common(12),
    }
    (ROOT / "checks_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    random.seed(0)


if __name__ == "__main__":
    main()
