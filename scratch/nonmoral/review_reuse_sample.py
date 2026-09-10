# ABOUTME: Verify frozen historical-example selection and reproduce local reuse-review counts.
# ABOUTME: Reads cached corpus/training bytes and manual decisions; makes no network or model calls.
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/nonmoral_investigation/20260908/reuse_sample"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    freeze = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    sample_path = OUT / "selected_rows.jsonl"
    assert digest(sample_path) == freeze["rows_sha256"]
    corpus_path = ROOT / freeze["corpus_path"]
    mixture_path = Path(freeze["mixture_path"])
    assert digest(corpus_path) == freeze["corpus_sha256"]
    assert digest(mixture_path) == freeze["mixture_sha256"]
    corpus = [json.loads(x) for x in corpus_path.read_text(encoding="utf-8").splitlines()]
    mixture = [json.loads(x) for x in mixture_path.read_text(encoding="utf-8").splitlines()]
    trained = {r["scenario_id"]: r for r in mixture if r["source"] == "nonmoral_deliberation"}
    assert len(trained) == 684
    selected = []
    for domain in freeze["domains"]:
        eligible = [r for r in corpus if r["metadata"]["domain"] == domain
                    and r["metadata"]["scenario_id"] in trained]
        eligible.sort(key=lambda r: hashlib.sha256(
            (freeze["salt"] + r["metadata"]["scenario_id"]).encode()).hexdigest())
        selected.extend(r["metadata"]["scenario_id"] for r in eligible[:4])
    assert selected == freeze["ids"]
    samples = {r["scenario_id"]: r for r in map(json.loads, sample_path.read_text(encoding="utf-8").splitlines())}
    for sid, row in samples.items():
        assert row["training_row"] == trained[sid]
        messages = row["corpus_row"]["messages"]
        assert all(m["content"] in trained[sid]["text"] for m in messages)
        assert messages[-1]["reasoning_content"] in trained[sid]["text"]
    decisions = json.loads((OUT / "decisions.json").read_text(encoding="utf-8"))["records"]
    assert {r["scenario_id"] for r in decisions} == set(selected)
    answer = lambda sid: samples[sid]["corpus_row"]["messages"][-1]["content"]
    result = {
        "original_denominator": 12,
        "source_identity_verified": 12,
        "prompt_completion": dict(Counter(r["prompt_completion"] for r in decisions)),
        "strict_pair_prompt": dict(Counter(r["strict_pair_prompt"] for r in decisions)),
        "unchanged_final_reusable": sum(r["unchanged_final_reusable"] for r in decisions),
        "unchanged_comparative_cot_approved": sum(r["unchanged_comparative_cot_approved"] for r in decisions),
        "unchanged_control_cot_approved": sum(r["unchanged_control_cot_approved"] for r in decisions),
        "final_contains_comparison": sum(r["final_contains_comparison"] for r in decisions),
        "mechanical_checks": {
            "creator_settings_bullets": len(re.findall(r"^- ", answer("t7_b03_s004"), re.M)),
            "profile_requested_characters": len("Controls visibility of your profile."),
            "profile_proposed_characters": len("Hidden profiles are excluded from search, but still open via direct link."),
            "game_forbidden_terms_in_final": {s: answer("t7_b04_s007").lower().count(s)
                for s in ("coalition", "victory points", "binding agreement")},
            "game_scoring_placeholders": len(re.findall(r"\[(?:game-specific source [12]|bonus structure)\]", answer("t7_b04_s007"))),
            "requested_validate_categories": 11,
        },
        "note": "Eligibility counts summarize manual evidence, not automatically proven quality. No ODCV information enters selection.",
    }
    (OUT / "checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
