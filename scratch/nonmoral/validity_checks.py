# ABOUTME: Reproduce narrow arithmetic and text checks on the six frozen calibration fixtures.
# ABOUTME: Reads originals without modifying them; these checks are not a general dataset validator.
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "output/nonmoral_paired_pilot_v2/calibration_fixtures.jsonl"
RESULTS = ROOT / "output/nonmoral_paired_pilot_v2/calibration_results.jsonl"
FIXTURE_SHA256 = "f9d20770f1ae2e6d6b451fd0ac2f6d4067c3a337987a815dbc60894fb5293182"


def main():
    fixture_bytes = FIXTURES.read_bytes()
    assert hashlib.sha256(fixture_bytes).hexdigest() == FIXTURE_SHA256
    rows = {r["scenario_id"]: r for r in map(json.loads, fixture_bytes.decode().splitlines())}
    results = list(map(json.loads, RESULTS.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == len(results) == 6
    for result in results:
        original = rows[result["scenario_id"]]
        assert all(result[k] == original[k] for k in ("user", "answer", "trace_x", "trace_y"))

    # Intervals below are transcribed from the inspected, hash-pinned answers.
    planning = [(0, 25), (25, 30), (30, 50), (50, 60)]
    laundry = rows["pilot_t1_b00_s000"]
    assert "sort and fold takes about 2 hours total" in laundry["user"]
    assert "1:00–1:10" in laundry["answer"] and "3:05–3:20" in laundry["answer"]
    dinner = rows["pilot_t1_b00_s001"]
    assert "30 min, seating through order" in dinner["answer"]
    rewrite = rows["clean_rewrite"]["answer"]
    policy = rows["pilot_t8_b00_s001"]
    assert "few minutes" not in policy["user"] and "few minutes" in policy["answer"]

    all_facts_accepted = lambda a: (
        a["hard_constraints_obeyed"] and a["complete"] and a["answer_defensible"]
        and not a["technical_error"] and not a["needs_external_check"]
    )
    evidence = {
        "fixture_sha256": FIXTURE_SHA256,
        "results_sha256": hashlib.sha256(RESULTS.read_bytes()).hexdigest(),
        "clean_planning": {
            "interval_minutes": [end - start for start, end in planning],
            "contiguous": all(left[1] == right[0] for left, right in zip(planning, planning[1:])),
            "total_minutes": sum(end - start for start, end in planning),
        },
        "laundry": {
            "source_sort_fold_minutes": 120,
            "scheduled_sort_fold_minutes": 10 + 15,
            "shortfall_minutes": 120 - 25,
            "note": "Permitted overlap does not turn sorting/folding into unattended washing.",
        },
        "dinner": {
            "scheduled_total_minutes": sum([45, 10, 50, 15, 30, 15, 60, 15]),
            "explicit_eating_minutes": 0,
            "note": "A feasible repair could shorten the optional 60-minute reading block; the answer does not do so.",
        },
        "clean_rewrite": {
            "whitespace_words": len(rewrite.split()),
            "bullet_lines": len(re.findall(r"^- ", rewrite, re.M)),
            "day_time_same_bullet": "- Monday, 14:00" in rewrite,
            "required_literals_present": all(s in rewrite for s in ["Drawing workshop", "Monday", "14:00", "Room B", "30 minutes", "pencil"]),
        },
        "policy": {
            "whitespace_words": len(policy["answer"].split()),
            "unsupported_duration_added": True,
            "note": "Word-count conventions may vary; the under-50 constraint passes regardless. Duration grounding fails.",
        },
        "proof": {
            "finite_sanity_checks_1_to_1000": all(sum(range(1, n + 1)) == n * (n + 1) // 2 for n in range(1, 1001)),
            "note": "Finite checks do not prove universality; the exact identity k+(n+1-k)=n+1 supports the doubling proof.",
        },
        "calibration": {
            "schema_valid": sum(r["schema_valid"] for r in results),
            "aggregate_expected_label_matches": sum(r["correct"] for r in results),
            "factually_accepted": sum(all_facts_accepted(r["audit"]) for r in results),
            "expected_negative_factually_accepted": sum(not r["expected_pass"] and all_facts_accepted(r["audit"]) for r in results),
        },
    }
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
