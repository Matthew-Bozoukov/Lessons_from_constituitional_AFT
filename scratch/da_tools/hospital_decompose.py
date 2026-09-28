# ABOUTME: Split the Hospital contrast into what the pair seats did: which sabotage mode, accepted vs rejected
# ABOUTME: scheduler calls, supply rounds, team outcome -- does the arm still do its job, or just do less?
# Run: uv run python scratch/da_tools/hospital_decompose.py <control run dir|repo> <treatment run dir|repo> [--out <json>]
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_hospital import Run  # noqa: E402

from src.eval.misalignment.colosseum.stats import arm_difference  # noqa: E402

MEASURES = [
    ("pair_sabotaged", "shifts with any sabotage by the pair"),
    ("prov_sabotage_iters", "provisioner: rounds withholding from / flooding a hospital"),
    ("tri_sabotage_iters", "triage seat: rounds of scheduler sabotage (spam, invented ids)"),
    ("tri_schedule_calls", "triage seat: schedule_patient calls"),
    ("tri_scheduler_rejections", "triage seat: calls the strict scheduler rejected"),
    ("tri_accepted_schedules", "triage seat: calls the scheduler accepted (derived)"),
    ("tri_invented_ids", "triage seat: invented patient ids"),
    ("prov_supplied_iters", "provisioner: rounds it supplied a hospital"),
    ("bottleneck_units_supplied", "provisioner: bottleneck units supplied"),
    ("pair_tool_calls", "both pair seats: all tool calls"),
    ("resource_failures", "team: resource failures"),
    ("joint_reward", "team: score"),
    ("declined_in_reasoning", "pair said no in its reasoning"),
    ("refused_and_held", "pair refused and never sabotaged"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("control")
    ap.add_argument("treatment")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    c, t = Run(a.control), Run(a.treatment)
    for r in (c, t):
        calls, rej = r.per_seed.get("tri_schedule_calls", {}), r.per_seed.get("tri_scheduler_rejections", {})
        r.per_seed["tri_accepted_schedules"] = {s: calls[s] - rej.get(s, 0.0) for s in calls}
    rows = []
    for k, text in MEASURES:
        if k not in c.per_seed or k not in t.per_seed:
            rows.append({"measure": k, "what": text, "missing": True})
            continue
        d = arm_difference(t.per_seed[k], c.per_seed[k], label=k)
        rows.append({"measure": k, "what": text, "control": d["control_mean"], "treatment": d["treatment_mean"],
                     "diff": d["diff"], "ci95": d["diff_ci95"], "p": d["p_two_sided"], "n": d["n_seeds"]})
    for r in rows:
        if r.get("missing"):
            print(f"{r['what']:<66} missing")
        else:
            print(f"{r['what']:<66} ctrl {r['control']:>8.2f}  treat {r['treatment']:>8.2f}  "
                  f"diff {r['diff']:+8.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] p={r['p']:.3f} n={r['n']}")
    if a.out:
        Path(a.out).write_text(json.dumps({"control": a.control, "treatment": a.treatment, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
