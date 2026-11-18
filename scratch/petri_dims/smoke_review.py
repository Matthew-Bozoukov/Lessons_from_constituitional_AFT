# ABOUTME: Writes a human-readable review of a Petri validation smoke: every validator reply next to the blind label,
# ABOUTME: plus one complete validator prompt exactly as sent. Run: uv run python scratch/petri_dims/smoke_review.py
import glob
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.eval.misalignment.petri import validate as v

SMOKE = "output/petri_validation_smoke"
LABELS = "output/petri/2026-11-18_petri_flag_audit/labels_joined.jsonl"

blind = {(j["run"], j["seed"]): j for j in map(json.loads, open(LABELS))}
out = [
    "# Petri validation smoke (first 5 flagged audits per arm): every validator reply",
    "",
]
for f in sorted(glob.glob(f"{SMOKE}/*/results/validation.jsonl")):
    run = Path(f).parts[-3][11:]
    for r in map(json.loads, open(f)):
        b = blind.get((run, r["seed"]), {})
        out += [
            f"## {b.get('arm', '?')} | {run} | {r['seed']}",
            f"final **{r['final']}** (round {r['round']}, execution slip {r['execution_slip']}) | blind label "
            f"{b.get('label')} ({b.get('severity')}): {b.get('what_happened', '')}",
            "",
        ]
        for name, reps in (("Round 1", r["round1"]), ("Round 2", r["round2"])):
            for rep in reps:
                out.append(
                    f"- {name} {rep['verdict']} (slip {rep['execution_slip']}, lines {rep.get('constitution_lines')}): "
                    f'{rep.get("rationale", "")} | quote: "{rep.get("key_quote", "")}"'
                )
        out.append("")
Path(f"{SMOKE}/SMOKE_REVIEW.md").write_text("\n".join(out))

ctrl = f"{SMOKE}/2026-11-18_2026-11-17-petri-qwen36-0-plain/results/validation.jsonl"
first = json.loads(open(ctrl).readline())
transcript = Path(
    hf_hub_download(
        "dougalldeepmind/2026-11-17-petri-qwen36-0-plain",
        f"rollouts/{first['seed']}.md",
        repo_type="dataset",
    )
).read_text()
constitution = Path(
    "constitutions/claude_distilled_09_principles/constitution.md"
).read_text()
Path(f"{SMOKE}/EXAMPLE_PROMPT.md").write_text(
    "# SYSTEM\n\n"
    + v.SYSTEM.format(constitution=v.numbered(constitution))
    + "\n\n# USER\n\n"
    + v.ROUND1.format(
        principles=v.PRINCIPLES, tenet=first["tenet"], transcript=transcript
    )
)
print("written", first["seed"])
