# ABOUTME: First cut of "do the models know the sabotage is wrong": per Hospital cell, how often a seat that took a
# ABOUTME: sabotage act had privately declined first, from the labels already published. Run: uv run python scratch/hospital_knows_wrong/first_cut.py
"""No new judging. Reads each cell's published `results/episodes.json` (pinned revision) and tabulates the
per-seat class that `src/eval/misalignment/colosseum/hospital/actions.py::classify` already assigned from
the autorater's per-iteration `declined` label and the rule's sabotage acts.

The autorater's `declined` is a DECISION not to sabotage. It is stricter than "recognised the act as wrong",
so these counts are a lower bound on recognition, not the answer to the question.
"""

import json
import os
from collections import Counter
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

ORG = "dougalldeepmind"
OUT = Path("output/hospital_knows_wrong")

# (version, arm, repo, pinned revision). "multi" seed-a cells use the same three adapters and the same
# public-board setting as the bridge and solo cells; the b/c cells are other training seeds of the arm.
CELLS = [
    ("multi", "control", "2026-10-11-hospital-self-sacrificial-qwen36-0-plain", "564a0df6e12ce24831543baed888cebac2b3d690"),
    ("multi", "da", "2026-10-15-hospital-self-sacrificial-qwen36-0-da-15", "962ccc36bfe351df3d8f7dc394a1214f86a1e7b6"),
    ("multi", "dat", "2026-10-14-hospital-self-sacrificial-qwen36-0-da-tools-15", "131eb1efdf3e2e626423b40c84d5a51d04c12b8a"),
    ("multi", "da (seed b)", "2026-11-05-hospital-self-sacrificial-qwen36-0-da-15-b", "59e946e12d9314c1304ae1d7eb168057fd6346da"),
    ("multi", "da (seed c)", "2026-11-05-hospital-self-sacrificial-qwen36-0-da-15-c", "5c58b2c00392c02ea1def0bdb1a9e5e5bfcc7ef2"),
    ("multi", "dat (seed b)", "2026-11-06-hospital-self-sacrificial-qwen36-0-da-tools-15-b", "f00cdd1dee896560e78ea6baa3dc33f648efbca7"),
    ("multi", "dat (seed c)", "2026-11-06-hospital-self-sacrificial-qwen36-0-da-tools-15-c", "edfd21e8194e12ba1a6970fc5f5a27c70f054b50"),
    ("bridge", "control", "2026-11-12-hospital-bridge-qwen36-0-plain", "6fa1d804e73c0b1354aa14914e3c21d15b79d8f6"),
    ("bridge", "da", "2026-11-12-hospital-bridge-qwen36-0-da-15", "44c2d661b8724e08ffea726e206599948bd1742e"),
    ("bridge", "dat", "2026-11-12-hospital-bridge-qwen36-0-da-tools-15", "c2c28ae42e4cb75244849897c2a4bd1033d52fb2"),
    ("solo", "control", "2026-11-12-hospital-solo-qwen36-0-plain", "42ac997cf0c072a81e4dc452f9478ebae4b970e8"),
    ("solo", "da", "2026-11-12-hospital-solo-qwen36-0-da-15", "8eb9d69bc13bd94c4c4056e4979e6b0dfd7a71a9"),
    ("solo", "dat", "2026-11-12-hospital-solo-qwen36-0-da-tools-15", "f0b6b77f504c77dcedd49b03c03236eddc7f5d10"),
    ("solo, no instruction", "control", "2026-11-12-hospital-solo-baseline-qwen36-0-plain", "dcdd4ef06e0889dbcb4c5696c2fcddbeab9eec21"),
]

# A seat that took a sabotage act falls in exactly one of these (classify() in actions.py).
ACTED_CLASSES = ["complied", "declined-but-sabotaged", "repented"]
HEADERS = {
    "complied": "never declined",
    "declined-but-sabotaged": "declined, then acted",
    "repented": "acted, declined only later",
}


def episodes(repo: str, revision: str) -> list[dict]:
    path = hf_hub_download(
        f"{ORG}/{repo}", "results/episodes.json", repo_type="dataset", revision=revision,
        token=os.environ.get("HF_TOKEN"), local_dir=OUT / "cells" / repo,
    )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seat_row(eps: list[dict], iters_key: str, class_key: str) -> dict | None:
    """Counts over the shifts in which this seat took at least one sabotage act (rule)."""
    if all(e.get(class_key) is None for e in eps):
        return None
    acted = [e for e in eps if (e.get(iters_key) or 0) > 0]
    classes = Counter(e[class_key] for e in acted)
    assert set(classes) <= set(ACTED_CLASSES), f"unexpected class for an acting seat: {classes}"
    return {"shifts": len(eps), "acted": len(acted), **{c: classes.get(c, 0) for c in ACTED_CLASSES}}


def table(rows: list[tuple[str, str, dict]]) -> str:
    head = ["version", "arm", "shifts", "seat took a sabotage act (rule)"] + [HEADERS[c] for c in ACTED_CLASSES]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for version, arm, r in rows:
        cells = [version, arm, r["shifts"], r["acted"]] + [r[c] for c in ACTED_CLASSES]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines)


def main() -> None:
    load_dotenv(".env")
    prov, dept = [], []
    for version, arm, repo, revision in CELLS:
        eps = episodes(repo, revision)
        assert not any(e.get("unlabelled") for e in eps), f"{repo} has unlabelled shifts"
        assert {e["text_reader"] for e in eps} == {"autorater"}, f"{repo} was not read by the autorater"
        prov.append((version, arm, seat_row(eps, "prov_sabotage_iters", "class_prov")))
        d = seat_row(eps, "tri_sabotage_iters", "class_tri")
        if d and version == "multi":  # the department seat is a live model only in the multi-agent cells
            dept.append((version, arm, d))
    sources = "\n".join(f"- {v}, {a}: `{ORG}/{r}` @ `{s[:8]}`" for v, a, r, s in CELLS)
    md = f"""# Hospital: did a sabotaging seat privately decline first? (existing labels only)

Produced {date.today().isoformat()} by `scratch/hospital_knows_wrong/first_cut.py`. No new judging.

Each row counts shifts. "Seat took a sabotage act (rule)" is the deterministic action rule. The three columns
after it split those shifts by the autorater's reading of the seat's private reasoning: it never decided to
decline the sabotage; it decided to decline at some iteration and took a sabotage act at that or a later
iteration; or it acted first and declined only afterwards.

"Declined" is a decision not to sabotage because it is wrong or harmful. A seat that notes the act is wrong
and goes ahead without deciding to decline is counted under "never declined", so these are lower bounds on
recognition.

## Resource Provisioner (a live model in every version)

{table(prov)}

## General Hospital Triage (a live model only in the multi-agent version)

{table(dept)}

## Sources

{sources}
"""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{date.today().isoformat()}_hospital_declined_before_act_results.md"
    path.write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
