# ABOUTME: One-off for the 2026-09-24 DA arm: merge the first attempt's finished pushy_base condition
# ABOUTME: into the same_self rerun's packaged run dir and publish ONE run under the contract.
"""The DA arm ran twice: attempt one finished and packaged pushy_base (120/120 real cells) before
the laptop slept and the pod died; the rerun did same_self only, with --no-push. This puts the two
conditions into one run dir — the layout src/eval/layout.py enforces — rebuilds the summary the
way peer/runner.py would have, and pushes it with the card and tags run_eval would have written.

    uv run python scratch/odcv_peer_merge_publish.py <same_self run dir> <pushy run dir> [--push]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from omegaconf import OmegaConf

from src.eval.layout import assert_layout
from src.eval.misalignment.odcv.peer import facts as team_facts
from src.eval.misalignment.odcv.peer.runner import cell_id, seat_string
from src.eval.run_eval import _card_fields, _results_markdown
from src.infra.huggingface import hf_repo_id
from src.naming import eval_name

NAME = "odcv_peer"


def condition_summary(
    run: Path, name: str, cond_cfg, model_key: str, model_name: str
) -> dict:
    """What peer/runner.py puts under summary['conditions'][name], from the packaged files."""
    res = run / "results" / name
    results = json.loads((res / "results.json").read_text())
    progress = (
        json.loads((res / "progress_results.json").read_text())
        if (res / "progress_results.json").is_file()
        else {}
    )
    tr = json.loads((res / "team_results.json").read_text())
    cells = tr["cells"]
    passes = sorted({p.name for p in (run / "rollouts" / name).glob("*/*/pass*")})
    return {
        "label": str(cond_cfg.get("label", name)),
        "cell_ids": sorted(
            {c.get("cell_id") or cell_id(model_key, name, c["variant"]) for c in cells}
        ),
        "seats": sorted({f"{c.get('seat1')} + {c.get('seat2')}" for c in cells}),
        "teammate": tr["teammate"],
        "variants": list(cond_cfg.variants),
        "measured_prompt": str(cond_cfg.get("measured_prompt", "cell")),
        "measured": results["ours"],
        "n_judged": results["n_judged"],
        "progress": progress.get("ours"),
        "submission": None,
        "team": tr["metrics"],
        "by_variant": tr.get("by_variant"),
        "passes": {
            "requested": len(passes),
            "kept": len(passes),
            "n_transcripts": len(cells),
        },
    }


def main(new: Path, old: Path, push: bool) -> None:
    meta = json.loads((new / "metadata" / "run_meta.json").read_text())
    cfg = OmegaConf.create(meta["config"])
    conds = {str(c.name): c for c in cfg.conditions}
    model_key = str(cfg.model_key)
    model_name = str(cfg.model)

    # 1. pushy_base's three published parts move in beside same_self's.
    for part in ("rollouts", "results", "metadata"):
        src, dst = old / part / "pushy_base", new / part / "pushy_base"
        assert src.is_dir(), src
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print("copied", part + "/pushy_base")

    # 2. the run summary, as the runner builds it, over both conditions.
    summary = json.loads((new / "results" / "results.json").read_text())
    summary["conditions"] = {
        n: condition_summary(new, n, conds[n], model_key, model_name)
        for n in ("pushy_base", "same_self")
    }
    pooled = []
    for n in summary["conditions"]:
        tr = json.loads((new / "results" / n / "team_results.json").read_text())
        if tr["teammate"].get("kind") == "live":
            pooled += tr["cells"]
    summary["team_pooled"] = team_facts.metrics(
        pooled, "live-teammate conditions pooled"
    )
    summary["merged_from"] = {
        "pushy_base": str(old),
        "same_self": str(new),
        "note": "attempt one finished pushy_base, then its pod died under a sleeping driver; "
        "same_self was rerun on a second pod (2026-09-24)",
    }
    (new / "results" / "results.json").write_text(json.dumps(summary, indent=2))
    (new / "results" / "results.md").write_text(
        _results_markdown(meta["target"], meta["mode"], summary)
    )
    assert_layout(new)
    print("layout ok; conditions:", list(summary["conditions"]))
    for n, s in summary["conditions"].items():
        print(
            f"  {n}: MR {s['measured']['overall']['mr_pct']}% | hit {s['team']['hit_rate']['pct']}% | "
            f"misalignment {s['team']['misalignment_rate']['pct']}% | own {s['team']['own_violation_rate']['pct']}%"
        )

    # 3. the card and tags run_eval writes, then the push.
    card = _card_fields(
        NAME,
        cfg,
        meta["command"],
        experiment=f"{NAME} eval of {meta['target']} (mode={meta['mode']})",
        models=json.dumps(
            {
                "target": meta["target"],
                "target_revision": meta["target_revision"],
                "base": meta["base_model"],
                "base_revision": meta["base_model_revision"],
            }
        ),
        source_revision=meta["git_sha"],
    )
    tags = ["eval-run", f"eval:{NAME}", f"model:{model_key}", f"mode:{meta['mode']}"]
    repo_id = hf_repo_id(eval_name(NAME, model_key))
    (new / "metadata" / "card.json").write_text(
        json.dumps({"repo_id": repo_id, "fields": card, "tags": tags}, indent=2)
    )
    print("repo:", repo_id)
    if push:
        from src.infra.huggingface import push_run_dir

        print(
            ">>> pushed", push_run_dir(new, repo_id, card, front_matter={"tags": tags})
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), "--push" in sys.argv)
