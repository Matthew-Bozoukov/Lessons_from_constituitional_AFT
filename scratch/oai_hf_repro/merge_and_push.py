# ABOUTME: Merges oai_hf run dirs that each ran a subset of the steps (one pod per step, all --no-push)
# ABOUTME: into ONE run in the contract layout, rebuilt from their Inspect logs, and pushes it once.
# Run: uv run python scratch/oai_hf_repro/merge_and_push.py <out_dir> <part_run_dir> [<part_run_dir> ...] [--push]
"""Why this exists: run_eval drives one vLLM server per invocation, and on 2026-11-13 one
H200 could not serve all four steps without slowing every agent ~3x (prefill-bound), which
would have let the task wall clocks, not the model, decide how far runs got. So each step
ran as its own `uv run evals ... only_steps=[stepN] --no-push` against its own pod. All
parts share one target, mode, harness commit and judge; this script checks that, rebuilds
rollouts/results from the union of their Inspect logs with the eval's own exporter, keeps
every part's run_meta.json, and publishes under the name run_eval would have used, with
run_eval's own card fields and tags.
"""

import json
import shutil
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log
from omegaconf import OmegaConf

from src.eval.layout import assert_layout, publish_layout, run_tags
from src.eval.misalignment.oai_hf.export import export
from src.eval.run_eval import _card_fields, _results_markdown, _run_repo
from src.infra.huggingface import hf_repo_id, push_run_dir

STEP = {
    "step1_gdrive_xlsx": "step1",
    "step2_reachout": "step2",
    "step3_egress_share": "step3",
    "step4_hf_cheat_chain": "step4",
}


def main(argv: list[str]) -> None:
    push = "--push" in argv
    args = [a for a in argv if a != "--push"]
    out_dir, parts = Path(args[0]), [Path(p) for p in args[1:]]
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; pick a new merged dir")
    metas = [json.loads((p / "metadata" / "run_meta.json").read_text()) for p in parts]
    same = (
        "target",
        "target_revision",
        "mode",
        "base_model",
        "base_model_revision",
        "model_key",
    )
    for k in same:
        vals = {str(m.get(k)) for m in metas}
        assert len(vals) == 1, f"parts disagree on {k}: {vals}"
    for k in ("harness", "judge_model"):
        vals = {
            json.dumps(
                OmegaConf.create(m["config"]).get(k)
                and OmegaConf.to_container(OmegaConf.create(m["config"])[k])
                if k == "harness"
                else m["config"][k]
            )
            for m in metas
        }
        assert len(vals) == 1, f"parts disagree on {k}: {vals}"

    rollout_dir, results_dir, metadata_dir = publish_layout(out_dir)
    step_logs: dict[str, list] = {}
    for p in parts:
        for f in sorted((p / "metadata" / "inspect").glob("*.eval")):
            log = read_eval_log(str(f))
            step = STEP[log.eval.task.split("/")[-1]]
            assert step not in step_logs or not any(
                x.eval.task == log.eval.task for x in step_logs[step]
            ), f"{step} appears in two parts"
            step_logs.setdefault(step, []).append(log)
            dest = metadata_dir / "inspect" / p.name
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest / f.name)
    m0 = metas[0]
    cfg0 = OmegaConf.create(m0["config"])
    run_info = {
        "target": f"openai-api/vllm/{m0['model_key'].replace('-', '_')}",
        "mode": m0["mode"],
        "judge_model": str(cfg0.judge_model),
        "harness": OmegaConf.to_container(cfg0.harness),
        "merged_from": [str(p) for p in parts],
        "parts": [
            {
                "run_dir": str(p),
                "command": m.get("command"),
                "only_steps": m["config"].get("only_steps"),
                "inspect": m["config"].get("inspect"),
                "serving": m["config"].get("serving"),
            }
            for p, m in zip(parts, metas)
        ],
        "inspect_logs": "metadata/inspect",
    }
    summary = export(step_logs, rollout_dir, results_dir, run_info)
    summary = {"target": m0["target"], "mode": m0["mode"], **summary}
    (results_dir / "results.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    (results_dir / "results.md").write_text(
        _results_markdown(m0["target"], m0["mode"], summary)
    )
    merged_meta = {
        **m0,
        "command": " && ".join(str(m.get("command")) for m in metas),
        "parts": run_info["parts"],
    }
    (metadata_dir / "run_meta.json").write_text(
        json.dumps(merged_meta, indent=2, default=str)
    )
    for p, m in zip(parts, metas):
        (metadata_dir / f"run_meta_{p.name}.json").write_text(
            json.dumps(m, indent=2, default=str)
        )
    assert_layout(out_dir)
    print(f">>> merged {len(parts)} parts -> {out_dir}: steps {sorted(step_logs)}")
    if not push:
        return
    repo_id = hf_repo_id(_run_repo("oai_hf", m0["model_key"], ""))
    card = _card_fields(
        "oai_hf",
        cfg0,
        merged_meta["command"],
        experiment=f"oai_hf eval of {m0['target']} (mode={m0['mode']}); steps run on separate pods and merged",
        models=json.dumps(
            {
                "target": m0["target"],
                "target_revision": m0["target_revision"],
                "base": m0["base_model"],
                "base_revision": m0["base_model_revision"],
            }
        ),
        source_revision=m0["git_sha"],
    )
    url = push_run_dir(
        out_dir,
        repo_id,
        card,
        front_matter={"tags": run_tags("oai_hf", m0["model_key"], m0["mode"])},
    )
    print(f">>> pushed {url}")


if __name__ == "__main__":
    main(sys.argv[1:])
