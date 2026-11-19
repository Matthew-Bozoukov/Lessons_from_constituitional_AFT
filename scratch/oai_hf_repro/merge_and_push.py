# ABOUTME: Merges oai_hf run dirs that each ran a subset of the steps (pods per step, Docker crashes in
# ABOUTME: between) into ONE run of `n` complete runs per step, in the contract layout, and pushes it once.
# Run: uv run python scratch/oai_hf_repro/merge_and_push.py <out_dir> <part_dir> [<part_dir> ...] [--n 64] [--push]
"""Why this exists (2026-11-13): run_eval drives one vLLM server per invocation, and one H200
could not serve all four steps without slowing every agent ~3x, so each step ran as its own
`uv run evals ... only_steps=[...] --no-push` against its own pod. Docker Desktop on the
driver laptop then died twice under load, killing every run in flight, so each step's runs
come from several invocations.

Selection rule, fixed before looking at any result: parts are taken in the order given
(chronological); within a part, samples in epoch order (the order Inspect ASSIGNED them,
not the order they finished); only samples that ended without an error are kept; the first
`n` per step are taken. A sample cut off by a crash is dropped, not counted as a refusal.
Each kept run is renumbered epoch 1..n within its step, and `provenance.jsonl` maps every
renumbered run back to its source log and original epoch.

Parts must share target, revision, mode, base model, harness commit and judge (checked).
Rollouts and results are rebuilt with the eval's own exporter; every part's run_meta.json is
kept; the push uses run_eval's own name, card fields and tags.
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
SAME = (
    "target",
    "target_revision",
    "mode",
    "base_model",
    "base_model_revision",
    "model_key",
)


def main(argv: list[str]) -> None:
    push = "--push" in argv
    n_cap = int(argv[argv.index("--n") + 1]) if "--n" in argv else 64
    args = [
        a
        for i, a in enumerate(argv)
        if a != "--push" and a != "--n" and (i == 0 or argv[i - 1] != "--n")
    ]
    out_dir, parts = Path(args[0]), [Path(p) for p in args[1:]]
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; pick a new merged dir")
    metas = [
        json.loads(
            next(
                f
                for f in (p / "metadata" / "run_meta.json", p / "run_meta.json")
                if f.exists()
            ).read_text()
        )
        for p in parts
    ]
    for k in SAME:
        vals = {str(m.get(k)) for m in metas}
        assert len(vals) == 1, f"parts disagree on {k}: {vals}"
    for k in ("harness", "judge_model"):
        vals = {json.dumps(m["config"][k], sort_keys=True) for m in metas}
        assert len(vals) == 1, f"parts disagree on {k}: {vals}"

    rollout_dir, results_dir, metadata_dir = publish_layout(out_dir)
    kept: dict[str, list] = {}  # step -> [(log, sample, source)]
    logs_by_step: dict[str, list] = {}
    dropped = {}
    for p in parts:
        for f in sorted((p / "metadata" / "inspect").glob("*.eval")):
            log = read_eval_log(str(f))
            step = STEP[log.eval.task.split("/")[-1]]
            dest = metadata_dir / "inspect" / p.name
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest / f.name)
            for s in sorted(log.samples or [], key=lambda s: (s.epoch, str(s.id))):
                if s.error is not None:
                    dropped[step] = dropped.get(step, 0) + 1
                    continue
                if len(kept.get(step, [])) >= n_cap:
                    continue
                kept.setdefault(step, []).append((log, s, f"{p.name}/{f.name}"))
    provenance = []
    for step, items in kept.items():
        # one EvalLog per step, its samples the kept ones renumbered 1..n
        base = items[0][0]
        samples = []
        for i, (log, s, src) in enumerate(items, start=1):
            samples.append(s.model_copy(update={"epoch": i}))
            provenance.append(
                {
                    "step": step,
                    "epoch": i,
                    "source": src,
                    "source_epoch": s.epoch,
                    "source_task_args": log.eval.task_args,
                }
            )
        logs_by_step[step] = [base.model_copy(update={"samples": samples})]
    with (metadata_dir / "provenance.jsonl").open("w") as fh:
        for r in provenance:
            fh.write(json.dumps(r, default=str) + "\n")

    m0 = metas[0]
    cfg0 = OmegaConf.create(m0["config"])
    counts = {s: len(v) for s, v in kept.items()}
    run_info = {
        "target": (
            f"openrouter/{m0['target'].split(':', 1)[1]}"
            if ":" in str(m0["target"])
            else f"openai-api/vllm/{m0['model_key'].replace('-', '_')}"
        ),
        "mode": m0["mode"],
        "judge_model": str(cfg0.judge_model),
        "harness": OmegaConf.to_container(cfg0.harness),
        "kept_per_step": counts,
        "dropped_errored_per_step": dropped,
        "selection_rule": "parts in order given; within a part by assigned epoch; error-free only; first n per step",
        "parts": [
            {
                "run_dir": str(p),
                "command": m.get("command"),
                "only_steps": m["config"].get("only_steps"),
                "n": m["config"].get("n"),
                "inspect": m["config"].get("inspect"),
                "serving": m["config"].get("serving"),
            }
            for p, m in zip(parts, metas)
        ],
        "inspect_logs": "metadata/inspect (every part's full log, including dropped samples)",
    }
    summary = export(logs_by_step, rollout_dir, results_dir, run_info)
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
    print(
        f">>> merged {len(parts)} parts -> {out_dir}: kept {counts}, dropped (errored) {dropped}"
    )
    short = {s: c for s, c in counts.items() if c < n_cap}
    if short:
        print(f"!!! fewer than {n_cap} complete runs for {short}")
    if not push:
        return
    repo_id = hf_repo_id(_run_repo("oai_hf", m0["model_key"], ""))
    card = _card_fields(
        "oai_hf",
        cfg0,
        merged_meta["command"],
        experiment=(
            f"oai_hf eval of {m0['target']} (mode={m0['mode']}); merged from {len(parts)} "
            f"run parts: {', '.join(p.name for p in parts)} ({run_info['selection_rule']})"
        ),
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
