# ABOUTME: Audits and verifies publication of the two remaining DA ODCV evaluations.
# ABOUTME: Can publish a locally completed run without regenerating or rejudging any cell.
import argparse
import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd
from omegaconf import OmegaConf

from src.eval.layout import assert_layout
from src.eval.misalignment.odcv.odcv import VARIANTS, scenario_names
from src.eval.run_eval import _card_fields, _run_repo
from src.infra import runpod
from src.infra.huggingface import hf_api, hf_repo_id, push_run_dir


def main(key, publish=False):
    plan = OmegaConf.load("scratch/da_supervision/odcv_remaining_plan.yaml")
    arm = next(a for a in plan.arms if a.key == key)
    owner_dir = Path(plan.output_root) / key
    state = json.loads((owner_dir / "status.json").read_text())
    assert not runpod._parent_alive(state["pid"]), (
        "Wait for the owner to finish before publishing"
    )
    assert state.get("terminated"), "Owner must verify teardown first"
    assert state["owned_pod"] not in {p["id"] for p in runpod.active_pods()}
    roots = list((owner_dir / "eval").glob("*"))
    assert len(roots) == 1
    root = roots[0]
    assert_layout(root)
    meta = json.loads((root / "metadata/run_meta.json").read_text())
    result = json.loads((root / "results/results.json").read_text())
    progress = json.loads((root / "results/progress_results.json").read_text())
    cfg = meta["config"]
    assert meta["target"] == arm.target and meta["target_revision"] == arm.revision
    assert meta["base_model_revision"] == arm.base_revision and meta["mode"] == "think"
    cot_receipt = json.loads(
        Path(
            "output/da_supervision/odcv_2026-09-16/cot/publication_receipt.json"
        ).read_text()
    )
    cot_meta = json.loads(
        (Path(cot_receipt["run_root"]) / "metadata/run_meta.json").read_text()
    )
    for field in (
        "temperature",
        "passes",
        "judges",
        "progress_judges",
        "serving",
        "scenario_timeout_s",
    ):
        assert cfg[field] == cot_meta["config"][field], field
    expected = {
        (v, s) for v in VARIANTS for s in scenario_names(Path(cfg["bench_dir"]), v)
    }
    assert len(expected) == 80
    rows = []
    for f in sorted((root / "rollouts").rglob("messages_record.txt")):
        v, scenario, pass_name, _ = f.relative_to(root / "rollouts").parts
        assert pass_name == "pass1"
        data = f.read_bytes()
        text = data.decode("utf-8")
        assert "role: assistant" in text
        notes = [line for line in text.splitlines() if "[Harness]" in line]
        rows.append(
            dict(
                variant=v,
                scenario=scenario,
                bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                budget_notes=[
                    n
                    for n in notes
                    if "token limit" in n or "finish_reason=length" in n
                ],
                timeout=any(
                    "timeout" in n.lower() or "timed out" in n.lower() for n in notes
                ),
            )
        )
    assert len(rows) == 80 and {(r["variant"], r["scenario"]) for r in rows} == expected
    assert result["n_judged"] == progress["n_judged"] == 80
    assert result["n_dropped_all_na"] == progress["n_dropped_all_na"] == 0
    assert result["passes"]["requested"] == result["passes"]["kept"] == 1
    assert result["passes"]["n_transcripts"] == 80
    assert all(
        a["missing_cells"] == 0 and a["clean"] for a in result["passes"]["audits"]
    )
    audit = dict(
        verified_epoch=time.time(),
        expected_cells=80,
        actual_cells=80,
        missing=0,
        budget_cutoff_count=sum(bool(r["budget_notes"]) for r in rows),
        timeout_count=sum(r["timeout"] for r in rows),
        rows=rows,
    )
    gpu_upper = (
        (state["termination_verified_epoch"] - state["created_epoch"])
        / 3600
        * state["hourly_usd"]
    )
    cost = dict(
        gpu_usd_upper_estimate=gpu_upper,
        caveat="Allocation-to-final-teardown-verification upper estimate; GPU was released before local judging. Excludes storage. Shared OpenRouter account usage deltas in original results are not attributable costs for this run.",
        original_usage_fields_preserved=True,
    )
    for name, data in [
        ("generation_audit.json", audit),
        ("cost_accounting.json", cost),
    ]:
        (root / "metadata" / name).write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
    api = hf_api()
    repo = hf_repo_id(_run_repo("odcv", result["model_key"], ""))
    if publish:
        card = _card_fields(
            "odcv",
            OmegaConf.create(cfg),
            meta["command"],
            experiment=f"odcv eval of {meta['target']} (mode=think)",
            models=json.dumps(
                dict(
                    target=meta["target"],
                    target_revision=meta["target_revision"],
                    base=meta["base_model"],
                    base_revision=meta["base_model_revision"],
                )
            ),
            source_revision=meta["git_sha"],
        )
        card["cost_accounting"] = cost["caveat"]
        push_run_dir(
            root,
            repo,
            card,
            front_matter={
                "tags": [
                    "eval-run",
                    "eval:odcv",
                    f"model:{result['model_key']}",
                    "mode:think",
                ]
            },
        )
    else:
        card_path = root / "README.md"
        card = card_path.read_text(encoding="utf-8")
        if cost["caveat"] not in card:
            card_path.write_text(
                card + "\nCost accounting: " + cost["caveat"] + "\n", encoding="utf-8"
            )
        files = [
            "README.md",
            "metadata/generation_audit.json",
            "metadata/cost_accounting.json",
        ]
        api.create_commit(
            repo_id=repo,
            repo_type="dataset",
            commit_message="Add transcript and cost attribution audits",
            operations=[
                CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(root / f))
                for f in files
            ],
        )
    info = api.dataset_info(repo, files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings}
    files = list((root / "rollouts").rglob("messages_record.txt"))
    files += [
        root / f
        for f in (
            "metadata/run_meta.json",
            "results/results.json",
            "results/progress_results.json",
            "metadata/generation_audit.json",
            "metadata/cost_accounting.json",
            "README.md",
        )
    ]
    for f in files:
        data = f.read_bytes()
        r = remote[f.relative_to(root).as_posix()]
        assert r.size == len(data), str(f)
        if r.lfs:
            assert r.lfs.sha256 == hashlib.sha256(data).hexdigest(), str(f)
        else:
            assert (
                r.blob_id
                == hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            ), str(f)
    receipt = dict(
        verified_epoch=time.time(),
        repo=repo,
        revision=info.sha,
        transcript_hashes_verified=80,
        owned_pod_gone=True,
        run_root=str(root),
        mr=result["ours"],
        progress=progress["ours"],
        submission=result["submission"],
        passes=result["passes"],
        budget_cutoff_count=audit["budget_cutoff_count"],
        timeout_count=audit["timeout_count"],
        gpu_usd_upper_estimate=gpu_upper,
    )
    (owner_dir / "publication_receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in receipt.items()
                if k not in ("mr", "progress", "submission", "passes")
            }
        )
    )
    print(
        "MR",
        result["ours"]["overall"]["mr_pct"],
        "TP",
        progress["ours"]["overall"]["tp_mean"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=["answer", "empty"])
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    main(args.arm, publish=args.publish)
