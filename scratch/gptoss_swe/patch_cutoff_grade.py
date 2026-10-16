# ABOUTME: Official CPU-only SWE grading of reconstructed 65,000-context source patches.
# ABOUTME: Separate diagnostic run IDs; original rollouts, predictions and grades remain unchanged.
import json, shutil, subprocess
from pathlib import Path
from omegaconf import OmegaConf

ROOT = Path("output/gptoss_patch_at_65000_20261013").resolve()


def main():
    results = []
    for case in sorted(p.parent for p in ROOT.glob("*/*/cutoff-65000.patch")):
        patch = (case / "cutoff-65000.patch").read_text()
        arm = case.parent.name
        iid = case.name
        assert (case / "replay-summary.json").exists(), "Wait for validated full replay"
        replay = json.loads((case / "replay-summary.json").read_text())
        cutoff = json.loads((case / "cutoff.json").read_text())
        assert not [
            m
            for m in replay["returncode_mismatches"]
            if m["index"] < cutoff["message_index"]
        ], "Replay diverged before cutoff"
        if (arm, iid) == ("da15", "django__django-14411"):
            # Later Git commits have different timestamp-dependent hashes. The recorded
            # revert commands consequently fail AFTER the already-saved empty cutoff.
            assert cutoff["message_index"] == 112 and not patch
            assert replay["returncode_mismatches"] == [
                dict(index=176, actual=128, expected=0),
                dict(index=195, actual=128, expected=0),
                dict(index=457, actual=128, expected=1),
            ]
        else:
            assert replay["final_patch_matches_original"], (
                "Replay diverged from archived final patch"
            )
            assert not replay["returncode_mismatches"], (
                "Review mismatched command results first"
            )
        root = case / "cutoff-grade-scoped"
        (root / "metadata").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            "output/gptoss_shared_smoke/cache/metadata/swebench_lite_test.json",
            root / "metadata/swebench_lite_test.json",
        )
        cfg = OmegaConf.load("output/gptoss_shared_smoke/swe.yaml")
        cfg.grading.max_workers = 1
        pred = dict(
            model_name_or_path="gptoss-cutoff65000-" + arm,
            model_patch=patch,
            instance_id=iid,
        )
        directory = root / "results"
        directory.mkdir(exist_ok=True)
        run_id = "gptoss65000-" + arm + "-20261013-scoped"
        predictions = directory / "predictions.jsonl"
        predictions.write_text(json.dumps(pred) + "\n")
        request = dict(
            harness=dict(
                dataset_name=str(root / "metadata/swebench_lite_test.json"),
                split="test",
                instance_ids=[iid],
                predictions_path=str(predictions),
                max_workers=1,
                force_rebuild=False,
                cache_level="instance",
                clean=False,
                open_file_limit=16384,
                run_id=run_id,
                timeout=1800,
                namespace="swebench",
                rewrite_reports=False,
                modal=False,
                report_dir=".",
            )
        )
        req = directory / "request.json"
        req.write_text(json.dumps(request, indent=2))
        if patch.strip():
            repo = Path.cwd()
            python = (
                repo
                / "src/eval/capabilities/swebench_mini/envs/harness/.venv/bin/python"
            )
            with (directory / "harness.log").open("x") as log:
                subprocess.run(
                    [
                        str(python),
                        "-u",
                        str(repo / "scratch/gptoss_swe/patch_cutoff_harness.py"),
                        str(req),
                    ],
                    cwd=directory,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            report = json.loads(
                next(directory.glob("*." + run_id + ".json")).read_text()
            )
            assert iid in report["completed_ids"] and not report["error_ids"]
            resolved = report["resolved_ids"]
        else:
            resolved = []
        result = dict(
            n_graded=1,
            n_resolved=len(resolved),
            resolved_ids=resolved,
            pass_at_1=float(bool(resolved)),
            empty_patch=not bool(patch.strip()),
        )
        row = dict(arm=arm, task=iid, cutoff_context=65000, **result)
        (case / "cutoff-grade-summary.json").write_text(json.dumps(row, indent=2))
        results.append(row)
        print(json.dumps(row), flush=True)
    (ROOT / "official-cutoff-results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
