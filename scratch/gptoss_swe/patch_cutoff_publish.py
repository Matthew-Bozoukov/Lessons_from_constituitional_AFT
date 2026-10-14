# ABOUTME: Publish the CPU-only historical 65,000-context patch audit and immutable evidence.
# ABOUTME: Original model trajectories and benchmark scores remain unchanged; no model calls.
import hashlib, json, shutil, subprocess, tarfile
from pathlib import Path
import requests
from src.infra.huggingface import push_run_dir, hf_api
from src.eval.layout import publish_layout, assert_layout, run_tags

ROOT = Path("output/gptoss_patch_at_65000_20261013")


def main():
    results = json.loads((ROOT / "official-cutoff-results.json").read_text())
    assert len(results) == 5 and sum(r["n_resolved"] for r in results) == 3
    out = ROOT / "publication"
    rollouts, scores, meta = publish_layout(out)
    report = """# Historical patches at 65,000 context tokens

Five limit-ending, officially resolved GPT-OSS attempts were identified across the completed saved sweeps (copies deduplicated). At the first current prompt + current response crossing of 65,000 tokens, three patches still pass and two are empty/unresolved. This is not cumulative token use.

| Run / arm | Task | Original ending | Pass at cutoff |
|---|---|---|---|
| 20-task Base | Django14411 | step500 | yes |
| 20-task Control | Django14411 | response16384 | yes |
| 20-task Control | Sphinx8721 | context131072 | yes |
| 20-task DA15 | Django14411 | context131072 | no: empty patch |
| Earlier Control smoke | sklearn12471 | context131072 | no: empty patch |

Recorded commands were replayed in the original digest-pinned Docker images without model calls. The patch is the tracked eligible source diff extracted by the original forced_patch function, before executing any tool action from the crossing response. When the prompt itself already exceeds 65,000, prior tools remain included. The three retained patches are byte-identical to their original passing patches and were officially graded again. Empty patches are unresolved under the standard harness wrapper and do not require test execution.

Four complete replays match their archived final patch with no command return-code differences. DA15's prefix has no return-code differences, but its later Git commits have different timestamp-dependent hashes: recorded reverts at message indices176,195,457 fail in replay, all after cutoff112. This suffix is explicitly not claimed equivalent; the already-saved empty cutoff patch is unaffected. Earlier failed edits were undone by git checkout at94, and subsequent edits before112 failed.

This is a fixed-trajectory cutoff audit, not a fresh 65K-model evaluation. Changed response allowance can change stochastic generation. Explicitly Submitted successes are outside this audit; no full-score prediction is established. Originals and original grades are preserved. No paid inference, judge calls, GPU rentals or outcome rerolls.
"""
    (scores / "results.md").write_text(report, encoding="utf-8")
    (scores / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    with tarfile.open(rollouts / "reconstruction-evidence.tar.gz", "w:gz") as tar:
        for arm in ["base", "control", "da15", "original-control"]:
            tar.add(ROOT / arm, arcname=arm)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    meta_info = dict(
        source_parent_commit=source,
        context_cutoff=65000,
        criterion="first current prompt_tokens + completion_tokens >= 65000; no crossing-response tools executed",
        model_calls=0,
        originals_unchanged=True,
        twenty_source=dict(
            repo="dougalldeepmind/2026-10-09-gptoss120b-base-control-da15-swe-smoke20",
            revision="09bc27d232fb52c6a6d46aa54f09c12e6ac4e9b5",
        ),
    )
    (meta / "run_meta.json").write_text(json.dumps(meta_info, indent=2))
    for name in [
        "historical-pass-inventory.json",
        "replay-summary.json",
        "response64k-control-state.json",
        "grading-inventory-recovery.json",
    ]:
        shutil.copyfile(ROOT / name, meta / name)
    for p in Path("scratch/gptoss_swe").glob("patch_cutoff_*.py"):
        shutil.copyfile(p, meta / p.name)
    hashes = {
        p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.rglob("*")
        if p.is_file() and p.name not in {"README.md", "sha256.json"}
    }
    (meta / "sha256.json").write_text(json.dumps(hashes, indent=2))
    fields = dict(
        experiment="Historical limit-ending GPT-OSS successes: patch correctness at 65,000 instantaneous context tokens",
        date_generated="2026-10-13",
        constitution="none added by this replay audit",
        source_repo=subprocess.check_output(
            ["git", "remote", "get-url", "origin"], text=True
        ).strip(),
        models="GPT-OSS-120B base, control, DA15; historical Tinker samples, no fresh inference",
        generation_config="No generation. Exact saved commands replayed in original image digests; cutoff65000 current prompt plus response.",
        schema="rollouts/reconstruction-evidence.tar.gz; results/results.json and .md; metadata/provenance and exact audit source",
        provenance=meta_info,
        caveats="Five outcome-selected attempts, not a new full benchmark. DA15 post-cutoff replay divergence disclosed in results.md.",
    )
    url = push_run_dir(
        out,
        "2026-10-13-swebench-gptoss120b-patch-cutoff65000",
        fields,
        front_matter={"tags": run_tags("swebench_mini", "gptoss120b", "diagnostic")},
        atomic_commit=True,
    )
    assert_layout(out)
    repo = url.split("/datasets/")[1]
    rev = hf_api().dataset_info(repo).sha
    for p in out.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(out).as_posix()
        r = requests.get(
            f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{rel}",
            timeout=(30, 180),
        )
        r.raise_for_status()
        assert (
            hashlib.sha256(r.content).digest()
            == hashlib.sha256(p.read_bytes()).digest()
        ), rel
    receipt = dict(
        repo=repo,
        revision=rev,
        complete=True,
        all_files_sha256_verified=True,
        url=url + "/tree/" + rev,
    )
    (ROOT / "publication.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
