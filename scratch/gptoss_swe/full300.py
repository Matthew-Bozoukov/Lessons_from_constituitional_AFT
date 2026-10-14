# ABOUTME: Prepare and run three fresh 300-task GPT-OSS arms on one qualified Linux CPU.
# ABOUTME: Separate preparation from explicit inference admission; retain every attempt and ledger.
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

from omegaconf import OmegaConf
from scratch.gptoss_swe import twenty as engine
from scratch.gptoss_swe.twenty import read, save, sha

C = OmegaConf.load(
    os.environ.get("GPTOSS_FULL300_CONFIG", "scratch/gptoss_swe/full300.yaml")
)
ROOT = Path(C.root)


def sources():
    return {
        p.as_posix(): sha(p)
        for parent in ("src", "scratch", "configs")
        for p in sorted(Path(parent).rglob("*"))
        if p.is_file()
        and (p.suffix in (".py", ".yaml", ".toml") or p.name == "uv.lock")
        and ".venv" not in p.parts
        and "__pycache__" not in p.parts
    }


def prepared_hashes():
    return {
        str(p.relative_to(ROOT)): sha(p)
        for a in C.targets
        for p in [
            ROOT / a / "config.yaml",
            ROOT / a / "swe/metadata/manifest.json",
            ROOT / a / "swe/metadata/images.json",
            ROOT / a / "swe/metadata/swebench_lite_test.json",
        ]
    }


def fresh_state(ids):
    assert len(ids) == len(set(ids)) == 300
    return dict(
        tasks={i: dict(status="pending", attempts=[]) for i in ids},
        pods=[],
        deadline=None,
        phase="inference",
        halt=None,
    )


def arm_config(baseline, arm, ids):
    cfg = OmegaConf.create(OmegaConf.to_container(baseline, resolve=True))
    assert sum(C.workers.values()) == 80 and set(C.targets) == {
        "base",
        "control",
        "da15",
    }
    assert cfg.worker.max_response_tokens == cfg.tinker.max_tokens == 16384
    assert (
        cfg.tinker.context_window
        == cfg.serving.context_window
        == C.context_tokens
        == 65000
    )
    assert cfg.worker.max_task_tokens == 262144 and cfg.worker.step_limit == 500
    assert (cfg.sampling.temperature, cfg.sampling.top_p, cfg.sampling.top_k) == (
        1,
        1,
        -1,
    )
    assert cfg.tinker.reasoning == "medium" and cfg.tinker.budget_usd is None
    assert (
        cfg.worker.model_request_timeout_seconds == 7200
        and cfg.worker.model_request_attempts == 2
    )
    assert (
        cfg.worker.max_infrastructure_attempts == 3
        and cfg.worker.max_infrastructure_failures == 6
    )
    assert (
        cfg.worker.tool_concurrency == 32 and cfg.worker.min_available_memory_gib == 32
    )
    assert cfg.grading.max_workers == 12 and cfg.grading.scoped_image_inventory
    fresh_state(ids)
    cfg.campaign = str(C.campaign) + "-" + arm
    cfg.run_name = "gptoss120b-" + arm + "-lite300-context65000"
    cfg.output_root = str(ROOT / arm / "swe")
    cfg.cached_campaign = str(C.input)
    cfg.instance_ids = ids
    cfg.subset.n = 300
    cfg.workers = int(C.workers[arm])
    cfg.tinker.budget_ledger = str(ROOT / arm / "budget.json")
    cfg.tinker.render_date = str(C.render_date)
    return cfg


def prepare():
    cache = Path(C.input)
    assert read(cache / "results/readiness.json")["status"] == "ready"
    assert read(cache.parent / "cpu-readiness-backup-verified.json")["revision"]
    baseline = OmegaConf.load(C.baseline)
    dataset = cache / "metadata/swebench_lite_test.json"
    assert sha(dataset) == baseline.dataset_sha256
    rows, images = read(dataset), read(cache / "metadata/images.json")
    from src.eval.capabilities.swebench_mini.fleet import scheduling_plan

    schedule = scheduling_plan(baseline, rows)
    ids = schedule["instance_ids"]
    fresh_state(ids)
    assert set(images) == set(ids)
    configs = {a: arm_config(baseline, a, ids) for a in C.targets}
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "prepare-claim.json").open("x") as f:
        json.dump(dict(pid=os.getpid(), at=time.time()), f)
    save(ROOT / "scheduling.json", schedule)
    save(
        ROOT / "selection.json",
        dict(
            ids=ids,
            dataset_revision=baseline.dataset_revision,
            dataset_sha256=sha(dataset),
            fresh_per_arm=300,
            retained_outcomes=0,
        ),
    )
    for arm, cfg in configs.items():
        r = ROOT / arm
        metadata = r / "swe/metadata"
        metadata.mkdir(parents=True)
        OmegaConf.save(cfg, r / "config.yaml")
        for name in ("images.json", "swebench_lite_test.json"):
            shutil.copyfile(cache / "metadata" / name, metadata / name)
        save(
            metadata / "manifest.json",
            dict(
                target=C.targets[arm],
                campaign=cfg.campaign,
                protocol=cfg.protocol,
                selected_ids=ids,
                retained_ids=[],
                config=OmegaConf.to_container(cfg, resolve=True),
            ),
        )
        save(metadata / "state.json", fresh_state(ids))
        save(r / "status.json", dict(phase="prepared", at=time.time()))
    save(ROOT / "runtime-source-hashes.json", sources())
    save(ROOT / "config.json", OmegaConf.to_container(C, resolve=True))
    save(ROOT / "prepared-input-hashes.json", prepared_hashes())
    archive = Path("/srv/lasr/staging/source.tar.gz")
    deployment = read("/srv/lasr/deployment.json")
    assert sha(archive) == deployment["source_sha256"]
    shutil.copyfile(archive, ROOT / "source.tar.gz")
    save(ROOT / "deployment.json", deployment)
    save(
        ROOT / "status.json",
        dict(phase="prepared_not_authorized_for_inference", at=time.time()),
    )


def qualify():
    """Run offline protocol/regression tests and the existing 80-environment CPU qualification."""
    assert not (ROOT / "owner-started.json").exists()
    assert read(ROOT / "runtime-source-hashes.json") == sources()
    assert read(ROOT / "prepared-input-hashes.json") == prepared_hashes()
    q = ROOT / "qualification"
    q.mkdir(exist_ok=True)
    info = json.loads(
        subprocess.check_output(["docker", "info", "--format", "{{json .}}"], text=True)
    )
    assert info["NCPU"] >= C.min_cpus and info["MemTotal"] / 2**30 >= C.min_ram_gib
    assert shutil.disk_usage(info["DockerRootDir"]).free / 2**30 >= C.min_free_gib
    cpu = "scratch/swebench_cpu_env/.venv/bin/python"
    agent = "src/eval/capabilities/swebench_mini/envs/agent/.venv/bin/python"
    commands = {
        "full300-tests": [
            cpu,
            "-m",
            "pytest",
            "scratch/gptoss_swe/test_full300.py",
            "tests/test_tinker_endpoint.py",
            "tests/test_gptoss_infra_optimizations.py",
            "tests/test_gptoss_finish_smoke.py",
            "scratch/gptoss_swe/test_parallel.py",
            "-q",
        ],
        "fleet-tests": [
            cpu,
            "-m",
            "pytest",
            "tests/test_swebench_fleet.py",
            "tests/test_swebench_session.py",
            "tests/test_swebench_admission.py",
            "scratch/test_swebench_lite.py",
            "-q",
        ],
        "shell": [
            cpu,
            "-m",
            "src.eval.capabilities.swebench_mini.fleet",
            "qualify-shell",
            "--config",
            "configs/eval/swebench_mini/lite.yaml",
        ],
        "capacity": [
            cpu,
            "-m",
            "scratch.swebench_cpu_load",
            "--output",
            str(q / "capacity"),
        ],
        "transport": [agent, "-m", "scratch.test_swebench_timeout_transport"],
        "protocol": [
            agent,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_swebench_protocol.py",
            "-v",
        ],
        "template": [agent, "-m", "scratch.swebench_protocol_template_check"],
        "agent-smoke": [cpu, "-m", "scratch.swebench_lite_smoke"],
        "gptoss-protocol": [
            "src/infra/endpoints/tinker_env/.venv/bin/python",
            "-m",
            "pytest",
            "tests/test_tinker_server.py",
            "tests/test_tinker_budget.py",
            "-q",
        ],
        "scoped-grader": [
            "src/eval/capabilities/swebench_mini/envs/harness/.venv/bin/python",
            "-m",
            "unittest",
            "scratch.gptoss_swe.test_scoped_harness",
            "-v",
        ],
    }
    receipts = {}
    for name, command in commands.items():
        path = q / (name + ".log")
        with path.open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        receipts[name] = dict(
            command=command, returncode=result.returncode, sha256=sha(path)
        )
        save(q / "checks.json", receipts)
        assert result.returncode == 0, f"{name} failed; preserve log and diagnose"
    smoke_receipts = []
    for line in (q / "agent-smoke.log").read_text().splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if (
            isinstance(value, dict)
            and value.get("status") == "passed"
            and value.get("root")
        ):
            smoke_receipts.append(value)
    assert len(smoke_receipts) == 1, "Missing unambiguous synthetic integration receipt"
    # This verifier checks all 526 commands, OOM counters, RAM floor, real-agent
    # results and source hashes; successful subprocess exit alone is insufficient.
    command = [
        cpu,
        "-m",
        "scratch.swebench_qualify_recipe",
        "--load-dir",
        str(q / "capacity"),
        "--smoke-root",
        smoke_receipts[0]["root"],
        "--test-log",
        str(q / "fleet-tests.log"),
        "--transport-log",
        str(q / "transport.log"),
        "--protocol-log",
        str(q / "protocol.log"),
        "--template-proof",
        "output/swebench-protocol-template-latest.json",
    ]
    with (q / "recipe.log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    receipts["recipe"] = dict(
        command=command, returncode=0, sha256=sha(q / "recipe.log")
    )
    readiness = read(Path(C.input) / "results/readiness.json")
    gold_source = (
        Path(C.input) / readiness["gold"]["artifact_directory"] / "request.json"
    )
    request = read(gold_source)
    request["harness"]["run_id"] = str(C.campaign) + "-scoped-gold"
    directory = q / "scoped-gold"
    directory.mkdir()
    save(directory / "request.json", request)
    repo = Path.cwd()
    command = [
        str(repo / "src/eval/capabilities/swebench_mini/envs/harness/.venv/bin/python"),
        str(repo / "scratch/gptoss_swe/scoped_harness.py"),
        "--request",
        str(directory / "request.json"),
    ]
    with (directory / "harness.log").open("x") as log:
        subprocess.run(
            command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    reports = list(directory.glob("*." + request["harness"]["run_id"] + ".json"))
    assert len(reports) == 1
    assert set(read(reports[0])["resolved_ids"]) == set(
        request["harness"]["instance_ids"]
    )
    receipts["scoped-gold"] = dict(
        command=command, returncode=0, report_sha256=sha(reports[0])
    )
    save(q / "checks.json", receipts)
    proof = dict(
        passed=True,
        at=time.time(),
        source_hashes=sources(),
        config_sha256=sha(ROOT / "config.json"),
        checks=receipts,
        host_cpus=info["NCPU"],
        host_memory=info["MemTotal"],
    )
    save(q / "proof.json", proof)
    from dotenv import load_dotenv
    from huggingface_hub import HfApi, hf_hub_download

    load_dotenv("/srv/lasr/credentials.env")
    package = q / "publication-package"
    package.mkdir()
    for name in ("proof.json", "checks.json"):
        shutil.copyfile(q / name, package / name)
    for name in (
        "selection.json",
        "config.json",
        "prepared-input-hashes.json",
        "deployment.json",
    ):
        shutil.copyfile(ROOT / name, package / name)
    for name in commands:
        shutil.copyfile(q / (name + ".log"), package / (name + ".log"))
    shutil.copyfile(q / "recipe.log", package / "recipe.log")
    shutil.copyfile(q / "capacity/results.json", package / "capacity-results.json")
    shutil.copyfile(reports[0], package / "scoped-gold-report.json")
    (package / "README.md").write_text(
        "---\nlicense: mit\ntags: [infrastructure-check, swebench-lite]\n---\n"
        "# CPU-only full GPT-OSS preparation\n\n300 pinned images, 80 environments, 32-command gate, "
        "offline Harmony tests, official gold and no-fix checks. No model inference.\n"
    )
    hashes = {p.name: sha(p) for p in package.iterdir()}
    api = HfApi(token=os.environ["HF_TOKEN"])
    repo = str(C.hf_repo) + "-infrastructure"
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    revision = api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=package,
        commit_message="Verify full-run CPU and protocol preparation",
    ).oid
    for name, digest in hashes.items():
        assert (
            sha(
                hf_hub_download(
                    repo,
                    name,
                    repo_type="dataset",
                    revision=revision,
                    token=os.environ["HF_TOKEN"],
                )
            )
            == digest
        )
    save(
        q / "publication.json",
        dict(repo=repo, revision=revision, sha256=hashes, verified=True),
    )
    save(q / "ready.json", proof | dict(qualification_revision=revision))


def configure():
    engine.C, engine.ROOT = C, ROOT


def grade():
    from scratch.gptoss_swe.run import grade_predictions

    for arm, target in C.targets.items():
        r = ROOT / arm
        cfg, state = (
            OmegaConf.load(r / "config.yaml"),
            read(r / "swe/metadata/state.json"),
        )
        assert not any(
            t["status"] in ("running", "pending") for t in state["tasks"].values()
        )
        preds = {
            i: t["attempts"][-1]["prediction"]
            for i, t in state["tasks"].items()
            if t["status"] == "valid"
        }
        save(r / "status.json", dict(phase="grading", at=time.time()))
        result = (
            grade_predictions(cfg, r / "swe", preds, str(C.campaign) + "-" + arm)
            if preds
            else dict(n_graded=0, n_resolved=0, resolved_ids=[])
        )
        save(
            r / "summary.json",
            dict(**result, target=target, n_selected=300, n_excluded=300 - len(preds)),
        )
        save(r / "status.json", dict(phase="graded", at=time.time()))


def publish():
    from huggingface_hub import HfApi, hf_hub_download

    summary = engine.status()
    for arm, stats in summary.items():
        assert not any(
            stats[k] for k in ("running", "waiting", "reservations", "awaiting_grading")
        )
        assert stats["graded"] + stats["infrastructure_excluded"] == 300
    save(ROOT / "summary.json", summary)
    out = ROOT / "package"
    for name in ("rollouts", "results", "metadata"):
        (out / name).mkdir(parents=True, exist_ok=True)
    secrets = [
        v.encode()
        for k, v in os.environ.items()
        if len(v) > 16 and ("KEY" in k or "TOKEN" in k)
    ]
    for arm in C.targets:
        with tarfile.open(out / "rollouts" / (arm + ".tar.gz"), "w:gz") as archive:
            for p in sorted((ROOT / arm).rglob("*")):
                if p.is_file():
                    assert not any(s in p.read_bytes() for s in secrets), (
                        "Secret found in evidence"
                    )
                    archive.add(
                        p, arcname=p.relative_to(ROOT).as_posix(), recursive=False
                    )
        save(out / "results" / (arm + ".json"), read(ROOT / arm / "summary.json"))
    for p in ROOT.iterdir():
        if p.is_file() and p.name != "publication.json":
            shutil.copyfile(p, out / "metadata" / p.name)
    shutil.copytree(
        ROOT / "qualification", out / "metadata/qualification", dirs_exist_ok=True
    )
    (out / "README.md").write_text(
        "---\nlicense: mit\ntags: [eval-run, swebench-lite, gptoss120b]\n---\n"
        "# GPT-OSS-120B full SWE-bench Lite at 65,000 context\n\n"
        "300 fresh tasks each for Base, Control and DA15. Exact targets and settings are in manifests. "
        "Medium Harmony 0.0.8; T1/p1/k disabled; current prompt plus response 65000; response 16384; "
        "generated/task 262144; 500 steps. All attempts and raw reasoning retained; no JSON repair or outcome rerolls. "
        "Official resolved scores include retained patches from limit endings. Submitted does not mean resolved. "
        "The pinned local HTTPBin fixture is shared with the qualified Qwen protocol; two requests no-fix passes "
        "are a known benchmark limitation. 80 total task workers, 32 commands, 12 grading workers. "
        "Inference accounting is fresh, uses confirmed cache tokens and conservative reservations, and is not an invoice. "
        "CPU and transfer charges are separate. Infrastructure exclusions are explicit in results.\n"
    )
    hashes = {
        p.relative_to(out).as_posix(): sha(p)
        for p in out.rglob("*")
        if p.is_file() and p.name != "file-hashes.json"
    }
    save(out / "metadata/file-hashes.json", hashes)
    hashes["metadata/file-hashes.json"] = sha(out / "metadata/file-hashes.json")
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(str(C.hf_repo), repo_type="dataset", exist_ok=True)
    revision = api.upload_folder(
        repo_id=str(C.hf_repo),
        repo_type="dataset",
        folder_path=out,
        commit_message="Publish full three-arm 65000-context SWE-bench Lite",
    ).oid
    receipt = dict(
        repo=str(C.hf_repo), revision=revision, complete=False, sha256=hashes
    )
    save(ROOT / "publication.json", receipt)
    for name, digest in hashes.items():
        assert (
            sha(
                hf_hub_download(
                    str(C.hf_repo),
                    name,
                    repo_type="dataset",
                    revision=revision,
                    token=os.environ["HF_TOKEN"],
                )
            )
            == digest
        )
    save(ROOT / "publication.json", receipt | dict(complete=True))


def run(authorized=False):
    assert authorized, "Preparation does not authorize paid inference"
    from dotenv import load_dotenv

    load_dotenv("/srv/lasr/credentials.env")
    ready = read(ROOT / "qualification/ready.json")
    assert ready["passed"] and ready["source_hashes"] == sources()
    assert read(ROOT / "qualification/publication.json")["verified"]
    assert ready["config_sha256"] == sha(ROOT / "config.json")
    assert read(ROOT / "config.json") == OmegaConf.to_container(C, resolve=True)
    assert read(ROOT / "prepared-input-hashes.json") == prepared_hashes()
    os.environ["TINKER_PRICE_RATES_JSON"] = json.dumps(
        OmegaConf.to_container(C.price_rates, resolve=True)
    )
    configure()
    engine.grade, engine.publish = grade, publish
    engine.run()


def table():
    configure()
    data = engine.status()
    print("| Metric | Base | Control | DA15 |\n|---|---:|---:|---:|")
    for label, key in [
        ("Officially resolved", "success"),
        ("Officially graded", "graded"),
        ("Submitted endings", "submitted"),
        ("Awaiting grading", "awaiting_grading"),
        ("Running", "running"),
        ("Waiting", "waiting"),
        ("Infrastructure exclusions", "infrastructure_excluded"),
        ("Accounted USD", "cost_usd"),
    ]:
        print(
            "| "
            + label
            + " | "
            + " | ".join(str(round(data[a][key], 4)) for a in C.targets)
            + " |"
        )
    for reason in (
        "context_limit",
        "response_token_limit",
        "step_limit",
        "task_token_limit",
        "other_limit",
    ):
        print(
            "| "
            + reason
            + " | "
            + " | ".join(str(data[a]["limits"].get(reason, 0)) for a in C.targets)
            + " |"
        )


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv("/srv/lasr/credentials.env")
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "qualify", "status", "run"])
    parser.add_argument("--authorize-inference", action="store_true")
    args = parser.parse_args()
    if args.action == "run":
        run(args.authorize_inference)
    elif args.action == "status":
        table()
    else:
        globals()[args.action]()
