# ABOUTME: Persist and back up CPU readiness checks before renting any inference GPU.
# ABOUTME: Reuses the existing image naming and official grading wrappers with frozen data.
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import ssl
import subprocess
import time
from urllib.request import urlopen

from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.eval.capabilities.swebench_mini.images import image_name
from src.eval.capabilities.swebench_mini.grade import HARNESS_ENV, report_path
from src.eval.docker import docker_preflight
from src.infra.huggingface import push_run_dir, hf_api, hf_download, hf_repo_id
from src.naming import artifact_name


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


from src.eval.capabilities.swebench_mini.fixture import ensure_fixture


def fixture_gold(cfg, dataset_path, instance_ids, out_dir, fixture, *, no_fix=False):
    """Run the existing pinned official harness with an explicitly recorded environment change."""
    out_dir.mkdir(parents=True, exist_ok=False)
    run_id = f"{'no_fix' if no_fix else 'gold'}_local_httpbin_{len(instance_ids)}"
    predictions = "gold"
    if no_fix:
        # A nonempty, harmless patch makes the harness execute tests instead of skipping
        # empty submissions; no candidate source code or benchmark tests are changed.
        patch = ("diff --git a/.lasr-httpbin-negative-control b/.lasr-httpbin-negative-control\n"
                 "new file mode 100644\n--- /dev/null\n+++ b/.lasr-httpbin-negative-control\n"
                 "@@ -0,0 +1 @@\n+Environment validation only; no source or test changes.\n")
        path = out_dir / "predictions.jsonl"
        path.write_text("".join(json.dumps({"instance_id": iid, "model_name_or_path": "environment-negative-control",
                                            "model_patch": patch}) + "\n" for iid in instance_ids))
        predictions = str(path)
    request = {"fixture": fixture, "harness": {
        "dataset_name": str(dataset_path), "split": cfg.split, "instance_ids": instance_ids,
        "predictions_path": predictions, "max_workers": cfg.gold_workers, "force_rebuild": False,
        "cache_level": "instance", "clean": False, "open_file_limit": 16384, "run_id": run_id,
        "timeout": cfg.gold_timeout_seconds, "namespace": "swebench", "rewrite_reports": False,
        "modal": False, "report_dir": "."}}
    atomic_json(out_dir / "request.json", request)
    wrapper = Path(__file__).with_name("swebench_local_httpbin.py").resolve()
    with (out_dir / "gold_check.log").open("w") as log:
        proc = subprocess.run([str(HARNESS_ENV / ".venv/bin/python"), str(wrapper), "--request",
                               str(out_dir / "request.json")], cwd=out_dir, stdout=log,
                              stderr=subprocess.STDOUT, timeout=cfg.gold_timeout_seconds)
    report_file = report_path(out_dir, run_id)
    assert proc.returncode == 0 and report_file, f"Gold harness failed: {out_dir}"
    report = json.loads(report_file.read_text())
    resolved = set(report.get("resolved_ids", [])) & set(instance_ids)
    expected = set(cfg.expected_requests_no_fix_resolved) if no_fix else set(instance_ids)
    completed = set(report.get("completed_ids", []))
    return {"passed": resolved == expected and completed == set(instance_ids) and not report.get("error_ids"),
            "n_requested": len(instance_ids), "resolved_ids": sorted(resolved),
            "n_resolved": len(resolved), "unresolved_gold": sorted(set(instance_ids) - resolved),
            "harness_version": "4.1.0", "dataset": str(dataset_path), "report_file": report_file.name,
            "protocol_deviation": "requests resolves httpbin.org locally with a recorded trusted CA",
            "wrapper_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scratch/swebench_cpu.yaml")
    args = parser.parse_args()
    load_dotenv("/srv/lasr/credentials.env")
    cfg = OmegaConf.load(args.config)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    receipt = json.loads(Path(cfg.receipt).read_text())
    old_meta = out / "metadata/run_meta.json"
    if old_meta.exists():
        previous = json.loads(old_meta.read_text())
        assert (previous["dataset"], previous["revision"]) == (cfg.dataset, cfg.revision), "Resume dataset changed"
    now = datetime.now(timezone.utc)
    assert now < datetime.fromisoformat(receipt["stop_at"]), "VM expiry reached"
    date = receipt["created_at"][:10]
    repo = hf_repo_id(artifact_name(f"swebench-cpu-readiness-{receipt['instance_id']}", date=date))
    state = {"status": "preparing", "dataset": cfg.dataset, "revision": cfg.revision,
             "instance_id": receipt["instance_id"], "started_at": now.isoformat(),
             "code_sha": receipt["code_sha"], "stop_at": receipt["stop_at"],
             "preparation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    state["config_sha256"] = hashlib.sha256(OmegaConf.to_yaml(cfg).encode()).hexdigest()
    fields = {"experiment": "SWE-bench Lite CPU readiness; no model evaluation",
              "date_generated": date, "constitution": "none",
              "source_repo": f"Matthew-Bozoukov/teaching_claude_why_replication@{receipt['code_sha']}",
              "models": "none; gold reference patches only",
              "generation_config": OmegaConf.to_container(cfg),
              "schema": "metadata: frozen dataset and host checks; results: readiness; gold: harness logs",
              "provenance": "uv run --project scratch/swebench_cpu_env --frozen python -m scratch.swebench_cpu_prepare"}
    def backup():
        atomic_json(out / "results/readiness.json", state)
        push_run_dir(out, repo, fields, private=False,
                     front_matter={"tags": ["infrastructure-check", "swebench-lite"]})
    try:
        docker_preflight()
        info = json.loads(subprocess.check_output(["docker", "info", "--format", "{{json .}}"], text=True))
        assert info["NCPU"] >= cfg.min_cpus
        assert info["MemTotal"] / 2**30 >= cfg.min_ram_gib
        assert shutil.disk_usage(info["DockerRootDir"]).free / 2**30 >= cfg.min_free_gib
        subprocess.run(["docker", "run", "--rm", "hello-world"], check=True, timeout=180)
        assert subprocess.run(["systemctl", "is-active", "lasr-vast-expiry.timer"], capture_output=True).returncode == 0
        atomic_json(out / "metadata/host.json", {"cpus": info["NCPU"], "ram_bytes": info["MemTotal"],
                    "docker_version": info["ServerVersion"], "docker_root": info["DockerRootDir"],
                    "disk": shutil.disk_usage(info["DockerRootDir"])._asdict()})
        rows = list(load_dataset(cfg.dataset, revision=cfg.revision, split=cfg.split))
        assert len(rows) == cfg.expected_tasks == len({r["instance_id"] for r in rows})
        dataset_path = out / "metadata/swebench_lite_test.json"
        atomic_json(dataset_path, rows)
        state["dataset_sha256"] = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        atomic_json(out / "metadata/run_meta.json", state)
        backup()
        commit = hf_api().dataset_info(repo).sha
        downloaded = hf_download(repo, "metadata/swebench_lite_test.json", repo_type="dataset", revision=commit)
        assert hashlib.sha256(Path(downloaded).read_bytes()).hexdigest() == state["dataset_sha256"]
        state["hf_roundtrip_commit"] = commit
        print(f"HF round-trip passed: {repo}@{commit}", flush=True)
        manifest_path = out / "metadata/images.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        def pull(row):
            name = image_name(row)
            # Resume only if the saved immutable digest is still present locally.
            old = manifest.get(row["instance_id"])
            if old and subprocess.run(["docker", "image", "inspect", old["digest"]], capture_output=True).returncode == 0:
                subprocess.run(["docker", "tag", old["digest"], name], check=True)
                return row["instance_id"], old
            log = out / "pull_logs" / (row["instance_id"] + ".log")
            log.parent.mkdir(exist_ok=True)
            for attempt in range(cfg.pull_attempts):
                assert shutil.disk_usage(info["DockerRootDir"]).free / 2**30 > cfg.min_free_gib, "Disk reserve reached"
                try:
                    with log.open("a") as fh:
                        result = subprocess.run(["docker", "pull", name], stdout=fh, stderr=subprocess.STDOUT,
                                                timeout=cfg.pull_timeout_seconds)
                    if result.returncode == 0:
                        image = json.loads(subprocess.check_output(["docker", "image", "inspect", name], text=True))[0]
                        return row["instance_id"], {"name": name, "digest": image["RepoDigests"][0], "id": image["Id"]}
                except subprocess.TimeoutExpired:
                    pass
                if attempt + 1 < cfg.pull_attempts:
                    time.sleep(5)
            raise RuntimeError(f"Image pull failed: {name}; see {log}")
        failures = []
        chosen = [next(r for r in sorted(rows, key=lambda r: r["instance_id"]) if r["repo"] == repo_name)
                  for repo_name in cfg.gold_repos]
        # Every requests task exercises the new fixture; preserve the standard checks for other repos.
        chosen = list({r["instance_id"]: r for r in [*chosen, *[r for r in rows if r["repo"] == "psf/requests"]]}.values())
        # Prove grading before spending time/bandwidth caching the whole split.
        for row in chosen:
            iid, image = pull(row)
            manifest[iid] = image
            atomic_json(manifest_path, manifest)
        probe = out / "metadata/docker-volume-probe"
        probe.mkdir(exist_ok=True)
        subprocess.run(["docker", "run", "--rm", "--network", "none", "-v", f"{probe}:/probe",
                        image_name(chosen[0]), "/bin/sh", "-c", "printf 'persistent-volume-ok\\n' > /probe/result.txt"],
                       check=True, timeout=60)
        assert (probe / "result.txt").read_text() == "persistent-volume-ok\n"
        gold_dir = out / "gold_attempts" / now.strftime("%Y%m%dT%H%M%S%fZ")
        fixture = ensure_fixture(cfg, receipt)
        atomic_json(out / "metadata/httpbin_fixture.json", fixture)
        shutil.copyfile(fixture["certificate_path"], out / "metadata/httpbin-ca.pem")
        state["gold"] = fixture_gold(cfg, dataset_path, [r["instance_id"] for r in chosen], gold_dir, fixture)
        state["gold"]["artifact_directory"] = str(gold_dir.relative_to(out))
        if state["gold"]["passed"]:
            no_fix_dir = out / "no_fix_attempts" / now.strftime("%Y%m%dT%H%M%S%fZ")
            state["no_fix"] = fixture_gold(cfg, dataset_path,
                [r["instance_id"] for r in rows if r["repo"] == "psf/requests"], no_fix_dir, fixture, no_fix=True)
            state["no_fix"]["artifact_directory"] = str(no_fix_dir.relative_to(out))
            state["benchmark_limitations"] = (
                "Two requests tasks resolve without a source fix under the pinned test labels. "
                "All 300 tasks remain included; report these no-fix passes alongside model scores. "
                "The local HTTPBin environment is a declared deviation from public-service grading.")
        backup()
        print(f"Gold grading: {state['gold']['n_resolved']}/{len(chosen)}. "
              "Caching continues; every gold check must pass before readiness.", flush=True)
        last_backed_up = 0
        with ThreadPoolExecutor(max_workers=cfg.pull_workers) as pool:
            pending = [pool.submit(pull, row) for row in rows]
            for future in as_completed(pending):
                if future.cancelled():
                    continue
                try:
                    iid, image = future.result()
                    manifest[iid] = image
                    atomic_json(manifest_path, manifest)
                    print(f"Images ready {len(manifest)}/{len(rows)}: {iid}", flush=True)
                except Exception as exc:
                    failures.append(str(exc))
                    print(f"Image failure: {exc}", flush=True)
                    for queued in pending:
                        queued.cancel()
                if len(manifest) >= last_backed_up + 25:
                    state["images_ready"] = len(manifest)
                    backup()
                    last_backed_up = len(manifest)
        assert not failures, failures
        assert len(manifest) == len(rows)
        state["images_ready"] = len(manifest)
        assert shutil.disk_usage(info["DockerRootDir"]).free / 2**30 > cfg.min_free_gib
        assert state["gold"]["passed"], state["gold"]
        assert state["no_fix"]["passed"], state["no_fix"]
        assert datetime.now(timezone.utc) < datetime.fromisoformat(receipt["stop_at"]), "VM expiry reached"
        state["status"] = "ready"
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        backup()
        commit = hf_api().dataset_info(repo).sha
        for relative in ("results/readiness.json", "metadata/images.json", "metadata/httpbin_fixture.json"):
            downloaded = hf_download(repo, relative, repo_type="dataset", revision=commit)
            assert Path(downloaded).read_bytes() == (out / relative).read_bytes(), f"HF round-trip mismatch: {relative}"
        atomic_json(out.parent / "cpu-readiness-backup-verified.json", {"repo": repo, "revision": commit,
                    "readiness_sha256": hashlib.sha256((out / "results/readiness.json").read_bytes()).hexdigest(),
                    "verified_at": datetime.now(timezone.utc).isoformat()})
        print(json.dumps(state, indent=2), flush=True)
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        atomic_json(out / "results/readiness.json", state)
        try:
            backup()
        except Exception:
            print("HF backup failed; local failure receipt retained", flush=True)
        raise


if __name__ == "__main__":
    main()
