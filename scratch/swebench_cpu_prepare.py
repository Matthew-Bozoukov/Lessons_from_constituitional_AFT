# ABOUTME: Persist and back up CPU readiness checks before renting any inference GPU.
# ABOUTME: Reuses the existing image naming and official grading wrappers with frozen data.
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.eval.capabilities.swebench_mini.images import image_name
from src.eval.capabilities.swebench_mini.grade import verify_environment
from src.infra.huggingface import push_run_dir, hf_api, hf_download, hf_repo_id
from src.naming import artifact_name


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


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
        # Prove grading before spending time/bandwidth caching the whole split.
        for row in chosen:
            iid, image = pull(row)
            manifest[iid] = image
            atomic_json(manifest_path, manifest)
        gold_dir = out / "gold_attempts" / now.strftime("%Y%m%dT%H%M%S%fZ")
        state["gold"] = verify_environment(dataset=str(dataset_path), instance_ids=[r["instance_id"] for r in chosen],
                        out_dir=gold_dir, max_workers=cfg.gold_workers, cache_level="instance",
                        timeout=cfg.gold_timeout_seconds)
        state["gold"]["artifact_directory"] = str(gold_dir.relative_to(out))
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
        state["status"] = "ready"
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        backup()
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
