# ABOUTME: Freeze the union of exact prior context-limit cells and run three fresh matched passes.
# ABOUTME: Doubles only model context, isolates new budgets/evidence, and reuses the qualified local runner.
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import tarfile
import time

from omegaconf import OmegaConf
from scratch.gptoss_swe import odcv_full as full

CONFIG = Path("scratch/gptoss_swe/odcv_double_context.yaml")
OLD = Path("output/gptoss_odcv_full_20261012")


def selection():
    cells = {}
    for arm in ("base", "control", "da15"):
        root = OLD / arm / "odcv/rollouts"
        for path in root.glob("*/*/pass*/messages_record.txt"):
            text = path.read_text(encoding="utf-8", errors="replace")
            log = path.with_name("docker_output.log").read_text(
                encoding="utf-8", errors="replace"
            )
            if full.classify(text, log) != "context_limit":
                continue
            variant, scenario, repeat = path.relative_to(root).parts[:3]
            cells.setdefault(variant + "/" + scenario, []).append(
                dict(
                    arm=arm,
                    pass_name=repeat,
                    transcript_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
    assert len(cells) == 30 and sum(map(len, cells.values())) == 37
    return dict(sorted(cells.items()))


def configure():
    full.CONFIG = CONFIG
    full.CFG = OmegaConf.load(CONFIG)
    full.ROOT = Path(full.CFG.root)
    full.ARMS = dict(full.CFG.arms)
    full.LABEL = "lasr.campaign=" + str(full.CFG.campaign)
    full.LABEL_FILTER = "label=" + full.LABEL


def prepare():
    import requests

    configure()
    cfg = full.CFG
    root = full.ROOT
    assert not root.exists(), "Do not reset preparation or inference history"
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).strip()
    assert full.read(OLD / "publication.json")["complete"]
    chosen = selection()
    frozen = {v + "/" + s for v, names in cfg.include_scenarios.items() for s in names}
    assert (
        frozen == set(chosen)
        and cfg.context_window == 112000
        and full.expected_rollouts() == 90
    )
    assert (
        cfg.shared_inference_budget_usd + 3 * cfg.judge_budget_per_arm_usd
        == cfg.total_budget_usd
        == 30
    )
    assert not subprocess.check_output(
        ["docker", "ps", "-aq", "--filter", full.LABEL_FILTER], text=True
    ).strip()
    for port in cfg.ports.values():
        with socket.socket() as check:
            check.bind(("127.0.0.1", int(port)))
    oldcfg = OmegaConf.load("scratch/gptoss_swe/odcv_full.yaml")
    for key in (
        "arms",
        "passes",
        "temperature",
        "top_p",
        "top_k",
        "reasoning",
        "harmony_version",
        "cycles",
        "scenario_timeout_s",
        "strict_tool_validation",
        "system_preamble_file",
        "progress_judge",
        "render_date",
        "response_allowance",
    ):
        assert cfg[key] == oldcfg[key], key
    qualification = Path("output/gptoss_odcv_preflight_20261012")
    ready = full.read(qualification / "ready.json")
    assert ready["unique_sandbox_cells_passed"] == 80
    assert full.read(qualification / "capacity-48/receipt.json")["passed"]
    # The same host/containers/interface were qualified earlier today. Recheck unchanged execution files.
    hashes = {}
    with tarfile.open(OLD / "source.tar.gz") as prior_source:
        for name in ready["source_hashes"]:
            if not name.startswith("src/"):
                continue
            before = prior_source.extractfile(name).read().replace(b"\r\n", b"\n")
            after = Path(name).read_bytes().replace(b"\r\n", b"\n")
            hashes[name] = hashlib.sha256(after).hexdigest()
            assert after == before, (
                f"Actual56K runtime execution source changed: {name}"
            )
    price = requests.get(
        "https://tinker-docs.thinkingmachines.ai/tinker/models.json", timeout=30
    )
    price.raise_for_status()
    model = next(
        m for m in price.json() if m["tinker_id"] == "openai/gpt-oss-120b:peft:131072"
    )
    rates = dict(
        input_per_million=float(model["prefill"].strip("$")),
        cached_input_per_million=float(model["cached_prefill"].strip("$")),
        output_per_million=float(model["sample"].strip("$")),
    )
    assert rates == OmegaConf.to_container(cfg.price_rates)
    root.mkdir()
    full.save(
        root / "selection.json",
        dict(
            cells=chosen,
            passes=3,
            rollouts_per_arm=90,
            total=270,
            source_repo=oldcfg.hf_repo,
            source_revision=full.read(OLD / "publication.json")["revision"],
            rule="Union of exact scenario/variant cells with a prior context-limit ending in any arm/pass; run all three arms on identical union.",
        ),
    )
    lines = [
        "# 112K-context diagnostic: selected ODCV tasks",
        "",
        "Three fresh passes per arm for every exact scenario/variant below.",
        "",
    ]
    lines += ["- " + key for key in chosen]
    (root / "selected-tasks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    full.save(
        root / "pricing.json",
        dict(at=time.time(), model=model, rates=rates, url=price.url),
    )
    for arm in full.ARMS:
        (root / arm / "odcv").mkdir(parents=True)
        OmegaConf.save(full.arm_config(arm), root / arm / "odcv.yaml")
    args = [
        "docker",
        "run",
        "--rm",
        "--label",
        full.LABEL,
        "-v",
        f"{Path.cwd()}:/work",
        "-v",
        "lasr-gptoss-shim-env:/work/src/infra/endpoints/tinker_env/.venv",
        "-v",
        "lasr-gptoss-agent-env:/work/src/eval/capabilities/swebench_mini/envs/agent/.venv",
        "-v",
        "lasr-gptoss-hf-cache:/root/.cache/huggingface",
        full.IMAGE,
        "src/infra/endpoints/tinker_env/.venv/bin/python",
        "-m",
        "pytest",
        "tests/test_tinker_server.py",
        "tests/test_tinker_budget.py",
        "-q",
    ]
    with (root / "offline-interface-tests.log").open("w", encoding="utf-8") as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
    tests = (root / "offline-interface-tests.log").read_text()
    assert "passed" in tests and "failed" not in tests and "skipped" not in tests
    full.save(
        root / "qualification.json",
        dict(
            qualified=True,
            source_hashes=hashes,
            reused_capacity_receipt=str(qualification / "capacity-48/receipt.json"),
            cached_cells_qualified=80,
            offline_interface_tests=tests,
            docker_info=json.loads(
                subprocess.check_output(
                    ["docker", "info", "--format", "{{json .}}"], text=True
                )
            ),
        ),
    )
    full.save(
        root / "manifest.json",
        dict(
            source=full.source(),
            config=OmegaConf.to_container(cfg),
            targets=full.ARMS,
            authorization="User requested three fresh passes per arm on exact union of failed cells and approved $30 total; stop at cap.",
            total_budget_usd=30,
            automatic_rollout_retries=0,
            context_delta=dict(before=56000, after=112000),
            comparison_caveat=str(cfg.comparison_caveat),
        ),
    )
    full.save(
        root / "run_meta.json",
        dict(
            source=full.source(), config=OmegaConf.to_container(cfg), targets=full.ARMS
        ),
    )
    subprocess.run(
        [
            "git",
            "archive",
            "--format=tar.gz",
            "--output=" + str(root / "source.tar.gz"),
            "HEAD",
        ],
        check=True,
    )
    full.phase(root, "prepared")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run", "status", "publish"])
    args = parser.parse_args()
    configure()
    if args.action == "prepare":
        prepare()
    elif args.action == "status":
        print(json.dumps(full.status_data(), indent=2))
    else:
        getattr(full, args.action)()
