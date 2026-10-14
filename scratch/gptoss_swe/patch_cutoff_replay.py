# ABOUTME: Reconstruct source patches at the first 65,000-token prompt-plus-response crossing.
# ABOUTME: Execute only recorded tools in pinned isolated CPU containers; never call any model.
import concurrent.futures, copy, hashlib, json, subprocess, time
from pathlib import Path
from minisweagent.environments.docker import DockerEnvironment
from src.eval.capabilities.swebench_mini.fleet_task import (
    forced_patch,
    install_container_limits,
)

ROOT = Path("output/gptoss_patch_at_65000_20261013")
LABEL = "gptoss-patch65000-20261013"


def save(p, value):
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def replay(path):
    case = path.parent
    arm = case.parent.name
    iid = case.name
    claim = case / "replay-started.json"
    with claim.open("x") as f:
        json.dump(dict(at=time.time(), arm=arm, task=iid), f)
    trajectory = json.loads(path.read_text())
    request = json.loads((case / "request.json").read_text())
    scaffold = json.loads((case / "scaffold.json").read_text())
    msgs = trajectory["messages"]
    cross = next(
        (
            i
            for i, m in enumerate(msgs)
            if (u := m.get("extra", {}).get("response", {}).get("usage"))
            and u["prompt_tokens"] + u["completion_tokens"] >= 65000
        ),
        None,
    )
    assert cross is not None
    usage = msgs[cross]["extra"]["response"]["usage"]
    expected_tools = {
        m["tool_call_id"]: m
        for m in msgs
        if m["role"] == "tool" and m.get("tool_call_id")
    }
    if (case / "prediction.json").exists():
        prediction = json.loads((case / "prediction.json").read_text())
    else:
        prediction = next(
            r["prediction"]
            for r in json.loads((ROOT / arm / "selected.json").read_text())
            if r["task"] == iid
        )
    (case / "original-final.patch").write_text(
        prediction["model_patch"], encoding="utf-8"
    )
    image = json.loads(
        subprocess.check_output(["docker", "image", "inspect", request["image"]])
    )[0]
    assert request["image"] in image["RepoDigests"]
    config = copy.deepcopy(scaffold["effective_config"]["environment"])
    config.pop("environment_class", None)
    config.update(
        image=request["image"],
        run_args=[
            "--rm",
            "--network",
            "none",
            "--pull",
            "never",
            "--cpus",
            "2",
            "--memory",
            "4g",
            "--pids-limit",
            "512",
            "--label",
            "lasr_campaign=" + LABEL,
        ],
    )
    env = DockerEnvironment(**config)
    records = []
    cutoff = None
    try:
        install_container_limits(env.container_id, 2, config["timeout"])
        initial = subprocess.check_output(
            [
                "docker",
                "exec",
                env.container_id,
                "git",
                "-C",
                "/testbed",
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip()
        assert (
            subprocess.run(
                [
                    "docker",
                    "exec",
                    env.container_id,
                    "git",
                    "-C",
                    "/testbed",
                    "merge-base",
                    "--is-ancestor",
                    request["instance"]["base_commit"],
                    "HEAD",
                ]
            ).returncode
            == 0
        )
        for index, m in enumerate(msgs):
            if index == cross:
                cutoff = forced_patch(env.container_id)
                (case / "cutoff-65000.patch").write_text(cutoff, encoding="utf-8")
                save(
                    case / "cutoff.json",
                    dict(
                        context_cap=65000,
                        message_index=index,
                        prompt=usage["prompt_tokens"],
                        response=usage["completion_tokens"],
                        kind="prompt_already_full"
                        if usage["prompt_tokens"] >= 65000
                        else "response_truncated",
                        patch_sha256=hashlib.sha256(cutoff.encode()).hexdigest(),
                        commands_completed=len(records),
                    ),
                )
                print(
                    json.dumps(
                        dict(
                            event="cutoff_saved",
                            arm=arm,
                            task=iid,
                            commands=len(records),
                            patch_bytes=len(cutoff.encode()),
                        )
                    ),
                    flush=True,
                )
            for action in m.get("extra", {}).get("actions", []):
                started = time.time()
                result = env.execute(action)
                expected = expected_tools.get(action["tool_call_id"], {}).get(
                    "extra", {}
                )
                rec = dict(
                    index=index,
                    action=action,
                    seconds=time.time() - started,
                    returncode=result["returncode"],
                    expected_returncode=expected.get("returncode"),
                    output=result["output"],
                    expected_output_sha256=hashlib.sha256(
                        expected.get("raw_output", "").encode()
                    ).hexdigest(),
                    exception_info=result.get("exception_info"),
                )
                records.append(rec)
                with (case / "replay.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        assert cutoff is not None
        final = forced_patch(env.container_id)
        (case / "replayed-final.patch").write_text(final, encoding="utf-8")
        summary = dict(
            arm=arm,
            task=iid,
            context_cap=65000,
            crossing=usage,
            commands=len(records),
            returncode_mismatches=[
                dict(
                    index=r["index"],
                    actual=r["returncode"],
                    expected=r["expected_returncode"],
                )
                for r in records
                if r["returncode"] != r["expected_returncode"]
            ],
            final_patch_matches_original=final == prediction["model_patch"],
            cutoff_patch_matches_original=cutoff == prediction["model_patch"],
            initial_head=initial,
            image=request["image"],
            trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        save(case / "replay-summary.json", summary)
        print(json.dumps(summary), flush=True)
        return summary
    finally:
        cid = env.container_id
        env.container_id = None
        data = json.loads(subprocess.check_output(["docker", "inspect", cid]))[0]
        assert data["Config"]["Labels"]["lasr_campaign"] == LABEL
        subprocess.run(["docker", "rm", "-f", cid], check=True, capture_output=True)


if __name__ == "__main__":
    paths = sorted(ROOT.glob("*/*/checkpoint.traj.json"))
    assert len(paths) == 5
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(replay, paths))
    save(ROOT / "replay-summary.json", results)
