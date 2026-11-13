# ABOUTME: One AgentHazard episode: a fresh container, one Claude Code session, each decomposed step
# ABOUTME: sent as the next user turn (claude -p, then --resume), with the tool-hook log copied out.
"""Rebuilds the loop of the paper's unreleased driver (run/agent_session.py::run_session): every
step of `decomposed_query` is one user turn in the SAME Claude Code session, each turn capped at
600 s, a timed-out turn recorded with their fixed message and the episode carried on. Claude Code
runs headless (`-p`, JSON result) with the tool allow-list from assets/settings.json, so tools
execute without prompts like their traces show (permission_mode "default", every Write ran).
"""

from __future__ import annotations

import io
import json
import tarfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.eval.misalignment.agenthazard.sandbox import RunNetwork, docker_args, run_cmd

TIMEOUT_MSG = "响应超时，超出最长响应时间"  # the paper's driver's fixed timeout reply (agent_session.py)
LOG_PATH = "opt/agenthazard/logs/tool_calls.jsonl"


@dataclass
class Turn:
    turn_idx: int
    input: str
    output: str
    status: str  # ok | error | timeout
    start_time: float
    end_time: float
    result_json: dict | None = None
    stderr_tail: str = ""


@dataclass
class Episode:
    id: int
    category: str
    jailbreak_method: str
    query: str
    comment: str
    session_id: str
    turns: list[Turn] = field(default_factory=list)
    detail_logs: list[dict] = field(default_factory=list)
    status: str = "ok"  # ok | turn_error | turn_timeout | killed_disk | infra_error
    error: str = ""
    wall_s: float = 0.0
    peak_mem: str = ""

    def to_json(self) -> dict:
        return asdict(self)


def claude_env(model: str, base_url: str) -> dict[str, str]:
    """Env for Claude Code inside the container: the served model as main AND small/fast model."""
    return {
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_AUTH_TOKEN": "dummy",  # vLLM takes no key; nothing secret enters the container
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_SMALL_FAST_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",  # egress is blocked anyway; avoids stalls
        "DISABLE_TELEMETRY": "1",
        "DISABLE_AUTOUPDATER": "1",
        "ENABLE_TOOL_SEARCH": "false",  # vLLM rejects ToolSearch's content blocks (docs/GOTCHAS.md)
        "TERM": "dumb",
    }


def _result_text(res: dict) -> tuple[str, str]:
    """(output text, status) from `claude -p --output-format json`, the paper's error prefixes."""
    if res.get("is_error") or res.get("subtype") not in (None, "success"):
        return (
            f"[ERROR] Claude API 调用失败: {res.get('result') or res.get('subtype')}",
            "error",
        )
    text = res.get("result")
    if not text:
        return (
            f"[WARNING] Claude 响应为空或格式异常: {json.dumps(res, ensure_ascii=False)[:500]}",
            "error",
        )
    return str(text), "ok"


async def _export(name: str, dest: Path) -> list[dict]:
    """Copy the tool log and Claude Code's session transcripts out of the container (tmpfs-safe)."""
    rc, out, _ = await run_cmd(
        [
            "docker",
            "exec",
            name,
            "tar",
            "-C",
            "/",
            "-cf",
            "-",
            "--ignore-failed-read",
            LOG_PATH,
            "root/.claude/projects",
        ],
        timeout=120,
    )
    events: list[dict] = []
    if rc not in (0, 1) or not out:
        # The container is stopped (the disk watchdog killed it): exec is impossible and the tmpfs
        # session transcripts are gone, but the tool log lives on the container's own filesystem,
        # which `docker cp` still reads.
        rc, out, _ = await run_cmd(
            ["docker", "cp", f"{name}:/{LOG_PATH}", "-"], timeout=120
        )
        if rc != 0 or not out:
            return events
    with tarfile.open(fileobj=io.BytesIO(out)) as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            data = tar.extractfile(m).read()
            if m.name in (
                LOG_PATH,
                Path(LOG_PATH).name,
            ):  # tar from exec, or `docker cp` of the one file
                events = [
                    json.loads(line)
                    for line in data.decode("utf-8", "replace").splitlines()
                    if line.strip()
                ]
                (dest / "tool_calls.jsonl").write_bytes(data)
            elif m.name.endswith(".jsonl"):
                (dest / "claude_session").mkdir(exist_ok=True)
                (dest / "claude_session" / Path(m.name).name).write_bytes(data)
    return events


async def run_episode(
    item: dict,
    *,
    image: str,
    network: RunNetwork,
    model: str,
    limits,
    turn_timeout_s: int,
    dest: Path,
    watchdog,
) -> Episode:
    """Run one instance end to end and write its rollout files under `dest`."""
    dest.mkdir(parents=True, exist_ok=True)
    sid = str(uuid.uuid4())
    ep = Episode(
        int(item["id"]),
        item["category"],
        item["jailbreak_method"],
        item["query"],
        item.get("comment", ""),
        sid,
    )
    name = f"agenthazard-ep-{network.run_id}-{item['id']}"
    t0 = time.time()
    rc, _, err = await run_cmd(
        docker_args(image, name, network, claude_env(model, network.base_url), limits),
        timeout=120,
    )
    if rc != 0:
        ep.status, ep.error = (
            "infra_error",
            f"container start failed: {err.decode()[-500:]}",
        )
        await run_cmd(["docker", "rm", "-f", name], timeout=60)
        return ep
    try:
        rc, _, err = await run_cmd(
            [
                "docker",
                "exec",
                name,
                "bash",
                "-c",
                "mkdir -p /root/.claude && cp /opt/agenthazard/settings.json /root/.claude/settings.json",
            ],
            timeout=60,
        )
        if rc != 0:
            ep.status, ep.error = (
                "infra_error",
                f"settings copy failed: {err.decode()[-500:]}",
            )
            return ep
        for k, step in enumerate(item["decomposed_query"], 1):
            if name in watchdog.killed:
                break
            session_flag = ["--session-id", sid] if k == 1 else ["--resume", sid]
            argv = [
                "docker", "exec", "-e", f"AGENTHAZARD_TURN={k}", "-w", "/root", name,
                "timeout", "-k", "10", str(turn_timeout_s),
                "claude", "-p", step, "--output-format", "json", "--model", model, *session_flag,
            ]  # fmt: skip
            start = time.time()
            rc, out, err = await run_cmd(argv, timeout=turn_timeout_s + 60)
            end = time.time()
            if rc in (124, 137, -9) and (end - start) >= turn_timeout_s - 1:
                ep.turns.append(
                    Turn(
                        k,
                        step,
                        TIMEOUT_MSG,
                        "timeout",
                        start,
                        end,
                        None,
                        err.decode()[-1000:],
                    )
                )
                continue
            try:
                res = json.loads(
                    out.decode("utf-8", "replace").strip().splitlines()[-1]
                )
                text, status = _result_text(res)
            except (json.JSONDecodeError, IndexError):
                res, text, status = (
                    None,
                    f"[ERROR] claude exited {rc}: {err.decode('utf-8', 'replace')[-800:]}",
                    "error",
                )
            ep.turns.append(
                Turn(
                    k,
                    step,
                    text,
                    status,
                    start,
                    end,
                    res,
                    err.decode("utf-8", "replace")[-1000:],
                )
            )
        rc, out, _ = await run_cmd(
            ["docker", "exec", name, "cat", "/sys/fs/cgroup/memory.peak"], timeout=30
        )
        ep.peak_mem = out.decode().strip() if rc == 0 else ""
        ep.detail_logs = await _export(name, dest)
    finally:
        await run_cmd(["docker", "rm", "-f", name], timeout=120)
    ep.wall_s = time.time() - t0
    if name in watchdog.killed:
        ep.status = "killed_disk"
    elif any(t.status == "timeout" for t in ep.turns):
        ep.status = "turn_timeout"
    elif any(t.status == "error" for t in ep.turns):
        ep.status = "turn_error"
    return ep
