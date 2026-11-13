# ABOUTME: run() for AgentHazard (arXiv 2604.02947) under Claude Code: sandboxed episodes against the
# ABOUTME: served target, each judged at R1..R4 and round_all as soon as it ends; rollouts + ASR summary.
"""run() per the CLAUDE.md eval contract.

Shape of a run (`uv run evals --name agenthazard --target <hf> --server <pod>`): the vLLM server
run_eval starts is tunnelled to this machine; the sandbox image is built here; one internal Docker
network with a one-port proxy to that tunnel is created and its isolation proven
(`sandbox.check_isolation`); then `concurrency` episodes run at a time, each in its own container
(session.run_episode). Judging runs on a thread pool fed as each episode finishes, so a run ends
when its last episode does. Every instance leaves `rollouts/<id>/` (episode.json, tool_calls.jsonl,
claude_session/*.jsonl, transcript.md) and rows in `results/{episodes,judgments}.jsonl`.

Claude Code reaches the model through vLLM's Anthropic-compatible `/v1/messages` (the served name
is the arm's LoRA, `target.model_name`), so this eval accepts only a vLLM-served HF target.
"""

from __future__ import annotations

import asyncio
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from omegaconf import OmegaConf

from src.eval.layout import publish_layout
from src.eval.misalignment.agenthazard import judge as J
from src.eval.misalignment.agenthazard.data import claude_code_ids, load_dataset, select
from src.eval.misalignment.agenthazard.sandbox import (
    DiskWatchdog,
    RunNetwork,
    build_image,
    check_isolation,
)
from src.eval.misalignment.agenthazard.session import run_episode
from src.eval.misalignment.agenthazard.stats import summarize
from src.eval.misalignment.odcv.runner import container_host_address
from src.utils import transcript_markdown

SMOKE_N = 5


def _transcript(ep: dict, verdicts: list[dict], clip: int = 4000) -> str:
    """Readable rollout: the task, each step and reply, every tool call, the judge's verdicts."""
    tools = (
        "\n\n".join(
            f"[turn {e.get('turn')}] {e.get('type')} {e.get('tool_name')}\n"
            f"input: {json.dumps(e.get('tool_input'), ensure_ascii=False)[:clip]}"
            + (
                f"\nresponse: {json.dumps(e.get('tool_response'), ensure_ascii=False)[:clip]}"
                if "tool_response" in e
                else ""
            )
            + (f"\nerror: {str(e.get('error'))[:clip]}" if e.get("error") else "")
            for e in ep["detail_logs"]
        )
        or "(no tool calls)"
    )
    sections = [
        (
            2,
            "Instance",
            "json",
            json.dumps(
                {
                    k: ep[k]
                    for k in ("id", "category", "jailbreak_method", "query", "comment")
                },
                ensure_ascii=False,
                indent=2,
            ),
        ),
        (2, "Turns (step sent → Claude Code's final reply)", "text", ""),
    ]
    for t in ep["turns"]:
        sections += [
            (
                3,
                f"Turn {t['turn_idx']} — {t['status']}, {t['end_time'] - t['start_time']:.0f}s",
                "text",
                "",
            ),
            (4, "Step sent", "fenced", t["input"]),
            (4, "Reply", "fenced", t["output"]),
        ]
    sections += [
        (2, "Tool calls (hook log)", "fenced", tools),
        (
            2,
            "Judge verdicts",
            "json",
            json.dumps(verdicts, ensure_ascii=False, indent=2),
        ),
        (
            2,
            "Full Claude Code session",
            "text",
            "See claude_session/*.jsonl beside this file (includes reasoning blocks).",
        ),
    ]
    final = next((v for v in verdicts if v["round"] == "all"), {})
    return transcript_markdown(
        f"AgentHazard {ep['id']} — {ep['category']} / {ep['jailbreak_method']}",
        f"Episode status {ep['status']}; round_all harmful={final.get('harmful')} score={final.get('score')}.",
        sections,
    )


async def _run_all(
    items, *, cfg, image, model, host, port, rollouts: Path, results: Path, client
) -> tuple[list, list, dict]:
    limits = cfg.docker.limits
    episodes, judgments, lock = [], [], asyncio.Lock()
    executor = ThreadPoolExecutor(max_workers=int(cfg.judge.workers))
    loop = asyncio.get_running_loop()
    with RunNetwork(image, host, port) as net:
        iso = await check_isolation(image, net, limits)
        print(f">>> agenthazard: isolation checks passed {iso}", flush=True)
        wd = DiskWatchdog(
            net.run_id,
            float(cfg.docker.max_layer_bytes),
            float(cfg.docker.min_host_free_bytes),
        )
        wd_task = asyncio.create_task(wd.loop())
        sem = asyncio.Semaphore(int(cfg.concurrency))

        async def one(item: dict) -> None:
            async with sem:
                if wd.abort_reason:
                    raise RuntimeError(f"aborting run: {wd.abort_reason}")
                for attempt in range(2):  # infra (container start) failures only
                    ep = await run_episode(
                        item, image=image, network=net, model=model, limits=limits,
                        turn_timeout_s=int(cfg.turn_timeout_s), dest=rollouts / str(item["id"]), watchdog=wd,
                    )  # fmt: skip
                    if ep.status != "infra_error":
                        break
            ej = ep.to_json()
            verdicts: list[dict] = []
            by_k: dict[int, dict] = {}
            if ej["turns"]:
                for label, k in J.rounds_for(len(ej["turns"])):
                    base = {
                        "id": ej["id"],
                        "round": label,
                        "k": k,
                        "category": ej["category"],
                        "jailbreak_method": ej["jailbreak_method"],
                    }
                    if (
                        k in by_k
                    ):  # round_all of a <=4-step instance is its R_k: judged once
                        verdicts.append(
                            {
                                **base,
                                **{
                                    x: by_k[k][x]
                                    for x in (
                                        "harmful",
                                        "score",
                                        "reason",
                                        "judge_error",
                                    )
                                },
                                "cost": 0.0,
                                "shared": True,
                            }
                        )
                        continue
                    prompt = J.build_prompt(ej, k, int(cfg.judge.max_field_chars))
                    v = await loop.run_in_executor(
                        executor,
                        lambda p=prompt: J.judge(
                            client, p, model=str(cfg.judge.model), temperature=float(cfg.judge.temperature),
                            max_tokens=int(cfg.judge.max_tokens), attempts=int(cfg.judge.attempts),
                        ),
                    )  # fmt: skip
                    by_k[k] = v
                    verdicts.append({**base, **v, "shared": False})
            (rollouts / str(ej["id"]) / "episode.json").write_text(
                json.dumps(ej, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            (rollouts / str(ej["id"]) / "transcript.md").write_text(
                _transcript(ej, verdicts), encoding="utf-8"
            )
            row = {
                "id": ej["id"], "category": ej["category"], "jailbreak_method": ej["jailbreak_method"], "status": ej["status"],
                "error": ej["error"], "n_turns": len(ej["turns"]), "n_timeout": sum(t["status"] == "timeout" for t in ej["turns"]),
                "n_error": sum(t["status"] == "error" for t in ej["turns"]), "n_tool_events": len(ej["detail_logs"]),
                "wall_s": round(ej["wall_s"], 1), "peak_mem_bytes": int(ej["peak_mem"]) if ej["peak_mem"].isdigit() else None,
            }  # fmt: skip
            async with lock:
                episodes.append(row)
                judgments.extend(verdicts)
                with open(results / "episodes.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                with open(results / "judgments.jsonl", "a", encoding="utf-8") as f:
                    for v in verdicts:
                        f.write(json.dumps(v, ensure_ascii=False) + "\n")
                final = next((v for v in verdicts if v["round"] == "all"), {})
                done_h = sum(
                    1 for j in judgments if j["round"] == "all" and j.get("harmful")
                )
                print(
                    f">>> agenthazard [{len(episodes)}/{len(items)}] id={ej['id']} {ej['status']} "
                    f"{row['wall_s']:.0f}s tools={row['n_tool_events']} harmful={final.get('harmful')} "
                    f"score={final.get('score')} | running harmful {done_h}/{len(episodes)}",
                    flush=True,
                )

        try:
            await asyncio.gather(*(one(i) for i in items))
        finally:
            wd_task.cancel()
            executor.shutdown(wait=True)
    health = {"isolation_checks": iso, "killed_disk": sorted(wd.killed)}
    return episodes, judgments, health


def run(target, cfg, out_dir: Path) -> dict:
    """Run AgentHazard against one vLLM-served target (CLAUDE.md contract).

    Returns:
        Summary: ASR (+95% CI) at round_all and R1..R4, by category and strategy, mean severity,
        episode/turn error and timeout counts, judge cost, Claude Code version, isolation results.
    """
    from src.infra.endpoints.openrouter import OpenRouterClient

    cfg = OmegaConf.merge(cfg)
    if getattr(target, "is_api", False):
        raise SystemExit(
            "agenthazard drives Claude Code against vLLM's /v1/messages; an API target is not supported"
        )
    rollouts, results, metadata = publish_layout(out_dir)
    if any(results.iterdir()) or any(rollouts.iterdir()):
        raise SystemExit(
            f"agenthazard needs a fresh run directory; {out_dir} already holds episodes"
        )
    smoke = bool(cfg.get("smoke", False))
    rows = load_dataset(str(cfg.dataset.repo), str(cfg.dataset.revision))
    items = select(
        rows,
        claude_code_ids(),
        int(cfg.n_per_category),
        int(cfg.seed),
        limit=SMOKE_N if smoke else cfg.get("limit"),
    )
    image = str(cfg.docker.image)
    image_id = build_image(image, str(cfg.claude_code_version))
    port = urlsplit(target.base_url).port  # first access boots or LoRA-swaps the server
    (metadata / "protocol.json").write_text(
        json.dumps(
            {
                "paper": "arXiv 2604.02947 (AgentHazard), Claude Code framework",
                "dataset": {
                    "repo": str(cfg.dataset.repo),
                    "revision": str(cfg.dataset.revision),
                },
                "instance_ids": [int(i["id"]) for i in items],
                "claude_code_version": str(cfg.claude_code_version),
                "image": image,
                "image_id": image_id,
                "served_model_name": target.model_name,
                "config": OmegaConf.to_container(cfg, resolve=True),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f">>> agenthazard: {len(items)} instances | Claude Code {cfg.claude_code_version} | model {target.model_name} "
        f"via {container_host_address()}:{port} | concurrency {cfg.concurrency} | judge {cfg.judge.model}",
        flush=True,
    )
    episodes, judgments, health = asyncio.run(
        _run_all(
            items, cfg=cfg, image=image, model=target.model_name, host=container_host_address(), port=port,
            rollouts=rollouts, results=results, client=OpenRouterClient(),
        )
    )  # fmt: skip
    summary = summarize(judgments, episodes)
    walls = [e["wall_s"] for e in episodes]
    mems = [e["peak_mem_bytes"] for e in episodes if e["peak_mem_bytes"]]
    summary.update(
        {
            "claude_code_version": str(cfg.claude_code_version),
            "served_model_name": target.model_name,
            "concurrency": int(cfg.concurrency),
            "episode_wall_s_median": statistics.median(walls) if walls else None,
            "episode_peak_mem_mb_max": round(max(mems) / 1e6) if mems else None,
            **health,
        }
    )
    return summary
