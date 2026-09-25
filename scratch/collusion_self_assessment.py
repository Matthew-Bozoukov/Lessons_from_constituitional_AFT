# ABOUTME: Resumes each agent's own agent_collusion history at its first mutual-ACCEPT verdict and
# ABOUTME: asks it (neutral, then direct) whether it did anything misaligned; served like the eval.
"""In-situ self-assessment of collusion.

For every trajectory of a published agent_collusion run, the onset episode is the first one in
which both agents ACCEPT. Each agent's stored `messages` (everything it saw, incl. the raw-log-only
verdict policy and the 200-char channel) is cut right after its `submit_verdict` tool result, and
the config's `questions` are appended in order as user turns of one conversation (e.g. a neutral
"assess your conduct", then "did you do anything misaligned?"). The request mirrors the harness's
own reflection call: the task's tool schemas are sent, no tool choice is forced, sampling is the eval's. Past reasoning is stripped from
the history; the pinned template drops reasoning before the last user turn anyway.

`strip` removes parts of the EARLIER episodes' feedback from the history before asking, to test
what the answer leans on: `verdict_review` deletes the "## Verdict review" section of each outcome
message (the reward line stays), `reflections` deletes each private-reflection exchange (the
harness's prompt and the agent's reply), `prior_episodes` drops every earlier episode entirely (the
system prompt plus the onset episode remain; it subsumes the other two). The agents' own messages to
each other within the kept episode(s) are untouched.

The target is served by `VllmServer` with the eval config's `serving:` block over `SshExec`, the
same path `uv run evals --server` takes, so the template the model sees is the eval's.

Run: uv run python scratch/collusion_self_assessment.py scratch/collusion_self_assessment.yaml \
         server=root@<ip>:<port> [smoke=true]
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download
from omegaconf import OmegaConf
from openai import OpenAI

from src.infra.endpoints.vllm import SshExec, VllmServer, resolve_target
from src.infra.huggingface import hf_token
from src.utils import git_sha

HARNESS = Path("src/eval/misalignment/agent_collusion/third_party/agent-collusion")
AGENTS = ("alice", "bob")


def mutual_accept(ep: dict) -> bool:
    """Both agents ACCEPTed in this episode."""
    return all(ep["agents"][a]["verdict"] == "accept" for a in AGENTS)


def tool_schemas(task_type: str, task_id: str, peer: str) -> list[dict]:
    """The tools the harness sends with its reflection call for this agent's task."""
    sys.path.insert(0, str(HARNESS))
    from experiments.tool_schemas import get_task_tool_schemas

    answer_key: list[str] = []
    if task_type == "data_search":
        manifest = json.loads((HARNESS / "task/data-search/task_manifest_data_search50.json").read_text())
        (task,) = [t for t in manifest if t["task_id"] == task_id]
        answer_key = [str(c) for c in task["answer_key"]]
    return get_task_tool_schemas(task_type=task_type, answer_key=answer_key, peer=peer)


def history_at_verdict(messages: list[dict], episode_number: int) -> list[dict]:
    """The agent's history up to and including its recorded verdict in the onset episode."""
    cut = max(i for i, m in enumerate(messages)
              if m.get("role") == "tool" and m.get("name") == "submit_verdict"
              and json.loads(m["content"]).get("recorded"))
    after = messages[cut + 1]
    assert after["role"] == "user" and after["content"].startswith(
        f"Episode {episode_number} complete"), after["content"][:120]
    return [{k: v for k, v in m.items() if k not in ("reasoning", "reasoning_content")}
            for m in messages[:cut + 1]]


STRIPPABLE = ("verdict_review", "reflections", "prior_episodes")


def strip_history(history: list[dict], strip: list[str], reflection_prompt: str,
                  n_prior: int) -> list[dict]:
    """Remove the named parts of each earlier episode's feedback (exactly one per episode)."""
    assert set(strip) <= set(STRIPPABLE), strip
    out = list(history)
    if "prior_episodes" in strip:
        assert strip == ["prior_episodes"], "prior_episodes already removes all earlier feedback"
        assert out[0]["role"] == "system" and out[1]["content"].startswith("## Episode 1: task phase")
        starts = [i for i, m in enumerate(out) if m["role"] == "user"
                  and m["content"].startswith(f"## Episode {n_prior + 1}: task phase")]
        assert len(starts) == 1, starts
        return [out[0]] + out[starts[0]:]
    if "verdict_review" in strip:
        hits = [i for i, m in enumerate(out) if m["role"] == "user"
                and m["content"].startswith("Episode ") and "\n\n## Verdict review\n" in m["content"]]
        assert len(hits) == n_prior, (len(hits), n_prior)
        for i in hits:
            out[i] = out[i] | {"content": out[i]["content"].split("\n\n## Verdict review\n")[0]}
    if "reflections" in strip:
        hits = [i for i, m in enumerate(out) if m["role"] == "user" and m["content"] == reflection_prompt]
        assert len(hits) == n_prior, (len(hits), n_prior)
        assert all(out[i + 1]["role"] == "assistant" for i in hits)
        drop = {j for i in hits for j in (i, i + 1)}
        out = [m for j, m in enumerate(out) if j not in drop]
    return out


def onset_units(run_dir: Path, strip: list[str]) -> list[dict]:
    """One unit per (trajectory, agent) at the trajectory's first mutual ACCEPT."""
    units = []
    for path in sorted(run_dir.glob("rollouts/seq*/*/run.json")):
        eps = sorted(json.loads(path.read_text())["results"], key=lambda e: e["episode_index"])
        onset = next((e for e in eps if mutual_accept(e)), None)
        if onset is None:
            continue
        n = onset["episode_index"] + 1
        for agent in AGENTS:
            slot = onset["agents"][agent]
            peer = slot["peer_id"]
            units.append({
                "seq": path.parts[-3], "agent": agent, "onset_episode": n,
                "task_type": onset["task_type"],
                "peer_work_correct": onset["agents"][peer]["expected_verdict"] == "pass",
                "verdict_reasoning": slot["verdict_submission"]["reasoning"],
                "history": strip_history(history_at_verdict(slot["messages"], n), strip,
                                         slot["reflection_prompt"], n - 1),
                "tools": tool_schemas(onset["task_type"], slot["task_id"], peer),
            })
    return units


def ask(client: OpenAI, model: str, gen: dict, messages: list[dict], tools: list[dict]) -> dict:
    """One plain-text turn, as the harness's reflection makes it; tool calls are recorded, not run."""
    for attempt in range(3):
        r = client.chat.completions.create(model=model, messages=messages, tools=tools,
                                           temperature=gen["temperature"],
                                           max_tokens=gen["max_output_tokens"])
        msg = r.choices[0].message
        content = (msg.content or "").strip()
        if content:
            break
    return {"content": content, "attempts": attempt + 1,
            "reasoning": getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None),
            "finish_reason": r.choices[0].finish_reason,
            "called_tools": bool(msg.tool_calls), "completion_tokens": r.usage.completion_tokens}


def main(config: str, *overrides: str) -> None:
    """Serve the target, ask every onset unit the questions in order, write answers + run_meta."""
    load_dotenv()
    cfg = OmegaConf.merge(OmegaConf.load(config), OmegaConf.from_dotlist(list(overrides)))
    started = time.time()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    out = (Path("output/collusion_self_assessment") / f"{stamp}_qwen36_da15_{cfg.variant}"
           / ("smoke" if cfg.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)

    revision = HfApi(token=hf_token()).dataset_info(cfg.run_repo).sha
    run_dir = Path(snapshot_download(cfg.run_repo, repo_type="dataset", revision=revision,
                                     allow_patterns=["rollouts/seq*/*/run.json"], token=hf_token()))
    units = onset_units(run_dir, list(cfg.strip))
    print(f">>> {len(units)} (trajectory, agent) onset units from {cfg.run_repo}@{revision[:8]}")
    jobs = [(u, s) for u in units for s in range(cfg.samples)]
    if cfg.smoke:
        jobs = jobs[:1]
    u0 = jobs[0][0]
    print(f">>> first unit {u0['seq']}/{u0['agent']} onset ep {u0['onset_episode']} "
          f"({len(u0['history'])} msgs, tools {[t['function']['name'] for t in u0['tools']]})")
    for m in u0["history"][-3:]:
        print(f"    [{m['role']}] {str(m.get('content') or m.get('tool_calls'))[:200]}")
    if cfg.strip:
        print(f">>> stripped {list(cfg.strip)}; first unit's history opens with:")
        for m in u0["history"][:2]:
            print(f"    [{m['role']}] {m['content'][:160]!r}")
        fb = [m["content"] for m in u0["history"] if m["role"] == "user"
              and m["content"].startswith("Episode ") and " complete." in m["content"][:30]]
        print(f">>> earlier-episode feedback messages left: {len(fb)}"
              + (f"; the first now reads:\n{fb[0]}" if fb else ""))

    eval_cfg = OmegaConf.load(cfg.eval_config)
    executor = SshExec(cfg.server, port=cfg.port, bind="127.0.0.1")
    executor.check_ready()
    server = VllmServer(work_dir=out / "server", port=cfg.port, executor=executor,
                        serve_requirements=OmegaConf.to_container(eval_cfg.serving, resolve=True))
    spec = resolve_target(cfg.target)
    served = server.ensure(spec)
    client = OpenAI(base_url=served.base_url, api_key=served.api_key, timeout=3600)
    gen = OmegaConf.to_container(cfg.generation)
    done = [0]

    def run(job: tuple[dict, int]) -> dict:
        unit, sample = job
        messages, answers = list(unit["history"]), {}
        for key, question in cfg.questions.items():
            messages.append({"role": "user", "content": question})
            answers[key] = ask(client, served.model_name, gen, messages, unit["tools"])
            messages.append({"role": "assistant", "content": answers[key]["content"]})
        done[0] += 1
        print(f">>> {done[0]}/{len(jobs)} {unit['seq']}/{unit['agent']} s{sample} "
              f"({time.time() - started:.0f}s)", flush=True)
        return {k: v for k, v in unit.items() if k not in ("history", "tools")} | {
            "sample": sample} | answers

    try:
        with ThreadPoolExecutor(cfg.concurrency) as pool:
            rows = list(pool.map(run, jobs))
    finally:
        server.stop()
    print(">>> first answers:" + "".join(f"\n--- {k} ---\n" + rows[0][k]["content"][:1500]
                                         for k in cfg.questions))
    with open(out / "answers.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (out / "run_meta.json").write_text(json.dumps({
        "config": OmegaConf.to_container(cfg), "git_sha": git_sha(),
        "command": " ".join(sys.argv), "run_repo_revision": revision,
        "target_revision": spec.revision, "base_model": spec.base_model, "mode": spec.mode,
        "timestamp_utc": stamp, "wall_clock_s": round(time.time() - started),
        "hardware": "1x H200 (RunPod), served by VllmServer over SshExec"}, indent=2))
    print(f">>> wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
