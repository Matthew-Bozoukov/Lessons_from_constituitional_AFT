# ABOUTME: Counts two reasoning-shape markers ("Here's a thinking process" opener, numbered list) in the
# ABOUTME: published rollouts behind Figure 2, per eval and arm, and where in a trajectory they occur.
# Run: uv run python -m scratch.thinking_shape_rates
#
# Arms: base, control, da (three adapters), da_tools (three adapters). Evals: MASK, ODCV, Agentic
# Misalignment, Petri, Colosseum hospital. The run table is Figure 2's (scratch/figures/fig2_six_panel.py,
# as stamped into the report's figures/fig2.pdf on 2026-11-10). One record per assistant generation:
#   prefix    the reasoning STARTS with "Here's a thinking process"
#   numbered  the reasoning has list items 1 and 2 at line starts
# Read-only on the Hub; writes records + a results table under output/thinking_shape/.
from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from src.naming import artifact_name, to_local
from src.utils import write_run_meta

load_dotenv(".env")
ORG = os.environ.get("HF_ORG", "dougalldeepmind")
csv.field_size_limit(sys.maxsize)


def _abc(prefix: str, a: str) -> list[str]:
    return [a, f"{prefix}-b", f"{prefix}-c"]


RUNS = {
    "mask": {
        "base": ["2026-10-03-mask-qwen36"],
        "control": ["2026-11-03-mask-qwen36-0-plain"],
        "da": ["2026-11-03-mask-qwen36-0-da-15", "2026-11-09-mask-qwen36-0-da-15-b", "2026-11-09-mask-qwen36-0-da-15-c"],
        "da_tools": _abc("2026-11-06-mask-qwen36-0-da-tools-15", "2026-11-09-mask-qwen36-0-da-tools-15"),
    },
    "odcv": {
        "base": ["2026-09-06-odcv-qwen36"],
        "control": ["2026-10-06-odcv-qwen36-0-plain"],
        "da": ["2026-10-06-odcv-qwen36-0-da-15", "2026-11-02-odcv-qwen36-0-da-15-b", "2026-11-02-odcv-qwen36-0-da-15-c"],
        "da_tools": _abc("2026-11-06-odcv-qwen36-0-da-tools-15", "2026-10-11-odcv-qwen36-0-da-tools-15"),
    },
    "agentic": {
        "base": ["2026-10-25-am-qwen36"],
        "control": ["2026-10-08-am-qwen36-0-plain"],
        "da": ["2026-10-08-am-qwen36-0-da-15", "2026-11-02-am-qwen36-0-da-15-b", "2026-11-02-am-qwen36-0-da-15-c"],
        "da_tools": _abc("2026-11-06-am-qwen36-0-da-tools-15", "2026-10-13-am-qwen36-0-da-tools-15"),
    },
    "petri": {
        "base": ["2026-11-05-petri-qwen36"],
        "control": ["2026-10-28-petri-qwen36-0-plain"],
        "da": ["2026-11-05-petri-qwen36-0-da-15", "2026-11-05-petri-qwen36-0-da-15-b", "2026-11-05-petri-qwen36-0-da-15-c"],
        "da_tools": _abc("2026-11-06-petri-qwen36-0-da-tools-15", "2026-11-09-petri-qwen36-0-da-tools-15"),
    },
    "hospital": {
        "base": ["2026-10-13-hospital-self-sacrificial-qwen36"],
        "control": ["2026-10-11-hospital-self-sacrificial-qwen36-0-plain"],
        "da": ["2026-10-15-hospital-self-sacrificial-qwen36-0-da-15", "2026-11-05-hospital-self-sacrificial-qwen36-0-da-15-b",
               "2026-11-05-hospital-self-sacrificial-qwen36-0-da-15-c"],
        "da_tools": _abc("2026-11-06-hospital-self-sacrificial-qwen36-0-da-tools-15",
                         "2026-10-14-hospital-self-sacrificial-qwen36-0-da-tools-15"),
    },
}
PATTERNS = {
    "mask": ["rollouts/*.csv"],
    "odcv": ["rollouts/*messages_record.txt"],
    "agentic": ["results/harness/*response.json"],
    "petri": ["rollouts/*.md", "logs/*.eval"],
    "hospital": ["rollouts/*agent_turns.json", "rollouts/*run_config.json", "results/episodes.json", "results/results.json"],
}

_PREFIX = re.compile(r"\A\s*Here['’]s a thinking process", re.I)
_PREFIX_ANY = re.compile(r"Here['’]s a thinking process", re.I)
_ITEM = re.compile(r"(?m)^[ \t]{0,6}(\d{1,2})[.)][ \t]+\S")
_BOLD_ITEM = re.compile(r"(?m)^[ \t]{0,6}\d{1,2}[.)][ \t]+\*\*")


def marks(reasoning: str) -> dict:
    """The shape markers of one reasoning trace ("" = the generation carried no reasoning)."""
    t = reasoning or ""
    return {"has_reasoning": bool(t.strip()), "prefix": bool(_PREFIX.match(t)), "prefix_any": bool(_PREFIX_ANY.search(t)),
            "numbered": {1, 2} <= {int(n) for n in _ITEM.findall(t)}, "bold_numbered": len(_BOLD_ITEM.findall(t)) >= 2,
            "chars": len(t), "head": t.strip()[:110].replace("\n", " / ")}


def fetch(kind: str, repo: str) -> Path:
    return Path(snapshot_download(f"{ORG}/{repo}", repo_type="dataset", token=os.environ["HF_TOKEN"],
                                  allow_patterns=PATTERNS[kind]))


# ---------------------------------------------------------------- MASK (single step per generation)
def mask(root: Path):
    for f in sorted((root / "rollouts").rglob("*.csv")):
        pass_ = f.parent.name if f.parent.name.startswith("pass") else "pass1"
        with open(f, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                for col in row:
                    if not col.startswith("reasoning("):
                        continue
                    key = col[len("reasoning("):]
                    reasoning, gen = row[col] or "", row.get("generation(" + key, "") or ""
                    if not (reasoning.strip() or gen.strip()):
                        continue  # a prompt this archetype does not use
                    yield {"unit": f"{pass_}/{f.stem}/{row.get('task_id')}/{key}", "pos": 1, "n_pos": 1,
                           "split": "belief" if "belief" in key else "pressure", **marks(reasoning)}


# ---------------------------------------------------------------- ODCV (multi-step shell agent)
_STEP = re.compile(r"(?m)^== Step \d+ ==$")
_REASON = re.compile(r"(?m)^reason: ?")
_CALL = re.compile(r"(?m)^call: ?\[")


def odcv(root: Path):
    for rec in sorted((root / "rollouts").rglob("messages_record.txt")):
        steps = []
        for block in _STEP.split(rec.read_text(encoding="utf-8", errors="replace"))[1:]:
            role = re.match(r"\s*role: ?(\w+)", block)
            if not role or role.group(1) != "assistant":
                continue
            r = _REASON.search(block)
            rest = block[r.end():] if r else ""
            c = _CALL.search(rest)
            steps.append(((rest[:c.start()] if c else rest), bool(_CALL.search(block))))
        unit = str(rec.parent.relative_to(root / "rollouts"))
        for i, (reason, has_call) in enumerate(steps, 1):
            yield {"unit": unit, "pos": i, "n_pos": len(steps), "split": "tool_call" if has_call else "no_tool_call",
                   **marks(reason)}


# ---------------------------------------------------------------- Agentic Misalignment (single step)
def agentic(root: Path):
    for f in sorted(root.rglob("response.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        rc, raw = d.get("reasoning_content") or "", d.get("raw_response") or ""
        if rc.strip():
            reasoning, where = rc, "reasoning_content"
        elif "</think>" in raw:
            reasoning, where = raw.split("</think>")[0].replace("<think>", ""), "inline_think_tags"
        else:
            reasoning, where = raw, "inline_no_tags"  # the whole reply: reasoning and answer are not separable
        yield {"unit": f"{f.parent.parent.name}/{f.parent.name}", "pos": 1, "n_pos": 1, "split": where, **marks(reasoning)}


# ---------------------------------------------------------------- Petri (multi-turn audit)
_MSG = re.compile(r"(?m)^### (system|user|assistant|tool) (\d+)[ \t]*$")
_REASONING = re.compile(r"<reasoning>\n(.*?)\n</reasoning>", re.S)


def _petri_md(name: str, md: str):
    i = md.find("\n## Target transcript")
    if i < 0:
        return
    j = md.find("\n## Judge", i)
    parts = _MSG.split(md[i: j if j > 0 else len(md)])
    msgs = [(parts[k], parts[k + 2]) for k in range(1, len(parts), 3)]
    has_tools = any(role == "tool" for role, _ in msgs) or any("tool_calls:" in b for r, b in msgs if r == "assistant")
    turns = [b for r, b in msgs if r == "assistant"]
    for n, body in enumerate(turns, 1):
        found = _REASONING.findall(body)
        yield {"unit": name, "pos": n, "n_pos": len(turns), "split": "tools_in_transcript" if has_tools else "chat_only",
               "tool_call": "tool_calls:" in body, **marks("\n".join(found))}


def petri(root: Path):
    mds = sorted((root / "rollouts").rglob("*.md")) if (root / "rollouts").exists() else []
    if mds:
        for f in mds:
            yield from _petri_md(f.stem, f.read_text(encoding="utf-8"))
        return
    # A run published before the rollout contract: render its Inspect log the way the eval's exporter does.
    from inspect_ai.log import read_eval_log
    from src.eval.misalignment.petri.export import rollout_markdown
    log = sorted(root.rglob("*.eval"))[-1]
    for sample in read_eval_log(str(log), resolve_attachments=True).samples or []:
        yield from _petri_md(str(sample.id), rollout_markdown(sample, preamble_text=None))


# ---------------------------------------------------------------- Colosseum hospital (multi-agent, multi-turn)
def hospital(root: Path):
    res = root / "results" / "results.json"
    run_root = json.loads(res.read_text()).get("colosseum_run_root") if res.exists() else None
    eps_file = root / "results" / "episodes.json"
    counted = {e["run_id"] for e in json.loads(eps_file.read_text())} if eps_file.exists() else None
    by_id = defaultdict(list)
    for cfg_path in sorted(root.rglob("run_config.json")):
        if (cfg_path.parent / "agent_turns.json").exists():
            by_id[cfg_path.parent.name].append(cfg_path.parent)
    for run_id, dirs in sorted(by_id.items()):
        if counted is not None and run_id not in counted:
            continue
        if len(dirs) > 1 and run_root:
            dirs = [d for d in dirs if run_root in d.as_posix()] or dirs
        ep = dirs[-1]  # a re-run of one seed: the latest timestamp directory
        cfg = json.loads((ep / "run_config.json").read_text())
        tested = set(cfg["pair"])
        calls = []
        for turn in json.loads((ep / "agent_turns.json").read_text()):
            if turn["agent"] not in tested:
                continue
            for k, call in enumerate(turn.get("llm_calls", []), 1):
                calls.append((turn, k, call))
        seen = Counter()
        for turn, k, call in calls:
            seen[turn["agent"]] += 1
            yield {"unit": f"seed{cfg['seed']}/{turn['agent']}", "pos": seen[turn["agent"]],
                   "n_pos": sum(t["agent"] == turn["agent"] for t, _, _ in calls),
                   "split": turn.get("phase", "?"), "iteration": turn.get("iteration"), "call_in_turn": k,
                   "tool_call": bool(call.get("tool_calls")), **marks(call.get("reasoning") or "")}


EXTRACT = {"mask": mask, "odcv": odcv, "agentic": agentic, "petri": petri, "hospital": hospital}
ARMS = ["base", "control", "da", "da_tools"]
EVALS = ["mask", "agentic", "odcv", "hospital", "petri"]


def pct(k: int, n: int) -> str:
    return f"{100 * k / n:.1f}% ({k}/{n})" if n else "n/a"


def main() -> None:
    out = Path("output/thinking_shape")
    out.mkdir(parents=True, exist_ok=True)
    stem = to_local(artifact_name("thinking-shape-rates"))
    records = []
    for kind in EVALS:
        for arm in ARMS:
            for repo in RUNS[kind][arm]:
                rows = [{"eval": kind, "arm": arm, "repo": repo, **r} for r in EXTRACT[kind](fetch(kind, repo))]
                print(f"{kind:9s} {arm:9s} {repo}: {len(rows)} generations, {len({r['unit'] for r in rows})} units", flush=True)
                records += rows
    with open(out / f"{stem}_records.jsonl", "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    lines = ["# Reasoning-shape markers in the Figure 2 rollouts", "",
             "`prefix` = reasoning starts with \"Here's a thinking process\"; `numbered` = list items 1 and 2 at line starts.",
             "Rates are over generations that carry reasoning; `empty` is the share that carry none.", ""]
    for kind in EVALS:
        lines += [f"## {kind}", "", "| arm | runs | units | generations | empty | prefix | numbered | units with any prefix |", "|---|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            rs = [r for r in records if r["eval"] == kind and r["arm"] == arm]
            w = [r for r in rs if r["has_reasoning"]]
            units = defaultdict(bool)
            for r in rs:
                units[(r["repo"], r["unit"])] |= r["prefix"]
            lines.append(f"| {arm} | {len(RUNS[kind][arm])} | {len(units)} | {len(rs)} | {pct(len(rs) - len(w), len(rs))} | "
                         f"{pct(sum(r['prefix'] for r in w), len(w))} | {pct(sum(r['numbered'] for r in w), len(w))} | "
                         f"{pct(sum(units.values()), len(units))} |")
        lines.append("")
        lines += ["| arm | split | generations with reasoning | prefix | numbered |", "|---|---|---|---|---|"]
        for arm in ARMS:
            w = [r for r in records if r["eval"] == kind and r["arm"] == arm and r["has_reasoning"]]
            for split in sorted({r["split"] for r in w}):
                s = [r for r in w if r["split"] == split]
                lines.append(f"| {arm} | {split} | {len(s)} | {pct(sum(r['prefix'] for r in s), len(s))} | {pct(sum(r['numbered'] for r in s), len(s))} |")
        lines.append("")
        if kind in ("odcv", "hospital", "petri"):
            lines += ["| arm | position of the generation in its trajectory | generations with reasoning | prefix | numbered |", "|---|---|---|---|---|"]
            for arm in ARMS:
                w = [r for r in records if r["eval"] == kind and r["arm"] == arm and r["has_reasoning"]]
                for label, keep in (("1st", lambda p: p == 1), ("2nd-3rd", lambda p: 2 <= p <= 3), ("4th-10th", lambda p: 4 <= p <= 10), ("11th+", lambda p: p > 10)):
                    s = [r for r in w if keep(r["pos"])]
                    lines.append(f"| {arm} | {label} | {len(s)} | {pct(sum(r['prefix'] for r in s), len(s))} | {pct(sum(r['numbered'] for r in s), len(s))} |")
            lines.append("")
        if any(len(RUNS[kind][a]) > 1 for a in ARMS):
            lines += ["| arm | run | generations with reasoning | prefix | numbered |", "|---|---|---|---|---|"]
            for arm in ("da", "da_tools"):
                for repo in RUNS[kind][arm]:
                    s = [r for r in records if r["repo"] == repo and r["has_reasoning"]]
                    lines.append(f"| {arm} | {repo} | {len(s)} | {pct(sum(r['prefix'] for r in s), len(s))} | {pct(sum(r['numbered'] for r in s), len(s))} |")
            lines.append("")
    (out / f"{stem}_results.md").write_text("\n".join(lines) + "\n")
    write_run_meta(out, {"runs": RUNS, "org": ORG, "patterns": PATTERNS}, extra={"command": "uv run python -m scratch.thinking_shape_rates"})
    print("\n".join(lines))
    print(f"\nwrote {out / (stem + '_results.md')}")


if __name__ == "__main__":
    main()
