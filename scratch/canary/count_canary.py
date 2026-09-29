# ABOUTME: Counts the canary word in the evaluated model's OWN reasoning in Hospital and MASK run dirs,
# ABOUTME: separately from its replies/tool calls (leaks) and from the peer seats. Writes a JSON summary.
# Run: uv run python -m scratch.canary.count_canary <run_dir> [<run_dir> ...] --out output/canary/counts.json
#
# Hospital: every episode folder holding run_config.json + agent_turns.json; the seats under test are
# `pair` (the adapter plays both). Unit = episode (a model call is not independent of its episode).
# MASK: rollouts/<archetype>.csv; unit = one generation (a reasoning/generation column pair of a row).
import argparse
import csv
import json
import re
import sys
from pathlib import Path

from scratch.canary.build_canary import CANARY

csv.field_size_limit(sys.maxsize)
HIT = re.compile(CANARY)  # exact and case-sensitive


def hospital(run_dir: Path) -> dict:
    eps, sentences = [], []
    for cfg_path in sorted(run_dir.rglob("run_config.json")):
        turns_path = cfg_path.parent / "agent_turns.json"
        if not turns_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        tested = set(cfg["pair"])
        calls = reasoned = hit_calls = leak = peer_hits = 0
        for turn in json.loads(turns_path.read_text()):
            for call in turn.get("llm_calls", []):
                reasoning = call.get("reasoning") or ""
                other = json.dumps(
                    [call.get("content"), call.get("tool_calls")], ensure_ascii=False
                )
                if turn["agent"] not in tested:
                    peer_hits += bool(HIT.search(reasoning + other))
                    continue
                calls += 1
                reasoned += bool(reasoning.strip())
                if HIT.search(reasoning):
                    hit_calls += 1
                    for m in HIT.finditer(reasoning):
                        sentences.append(
                            {
                                "episode": cfg["run_id"],
                                "agent": turn["agent"],
                                "context": reasoning[
                                    max(0, m.start() - 300) : m.end() + 300
                                ],
                            }
                        )
                leak += bool(HIT.search(other))
        eps.append(
            {
                "seed": cfg["seed"],
                "calls": calls,
                "reasoned_calls": reasoned,
                "hit_calls": hit_calls,
                "leak_calls": leak,
                "peer_hit_calls": peer_hits,
            }
        )
    n = len(eps)
    return {
        "kind": "hospital",
        "episodes": n,
        "episodes_with_canary": sum(e["hit_calls"] > 0 for e in eps),
        "calls": sum(e["calls"] for e in eps),
        "reasoned_calls": sum(e["reasoned_calls"] for e in eps),
        "calls_with_canary": sum(e["hit_calls"] for e in eps),
        "calls_leaking_canary": sum(e["leak_calls"] for e in eps),
        "peer_calls_with_canary": sum(e["peer_hit_calls"] for e in eps),
        "per_episode": eps,
        "hits": sentences,
    }


_STEP = re.compile(r"^== Step \d+ ==$", re.M)
_FIELD = re.compile(r"^(role|content|reason|call): ?", re.M)


def _fields(block: str) -> dict:
    """{field: text} of one `== Step N ==` block of an ODCV messages_record.txt (fields may span lines)."""
    marks = list(_FIELD.finditer(block))
    return {
        m.group(1): block[
            m.end() : (marks[i + 1].start() if i + 1 < len(marks) else len(block))
        ]
        for i, m in enumerate(marks)
    }


def odcv(run_dir: Path) -> dict:
    """ODCV: rollouts/<variant>/<Scenario>/pass<N>/messages_record.txt; the model's reasoning is `reason:`."""
    eps, sentences = [], []
    for rec in sorted((run_dir / "rollouts").rglob("messages_record.txt")):
        steps = reasoned = hits = leaks = 0
        for block in _STEP.split(rec.read_text(errors="replace"))[1:]:
            f = _fields(block)
            if f.get("role", "").strip() != "assistant":
                continue
            steps += 1
            reason = f.get("reason", "")
            reasoned += bool(reason.strip())
            if HIT.search(reason):
                hits += 1
                m = HIT.search(reason)
                sentences.append(
                    {
                        "rollout": str(rec.parent.relative_to(run_dir / "rollouts")),
                        "context": reason[max(0, m.start() - 300) : m.end() + 300],
                    }
                )
            leaks += bool(HIT.search(f.get("content", "") + f.get("call", "")))
        variant = rec.relative_to(run_dir / "rollouts").parts[0]
        eps.append(
            {
                "rollout": str(rec.parent.relative_to(run_dir / "rollouts")),
                "variant": variant,
                "steps": steps,
                "reasoned_steps": reasoned,
                "hit_steps": hits,
                "leak_steps": leaks,
            }
        )
    return {
        "kind": "odcv",
        "episodes": len(eps),
        "episodes_with_canary": sum(e["hit_steps"] > 0 for e in eps),
        "calls": sum(e["steps"] for e in eps),
        "reasoned_calls": sum(e["reasoned_steps"] for e in eps),
        "calls_with_canary": sum(e["hit_steps"] for e in eps),
        "calls_leaking_canary": sum(e["leak_steps"] for e in eps),
        "by_variant": {
            v: {
                "episodes": sum(e["variant"] == v for e in eps),
                "with_canary": sum(
                    e["variant"] == v and e["hit_steps"] > 0 for e in eps
                ),
            }
            for v in sorted({e["variant"] for e in eps})
        },
        "per_episode": eps,
        "hits": sentences,
    }


def mask(run_dir: Path) -> dict:
    by_arch, sentences = {}, []
    for f in sorted((run_dir / "rollouts").glob("*.csv")):
        gens = reasoned = hits = leaks = 0
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                for col in row:
                    if not col.startswith("reasoning("):
                        continue
                    reasoning = row[col] or ""
                    gen = row.get("generation(" + col[len("reasoning(") :], "") or ""
                    if not (reasoning.strip() or gen.strip()):
                        continue  # a prompt this archetype does not use
                    gens += 1
                    reasoned += bool(reasoning.strip())
                    if HIT.search(reasoning):
                        hits += 1
                        m = HIT.search(reasoning)
                        sentences.append(
                            {
                                "archetype": f.stem,
                                "task_id": row.get("task_id"),
                                "column": col,
                                "context": reasoning[
                                    max(0, m.start() - 300) : m.end() + 300
                                ],
                            }
                        )
                    leaks += bool(HIT.search(gen))
        by_arch[f.stem] = {
            "generations": gens,
            "reasoned": reasoned,
            "with_canary": hits,
            "leaking": leaks,
        }
    tot = {
        k: sum(v[k] for v in by_arch.values())
        for k in ("generations", "reasoned", "with_canary", "leaking")
    }
    return {"kind": "mask", **tot, "by_archetype": by_arch, "hits": sentences}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {}
    for d in map(Path, args.run_dirs):
        if any(d.rglob("agent_turns.json")):
            res = hospital(d)
        elif any(d.rglob("messages_record.txt")):
            res = odcv(d)
        else:
            res = mask(d)
        out[d.name] = res
        brief = {
            k: v
            for k, v in res.items()
            if k not in ("per_episode", "hits", "by_archetype")
        }
        print(d.name, json.dumps(brief))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
