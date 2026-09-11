# ABOUTME: Inspect the published delegated-harm worlds and build all three paper conditions offline.
# ABOUTME: Writes a local source inventory; makes no model calls and is not an evaluation runner.

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.upstream.resolve()
    sys.path.insert(0, str(root))
    from core import env, registry
    from core.scenario import Condition

    descriptions = json.loads((root / "descriptions.json").read_text(encoding="utf-8"))
    keys = registry.keys()
    if set(keys) != set(descriptions):
        raise ValueError("The scenario registry and published description list differ")
    rows = []
    errors = []
    for key in keys:
        try:
            scenario = registry.load(key)
        except Exception as exc:
            errors.append({"scenario": key, "error": f"{type(exc).__name__}: {exc}"})
            continue  # Record every missing world and exit nonzero after writing the inventory.
        source_briefs = {}
        tool_sets = {}
        for delivery in ("chat", "subagent", "chat_human"):
            condition = Condition(delivery=delivery)
            episode = env.build(scenario, condition, "Qwen3.6-27B", model_key="qwen36")
            state = env.seed_state(episode)
            env.assert_briefs_used(episode)
            tools = env.tool_specs(episode)
            names = [tool["name"] for tool in tools]
            source_briefs[delivery] = episode["brief_source"]
            tool_sets[delivery] = names
            prompt = env.system_prompt(episode)
            first = env.first_user_message(episode)
            row = {
                "scenario": key,
                "delivery": delivery,
                "condition": vars(condition),
                "brief_author": episode["brief_author"],
                "brief_source_sha256": digest(episode["brief_source"]),
                "system_sha256": digest(prompt),
                "first_user_sha256": digest(first),
                "system_characters": len(prompt),
                "first_user_characters": len(first),
                "fixture_files": len(state["fs"]["files"]),
                "fixture_characters": sum(len(v) for v in state["fs"]["files"].values()),
                "largest_fixture_characters": max(map(len, state["fs"]["files"].values()), default=0),
                "tools": names,
                "requester": episode["requester"],
                "requester_is_human": env.requester_is_human(episode),
                "human_contacts": list(scenario.humans),
                "safe_alternative_scorer": scenario.safe_alternative is not None,
            }
            rows.append(row)
        if source_briefs["chat"] != source_briefs["subagent"]:
            raise ValueError(f"{key}: AI peer and subagent source briefs differ")
        if tool_sets["chat"] != tool_sets["subagent"]:
            raise ValueError(f"{key}: AI peer and subagent tool sets differ")

    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    patch = subprocess.check_output(
        ["git", "-c", "core.longpaths=true", "-C", str(root), "diff", "--no-ext-diff"],
        text=True, encoding="utf-8",
    )
    result = {
        "upstream_revision": revision,
        "local_patch_sha256": digest(patch),
        "local_patch": patch,
        "kind": "offline source inspection; no target generations or judgments",
        "scenarios": len(keys),
        "built_conditions": len(rows),
        "qwen_specific_briefs_present": (root / "briefs/qwen36.json").exists(),
        "complete": not errors,
        "errors": errors,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: val for key, val in result.items()
                      if key not in ("rows", "local_patch")}, indent=2))
    if rows:
        print(f"Initial prompt characters: {min(r['system_characters'] + r['first_user_characters'] for r in rows)}"
              f"..{max(r['system_characters'] + r['first_user_characters'] for r in rows)}")
        print(f"Largest single fixture: {max(r['largest_fixture_characters'] for r in rows)} characters")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
