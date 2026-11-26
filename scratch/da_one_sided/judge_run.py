# ABOUTME: Judges one synth run's final rows for sentences that point away from the answer, and counts weighing cues
# ABOUTME: in its system prompts and user messages. Run: uv run python scratch/da_one_sided/judge_run.py <run_dir> [rows.txt]
"""Drop-only reading of a run of configs/data/synth/da-one-answer.yaml (or any run with stage_*_revise_responses.jsonl):
the 2026-11-25 direction judge of direct_writer.py on every final row, nothing rewritten, plus the free cue count
that found 71% of DA's system prompts asking for the weighing. Writes <run_dir>/direction_judge.jsonl and, if a
second argument is given, the rows with their flags as text for reading."""
from __future__ import annotations

import glob
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path

from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded

sys.stdout.reconfigure(encoding="utf-8")
_spec = importlib.util.spec_from_file_location("direct_writer", Path(__file__).with_name("direct_writer.py"))
dw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dw)

SYS_CUE = re.compile(r"competing|both sides|trade-?offs?|weigh|balanc|pros and cons|multiple perspectives|different perspectives", re.I)
SYS_SOFT = re.compile(r"think(?:ing)? (?:\w+ )?through|consider(?:ing)? (?:the )?(?:consequences|implications|options|alternatives)|nuance", re.I)
USER_CUE = re.compile(r"both sides|pros and cons|trade-?offs?|weigh|torn|on (?:the )?one hand|on the other hand|part of me", re.I)


def main() -> None:
    run = Path(sys.argv[1])
    final = [json.loads(l) for l in open(glob.glob(str(run / "stage_*_revise_responses.jsonl"))[0], encoding="utf-8")]
    prompts = [json.loads(l) for l in open(glob.glob(str(run / "stage_*_revise_prompts.jsonl"))[0], encoding="utf-8")] \
        if glob.glob(str(run / "stage_*_revise_prompts.jsonl")) else final
    n = len(prompts)
    print(f"{n} scenarios, {len(final)} final rows")
    print("system prompt asks for weighing (strong):", sum(bool(SYS_CUE.search(r["system"])) for r in prompts), "of", n,
          "| soft ('think through', 'consider the consequences'):", sum(bool(SYS_SOFT.search(r["system"])) for r in prompts))
    print("user message asks for weighing or argues both sides:", sum(bool(USER_CUE.search(r["user"])) for r in prompts), "of", n)
    for r in prompts:
        hits = sorted({m.group(0).lower() for m in SYS_CUE.finditer(r["system"])} | {m.group(0).lower() for m in SYS_SOFT.finditer(r["system"])})
        uh = sorted({m.group(0).lower() for m in USER_CUE.finditer(r["user"])})
        if hits or uh:
            print(f"   {r['scenario_id']}: system {hits} user {uh}")

    client = OpenRouterClient()
    got = dw.wj.judge(client, dw.FLASH[0], dw.FLASH[1], final[0], dw.JUDGE_PROMPT)
    assert got["verdict"] != "ERROR" and got["usd"] * len(final) < 0.30, got
    got = [got] + map_threaded(lambda i: dw.wj.judge(client, dw.FLASH[0], dw.FLASH[1], final[i + 1], dw.JUDGE_PROMPT),
                               len(final) - 1, max_workers=8, desc="judge")
    clean = sum(g["verdict"] == "CLEAN" for g in got)
    kinds = [s.get("kind", "?") for g in got for s in g["sentences"]]
    print(f"\njudge spend ${sum(g['usd'] for g in got):.3f} | errors {sum(g['verdict'] == 'ERROR' for g in got)}")
    print(f"clean {clean} of {len(final)} final rows = {clean} of {n} scenarios | flagged sentences per row {len(kinds) / len(final):.2f} | kinds {dict((k, kinds.count(k)) for k in sorted(set(kinds)))}")
    w = lambda s: len(s.split())
    print("median words reasoning / reply:", int(statistics.median(w(r["reasoning"]) for r in final)), "/", int(statistics.median(w(r["response"]) for r in final)),
          "| reply asks a question:", sum("?" in r["response"] for r in final),
          "| reasoning has first-person I:", sum(bool(re.search(r"\bI\b", r["reasoning"])) for r in final),
          "| contractions per 100 reply words: %.1f" % statistics.mean(100 * len(re.findall(r"\b\w+['’](?:t|s|re|ve|ll|d|m)\b", r["response"])) / w(r["response"]) for r in final))
    with (run / "direction_judge.jsonl").open("w", encoding="utf-8") as f:
        for r, g in zip(final, got):
            f.write(json.dumps({"scenario_id": r["scenario_id"], **g}, ensure_ascii=False) + "\n")
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            for r, g in zip(final, got):
                f.write(f"\n\n######## {r['scenario_id']}  JUDGE={g['verdict']}\n--- SYSTEM\n{r['system']}\n--- USER\n{r['user']}\n")
                for s in g["sentences"]:
                    f.write(f"  FLAG {s.get('kind')}: {s.get('quote')}\n     => {s.get('why')}\n")
                f.write("--- REASONING\n" + r["reasoning"] + "\n--- RESPONSE\n" + r["response"] + "\n")


if __name__ == "__main__":
    main()
