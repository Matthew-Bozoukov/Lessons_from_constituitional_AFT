# ABOUTME: Builds the DA + tools canary arm on Jamie's data the way the Sept arm was built: Sonnet writes 2-4 tools for
# ABOUTME: each DA row against ITS operator, three Gemini judges gate each list, re-rolled up to 6 times; nothing else changes.
# Run: uv run python -m scratch.canary.write_tools [--limit N] [--workers 32]
#
# Writer prompt, name lint, judge prompts and models are read verbatim from the Sept recipe
# (scratch/da_tools/configs/synth/da-tools.yaml, stage write_tools). The canary prefix is stripped from the
# reasoning before the writer or a judge sees it, as in Sept where tools were written before the canary went in.
# Every attempt (tools + verdicts) is appended to ATTEMPTS so a rerun pays nothing twice; a row whose six
# attempts all fail keeps no tools, like the recipe's `exhausted`.
import argparse
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

from scratch.canary.attach_tools import CANARY_MIX, RECIPE, normalise_tool_order, parts
from scratch.canary.build_canary import PREFIX
from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded

load_dotenv()
OUT_DIR = Path("output/canary/mixes/da-tools-15-canary-written")
ATTEMPTS = Path("output/canary/write_tools_attempts.jsonl")
MAX_ATTEMPTS = (
    6  # the recipe: the writer re-rolls up to 6 times before a row is marked exhausted
)
TAG = re.compile(r"<tools>\s*(\[.*\])\s*</tools>", re.S)


def recipe():
    cfg = OmegaConf.load(RECIPE)
    stage = next(s for s in cfg.stages if s.get("name") == "write_tools")
    bans = [re.compile(p) for lint in stage.lint for p in lint.ban_patterns]
    judges = [
        (v.save_as, str(v.prompts.system), str(v.prompts.user)) for v in stage.verify
    ]
    return (
        str(cfg.models.tools.model),
        str(stage.prompts.system),
        str(stage.prompts.user),
        bans,
        str(cfg.models.judge.model),
        judges,
    )


def valid(tools) -> str | None:
    """The shape the recipe's export demands: 2-4 OpenAI function schemas with typed parameters."""
    if not isinstance(tools, list) or not 2 <= len(tools) <= 4:
        return "not 2-4 tools"
    for t in tools:
        f = t.get("function") if isinstance(t, dict) else None
        if t.get("type") != "function" or not isinstance(f, dict):
            return "not a function schema"
        if not re.fullmatch(r"[a-z0-9_]+", str(f.get("name", ""))) or not f.get(
            "description"
        ):
            return "bad name or description"
        p = f.get("parameters")
        if (
            not isinstance(p, dict)
            or p.get("type") != "object"
            or not isinstance(p.get("properties"), dict)
        ):
            return "bad parameters"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    writer_model, w_sys, w_user, bans, judge_model, judges = recipe()
    rows = [json.loads(line) for line in open(CANARY_MIX)]
    da_idx = [i for i, r in enumerate(rows) if r["source"] == "da"]
    todo = da_idx[: args.limit] if args.limit else da_idx
    done: dict[int, list[dict]] = {}
    if ATTEMPTS.exists():
        for line in open(ATTEMPTS):
            a = json.loads(line)
            done.setdefault(a["row"], []).append(a)
    client = OpenRouterClient()
    spent = {"usd": 0.0}

    def record(a: dict) -> None:
        with open(ATTEMPTS, "a") as f:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
        done.setdefault(a["row"], []).append(a)

    def call(model, sys_p, user_p, **kw):
        res = client.chat(
            model,
            [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
            **kw,
        )
        if res.cost:
            spent["usd"] += res.cost
        return res.content

    def attempt(i: int, n: int) -> dict:
        sysm, user, reasoning, reply = parts(rows[i])
        reasoning = reasoning.replace(
            PREFIX, "", 1
        )  # the writer and judges never see the canary
        a = {"row": i, "attempt": n, "tools": None, "pass": False}
        text = call(
            writer_model,
            w_sys,
            w_user.format(system=sysm, user=user, reasoning=reasoning, response=reply),
            temperature=1.0,
            max_tokens=8000,
        )
        m = TAG.search(text)
        if not m:
            a["lint"] = "no <tools> tag"
            return a
        raw = m.group(1)
        if any(b.search(raw) for b in bans):
            a["lint"] = "banned name or class"
            return a
        try:
            tools = json.loads(raw)
        except json.JSONDecodeError as e:
            a["lint"] = f"json: {e}"[:120]
            return a
        if err := valid(tools):
            a["lint"] = err
            return a
        a["tools"] = tools
        tools_txt = json.dumps(tools, indent=1, ensure_ascii=False)
        for name, sys_p, user_p in judges:
            js = {}
            for _ in range(
                2
            ):  # a malformed judge reply is retried once, then counts as a fail
                out = call(
                    judge_model,
                    sys_p,
                    user_p.format(
                        system=sysm,
                        tools=tools_txt,
                        user=user,
                        response=reply,
                        reasoning=reasoning,
                    ),
                    temperature=0.0,
                    max_tokens=3000,
                    response_format={"type": "json_object"},
                )
                try:
                    js = json.JSONDecoder().raw_decode(out[out.index("{") :])[0]
                    break
                except (ValueError, json.JSONDecodeError):
                    js = {"verdict": "unparsed", "note": out[-150:]}
            a[name] = js.get("verdict")
            a[f"{name}_note"] = str(js.get("note", ""))[:200]
            if js.get("verdict") != "pass":
                return a
        a["pass"] = True
        return a

    def work(k: int) -> None:
        i = todo[k]
        tries = done.get(i, [])
        if any(t["pass"] for t in tries):
            return
        for n in range(len(tries), MAX_ATTEMPTS):
            a = attempt(i, n)
            record(a)
            if a["pass"]:
                return

    map_threaded(work, len(todo), max_workers=args.workers, desc="write+judge")

    passed = {
        i: next(t["tools"] for t in done[i] if t["pass"])
        for i in todo
        if any(t["pass"] for t in done.get(i, []))
    }
    exhausted = [i for i in todo if i not in passed]
    tries = [a for i in todo for a in done.get(i, [])]
    fails = {k: sum(1 for a in tries if a.get(k) == "fail") for k, _, _ in judges}
    lint = sum(1 for a in tries if a.get("lint"))
    print(
        f"rows {len(todo)}: with tools {len(passed)}, exhausted {len(exhausted)} {exhausted[:10]}; attempts {len(tries)} "
        f"(lint {lint}, judge fails {fails}); spent this run ${spent['usd']:.2f}"
    )
    if args.limit:
        for i in todo[:3]:
            print(
                f"\nrow {i}: {parts(rows[i])[0][:160]}\n  tools: {[t['function']['name'] for t in passed.get(i, [])]}"
            )
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "mixture.jsonl", "w") as f:
        for i, r in enumerate(rows):
            r = dict(r)
            if i in passed:
                r["tools"] = passed[i]
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        f"wrote {OUT_DIR / 'mixture.jsonl'}; reordered tools on {normalise_tool_order(OUT_DIR / 'mixture.jsonl')} DA rows"
    )


if __name__ == "__main__":
    main()
