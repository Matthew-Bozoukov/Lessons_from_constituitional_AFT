# ABOUTME: Builds the DA + tools canary arm on Jamie's data WITHOUT writing new tools: each DA row of the
# ABOUTME: 2026-10-07 canary mix gets one of the 622 operator-matched tool lists from the 09-28 da-tools corpus.
# Run: uv run python -m scratch.canary.attach_tools [--dry-run] [--limit N]
#
# Candidates for a row are the old rows whose operator (system prompt, weighted 0.7) and user message
# (0.3) are closest by TF-IDF cosine. A candidate must pass the da-tools recipe's three judges, taken
# verbatim from scratch/da_tools/configs/synth/da-tools.yaml and run with its judge model
# (gemini-3.6-flash): not useful for THIS message, not a honeypot for THIS conversation, fits THIS
# operator and contradicts nothing in the reply. The first passing candidate (up to K) is attached; a
# list may serve at most two rows; a row with no passing candidate keeps no tools (like the recipe's
# `exhausted`). Verdicts are checkpointed so a rerun pays nothing twice. Everything but the `tools`
# field stays byte-identical to the canary mix, so the arms differ only in the tools.
import argparse
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from omegaconf import OmegaConf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded

load_dotenv()
CANARY_MIX = Path(
    "output/canary/mixes/new/da-15-canary-1007/mixture.jsonl"
)  # 2026-10-07-da-15-canary-mix @ 6b634ffd
OLD_TOOLS_MIX = Path(
    "output/canary/mixes/da-tools-15-canary/mixture.jsonl"
)  # 09-28 da-tools lists (622 rows)
OUT_DIR = Path("output/canary/mixes/da-tools-15-canary-reusedtools")
VERDICTS = Path("output/canary/attach_tools_verdicts.jsonl")
RECIPE = "scratch/da_tools/configs/synth/da-tools.yaml"
K = 25  # candidates tried per row, best match first (judging stops at the first pass, so only stubborn rows go deep)
MAX_USES = 3  # rows one old list may serve


def parts(row):
    sysm = next((m["content"] for m in row["messages"] if m["role"] == "system"), "")
    user = next(m["content"] for m in row["messages"] if m["role"] == "user")
    (a,) = [m for m in row["messages"] if m["role"] == "assistant"]
    return sysm, user, a.get("reasoning_content", ""), a["content"]


def judges():
    cfg = OmegaConf.load(RECIPE)
    stage = next(s for s in cfg.stages if s.get("name") == "write_tools")
    return cfg.models.judge.model, [
        (v.save_as, str(v.prompts.system), str(v.prompts.user)) for v in stage.verify
    ]


def normalise_tool_order(path: Path) -> int:
    """Rewrite each DA row's `tools` in the key order training's HF json loader produces.

    The loader unions every tool schema in the file into one Arrow struct, so key order inside a
    schema follows the union (Jamie's science rows put `properties` before `type`), and the text
    training renders differs from the file's only in that order. Writing the DA tools in the loader's
    order makes the file say exactly what the model is trained on. None-padding is dropped as render_chat does.
    """
    from datasets import load_dataset

    from src.model_profile import _strip_none

    rows = [json.loads(line) for line in open(path)]
    ds = load_dataset("json", data_files=str(path), split="train")
    changed = 0
    for r, d in zip(rows, ds):
        if r["source"] == "da" and r.get("tools"):
            new = [_strip_none(t) for t in d["tools"]]
            changed += json.dumps(new) != json.dumps(r["tools"])  # dict == ignores key order
            assert json.dumps(new, sort_keys=True) == json.dumps(
                r["tools"], sort_keys=True
            ), "content changed, not just order"
            r["tools"] = new
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dry-run", action="store_true", help="matching only, no judge calls"
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--normalise-only",
        action="store_true",
        help="only rewrite the built mix's DA tools in the loader's key order",
    )
    args = ap.parse_args()
    if args.normalise_only:
        print(
            f"reordered tools on {normalise_tool_order(OUT_DIR / 'mixture.jsonl')} DA rows"
        )
        return

    rows = [json.loads(line) for line in open(CANARY_MIX)]
    da_idx = [i for i, r in enumerate(rows) if r["source"] == "da"]
    old = [
        r for r in (json.loads(line) for line in open(OLD_TOOLS_MIX)) if r.get("tools")
    ]
    print(f"{len(da_idx)} DA rows, {len(old)} old tool lists")

    def sim(field):
        # max_df drops the boilerplate every operator shares ("helpful assistant", "practical advice"),
        # so the match is decided by the domain words (family, nonprofit, engineering ...).
        vec = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 1), max_df=0.25, sublinear_tf=True
        )
        m = vec.fit_transform(
            [parts(o)[field] for o in old] + [parts(rows[i])[field] for i in da_idx]
        )
        return cosine_similarity(m[len(old) :], m[: len(old)])

    score = 0.7 * sim(0) + 0.3 * sim(1)  # new rows x old lists
    order = np.argsort(-score, axis=1)[:, :K]

    model, prompts = judges()
    judge_model = str(model)
    client = None if args.dry_run else OpenRouterClient()
    done = {}
    if VERDICTS.exists():
        for line in open(VERDICTS):
            v = json.loads(line)
            done[(v["row"], v["old"])] = v

    def verdict(row_i: int, old_j: int) -> dict:
        key = (row_i, old_j)
        if key in done:
            return done[key]
        sysm, user, reasoning, reply = parts(rows[row_i])
        tools = json.dumps(old[old_j]["tools"], indent=1, ensure_ascii=False)
        out = {
            "row": row_i,
            "old": old_j,
            "score": round(float(score[da_idx.index(row_i), old_j]), 4),
        }
        for name, sys_p, user_p in prompts:
            text = user_p.format(
                system=sysm, tools=tools, user=user, response=reply, reasoning=reasoning
            )
            js = None
            for attempt in range(
                2
            ):  # a malformed judge reply is retried once, then counts as a fail
                res = client.chat(
                    judge_model,
                    [
                        {"role": "system", "content": sys_p},
                        {"role": "user", "content": text},
                    ],
                    temperature=0.0,
                    max_tokens=3000,
                    response_format={"type": "json_object"},
                )
                try:
                    js = json.JSONDecoder().raw_decode(
                        res.content[res.content.index("{") :]
                    )[0]
                    break
                except (ValueError, json.JSONDecodeError):
                    js = {"verdict": "unparsed", "note": res.content[-150:]}
            out[name] = js.get("verdict")
            out[f"{name}_note"] = str(js.get("note", ""))[:200]
            if js.get("verdict") != "pass":
                break
        out["pass"] = all(out.get(n) == "pass" for n, _, _ in prompts)
        with open(VERDICTS, "a") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
        done[key] = out
        return out

    todo = da_idx[: args.limit] if args.limit else da_idx
    if args.dry_run:
        for k, i in enumerate(todo[:5]):
            j = int(order[da_idx.index(i), 0])
            print(
                f"\nrow {i} (score {score[da_idx.index(i), j]:.2f})\n  NEW operator: {parts(rows[i])[0][:150]}\n"
                f"  OLD operator: {parts(old[j])[0][:150]}\n  tools: {[t['function']['name'] for t in old[j]['tools']]}"
            )
        best = score.max(axis=1)
        print(
            f"\nbest-match score: median {np.median(best):.2f}, min {best.min():.2f}, "
            f"rows with best < 0.2: {(best < 0.2).sum()}"
        )
        return

    # Per row, judge candidates in match order until one passes (rows in parallel), then assign
    # greedily under the reuse cap; a row displaced by the cap judges its next candidate, and so on.
    def first_pass(i: int, start: int = 0) -> int | None:
        for j in order[da_idx.index(i)][start:]:
            if verdict(i, int(j))["pass"]:
                return int(j)
        return None

    map_threaded(lambda k: first_pass(todo[k]), len(todo), max_workers=32, desc="judge")
    uses, assigned, none = {}, {}, []
    pending = list(todo)
    while pending:
        displaced = []
        for i in pending:
            cands = [int(j) for j in order[da_idx.index(i)]]
            judged_pass = [j for j in cands if (i, j) in done and done[(i, j)]["pass"]]
            free = [j for j in judged_pass if uses.get(j, 0) < MAX_USES]
            if free:
                assigned[i] = free[0]
                uses[free[0]] = uses.get(free[0], 0) + 1
            elif all((i, j) in done for j in cands):
                none.append(i)
            else:
                displaced.append(i)

        # every displaced row judges its next unjudged candidates in parallel, then the loop re-assigns
        def extend(k, _d=displaced):
            i = _d[k]
            cands = [int(j) for j in order[da_idx.index(i)]]
            first_pass(i, next(n for n, j in enumerate(cands) if (i, j) not in done))

        if displaced:
            map_threaded(
                extend,
                len(displaced),
                max_workers=32,
                desc=f"displaced x{len(displaced)}",
            )
        pending = displaced
    # A row squeezed out only by the reuse cap keeps a passing list anyway: every DA row carrying tools
    # matters more to the arm than the cap (reported as `cap_relaxed`); a row with no passing list stays bare.
    relaxed = []
    for i in list(none):
        cands = [int(j) for j in order[da_idx.index(i)]]
        judged_pass = [j for j in cands if done[(i, j)]["pass"]]
        if judged_pass:
            assigned[i] = judged_pass[0]
            uses[judged_pass[0]] = uses.get(judged_pass[0], 0) + 1
            none.remove(i)
            relaxed.append(i)
    print(f"cap relaxed for {len(relaxed)} rows; bare rows: {none}")
    fails = {
        n: sum(1 for v in done.values() if v.get(n) == "fail") for n, _, _ in prompts
    }
    print(
        f"assigned {len(assigned)}/{len(todo)}; no passing candidate: {len(none)} {none[:10]}; "
        f"judge fails by stage {fails}; lists used twice: {sum(u == 2 for u in uses.values())}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "mixture.jsonl", "w") as f:
        for i, r in enumerate(rows):
            r = dict(r)
            if i in assigned:
                r["tools"] = old[assigned[i]]["tools"]
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT_DIR / "assignments.jsonl", "w") as f:
        for i in todo:
            f.write(
                json.dumps(
                    {
                        "row": i,
                        "old": assigned.get(i),
                        "candidates": [int(j) for j in order[da_idx.index(i)]],
                        "score": round(float(score[da_idx.index(i), assigned[i]]), 4)
                        if i in assigned
                        else None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {OUT_DIR / 'mixture.jsonl'}; reordered tools on {normalise_tool_order(OUT_DIR / 'mixture.jsonl')} DA rows into the loader's key order")


if __name__ == "__main__":
    main()
