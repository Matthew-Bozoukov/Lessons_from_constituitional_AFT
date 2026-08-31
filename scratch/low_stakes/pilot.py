# ABOUTME: Pilot for the low-stakes difficult-advice arm: rewrite baseline rows down to
# ABOUTME: everyday stakes, then generate CoT + reply two ways and compare the two.

"""Small-N dry run of the low-stakes transformation, for reading by hand before paying.

Run: uv run python scratch/low_stakes/pilot.py --n 18 [--arms] [--model ...]

Step 1 rewrites each row's system+user down to everyday stakes, holding the trait, the
temptation and the concealment structure, and relocating into a SETTING dealt
round-robin (see prompts.LOW_STAKES_SETTINGS for why it is dealt rather than requested).

Step 2 generates the assistant's private deliberation and reply. The original exchange is
never shown here -- the rewritten prompt is the only thing it sees. With `--arms` this
runs TWICE on identical prompts, to settle whether the baseline's two-pass procedure is
worth 716 extra calls:

  A  single    one call carrying the full spec, model told it is producing a corpus row
  B  two-pass  the baseline's own draft_responses -> revise_responses, model in role as
               the assistant and told nothing about corpora

Comparison is on length, the baseline lint block, mean pairwise cosine between traces
within an arm (lower = more varied), and distinct openings. Arm B additionally reports
cosine(draft, revised), which is how much the second pass actually changed.

Output: output/low_stakes_pilot/<ts>/ -- pilot.jsonl (machine) + review.txt (human).
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import fire
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scratch.low_stakes.prompts import (  # noqa: E402
    DRAFT_SYSTEM,
    DRAFT_USER,
    LOW_STAKES_SETTINGS,
    RESPOND_INSTRUCTION,
    REVISE_SYSTEM,
    REVISE_USER,
    REWRITE_INSTRUCTION,
    TRAIT_BLOCK,
)
from src.data.synth import constitution, embeddings  # noqa: E402
from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402

SOURCE_REPO = "LASR-Callum/2026-08-21-sonnet45-difficult-advice-chunk-only-constitution-716"
SOURCE_FILE = "dataset.jsonl"
CONSTITUTION = "constitutions/claude_distilled_12_principles_mid/constitution.md"

# The baseline's own voice contract (revise_responses `lint:` in 2026-08-01_difficult_advice.yaml).
# Reported, not enforced, at pilot size -- a violation here is a prompt bug to look at.
BAN_PATTERNS = [
    r"\bmy (?:constitution|guidelines|rules|policies|instructions|training|constraints)\b",
    r"\bthe constitution\b",
    r"\bprinciple \d+\b",
    r"\b(?:i'm|i am) not (?:allowed|permitted|supposed)\b",
    r"^let me",
]

# Measured on the 708-row baseline corpus (scratch pull, 2026-08-26).
BASELINE_REASONING = (1641, 2920, 4414)


def _role(row: dict, name: str) -> str:
    return next(m["content"] for m in row["messages"] if m["role"] == name)


def _tagged(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    if not m:
        raise ValueError(f"no <{tag}> block in reply: {text[:200]!r}")
    return m.group(1).strip()


def _json_block(text: str) -> dict:
    """Parse the rewrite reply, tolerating a ```json fence and raw newlines in strings.

    `strict=False` because the rewritten user message is multi-paragraph and the model
    intermittently emits its newlines literally rather than escaped; that is a valid
    reading of the text and not worth a retry. Truncation is the failure worth retrying,
    so the error carries the length and tail needed to tell the two apart.
    """
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    try:
        return json.loads(t, strict=False)
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(
            f"{exc.msg} (reply was {len(t)} chars, tail={t[-120:]!r})", t, exc.pos)


def _lint(text: str) -> list[str]:
    return [p for p in BAN_PATTERNS if re.search(p, text, re.I | re.M)]


def _mean_pairwise_cosine(texts: list[str]) -> float:
    """Mean off-diagonal cosine over an arm's traces. Lower means more varied."""
    import numpy as np

    if len(texts) < 2:
        return float("nan")
    X = embeddings.embed(texts)
    S = X @ X.T
    iu = np.triu_indices(len(texts), k=1)
    return float(S[iu].mean())


def _paired_cosine(a: list[str], b: list[str]) -> list[float]:
    """Row-wise cosine between two aligned trace lists."""
    import numpy as np

    XA, XB = embeddings.embed(a), embeddings.embed(b)
    return [float(np.dot(XA[i], XB[i])) for i in range(len(a))]


def _opening(text: str, words: int = 8) -> str:
    return " ".join(re.findall(r"[a-z']+", text.lower())[:words])


def main(n: int = 18, model: str = "anthropic/claude-sonnet-5", arms: bool = True,
         seed: int = 0, workers: int = 8) -> None:
    client = OpenRouterClient()
    _, style_guidance = constitution.segment(CONSTITUTION)

    path = hf_hub_download(SOURCE_REPO, SOURCE_FILE, repo_type="dataset",
                           token=os.environ.get("HF_TOKEN"))
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]

    # Walk t1..t9 in order, taking successive rows per trait, so any n spans the whole
    # constitution rather than clustering in the low trait ids.
    by_trait: dict[str, list[dict]] = {}
    for r in rows:
        by_trait.setdefault(r["metadata"]["trait_id"], []).append(r)
    traits = sorted(by_trait, key=lambda x: int(x[1:]))
    picks = []
    for k in range((n + len(traits) - 1) // len(traits)):
        for t in traits:
            if len(picks) < n:
                picks.append(by_trait[t][(seed + k) % len(by_trait[t])])
    print(f"{len(rows)} source rows; piloting {len(picks)} across {len(traits)} traits")

    def rewrite(i: int) -> dict:
        r = picks[i]
        md = r["metadata"]
        prompt = REWRITE_INSTRUCTION.format(
            setting=LOW_STAKES_SETTINGS[i % len(LOW_STAKES_SETTINGS)],
            trait_name=md["trait_name"], trait_text=md["trait_text"],
            system=_role(r, "system"), user=_role(r, "user"))
        # 4000: 2000 then 3000 both truncated a long rewrite mid-JSON. The retry is for
        # that same failure (client.chat already retries transient errors).
        for attempt in range(3):
            res = client.chat(model=model,
                              messages=[{"role": "user", "content": prompt}],
                              temperature=1.0, max_tokens=4000)
            try:
                out = _json_block(res.content)
            except (json.JSONDecodeError, ValueError):
                if attempt == 2:
                    raise
                continue
            out["assigned_setting"] = LOW_STAKES_SETTINGS[i % len(LOW_STAKES_SETTINGS)].split(" --")[0]
            out["_cached"] = res.cached_tokens
            return out
        raise AssertionError("unreachable")

    def arm_a(i: int) -> dict:
        """One call, full spec, model told it is producing a training row."""
        md, new = picks[i]["metadata"], news[i]
        prompt = RESPOND_INSTRUCTION.format(
            style_guidance=style_guidance,
            trait_block=TRAIT_BLOCK.format(trait_name=md["trait_name"],
                                           trait_text=md["trait_text"]),
            system=new["system"], user=new["user"])
        # 8000: at 4000 a long deliberation ran past the cap and lost its <response>.
        for attempt in range(2):
            res = client.chat(model=model,
                              messages=[{"role": "user", "content": prompt}],
                              temperature=1.0, max_tokens=8000)
            try:
                return {"reasoning": _tagged(res.content, "reasoning"),
                        "response": _tagged(res.content, "response"),
                        "_cached": res.cached_tokens}
            except ValueError:
                if attempt:
                    raise
        raise AssertionError("unreachable")

    def arm_b(i: int) -> dict:
        """The baseline's draft -> revise, model in role as the assistant."""
        md, new = picks[i]["metadata"], news[i]

        def call(msgs: list[dict], tags: tuple[str, ...]) -> dict:
            for attempt in range(2):
                res = client.chat(model=model, messages=msgs, temperature=1.0,
                                  max_tokens=8000)
                try:
                    return {t: _tagged(res.content, t) for t in tags}
                except ValueError:
                    if attempt:
                        raise
            raise AssertionError("unreachable")

        draft = call([
            {"role": "system", "content": DRAFT_SYSTEM.format(
                system=new["system"], trait_name=md["trait_name"],
                trait_text=md["trait_text"], style_guidance=style_guidance)},
            {"role": "user", "content": DRAFT_USER.format(user=new["user"])},
        ], ("reasoning", "response"))
        revised = call([
            {"role": "system", "content": REVISE_SYSTEM},
            {"role": "user", "content": REVISE_USER.format(
                trait_name=md["trait_name"], trait_text=md["trait_text"],
                system=new["system"], user=new["user"],
                draft_reasoning=draft["reasoning"],
                draft_response=draft["response"])},
        ], ("reasoning", "response", "changes"))
        return {"reasoning": revised["reasoning"], "response": revised["response"],
                "changes": revised["changes"],
                "draft_reasoning": draft["reasoning"],
                "draft_response": draft["response"]}

    print("step 1  rewriting to everyday stakes (setting dealt round-robin) ...")
    news = map_threaded(rewrite, len(picks), max_workers=workers, desc="rewrite")
    print("step 2A single detailed call ...")
    outs_a = map_threaded(arm_a, len(picks), max_workers=workers, desc="arm A")
    outs_b = (map_threaded(arm_b, len(picks), max_workers=workers, desc="arm B")
              if arms else None)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("output/low_stakes_pilot") / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i, (src, new, a) in enumerate(zip(picks, news, outs_a)):
        md = src["metadata"]
        src_reasoning = next(m for m in src["messages"]
                             if m["role"] == "assistant")["reasoning_content"]
        rec = {
            "scenario_id": md["scenario_id"], "trait_id": md["trait_id"],
            "trait_name": md["trait_name"],
            "original": {"domain": md["domain"], "situation": md["situation"],
                         "shortcut": md["shortcut"], "system": _role(src, "system"),
                         "user": _role(src, "user"), "reasoning": src_reasoning,
                         "reasoning_len": len(src_reasoning)},
            "low_stakes": {k: new.get(k) for k in
                           ("domain", "assigned_setting", "situation", "shortcut",
                            "system", "user", "relocated", "worst_outcome")},
            "arm_a": {**a, "reasoning_len": len(a["reasoning"]),
                      "response_len": len(a["response"]),
                      "lint": _lint(a["reasoning"] + "\n" + a["response"])},
        }
        if outs_b:
            b = outs_b[i]
            rec["arm_b"] = {**b, "reasoning_len": len(b["reasoning"]),
                            "response_len": len(b["response"]),
                            "lint": _lint(b["reasoning"] + "\n" + b["response"])}
        records.append(rec)
    (out_dir / "pilot.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8")

    # --- comparison -------------------------------------------------------------------
    def arm_stats(key: str) -> dict:
        rs = [r[key]["reasoning"] for r in records]
        lens = sorted(len(x) for x in rs)
        return {
            "med_reasoning": lens[len(lens) // 2],
            "min_reasoning": lens[0], "max_reasoning": lens[-1],
            "med_response": sorted(r[key]["response_len"] for r in records)[len(rs) // 2],
            "lint": sum(bool(r[key]["lint"]) for r in records),
            "cosine": _mean_pairwise_cosine(rs),
            "distinct_openings": len({_opening(x) for x in rs}),
        }

    stats = {"A single": arm_stats("arm_a")}
    if outs_b:
        stats["B two-pass"] = arm_stats("arm_b")
        drift = _paired_cosine([r["arm_b"]["draft_reasoning"] for r in records],
                               [r["arm_b"]["reasoning"] for r in records])
        cross = _paired_cosine([r["arm_a"]["reasoning"] for r in records],
                               [r["arm_b"]["reasoning"] for r in records])

    lines = []
    for r in records:
        o, l = r["original"], r["low_stakes"]
        block = (f"{'=' * 94}\n{r['trait_id']}  {r['trait_name']}\n{r['scenario_id']}\n\n"
                 f"ORIGINAL  [{o['domain']}]  (reasoning {o['reasoning_len']} chars)\n"
                 f"  situation: {o['situation']}\n  shortcut : {o['shortcut']}\n\n"
                 f"--- original user ---\n{o['user']}\n\n{'-' * 94}\n"
                 f"LOW STAKES  [{l['domain']}]  relocated={l['relocated']}  "
                 f"assigned={l['assigned_setting']}\n"
                 f"  worst outcome: {l['worst_outcome']}\n"
                 f"  situation: {l['situation']}\n  shortcut : {l['shortcut']}\n\n"
                 f"--- new system ---\n{l['system']}\n\n--- new user ---\n{l['user']}\n")
        for key, label in (("arm_a", "ARM A (single call)"),
                           ("arm_b", "ARM B (draft -> revise)")):
            if key not in r:
                continue
            g = r[key]
            block += (f"\n{'-' * 94}\n{label}\n"
                      f"--- REASONING ({g['reasoning_len']} chars) ---\n{g['reasoning']}\n"
                      f"\n--- RESPONSE ({g['response_len']} chars) ---\n{g['response']}\n"
                      f"\nlint: {g['lint'] or 'none'}\n")
            if key == "arm_b":
                block += f"revise changed: {g['changes']}\n"
        lines.append(block)
    (out_dir / "review.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nbaseline corpus reasoning: min {BASELINE_REASONING[0]} "
          f"med {BASELINE_REASONING[1]} max {BASELINE_REASONING[2]}")
    print(f"{'arm':12s} {'med CoT':>8s} {'min':>6s} {'max':>6s} {'med resp':>9s} "
          f"{'lint':>5s} {'cosine':>7s} {'openings':>9s}")
    for name, s in stats.items():
        print(f"{name:12s} {s['med_reasoning']:8d} {s['min_reasoning']:6d} "
              f"{s['max_reasoning']:6d} {s['med_response']:9d} "
              f"{s['lint']:>2d}/{len(records):<2d} {s['cosine']:7.3f} "
              f"{s['distinct_openings']:>4d}/{len(records):<4d}")
    if outs_b:
        print(f"\ncosine(draft, revised) per row: mean {sum(drift) / len(drift):.3f}  "
              f"min {min(drift):.3f}  max {max(drift):.3f}   "
              f"(1.0 = the revise pass changed nothing)")
        print(f"cosine(arm A, arm B) per row:   mean {sum(cross) / len(cross):.3f}")

    doms = [r["low_stakes"]["domain"] for r in records]
    print(f"\ndistinct domains: {len(set(doms))}/{len(doms)}   "
          f"'community*': {sum(d.lower().startswith('community') for d in doms)}")
    print(f"relocated: {sum(bool(r['low_stakes']['relocated']) for r in records)}"
          f"/{len(records)}")
    print(f"\nwrote {out_dir / 'review.txt'}")


if __name__ == "__main__":
    fire.Fire(main)
