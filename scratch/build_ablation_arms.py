# ABOUTME: Build reasoning-property ablation arms: swap ONE reasoning property in a trace for
# ABOUTME: another, preserving every other property, the conclusion, and the reply verbatim.

"""Swap one reasoning property for another, holding everything else fixed.

For the arm "harm risk assessment removed", every row carrying that property has its private
reasoning rewritten so the property is gone and a replacement property is present instead.
Replacements are dealt round-robin across the other three properties, so no single one
dominates the arm and the arm is not secretly an "add downstream consequences" experiment.

What must NOT change, or the ablation stops being interpretable:
* the reply — it is the supervised half and it already follows from the decision
* the decision itself
* any OTHER property from the four that the row happens to carry (they are named in the
  prompt for each row so the model knows what to protect)
* the voice and roughly the length

Families are the k=150 clusters within 0.87 centroid cosine of each target, assigned to
their nearest target with a 0.02 margin (see the neighbour analysis).

Run:
  uv run python scratch/build_ablation_arms.py --smoke --arm harm
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.utils import extract_json, git_sha, timestamp  # noqa: E402  # noqa: F401

ROUTING = {"provider": {"ignore": ["Amazon Bedrock"]}, "reasoning": {"enabled": False}}

FAMILIES = {
    "harm": {"clusters": [30, 79],
             "name": "explicit multi-factor harm risk assessment",
             "desc": "weighing a harm by how likely it is, how severe it would be, whether it "
                     "is reversible, and how those factors combine into a risk judgement"},
    "downstream": {"clusters": [29, 14, 62, 48],
                   "name": "downstream second-order consequences",
                   "desc": "tracing what follows AFTER the immediate outcome — the knock-on "
                           "effects, what it sets in motion, where it leads over time"},
    "epistemic": {"clusters": [67, 133, 87],
                  "name": "acknowledgment of its own epistemic limits",
                  "desc": "naming what it cannot know, verify or see from where it sits, and "
                          "what that uncertainty means for how far its judgement can reach"},
    "thirdparty": {"clusters": [20, 138],
                   "name": "third-party harm identification",
                   "desc": "identifying who else is affected besides the person asking — the "
                           "absent parties who bear consequences they did not agree to"},
}

REWRITE = """You are rewriting an AI assistant's PRIVATE REASONING for a controlled \
experiment. Exactly one property of the reasoning changes. Everything else is held fixed.

REMOVE this property entirely:
  {remove_name} — {remove_desc}
Every trace of it must go: not just the vocabulary, but the underlying move.

PUT THIS IN ITS PLACE:
  {add_name} — {add_desc}
It must do the same work in the argument that the removed property did — carrying the \
deliberation to the same conclusion, at comparable depth and length. Not a bolted-on \
paragraph; the reasoning should read as though it were always argued this way.

PRESERVE, exactly as they are:
{preserve}
* the CONCLUSION. The reasoning reaches a decision; it must reach the same decision.
* the assistant's voice, register and roughly the length.
* every fact. Introduce nothing about the world, the people or the law that is not already \
in the exchange.

The assistant's reply is NOT being rewritten and is not shown to be changed — the rewritten \
reasoning must still lead naturally into the reply exactly as written.

Return exactly this and nothing else:
<reasoning>
the rewritten private reasoning
</reasoning>
<changes>
one sentence: what you removed and what now does its work
</changes>"""


def _exchange(row: dict) -> str:
    """Render one SFT row for the rewrite prompt.

    Args:
        row: A stage_7 SFT record.

    Returns:
        System, user, reasoning and reply in tagged sections.
    """
    m = {x["role"]: x for x in row["messages"]}
    return (f"<system_prompt>\n{m['system']['content']}\n</system_prompt>\n\n"
            f"<user_message>\n{m['user']['content']}\n</user_message>\n\n"
            f"<assistant_reasoning>\n{m['assistant'].get('reasoning_content', '')}\n"
            f"</assistant_reasoning>\n\n"
            f"<assistant_reply>\n{m['assistant']['content']}\n</assistant_reply>")


def _tagged(text: str, tag: str) -> str:
    """Extract one tagged block, failing loudly if absent.

    Args:
        text: Raw model output.
        tag: Tag name.

    Returns:
        Block contents.

    Raises:
        ValueError: If the tag is missing.
    """
    o, c = f"<{tag}>", f"</{tag}>"
    if o not in text or c not in text:
        raise ValueError(f"missing <{tag}> in {text[:200]!r}")
    return text.split(o, 1)[1].split(c, 1)[0].strip()


def main(
    arm: str = "harm",
    run_dir: str = "output/feature_discovery/20260812_092119",
    sft: str = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl",
    model: str = "anthropic/claude-sonnet-5",
    workers: int = 10,
    smoke: bool = False,
    limit: int | None = None,
    out_dir: str | None = None,
) -> None:
    """Rewrite every row carrying `arm`'s property, dealing replacements round-robin.

    Args:
        arm: Which family to remove (key of FAMILIES).
        run_dir: Feature-discovery run holding the cluster assignment.
        sft: The difficult-advice SFT file.
        model: OpenRouter rewrite model.
        workers: Concurrent requests.
        smoke: Rewrite 5 rows and print each before/after.
        limit: Rewrite only the first N rows.
        out_dir: Output directory.
    """
    d = Path(run_dir)
    fmap = json.loads((d / "feature_cluster_map.json").read_text())
    traces = [json.loads(x) for x in (d / "features.jsonl").read_text().splitlines() if x.strip()]
    rows = {json.loads(x)["metadata"]["scenario_id"]: json.loads(x)
            for x in Path(sft).read_text().splitlines() if x.strip()}
    tset = {t["scenario_id"]: {fmap[f] for f in t["features"]} for t in traces}

    target = FAMILIES[arm]
    others = [k for k in FAMILIES if k != arm]
    members = sorted(s for s, m in tset.items() if m & set(target["clusters"]))
    if smoke:
        members = members[:5]
    elif limit:
        members = members[:limit]
    # Round-robin so each replacement property gets an equal share of the arm.
    assign = {s: others[i % len(others)] for i, s in enumerate(members)}
    print(f"arm '{arm}' (remove {target['name']}): {len(members)} rows")
    from collections import Counter
    print(f"replacement split: {dict(Counter(assign.values()))}")

    out = Path(out_dir or f"output/ablation_arms/{arm}_{timestamp()}")
    out.mkdir(parents=True, exist_ok=True)
    client = OpenRouterClient()
    lock = threading.Lock()
    done: list[dict] = []

    def run(i: int) -> tuple[int, int]:
        sid = members[i]
        add = FAMILIES[assign[sid]]
        # Any OTHER listed property this row carries must survive the rewrite.
        keep = [FAMILIES[k]["name"] for k in others
                if tset[sid] & set(FAMILIES[k]["clusters"]) and k != assign[sid]]
        preserve = ("\n".join(f"* the reasoning's existing {n} — leave that move intact"
                              for n in keep)
                    or "* (this row carries none of the other studied properties)")
        sysp = REWRITE.format(remove_name=target["name"], remove_desc=target["desc"],
                              add_name=add["name"], add_desc=add["desc"], preserve=preserve)
        res = client.chat(model=model, temperature=1.0, max_tokens=3000, extra_body=ROUTING,
                          messages=[{"role": "system", "content": sysp},
                                    {"role": "user", "content": _exchange(rows[sid])}])
        old = rows[sid]["messages"][2].get("reasoning_content", "")
        new = _tagged(res.content, "reasoning")
        with lock:
            done.append({"scenario_id": sid, "removed": arm, "added": assign[sid],
                         "preserved": keep, "changes": _tagged(res.content, "changes"),
                         "old_words": len(old.split()), "new_words": len(new.split()),
                         "reasoning_before": old, "reasoning_after": new})
        return res.prompt_tokens, res.completion_tokens

    usage = map_threaded(run, len(members), max_workers=workers, desc=f"rewriting[{arm}]")
    tin, tout = sum(u[0] for u in usage), sum(u[1] for u in usage)
    (out / "rewrites.jsonl").write_text("".join(json.dumps(r) + "\n" for r in done))
    (out / "run_meta.json").write_text(json.dumps(
        {"git_sha": git_sha(), "timestamp_utc": timestamp(), "arm": arm, "model": model,
         "rows": len(members), "tokens_in": tin, "tokens_out": tout,
         "families": {k: v["clusters"] for k, v in FAMILIES.items()},
         "command": " ".join(sys.argv)}, indent=2))
    print(f"\ntokens: {tin / len(members):.0f} in, {tout / len(members):.0f} out per row")
    print(f"cost at Sonnet rates: ${tin / 1e6 * 3 + tout / 1e6 * 15:.3f} for {len(members)} rows")
    print(f"-> {out}")
    if smoke:
        for r in done:
            print(f"\n{'=' * 78}\n{r['scenario_id']}  remove={r['removed']}  add={r['added']}  "
                  f"preserve={r['preserved'] or 'none'}  {r['old_words']}->{r['new_words']}w")
            print(f"CHANGES: {r['changes']}")


if __name__ == "__main__":
    fire.Fire(main)
