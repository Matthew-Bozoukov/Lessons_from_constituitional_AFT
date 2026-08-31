# ABOUTME: Add explicit probability/severity/reversibility weighing to difficult-advice rows
# ABOUTME: whose reasoning treats a harm qualitatively, preserving the decision already made.

"""Augment the difficult-advice corpus with calibrated risk framing.

Cluster C30 ("Explicit multi-factor harm risk assessment") covers 196 of 2,202 traces: the
reasoning names how LIKELY a harm is, how BAD it would be, and whether it is REVERSIBLE.
The wider risk-reasoning family covers 1,046 traces, but most of those weigh harm
qualitatively ("this could really hurt her") without ever quantifying it.

Two passes, because forcing calibration onto a scenario without a probabilistic harm axis
manufactures analysis the situation does not support:

  GATE     does the decision actually hinge on a harm whose likelihood is uncertain and
           decision-relevant? Deontological dilemmas (is this a lie?) and certain harms
           are rejected here.
  REWRITE  add the weighing to the REASONING only, keeping the conclusion, the voice, and
           the reply untouched. No new facts about the world may be introduced.

The reply is deliberately not rewritten: it is the supervised half, and changing the
conclusion there would decouple it from a reasoning trace that already argued for it.

Run:
  uv run python scratch/augment_risk_calibration.py --smoke
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.utils import extract_json, git_sha, timestamp  # noqa: E402

ROUTING = {"provider": {"ignore": ["Amazon Bedrock"]}, "reasoning": {"enabled": False}}

GATE = """You are selecting training rows that could carry calibrated risk reasoning.

A row QUALIFIES only if the assistant's decision hinges on a harm whose LIKELIHOOD is \
genuinely uncertain and decision-relevant — the harm might or might not happen, and how \
likely it is should change what the assistant advises.

A row does NOT qualify if:
* the wrong is deontological — it is a lie, or a bypassed rule, and the probability of being \
caught is beside the point
* the harm is certain, or already happened
* the reasoning ALREADY weighs likelihood explicitly (words like likely, probability, odds, \
chance, worst case, how bad, reversible, recoverable)
* there is no real harm axis, only a preference or a values disagreement

Return ONLY:
{"qualifies": true|false,
 "harm": "<the uncertain harm, or empty>",
 "factors_available": ["likelihood"|"severity"|"reversibility"|"who_bears_it", ...],
 "why": "<one sentence, max 25 words>"}"""

REWRITE = """You revise an assistant's PRIVATE REASONING so that it weighs a risk \
explicitly, in the way a calibrated thinker would.

You are given the exchange and the reasoning as written. Rewrite the reasoning so it works \
through, in its own voice and in prose:
* how likely the harm actually is, given what is known — not a fake percentage, a judgement \
("this probably doesn't blow up; the failure mode is narrow but it exists")
* how bad it would be if it happened, and who absorbs it
* whether it can be undone, and what the cost of being wrong in each direction is

Hard constraints:
1. The CONCLUSION must not change. The reasoning already reaches a decision; it must reach \
the same one, for compatible reasons. The reply is not being rewritten and must still follow.
2. Introduce NO new facts about the world, the people, or the law. Work only from what the \
user said and what the reasoning already established. If you do not know a base rate, say \
so in the voice of the assistant rather than inventing one.
3. Keep the existing length roughly. Do not append a labelled section; weave the weighing \
into the deliberation where it belongs.
4. Keep the voice. This is private thinking, not a report.

Return your answer in exactly this form and nothing else:
<reasoning>
the rewritten private reasoning
</reasoning>
<changes>
one sentence on what you added
</changes>"""


def _exchange(row: dict) -> str:
    """Render one SFT row for the gate and rewrite prompts.

    Args:
        row: A stage_7 SFT record.

    Returns:
        The system prompt, user turn, reasoning and reply in tagged sections.
    """
    m = {x["role"]: x for x in row["messages"]}
    return (f"<system_prompt>\n{m['system']['content']}\n</system_prompt>\n\n"
            f"<user_message>\n{m['user']['content']}\n</user_message>\n\n"
            f"<assistant_reasoning>\n{m['assistant'].get('reasoning_content', '')}\n"
            f"</assistant_reasoning>\n\n"
            f"<assistant_reply>\n{m['assistant']['content']}\n</assistant_reply>")


def _tagged(text: str, tag: str) -> str:
    """Pull one tagged block out of a model response.

    Args:
        text: Raw model output.
        tag: Tag name.

    Returns:
        The block contents.

    Raises:
        ValueError: If the tag is absent, rather than silently returning the whole text.
    """
    open_, close = f"<{tag}>", f"</{tag}>"
    if open_ not in text or close not in text:
        raise ValueError(f"missing <{tag}> in: {text[:200]!r}")
    return text.split(open_, 1)[1].split(close, 1)[0].strip()


def main(
    sft: str = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl",
    run_dir: str = "output/feature_discovery/20260812_092119",
    cluster: int = 30,
    gate_model: str = "google/gemini-3-flash-preview",
    rewrite_model: str = "anthropic/claude-sonnet-5",
    workers: int = 20,
    limit: int | None = None,
    out_dir: str | None = None,
    smoke: bool = False,
) -> None:
    """Gate candidate rows, then rewrite the qualifying ones' reasoning.

    Args:
        sft: The difficult-advice SFT file.
        run_dir: Feature-discovery run (for the cluster assignment).
        cluster: Cluster whose framing is being spread.
        gate_model: Cheap model for the qualification pass.
        rewrite_model: Model that rewrites the reasoning.
        workers: Concurrent requests.
        limit: Gate only the first N candidates.
        out_dir: Output directory.
        smoke: Gate 12 rows, rewrite up to 2, print one rewrite in full.
    """
    d = Path(run_dir)
    fmap = json.loads((d / "feature_cluster_map.json").read_text())
    feats = [json.loads(x) for x in (d / "features.jsonl").read_text().splitlines() if x.strip()]
    has = {t["scenario_id"] for t in feats if any(fmap[f] == cluster for f in t["features"])}
    rows = {json.loads(x)["metadata"]["scenario_id"]: json.loads(x)
            for x in Path(sft).read_text().splitlines() if x.strip()}
    cands = sorted(set(rows) & {t["scenario_id"] for t in feats} - has)
    if smoke:
        cands = cands[:12]
    elif limit:
        cands = cands[:limit]
    print(f"rows already carrying C{cluster}: {len(has)}; candidates to gate: {len(cands)}")

    out = Path(out_dir or f"output/risk_calibration/{timestamp()}")
    out.mkdir(parents=True, exist_ok=True)
    client = OpenRouterClient()
    lock = threading.Lock()
    gated: list[dict] = []

    def gate(i: int) -> None:
        sid = cands[i]
        res = client.chat(model=gate_model, temperature=0.0, max_tokens=600,
                          extra_body=ROUTING,
                          messages=[{"role": "system", "content": GATE},
                                    {"role": "user", "content": _exchange(rows[sid])}])
        obj = extract_json(res.content)
        if "qualifies" not in obj:
            raise ValueError(f"gate omitted qualifies: {obj}")
        with lock:
            gated.append({"scenario_id": sid, **obj})

    map_threaded(gate, len(cands), max_workers=workers, desc="gating")
    qual = [g for g in gated if g["qualifies"]]
    (out / "gate.jsonl").write_text("".join(json.dumps(g) + "\n" for g in gated))
    print(f"\nqualified: {len(qual)}/{len(gated)} ({len(qual) / max(len(gated), 1):.0%})")
    if not qual:
        print("nothing to rewrite")
        return

    todo = qual[:2] if smoke else qual
    rewrites: list[dict] = []

    def rewrite(i: int) -> None:
        g = todo[i]
        row = rows[g["scenario_id"]]
        res = client.chat(model=rewrite_model, temperature=1.0, max_tokens=3000,
                          extra_body=ROUTING,
                          messages=[{"role": "system", "content": REWRITE},
                                    {"role": "user", "content": _exchange(row)}])
        new = _tagged(res.content, "reasoning")
        old = row["messages"][2].get("reasoning_content", "")
        with lock:
            rewrites.append({"scenario_id": g["scenario_id"], "harm": g.get("harm", ""),
                             "changes": _tagged(res.content, "changes"),
                             "old_words": len(old.split()), "new_words": len(new.split()),
                             "reasoning_before": old, "reasoning_after": new})

    map_threaded(rewrite, len(todo), max_workers=workers, desc="rewriting")
    (out / "rewrites.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rewrites))
    (out / "run_meta.json").write_text(json.dumps(
        {"git_sha": git_sha(), "timestamp_utc": timestamp(), "sft": sft, "cluster": cluster,
         "gate_model": gate_model, "rewrite_model": rewrite_model,
         "candidates": len(cands), "qualified": len(qual), "rewritten": len(rewrites),
         "command": " ".join(sys.argv)}, indent=2))
    print(f"rewrote {len(rewrites)} -> {out}")
    if rewrites:
        r = rewrites[0]
        print(f"\n--- {r['scenario_id']}: {r['old_words']} -> {r['new_words']} words ---")
        print(f"harm: {r['harm']}\nchanges: {r['changes']}")
        print(f"\nBEFORE:\n{r['reasoning_before'][:900]}")
        print(f"\nAFTER:\n{r['reasoning_after'][:1400]}")


if __name__ == "__main__":
    fire.Fire(main)
