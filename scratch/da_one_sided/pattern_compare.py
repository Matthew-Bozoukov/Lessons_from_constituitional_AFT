# ABOUTME: Rates one fixed list of recurring patterns over the one-sided corpus AND the original DA rows of the same
# ABOUTME: scenarios, with the synth engine's own pattern_scan classifiers, so the two shares are comparable.
"""Run (after `synth check ... --only pattern_scan` has left its scan batches in the run dir):

    uv run python scratch/da_one_sided/pattern_compare.py --run_dir output/synth_da_one_sided/<run>

WHY THIS EXISTS. The engine's pattern scan has three passes: scan, merge, rate. On 2026-11-20 the scan over this
corpus proposed 176 candidates in 12 batches and the merge (single-link at cosine 0.35) joined all of them into ONE
pattern, rated at 96%, which says nothing. The candidates themselves were fine, so they are consolidated here BY HAND
into PATTERNS (each entry names the scan candidates it stands for; their verbatim snippets become the classifier's
sanity examples) and handed to the engine through `patterns:`, the documented way to run one corpus's classifiers
over another. Nothing else is reimplemented: `_check_corpus_stage` does the rating.

The two-part "analysis then reply" candidate every batch reported at 25 of 25 is left out: it is the reasoning and
the reply concatenated into one text field, true of every row of both corpora.

Both corpora are rated on the SAME scenarios: the DA side is this run's export with each assistant turn replaced by
the original row's, in the same order, so the engine's seeded sample draws the same indices.
"""
import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
import edit  # noqa: E402

from src.data.synth.ours.cli import _check_corpus_stage, _load  # noqa: E402

# (name, category, description, [(scan batch, candidate name), ...])
PATTERNS = [
    ("key fact opener", "structural",
     "The text opens by restating or isolating ONE concrete fact or number from the person's situation, as a short "
     "declarative sentence that is treated as the crux, before any recommendation is given.",
     [(0, "key fact opens the response"), (1, "load-bearing fact isolated up front"),
      (2, "fact restated as anchor opening"), (6, "key fact isolated as organizing detail")]),
    ("names what it cannot know", "rhetorical",
     "The text explicitly states what the assistant does not know or cannot verify from its position, flagged as a "
     "limit on its own judgment rather than guessed at.",
     [(0, "explicit limits-of-knowledge disclaimer"), (1, "explicit knowledge-boundary disclaimer"),
      (3, "named unknowns disclaimer"), (7, "naming what cannot be known")]),
    ("closing question", "structural",
     "The final line of the reply is a direct question to the person, asking for a missing fact or inviting a next step.",
     [(0, "ends with a direct question"), (2, "closing question to user"),
      (3, "closing diagnostic question"), (7, "closing open question")]),
    ("bulleted or numbered steps", "structural",
     "The reply sets out its concrete next steps as a numbered or bulleted list (or bolded step headers) rather than "
     "in prose.",
     [(0, "numbered or bulleted concrete options"), (1, "numbered or bulleted action list"),
      (3, "numbered or bulleted action plan"), (9, "bolded action-step list")]),
    ("validates the pressure then pivots", "rhetorical",
     "The text affirms that the person's feeling, pressure, constraint or instinct is real or reasonable, and then "
     "pivots to the recommendation or to why the proposed action is still wrong.",
     [(1, "named emotional acknowledgment before pivot"), (2, "validates feeling then redirects"),
      (4, "validating the real cost upfront"), (6, "acknowledge real constraint then pivot")]),
    ("concedes the other side's reasoning", "rhetorical",
     "The text concedes that the reasoning, good faith or stated reasons behind the path it argues AGAINST are "
     "legitimate or understandable, before explaining why that path is still the wrong one.",
     [(5, "acknowledge good intent then redirect"), (7, "none-of-this-requires concession")]),
    ("decision is yours", "behavioural",
     "The text explicitly states that the final decision rests with the person being addressed, after having argued "
     "for one course of action.",
     [(1, "decision-belongs-to-you closer"), (4, "decision is yours to make disclaimer"),
      (6, "leaves final choice explicitly open"), (10, "decision is yours to make")]),
    ("take it to whoever has authority", "behavioural",
     "The core recommendation is to disclose, surface or escalate the issue to a specific party who holds authority "
     "over it, rather than resolve it unilaterally or stay silent.",
     [(0, "bring it to the real decision-maker"), (2, "routes decision to proper authority"),
      (3, "bring-to-authority-not-alone recommendation"), (7, "surfacing over silence directive")]),
    ("rejects the framing as a false choice", "rhetorical",
     "The text says the choice as the person framed it is not the real choice: it names two options presented as "
     "opposed and argues they are not actually in conflict, or splits the question into two separate questions.",
     [(1, "false dichotomy rejection"), (3, "false-dichotomy reframe"),
      (5, "false binary reframed as two questions"), (9, "reframe the false binary")]),
    ("X is not the same as Y", "rhetorical",
     "A contrastive sentence that separates two superficially similar things or relabels an action: 'X is not the "
     "same as Y', 'that's not A, it's B', 'those are different claims'.",
     [(2, "two things conflated correction"), (7, "X is not the same as Y formula"),
      (6, "not X but Y reframing"), (8, "that's not X, it's Y construction")]),
    ("it will surface later and cost more", "rhetorical",
     "The text argues that the concealment or shortcut does not remove the problem but defers it, and that its later "
     "discovery (by an audit, a review, a reporter, a colleague) will do more damage than disclosure now.",
     [(0, "discovery-later consequence argument"), (3, "discovery-will-surface-later warning"),
      (9, "secrecy surfaces later and costs more"), (11, "invisible-now visible-later discovery risk")]),
]


def build_patterns(run: Path) -> list[dict]:
    """PATTERNS as the engine's `patterns:` param, with the scan's own verbatim snippets as examples."""
    scans = [json.loads(line) for line in open(next(run.glob("*pattern_scan.scans.jsonl")), encoding="utf-8")]
    out = []
    for name, cat, desc, picks in PATTERNS:
        examples = []
        for b, cand in picks:
            hit = [p for p in scans[b]["patterns"] if p["name"] == cand]
            assert len(hit) == 1, f"scan batch {b} has no candidate named {cand!r}"
            examples += [str(e) for e in (hit[0].get("examples") or [])[:1]]
        assert examples, f"no examples for {name!r}"
        out.append({"name": name, "category": cat, "description": desc, "examples": examples, "scans": len(picks)})
    return out


def rate(cfg_path: str, run: Path, patterns: list[dict], sample: int) -> list[dict]:
    """The engine's rating pass over `run`'s export, with the supplied patterns. Returns the report's pattern rows."""
    cfg = _load(cfg_path)
    sc = next(s for s in cfg["stages"] if s.get("kind") == "corpus_check")
    for p in sc["properties"]:
        if p.get("property") == "pattern_scan":
            p["enabled"] = True
            p["params"] = {**(p.get("params") or {}), "patterns": patterns, "sample": sample}
    _check_corpus_stage(cfg, sc, run, None, only="pattern_scan")
    rep = json.loads((run / f"{sc['name']}_report.json").read_text(encoding="utf-8"))

    def find(o):
        if isinstance(o, dict):
            # The rated rows carry `pattern`; the supplied list echoed in the params carries `name`.
            got = o.get("patterns")
            if isinstance(got, list) and got and isinstance(got[0], dict) and "broad_share" in got[0]:
                return got
            for v in o.values():
                r = find(v)
                if r is not None:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = find(v)
                if r is not None:
                    return r
        return None

    return find(rep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--config", default="configs/data/synth/da-one-sided.yaml")
    ap.add_argument("--source", default="data/da_one_sided_source")
    ap.add_argument("--sample", type=int, default=90)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()

    run = Path(a.run_dir)
    patterns = build_patterns(run)
    out = Path("output") / f"{dt.date.today().isoformat()}_da_one_sided_pattern_compare"
    ours_dir, da_dir = out / "one_sided", out / "da_same_scenarios"
    for d in (ours_dir, da_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in open(run / "stage_5_export_sft.jsonl", encoding="utf-8")]
    src = {r["scenario_id"]: r for r in edit.load_src(a.source, None)}
    shutil.copyfile(run / "stage_5_export_sft.jsonl", ours_dir / "stage_5_export_sft.jsonl")
    with open(da_dir / "stage_5_export_sft.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            s = src[r["metadata"]["scenario_id"]]
            msgs = [dict(m) for m in r["messages"]]
            assert msgs[-1]["role"] == "assistant" and "reasoning_content" in msgs[-1]
            msgs[-1]["content"], msgs[-1]["reasoning_content"] = s["response"], s["reasoning"]
            f.write(json.dumps({**r, "messages": msgs}, ensure_ascii=False) + "\n")

    res = {"one_sided": rate(a.config, ours_dir, patterns, a.sample),
           "da": rate(a.config, da_dir, patterns, a.sample)}
    by = {k: {r["pattern"]: r for r in v} for k, v in res.items()}

    def cell(r: dict) -> str:
        lo, hi = r["broad_ci95"]
        flag = "" if r["reliable"] else " (classifier unreliable)"
        return f"{r['strict_share']:.0%} / {r['broad_share']:.0%} [{lo:.0%}-{hi:.0%}]{flag}"

    lines = [f"# Recurring patterns, one-sided vs DA on the same {a.sample} scenarios", "",
             f"Run: {run.as_posix()}. Classifier: the synth engine's pattern_scan rating pass. Each cell is the share "
             "of rows where the pattern is unambiguously present (strict) / loosely present (broad), with the 95% "
             "interval of the broad share.", "",
             "| pattern | DA strict / broad | one-sided strict / broad |", "|---|---|---|"]
    for name, *_ in PATTERNS:
        lines.append(f"| {name} | {cell(by['da'][name])} | {cell(by['one_sided'][name])} |")
    stamp = f"{dt.date.today().isoformat()}_da_one_sided_pattern_compare"
    (out / f"{stamp}_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / f"{stamp}.json").write_text(json.dumps({"run_dir": run.as_posix(), "sample": a.sample,
                                                   "patterns": patterns, "results": res}, indent=1,
                                                  ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
