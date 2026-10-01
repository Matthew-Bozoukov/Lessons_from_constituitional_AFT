# ABOUTME: Build the artifact showing one MoReBench dilemma, the model's answer, and every
# ABOUTME: rubric criterion it met or missed — the evidence that the metric scores coverage.
"""Emit the rubric-breakdown page for a MoReBench run.

    uv run python scratch/morebench/build_rubric_page.py [--task row_377]

The scores say da-15 59.78 and nosynth 56.81, and nothing in those numbers shows what a
point is. This page shows one dilemma end to end: the scenario, the answer, the eight
criteria it earned and the sixteen it did not, each with its weight. Three shorter cases
follow so the pattern is not resting on one example.
"""

from __future__ import annotations

import argparse
import html
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "output/morebench/2026-09-29_qwen36_0_da_15_135845"
OUT = Path("/tmp/claude-1000/-home-matthewb-git-repos-agent-interp-envs/"
           "2608ec10-a153-4ab4-ae39-b5f9dad687f7/scratchpad/art/rubric.html")
PREFIX = "Provide corresponding reasoning and decision for the following scenario."


def met(j: dict) -> bool:
    """Did the judge answer yes for this criterion?"""
    return "yes" in str(j["judgement"]).lower()


def scenario(text: str) -> str:
    """The dilemma without the benchmark's instruction prefix."""
    t = text.replace(PREFIX, "").replace("Scenario:", "", 1).strip()
    return t


def case_block(task: str, J: list[dict], R: dict, score: float, compact: bool = False) -> str:
    """One dilemma: prompt, answer, and the criterion ledger."""
    cs = [j for j in J if j["task_id"] == task]
    pos = [c for c in cs if float(c["criterion_weight"]) > 0]
    neg = [c for c in cs if float(c["criterion_weight"]) < 0]
    hit = [c for c in pos if met(c)]
    miss = [c for c in pos if not met(c)]
    tripped = [c for c in neg if met(c)]
    e = html.escape
    resp = R[task]["response"]
    dom = R[task].get("role_domain", "")
    typ = R[task].get("dilemma_type", "")

    def li(c: str, w: float, kind: str) -> str:
        return (f'<li class="{kind}"><span class="w">{"+" if w > 0 else ""}{int(w)}</span>'
                f'<span>{e(c)}</span></li>')

    earned = sum(float(c["criterion_weight"]) for c in hit)
    avail = sum(abs(float(c["criterion_weight"])) for c in cs)
    body = f"""
  <article class="case{' compact' if compact else ''}">
    <header class="casehead">
      <span class="tid">{e(task)}</span>
      <span class="meta">{e(dom)} · {e(typ)}</span>
      <span class="score">{score:.1f}<small>/100</small></span>
    </header>
    <div class="grid">
      <section class="pane">
        <p class="panelab">The dilemma</p>
        <div class="prose">{e(scenario(R[task]['prompt']))}</div>
      </section>
      <section class="pane">
        <p class="panelab">What the model answered <span class="count">{len(resp):,} chars</span></p>
        <div class="prose">{e(resp)}</div>
      </section>
    </div>
    <div class="ledger">
      <div class="side ok">
        <p class="panelab">Earned <span class="count">{len(hit)} of {len(pos)} positive criteria · {earned:.0f} of {avail:.0f} weight</span></p>
        <ul>{''.join(li(c['criterion'], float(c['criterion_weight']), 'ok') for c in hit)}</ul>
      </div>
      <div class="side no">
        <p class="panelab">Marked off for <span class="count">{len(miss)} missed</span></p>
        <ul>{''.join(li(c['criterion'], float(c['criterion_weight']), 'no') for c in miss)}</ul>
        {'<p class="neg">Negative criteria tripped: ' + str(len(tripped)) + '</p>' if neg else ''}
      </div>
    </div>
  </article>"""
    return body


def main() -> None:
    """Write the page."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="row_377")
    ap.add_argument("--also", nargs="*", default=["row_136", "row_16", "row_121"])
    args = ap.parse_args()

    J = [json.loads(l) for l in open(RUN / "rollouts/judgements.jsonl", encoding="utf-8")]
    R = {json.loads(l)["task_id"]: json.loads(l)
         for l in open(RUN / "rollouts/responses.jsonl", encoding="utf-8")}
    S = json.loads((RUN / "results/results.json").read_text())["task_scores"]

    by = defaultdict(list)
    for j in J:
        by[j["task_id"]].append(j)
    pos_per = [sum(1 for c in cs if float(c["criterion_weight"]) > 0) for cs in by.values()]
    hit_per = [sum(1 for c in cs if float(c["criterion_weight"]) > 0 and met(c)) for cs in by.values()]
    # how specific are the missed criteria, across the whole run
    all_miss = [c for cs in by.values() for c in cs
                if float(c["criterion_weight"]) > 0 and not met(c)]
    long_miss = sum(1 for c in all_miss if len(c["criterion"]) > 90)

    stats = {
        "tasks": len(by), "criteria": len(J),
        "pos_mean": st.mean(pos_per), "hit_mean": st.mean(hit_per),
        "long_pct": 100 * long_miss / len(all_miss),
    }

    cases = case_block(args.task, J, R, S[args.task])
    extra = "\n".join(case_block(t, J, R, S[t], compact=True) for t in args.also)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(HEAD.format(**stats) + cases + MID + extra + FOOT, encoding="utf-8")
    print(f">>> wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
    print(f">>> {stats['pos_mean']:.1f} positive criteria/dilemma, {stats['hit_mean']:.1f} met, "
          f"{stats['long_pct']:.0f}% of misses are long specific criteria")


HEAD = """<title>What a Point Costs</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
  :root{{
    --paper:#F2F4F3; --raise:#FFFFFF; --ink:#132A33; --body:#2D3F47; --muted:#64737A;
    --rule:#D8DEDC; --rule-strong:#B9C3C1; --quiet:#F6F7F6;
    --ok:#2F6B4F; --ok-soft:#E2EEE7; --no:#9B2C1F; --no-soft:#F6E4E0;
    --accent:#1E6B74;
    color-scheme:light;
  }}
  @media (prefers-color-scheme:dark){{ :root:not([data-theme="light"]){{
    --paper:#0E181D; --raise:#15232A; --ink:#E8EEEC; --body:#C2CFCE; --muted:#8A9A9D;
    --rule:#24353C; --rule-strong:#354A52; --quiet:#111D23;
    --ok:#6FBF93; --ok-soft:#162A22; --no:#E0775F; --no-soft:#2E1C18;
    --accent:#5FB9BF; color-scheme:dark; }} }}
  :root[data-theme="dark"]{{
    --paper:#0E181D; --raise:#15232A; --ink:#E8EEEC; --body:#C2CFCE; --muted:#8A9A9D;
    --rule:#24353C; --rule-strong:#354A52; --quiet:#111D23;
    --ok:#6FBF93; --ok-soft:#162A22; --no:#E0775F; --no-soft:#2E1C18;
    --accent:#5FB9BF; color-scheme:dark; }}
  *{{box-sizing:border-box}}
  body{{background:var(--paper); color:var(--body); margin:0;
    font-family:"IBM Plex Sans",system-ui,sans-serif; font-size:15px; line-height:1.6}}
  .wrap{{max-width:1120px; margin:0 auto; padding-inline:20px; padding-block:44px 72px}}
  h1,h2{{font-family:Spectral,Georgia,serif; color:var(--ink); margin:0; text-wrap:balance}}
  h1{{font-size:clamp(2rem,4.6vw,2.9rem); font-weight:600; line-height:1.1}}
  h2{{font-size:1.25rem; font-weight:600}}
  p{{margin:0}}
  .eyebrow{{font-family:"IBM Plex Mono",monospace; font-size:.72rem; letter-spacing:.13em;
    text-transform:uppercase; color:var(--muted)}}
  .lede{{font-size:1.1rem; max-width:66ch; margin-top:16px}}
  .facts{{display:flex; gap:26px; flex-wrap:wrap; margin-top:24px; padding-block:16px;
    border-block:1px solid var(--rule-strong)}}
  .fact b{{display:block; font-family:Spectral,Georgia,serif; font-size:1.7rem; color:var(--ink);
    line-height:1; font-variant-numeric:tabular-nums}}
  .fact span{{font-size:.84rem; color:var(--muted)}}
  .case{{margin-top:38px; background:var(--raise); border:1px solid var(--rule); border-radius:3px;
    padding:18px}}
  .casehead{{display:flex; align-items:baseline; gap:12px; flex-wrap:wrap;
    border-bottom:1px solid var(--rule); padding-bottom:11px}}
  .tid{{font-family:"IBM Plex Mono",monospace; font-size:.8rem; color:var(--accent)}}
  .meta{{font-size:.82rem; color:var(--muted)}}
  .score{{margin-left:auto; font-family:Spectral,Georgia,serif; font-size:1.6rem; color:var(--ink);
    font-variant-numeric:tabular-nums}}
  .score small{{font-size:.8rem; color:var(--muted); font-family:"IBM Plex Sans",sans-serif}}
  .grid{{display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:14px}}
  @media (max-width:820px){{.grid{{grid-template-columns:1fr}}}}
  .panelab{{font-family:"IBM Plex Mono",monospace; font-size:.69rem; letter-spacing:.09em;
    text-transform:uppercase; color:var(--muted)}}
  .count{{text-transform:none; letter-spacing:0; margin-left:7px; opacity:.85}}
  .prose{{white-space:pre-wrap; font-size:.9rem; margin-top:7px; background:var(--quiet);
    border-left:2px solid var(--rule-strong); padding:11px 13px; max-height:380px; overflow-y:auto}}
  .ledger{{display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:18px}}
  @media (max-width:820px){{.ledger{{grid-template-columns:1fr}}}}
  .side ul{{list-style:none; margin:8px 0 0; padding:0; display:flex; flex-direction:column; gap:7px}}
  .side li{{display:grid; grid-template-columns:30px 1fr; gap:9px; font-size:.87rem;
    padding:7px 9px; border-radius:2px; align-items:start}}
  li.ok{{background:var(--ok-soft)}} li.no{{background:var(--no-soft)}}
  .w{{font-family:"IBM Plex Mono",monospace; font-size:.76rem; font-variant-numeric:tabular-nums}}
  li.ok .w{{color:var(--ok)}} li.no .w{{color:var(--no)}}
  .neg{{font-size:.82rem; color:var(--muted); margin-top:9px}}
  .compact .prose{{max-height:200px}}
  .note{{font-size:.9rem; color:var(--muted); max-width:72ch; margin-top:14px}}
  .foot{{margin-top:54px; border-top:1px solid var(--rule-strong); padding-top:18px;
    font-size:.84rem; color:var(--muted)}}
  .foot code{{font-family:"IBM Plex Mono",monospace; color:var(--body)}}
</style>
<div class="wrap">
  <header>
    <p class="eyebrow">MoReBench · da-15 · rubric breakdown</p>
    <h1>What a Point Costs</h1>
    <p class="lede">MoReBench scores a response by how much of a per-dilemma rubric it covers.
      The case below is a competent answer that commits to a position and defends it, and it lost
      more than half the available weight — to criteria a municipal-lawyer would list, not to
      anything it got wrong.</p>
    <div class="facts">
      <span class="fact"><b>{pos_mean:.0f}</b><span>positive criteria per dilemma, on average</span></span>
      <span class="fact"><b>{hit_mean:.1f}</b><span>of them met, on average</span></span>
      <span class="fact"><b>{long_pct:.0f}%</b><span>of all missed criteria are long, named sub-points</span></span>
      <span class="fact"><b>{criteria:,}</b><span>criteria judged across {tasks} dilemmas</span></span>
    </div>
  </header>
"""

MID = """
  <h2 style="margin-top:46px">Three more, shorter</h2>
  <p class="note">Same pattern, picked by the same rule: a substantive answer whose every missed
    criterion is a long, specific sub-point. Scroll each panel to read in full.</p>
"""

FOOT = """
  <p class="foot">
    From <code>dougalldeepmind/2026-09-29-mrb-qwen36-0-da-15</code>, judged by
    <code>openai/gpt-oss-120b</code> as MoReBench specifies, one call per criterion, 0 unparsed.
    A dilemma's score is the share of available criterion weight earned: positive criteria pay their
    weight when met, negative criteria pay when avoided. Nothing here is reweighted or re-judged —
    these are the run's own verdicts.
    The run totals were 59.78 for da-15 and 56.81 for nosynth; on the benchmark's length-controlled
    variant the same runs are 34.02 and 34.44.
  </p>
</div>
"""

if __name__ == "__main__":
    main()
