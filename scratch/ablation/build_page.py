# ABOUTME: Build the artifact comparing an ablation run's rewritten reasoning and replies
# ABOUTME: against the source rows they were made from, paragraph by labelled paragraph.
"""What the ablation removed, side by side.

    uv run python scratch/ablation/build_page.py --run_dir output/synth_ablation/<run>

The arm is defined by a subtraction, so the only honest way to show it is next to what it
subtracted from. Each specimen puts the source reasoning beside the restructured one with
every paragraph numbered and the restructured ones labelled by the role they fill, then does
the same for the two replies.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path("/tmp/claude-1000/-home-matthewb-git-repos-agent-interp-envs/"
           "2608ec10-a153-4ab4-ae39-b5f9dad687f7/scratchpad/art/ablation_v2.html")

ROLE = {"bg": "background", "gap": "rest of the background",
        "harm": "the harmful thing", "plan": "what the reply will say"}


def paras(t: str) -> list[str]:
    """Paragraphs, blank-line separated."""
    return [p.strip() for p in re.split(r"\n\s*\n", t or "") if p.strip()]


def roles(n: int) -> list[str]:
    """The contract's role for each of n reasoning paragraphs, by position."""
    if n < 4:
        return ["bg"] * n
    return ["bg", "gap"] + ["harm"] * (n - 3) + ["plan"]


def col(label: str, body: list[str], kind: str, tags: list[str] | None = None) -> str:
    """One column of numbered paragraphs."""
    e = html.escape
    out = []
    for i, p in enumerate(body):
        tag = (f'<span class="role r-{tags[i]}">{ROLE[tags[i]]}</span>' if tags else "")
        out.append(f'<div class="p"><span class="pn">{i+1}</span>{tag}<p>{e(p)}</p></div>')
    chars = sum(len(p) for p in body)
    return (f'<section class="side {kind}"><p class="sl">{label}'
            f'<span class="ct">{len(body)}¶ · {chars:,}c</span></p>{"".join(out)}</section>')


def main() -> None:
    """Write the page."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    d = REPO / args.run_dir

    out_rows = [json.loads(l) for l in
                open(sorted(d.glob("stage_*_export_sft.jsonl"))[-1], encoding="utf-8")]
    src = {r["scenario_id"]: r for r in
           (json.loads(l) for l in
            open(sorted(d.glob("stage_*_load_source.jsonl"))[-1], encoding="utf-8"))}

    specs, ratios = [], []
    for n, r in enumerate(out_rows, 1):
        m = r["metadata"]
        sid = m["scenario_id"]
        s = src[sid]
        msgs = r["messages"]
        ab_r = next(x.get("reasoning_content") or "" for x in msgs if x.get("reasoning_content"))
        ab_p = next(x["content"] for x in reversed(msgs) if x["role"] == "assistant")
        user = next(x["content"] for x in msgs if x["role"] == "user")
        P = paras(ab_r)
        ratios.append(len(ab_r) / len(s["reasoning"]))
        e = html.escape
        specs.append(f"""
<article class="spec">
  <header class="sh">
    <span class="n">{n:02d}</span><span class="sid">{e(sid)}</span>
    <span class="chip">{e(m.get('trait_id','').upper())}</span>
    <span class="chip ai">{e(str(m.get('ai')))}</span>
    <span class="delta">{len(ab_r)/len(s['reasoning']):.2f}× the source reasoning</span>
  </header>
  <details class="ask"><summary>The person's message <span>{len(user):,}c</span></summary>
    <div class="askbody">{e(user)}</div></details>
  <p class="rowlab">Reasoning</p>
  <div class="pair">{col("Source — deliberated", paras(s["reasoning"]), "src")}
    {col("Ablated — background → harm → plan", P, "abl", roles(len(P)))}</div>
  <p class="rowlab">Reply</p>
  <div class="pair">{col("Source", paras(s["response"]), "src")}
    {col("Ablated", paras(ab_p), "abl")}</div>
</article>""")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(PAGE.format(specs="".join(specs), n=len(out_rows),
                               ratio=st.median(ratios)), encoding="utf-8")
    print(f">>> wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB), {len(out_rows)} specimens")


PAGE = """<title>What the Ablation Removes</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Archivo:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
  :root{{
    --bg:#F0F1F3; --card:#FFFFFF; --ink:#15181E; --body:#353B45; --soft:#6A7180;
    --line:#DCDFE4; --line2:#B9BFC9; --well:#F6F7F9;
    --src:#9A6B22; --src-well:#FBF4E9; --abl:#1F6670; --abl-well:#E8F1F2;
    --r-bg:#6B7280; --r-gap:#6B7280; --r-harm:#9B3B2C; --r-plan:#1F6670;
    color-scheme:light;
  }}
  @media (prefers-color-scheme:dark){{ :root:not([data-theme="light"]){{
    --bg:#0F1216; --card:#171B21; --ink:#EBEEF2; --body:#C3CAD4; --soft:#8B93A1;
    --line:#262C35; --line2:#3A424E; --well:#13171C;
    --src:#D6A55C; --src-well:#20190E; --abl:#69B4BE; --abl-well:#0F1E20;
    --r-bg:#8B93A1; --r-gap:#8B93A1; --r-harm:#DE8574; --r-plan:#69B4BE;
    color-scheme:dark; }} }}
  :root[data-theme="dark"]{{
    --bg:#0F1216; --card:#171B21; --ink:#EBEEF2; --body:#C3CAD4; --soft:#8B93A1;
    --line:#262C35; --line2:#3A424E; --well:#13171C;
    --src:#D6A55C; --src-well:#20190E; --abl:#69B4BE; --abl-well:#0F1E20;
    --r-bg:#8B93A1; --r-gap:#8B93A1; --r-harm:#DE8574; --r-plan:#69B4BE;
    color-scheme:dark; }}
  *{{box-sizing:border-box}}
  body{{margin:0; background:var(--bg); color:var(--body);
    font-family:Archivo,system-ui,sans-serif; font-size:15px; line-height:1.6}}
  h1{{font-family:Fraunces,Georgia,serif; font-weight:600; color:var(--ink); margin:0;
    font-size:clamp(2rem,4.5vw,2.9rem); line-height:1.08; text-wrap:balance}}
  p{{margin:0}}
  .wrap{{max-width:1280px; margin:0 auto; padding:42px 20px 72px}}
  .eyebrow{{font-family:"IBM Plex Mono",monospace; font-size:.71rem; letter-spacing:.14em;
    text-transform:uppercase; color:var(--soft)}}
  .lede{{margin-top:15px; max-width:68ch; font-size:1.07rem}}
  .contract{{margin-top:22px; background:var(--card); border:1px solid var(--line);
    border-radius:3px; padding:14px 16px; display:grid; gap:7px;
    font-family:"IBM Plex Mono",monospace; font-size:.8rem}}
  .contract div{{display:grid; grid-template-columns:118px 1fr; gap:12px}}
  .contract b{{color:var(--abl); font-weight:500}}
  .note{{margin-top:16px; padding-top:14px; border-top:1px solid var(--line2);
    font-size:.88rem; color:var(--soft); max-width:76ch}}
  .spec{{margin-top:30px; background:var(--card); border:1px solid var(--line);
    border-radius:3px; padding:16px}}
  .sh{{display:flex; gap:10px; align-items:baseline; flex-wrap:wrap;
    border-bottom:1px solid var(--line); padding-bottom:10px}}
  .n{{font-family:Fraunces,Georgia,serif; font-size:1.3rem; color:var(--ink)}}
  .sid{{font-family:"IBM Plex Mono",monospace; font-size:.75rem; color:var(--soft)}}
  .chip{{font-family:"IBM Plex Mono",monospace; font-size:.68rem; padding:2px 7px;
    border:1px solid var(--line2); border-radius:2px; color:var(--soft)}}
  .chip.ai{{color:var(--abl); border-color:var(--abl)}}
  .delta{{margin-left:auto; font-family:"IBM Plex Mono",monospace; font-size:.74rem;
    color:var(--soft)}}
  .ask{{margin-top:12px; background:var(--well); border-left:2px solid var(--line2)}}
  .ask summary{{cursor:pointer; padding:8px 11px; font-size:.85rem; color:var(--soft)}}
  .ask summary span{{font-family:"IBM Plex Mono",monospace; font-size:.73rem; margin-left:7px}}
  .askbody{{white-space:pre-wrap; font-size:.86rem; padding:0 13px 12px 13px; max-height:300px;
    overflow-y:auto}}
  .rowlab{{font-family:"IBM Plex Mono",monospace; font-size:.69rem; letter-spacing:.11em;
    text-transform:uppercase; color:var(--soft); margin:16px 0 8px}}
  .pair{{display:grid; grid-template-columns:1fr 1fr; gap:14px}}
  @media (max-width:900px){{.pair{{grid-template-columns:1fr}}}}
  .side{{border-left:2px solid; padding:10px 12px}}
  .side.src{{background:var(--src-well); border-left-color:var(--src)}}
  .side.abl{{background:var(--abl-well); border-left-color:var(--abl)}}
  .sl{{font-family:"IBM Plex Mono",monospace; font-size:.7rem; letter-spacing:.06em;
    text-transform:uppercase; margin-bottom:9px}}
  .side.src .sl{{color:var(--src)}} .side.abl .sl{{color:var(--abl)}}
  .ct{{float:right; letter-spacing:0; text-transform:none; opacity:.85}}
  .p{{display:grid; grid-template-columns:22px 1fr; gap:8px; margin-bottom:11px;
    font-size:.875rem}}
  .pn{{font-family:"IBM Plex Mono",monospace; font-size:.72rem; color:var(--soft);
    padding-top:2px}}
  .role{{grid-column:2; font-family:"IBM Plex Mono",monospace; font-size:.64rem;
    letter-spacing:.07em; text-transform:uppercase; margin-bottom:3px; display:block}}
  .r-bg,.r-gap{{color:var(--r-bg)}} .r-harm{{color:var(--r-harm)}} .r-plan{{color:var(--r-plan)}}
  .p p{{grid-column:2}}
  .foot{{margin-top:44px; padding-top:16px; border-top:1px solid var(--line2);
    font-size:.85rem; color:var(--soft); max-width:84ch}}
  .foot code{{font-family:"IBM Plex Mono",monospace; color:var(--body)}}
</style>
<div class="wrap">
  <header>
    <p class="eyebrow">configs/data/synth/ablation.yaml · smoke · gemini-3-flash-preview</p>
    <h1>What the Ablation Removes</h1>
    <p class="lede">The difficult-advice recipe calls open deliberation about the tension
      “the ingredient that matters most”. This arm removes exactly that and keeps everything
      else — the same scenario, the same facts, the same refusal — by rewriting the reasoning
      into a fixed order. Two further changes on 2026-10-04: the rewrite is no longer required
      to match the length of what it rewrote, and the reply now goes straight from background
      to advice, with the harm explanation removed from what the person sees.</p>
    <div class="contract">
      <div><b>reasoning P1</b><span>background: the situation restated, no evaluation</span></div>
      <div><b>P2</b><span>whatever the background still lacks</span></div>
      <div><b>P3…Pn-1</b><span>what the harmful thing being contemplated actually is</span></div>
      <div><b>Pn</b><span>a direct statement of what the reply will say</span></div>
      <div><b>reply P1</b><span>background: the situation said back to them</span></div>
      <div><b>reply rest</b><span>the advice, starting immediately — the refusal and the
        alternative, in full</span></div>
      <div><b>reply: absent</b><span>any account of WHY the harmful path is harmful. That
        material stays in the private reasoning and never reaches the person.</span></div>
    </div>
    <p class="note">{n} specimens from the current 1,283-row corpus, spanning the
      <code>ai</code> axis. The ablated reasoning runs <b>{ratio:.2f}×</b> the source's length
      now that no length floor is enforced — so a downstream difference between these arms
      cannot be read as “the deliberation mattered” until length is controlled. A fifth row
      was dropped by the lint for reaching for banned weighing vocabulary in its
      reasoning.</p>
  </header>
  {specs}
  <p class="foot">
    Source rows are <code>dougalldeepmind/2026-10-02-da-synth</code> stage 6, reused verbatim
    via <code>load_source_run</code>; only <code>reasoning</code> and <code>response</code> are
    rewritten, by <code>google/gemini-3-flash-preview</code>. Paragraph roles on the ablated
    side are assigned by position from the contract, and each row also carries the model's own
    <code>ablation_paragraph_map</code>. Text is verbatim on both sides.
  </p>
</div>
"""

if __name__ == "__main__":
    main()
