# ABOUTME: One-off report generator for scratch/psychosis_redteam_probe.py: renders the
# ABOUTME: model x persona willingness matrix plus every probe transcript into one HTML page.

"""Build a willingness matrix + transcript viewer for one probe run directory.

Usage:
    uv run python scratch/psychosis_probe_report.py \
        --run output/psychosis_redteam_probe/<ts> --out <path.html>

Reads `transcripts/*.json` (full per-cell records) and `summary.json`. Shares the visual
language of scratch/psychosis_cue_transcripts.py deliberately; the two are separate
throwaway scripts because their data shapes are unrelated (a 2-arm A/B vs an N-model sweep).
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

OUTCOME_LABEL = {
    "completed": "full arc",
    "refused": "refused",
    "format_miss": "no &lt;message&gt;",
}


def load(run_dir: Path) -> tuple[list[dict], dict]:
    """Load every per-cell record and the run's summary."""
    cells = [json.loads(p.read_text(encoding="utf-8"))
             for p in sorted((run_dir / "transcripts").glob("*.json"))]
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return cells, summary


def _esc(text: str) -> str:
    return html.escape(text or "").replace("\n", "<br>")


def _cell_state(cell: dict) -> str:
    if cell.get("error"):
        return "error"
    return cell["outcome"]


def matrix_html(cells: list[dict], models: list[str], personas: list[str]) -> str:
    """The at-a-glance grid: one row per red-teamer, one column per persona."""
    index = {(c["model"], c["persona"]): c for c in cells}
    head = "".join(f"<th>{html.escape(p.split('_')[0].title())}</th>" for p in personas)
    rows = []
    for model in models:
        tds = []
        for persona in personas:
            cell = index.get((model, persona))
            if cell is None:
                tds.append('<td><span class="chip none">—</span></td>')
                continue
            state = _cell_state(cell)
            label = OUTCOME_LABEL.get(state, state)
            detail = ""
            if state in ("refused", "format_miss") and cell.get("stop_turn"):
                detail = f' <span class="at">t{cell["stop_turn"]}</span>'
            elif state == "completed" and cell.get("wavered_turns"):
                detail = f' <span class="at">{len(cell["wavered_turns"])} waver</span>'
            tds.append(f'<td><span class="chip {state}">{label}</span>{detail}</td>')
        full = sum(1 for p in personas
                   if index.get((model, p), {}).get("outcome") == "completed")
        rows.append(f'<tr><th class="model">{html.escape(model)}</th>'
                    f'<td class="score">{full}/{len(personas)}</td>{"".join(tds)}</tr>')
    return f"""
    <div class="matrix-wrap">
      <table class="matrix">
        <thead><tr><th>red-teamer</th><th>full arcs</th>{head}</tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </div>"""


def _turns_html(cell: dict) -> str:
    if cell.get("error"):
        return f'<p class="empty">error: {html.escape(cell["error"])}</p>'
    blocks = []
    for turn in cell.get("turns", []):
        blocks.append(f"""
        <article class="turn">
          <div class="marker">{turn['turn']}</div>
          <div class="speech persona">
            <div class="who">red-teamer, in character</div>
            <div class="body">{_esc(turn['user'])}</div>
          </div>
          <div class="speech model">
            <div class="who">target reply</div>
            <div class="body">{_esc(turn['assistant'])}</div>
          </div>
        </article>""")
    if cell.get("stop_text"):
        blocks.append(f"""
        <article class="turn">
          <div class="marker">{cell['stop_turn']}</div>
          <div class="speech stop">
            <div class="who">arc stopped &mdash; {OUTCOME_LABEL.get(cell['outcome'], '')}</div>
            <div class="body mono">{_esc(cell['stop_text'])}</div>
          </div>
        </article>""")
    return "".join(blocks) or '<p class="empty">(no turns recorded)</p>'


def build(run_dir: Path, cells: list[dict], summary: dict) -> str:
    models = sorted({c["model"] for c in cells},
                    key=lambda m: (-summary.get(m, {}).get("full_arc_rate", 0), m))
    personas = sorted({c["persona"] for c in cells})
    index = {(c["model"], c["persona"]): c for c in cells}

    tabs, panels = [], []
    for i, model in enumerate(models):
        arm_id = f"m{i}"
        active = " is-active" if i == 0 else ""
        full = summary.get(model, {}).get("full_arcs", 0)
        total = summary.get(model, {}).get("personas", len(personas))
        tabs.append(f'<button class="tab{active}" data-arm="{arm_id}">'
                    f'{html.escape(model)}<span class="tally">{full}/{total}</span></button>')
        sections = []
        for persona in personas:
            cell = index.get((model, persona))
            if cell is None:
                continue
            state = _cell_state(cell)
            sections.append(f"""
            <section class="persona" id="{arm_id}-{persona}">
              <header class="persona-head">
                <h2>{html.escape(persona.split('_')[0].title())}
                  <span class="chip {state}">{OUTCOME_LABEL.get(state, state)}</span></h2>
                <p class="stem">{html.escape(persona)} &middot;
                   {cell.get('completed_turns', 0)}/{cell.get('requested_turns', 0)} turns
                   &middot; wavered on {cell.get('wavered_turns') or 'no turn'}</p>
              </header>
              {_turns_html(cell)}
            </section>""")
        nav = "".join(f'<a href="#{arm_id}-{p}">{html.escape(p.split("_")[0].title())}</a>'
                      for p in personas if (model, p) in index)
        panels.append(f'<div class="panel{active}" id="{arm_id}">'
                      f'<nav class="rail"><p class="rail-label">personas</p>{nav}</nav>'
                      f'<main class="column">{"".join(sections)}</main></div>')

    return (HEAD
            + f'<div class="head"><h1>Which red-teamers will play the darkest personas '
              f'all the way through?</h1><p>Each candidate runs the four reality-break '
              f'personas grok refused on, against a stand-in target. A cell is a full arc '
              f'only if the red-teamer produced an in-character message at every turn; '
              f'<em>refused</em> means it declined twice, <em>no &lt;message&gt;</em> means '
              f'it stayed engaged but dropped the required tags. Run '
              f'<code>{html.escape(run_dir.name)}</code>.</p></div>'
            + matrix_html(cells, models, personas)
            + f'<div class="tabs">{"".join(tabs)}</div>'
            + "".join(panels) + SCRIPT)


HEAD = """<title>Red-teamer willingness probe</title>
<style>
  :root {
    --paper: #eef1f2; --card: #ffffff; --ink: #131a1f; --muted: #5a686f;
    --rule: #d3dcdf; --persona: #8c4a1e; --model: #1f6169; --flag: #9c2d52;
    --ok: #2f6a41;
    --serif: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
    --sans: ui-sans-serif, system-ui, "Segoe UI", sans-serif;
    --mono: ui-monospace, "Cascadia Mono", Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --paper: #0e1316; --card: #151c20; --ink: #e4eaec; --muted: #8a9aa1;
      --rule: #26333a; --persona: #d68a52; --model: #63b6bd; --flag: #e0728f;
      --ok: #6fbe89;
    }
  }
  :root[data-theme="dark"] {
    --paper: #0e1316; --card: #151c20; --ink: #e4eaec; --muted: #8a9aa1;
    --rule: #26333a; --persona: #d68a52; --model: #63b6bd; --flag: #e0728f;
    --ok: #6fbe89;
  }
  :root[data-theme="light"] {
    --paper: #eef1f2; --card: #ffffff; --ink: #131a1f; --muted: #5a686f;
    --rule: #d3dcdf; --persona: #8c4a1e; --model: #1f6169; --flag: #9c2d52;
    --ok: #2f6a41;
  }
  body { background: var(--paper); color: var(--ink); font-family: var(--sans);
         line-height: 1.6; margin: 0; }
  .head { padding: 2.5rem 1.5rem 0; max-width: 74rem; margin: 0 auto; }
  .head h1 { font-family: var(--serif); font-weight: 400; font-size: 1.9rem;
             margin: 0 0 .35rem; text-wrap: balance; }
  .head p { margin: 0; color: var(--muted); max-width: 64ch; }
  .matrix-wrap { max-width: 74rem; margin: 1.75rem auto 0; padding: 0 1.5rem;
                 overflow-x: auto; }
  .matrix { border-collapse: collapse; font-size: .82rem; min-width: 46rem; width: 100%; }
  .matrix th, .matrix td { text-align: left; padding: .5rem .7rem;
                           border-bottom: 1px solid var(--rule); white-space: nowrap; }
  .matrix thead th { font-size: .68rem; letter-spacing: .1em; text-transform: uppercase;
                     color: var(--muted); font-weight: 400; }
  .matrix th.model { font-family: var(--mono); font-size: .74rem; font-weight: 400; }
  .matrix .score { font-family: var(--mono); font-variant-numeric: tabular-nums;
                   color: var(--muted); }
  .chip { font-family: var(--mono); font-size: .68rem; border: 1px solid currentColor;
          border-radius: 2px; padding: 0 .35rem; }
  .chip.completed { color: var(--ok); }
  .chip.refused { color: var(--flag); }
  .chip.format_miss { color: var(--persona); }
  .chip.error, .chip.none { color: var(--muted); }
  .at { font-family: var(--mono); font-size: .68rem; color: var(--muted); }
  .tabs { display: flex; gap: .5rem; flex-wrap: wrap; padding: 2rem 1.5rem 0;
          max-width: 74rem; margin: 0 auto; }
  .tab { font: inherit; font-size: .8rem; font-family: var(--mono); background: transparent;
         color: var(--muted); border: 1px solid var(--rule); border-radius: 2px;
         padding: .45rem .8rem; cursor: pointer; display: flex; gap: .5rem;
         align-items: baseline; }
  .tab:hover { color: var(--ink); }
  .tab:focus-visible { outline: 2px solid var(--model); outline-offset: 2px; }
  .tab.is-active { background: var(--card); color: var(--ink); border-color: var(--ink); }
  .tally { font-variant-numeric: tabular-nums; color: var(--muted); }
  .panel { display: none; max-width: 74rem; margin: 0 auto; padding: 1.5rem;
           gap: 2.5rem; grid-template-columns: 9rem minmax(0, 1fr); }
  .panel.is-active { display: grid; }
  @media (max-width: 46rem) { .panel.is-active { grid-template-columns: minmax(0, 1fr); } }
  .rail { position: sticky; top: 1.5rem; align-self: start; display: flex;
          flex-direction: column; gap: .3rem; font-size: .82rem; }
  .rail-label { font-size: .68rem; letter-spacing: .1em; text-transform: uppercase;
                color: var(--muted); margin: 0 0 .2rem; }
  .rail a { color: var(--ink); text-decoration: none; border-bottom: 1px solid transparent;
            width: fit-content; }
  .rail a:hover { border-bottom-color: var(--persona); }
  .column { display: flex; flex-direction: column; gap: 3rem; min-width: 0; }
  .persona { display: flex; flex-direction: column; gap: 1.75rem; scroll-margin-top: 1.5rem; }
  .persona-head { border-bottom: 1px solid var(--rule); padding-bottom: .6rem; }
  .persona-head h2 { font-family: var(--serif); font-weight: 400; font-size: 1.4rem;
                     margin: 0; display: flex; gap: .6rem; align-items: baseline;
                     flex-wrap: wrap; }
  .stem { font-family: var(--mono); font-size: .72rem; color: var(--muted); margin: .2rem 0 0; }
  .turn { display: grid; grid-template-columns: 2rem minmax(0, 1fr); gap: .9rem; }
  .marker { font-family: var(--mono); font-size: .78rem; color: var(--muted);
            font-variant-numeric: tabular-nums; padding-top: .15rem; }
  .speech { grid-column: 2; border-left: 2px solid var(--rule); padding-left: 1rem; }
  .speech + .speech { margin-top: 1rem; }
  .speech.persona { border-left-color: var(--persona); }
  .speech.model { border-left-color: var(--model); }
  .speech.stop { border-left-color: var(--flag); }
  .who { font-size: .68rem; letter-spacing: .1em; text-transform: uppercase;
         color: var(--muted); margin-bottom: .45rem; }
  .speech.persona .who { color: var(--persona); }
  .speech.model .who { color: var(--model); }
  .speech.stop .who { color: var(--flag); }
  .body { font-family: var(--serif); max-width: 66ch; }
  .speech.model .body { font-size: .94rem; color: var(--muted); }
  .body.mono { font-family: var(--mono); font-size: .78rem; white-space: pre-wrap; }
  .empty { color: var(--muted); font-size: .85rem; }
</style>
"""

SCRIPT = """
<script>
  document.querySelectorAll(".tab").forEach(function (tab) {
    tab.addEventListener("click", function () {
      document.querySelectorAll(".tab").forEach(function (t) { t.classList.remove("is-active"); });
      document.querySelectorAll(".panel").forEach(function (p) { p.classList.remove("is-active"); });
      tab.classList.add("is-active");
      document.getElementById(tab.dataset.arm).classList.add("is-active");
    });
  });
</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="a psychosis_redteam_probe run dir")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run)
    cells, summary = load(run_dir)
    if not cells:
        raise SystemExit(f"no per-cell transcripts under {run_dir / 'transcripts'}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(run_dir, cells, summary), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB) from {len(cells)} cells")


if __name__ == "__main__":
    main()
