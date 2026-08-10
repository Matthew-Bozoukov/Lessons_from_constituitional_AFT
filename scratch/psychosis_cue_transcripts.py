# ABOUTME: One-off report generator for the 2026-08-10 continuation-cue A/B: renders both
# ABOUTME: psychosis runs' rollouts into one self-contained HTML transcript viewer.

"""Build a side-by-side transcript page for two psychosis run directories.

Usage:
    uv run python scratch/psychosis_cue_transcripts.py \
        --arm "upstream cue:output/psychosis/.../<ts_baseline>" \
        --arm "modified cue:output/psychosis/.../<ts_modified>" \
        --out <path.html>

Reads only `rollouts/*.json` (written the moment each conversation finishes), so it
works on a run whose judging was disabled or whose epilogue crashed.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

# The two failure modes the modified cue targets, plus the one it may introduce.
PREAMBLE_RE = re.compile(r"^\s*(\*|\(|_)")
LABEL_RE = re.compile(r"^\s*[A-Z][a-z]+\s*:")


def flags(message: str, persona_first_name: str) -> list[str]:
    """Artifacts that mark a message as not reading like something a user would type."""
    found = []
    if PREAMBLE_RE.match(message):
        found.append("narrator preamble")
    if LABEL_RE.match(message):
        found.append("speaker label")
    if re.search(rf"\b{persona_first_name}\b", message):
        found.append("names own persona")
    return found


def load_arm(run_dir: Path) -> list[dict]:
    """Load one run's persona rollouts, in stem order, with per-message flags attached."""
    personas = []
    for path in sorted((run_dir / "rollouts").glob("*.json")):
        if path.stem.endswith("_error"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        first = data["character"].split("_")[0].title()
        for turn in data["turns"]:
            turn["flags"] = flags(turn["user"], first)
        personas.append(data)
    return personas


def _esc(text: str) -> str:
    return html.escape(text or "").replace("\n", "<br>")


def _turn_html(turn: dict) -> str:
    chips = "".join(f'<span class="flag">{html.escape(f)}</span>' for f in turn["flags"])
    think = turn.get("think") or ""
    think_html = (f'<details class="think"><summary>target reasoning '
                  f'({len(think)} chars)</summary><div>{_esc(think)}</div></details>'
                  if think.strip() else "")
    return f"""
      <article class="turn">
        <div class="marker">{turn['turn']}</div>
        <div class="speech persona">
          <div class="who">red-teamer, in character{chips}</div>
          <div class="body">{_esc(turn['user'])}</div>
        </div>
        <div class="speech model">
          <div class="who">target reply</div>
          {think_html}
          <div class="body">{_esc(turn['assistant'])}</div>
        </div>
      </article>"""


def _persona_html(persona: dict, arm_id: str) -> str:
    stem = persona["character"]
    n_flagged = sum(1 for t in persona["turns"] if t["flags"])
    turns = "".join(_turn_html(t) for t in persona["turns"])
    return f"""
    <section class="persona" id="{arm_id}-{stem}">
      <header class="persona-head">
        <h2>{html.escape(stem.split('_')[0].title())}</h2>
        <p class="stem">{html.escape(stem)} &middot; {len(persona['turns'])} turns &middot;
           {n_flagged} flagged</p>
      </header>
      {turns}
    </section>"""


def build(arms: list[tuple[str, Path]]) -> str:
    """Render the full page: one panel per arm, persona sections within."""
    loaded = [(label, load_arm(path), path) for label, path in arms]

    tabs, panels, stats = [], [], []
    for index, (label, personas, path) in enumerate(loaded):
        arm_id = f"arm{index}"
        active = " is-active" if index == 0 else ""
        total = sum(len(p["turns"]) for p in personas)
        flagged = sum(1 for p in personas for t in p["turns"] if t["flags"])
        tabs.append(f'<button class="tab{active}" data-arm="{arm_id}">{html.escape(label)}'
                    f'<span class="tally">{flagged}/{total} flagged</span></button>')
        nav = "".join(f'<a href="#{arm_id}-{p["character"]}">'
                      f'{html.escape(p["character"].split("_")[0].title())}</a>'
                      for p in personas)
        body = "".join(_persona_html(p, arm_id) for p in personas)
        panels.append(f"""
        <div class="panel{active}" id="{arm_id}">
          <nav class="rail"><p class="rail-label">personas</p>{nav}
            <p class="rail-label">run</p><p class="path">{html.escape(path.name)}</p></nav>
          <main class="column">{body}</main>
        </div>""")
        stats.append((label, flagged, total))

    return TEMPLATE.format(tabs="".join(tabs), panels="".join(panels))


TEMPLATE = """<title>Psychosis red-teamer — continuation cue A/B</title>
<style>
  :root {{
    --paper: #eef1f2; --card: #ffffff; --ink: #131a1f; --muted: #5a686f;
    --rule: #d3dcdf; --persona: #8c4a1e; --model: #1f6169; --flag: #9c2d52;
    --serif: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
    --sans: ui-sans-serif, system-ui, "Segoe UI", sans-serif;
    --mono: ui-monospace, "Cascadia Mono", Consolas, monospace;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --paper: #0e1316; --card: #151c20; --ink: #e4eaec; --muted: #8a9aa1;
      --rule: #26333a; --persona: #d68a52; --model: #63b6bd; --flag: #e0728f;
    }}
  }}
  :root[data-theme="dark"] {{
    --paper: #0e1316; --card: #151c20; --ink: #e4eaec; --muted: #8a9aa1;
    --rule: #26333a; --persona: #d68a52; --model: #63b6bd; --flag: #e0728f;
  }}
  :root[data-theme="light"] {{
    --paper: #eef1f2; --card: #ffffff; --ink: #131a1f; --muted: #5a686f;
    --rule: #d3dcdf; --persona: #8c4a1e; --model: #1f6169; --flag: #9c2d52;
  }}
  body {{ background: var(--paper); color: var(--ink); font-family: var(--sans);
          line-height: 1.6; margin: 0; }}
  .head {{ padding: 2.5rem 1.5rem 0; max-width: 74rem; margin: 0 auto; }}
  .head h1 {{ font-family: var(--serif); font-weight: 400; font-size: 1.9rem;
              margin: 0 0 .35rem; text-wrap: balance; }}
  .head p {{ margin: 0; color: var(--muted); max-width: 62ch; }}
  .tabs {{ display: flex; gap: .5rem; flex-wrap: wrap; padding: 1.5rem 1.5rem 0;
           max-width: 74rem; margin: 0 auto; }}
  .tab {{ font: inherit; font-size: .85rem; background: transparent; color: var(--muted);
          border: 1px solid var(--rule); border-radius: 2px; padding: .5rem .9rem;
          cursor: pointer; display: flex; gap: .6rem; align-items: baseline; }}
  .tab:hover {{ color: var(--ink); }}
  .tab:focus-visible {{ outline: 2px solid var(--model); outline-offset: 2px; }}
  .tab.is-active {{ background: var(--card); color: var(--ink); border-color: var(--ink); }}
  .tally {{ font-family: var(--mono); font-size: .72rem; font-variant-numeric: tabular-nums;
            color: var(--muted); }}
  .panel {{ display: none; max-width: 74rem; margin: 0 auto; padding: 1.5rem;
            gap: 2.5rem; grid-template-columns: 10rem minmax(0, 1fr); }}
  .panel.is-active {{ display: grid; }}
  @media (max-width: 46rem) {{ .panel.is-active {{ grid-template-columns: minmax(0, 1fr); }} }}
  .rail {{ position: sticky; top: 1.5rem; align-self: start; display: flex;
           flex-direction: column; gap: .3rem; font-size: .82rem; }}
  .rail-label {{ font-size: .68rem; letter-spacing: .1em; text-transform: uppercase;
                 color: var(--muted); margin: .8rem 0 .2rem; }}
  .rail a {{ color: var(--ink); text-decoration: none; border-bottom: 1px solid transparent;
             width: fit-content; }}
  .rail a:hover {{ border-bottom-color: var(--persona); }}
  .path {{ font-family: var(--mono); font-size: .7rem; color: var(--muted);
           margin: 0; word-break: break-all; }}
  .column {{ display: flex; flex-direction: column; gap: 3rem; min-width: 0; }}
  .persona-head {{ border-bottom: 1px solid var(--rule); padding-bottom: .6rem; }}
  .persona-head h2 {{ font-family: var(--serif); font-weight: 400; font-size: 1.4rem;
                      margin: 0; }}
  .stem {{ font-family: var(--mono); font-size: .72rem; color: var(--muted); margin: .2rem 0 0; }}
  .persona {{ display: flex; flex-direction: column; gap: 1.75rem; scroll-margin-top: 1.5rem; }}
  .turn {{ display: grid; grid-template-columns: 2rem minmax(0, 1fr); gap: .9rem; }}
  .marker {{ font-family: var(--mono); font-size: .78rem; color: var(--muted);
             font-variant-numeric: tabular-nums; padding-top: .15rem; }}
  .speech {{ grid-column: 2; border-left: 2px solid var(--rule); padding-left: 1rem; }}
  .speech + .speech {{ margin-top: 1rem; }}
  .speech.persona {{ border-left-color: var(--persona); }}
  .speech.model {{ border-left-color: var(--model); }}
  .who {{ font-size: .68rem; letter-spacing: .1em; text-transform: uppercase;
          color: var(--muted); margin-bottom: .45rem; display: flex; gap: .4rem;
          flex-wrap: wrap; align-items: center; }}
  .speech.persona .who {{ color: var(--persona); }}
  .speech.model .who {{ color: var(--model); }}
  .flag {{ font-family: var(--mono); font-size: .62rem; letter-spacing: 0;
           text-transform: none; color: var(--flag); border: 1px solid var(--flag);
           border-radius: 2px; padding: 0 .35rem; }}
  .body {{ font-family: var(--serif); max-width: 66ch; }}
  .speech.model .body {{ font-size: .94rem; color: var(--muted); }}
  .think {{ margin-bottom: .6rem; }}
  .think summary {{ font-family: var(--mono); font-size: .7rem; color: var(--muted);
                    cursor: pointer; }}
  .think div {{ font-family: var(--mono); font-size: .74rem; color: var(--muted);
                white-space: pre-wrap; border-left: 1px solid var(--rule);
                padding-left: .8rem; margin-top: .5rem; overflow-x: auto; }}
</style>
<div class="head">
  <h1>Red-teamer transcripts: upstream cue vs modified cue</h1>
  <p>Five personas, four turns each, against <code>qwen/qwen3-32b</code>. Red-teamer
     <code>nousresearch/hermes-4-405b</code> at temperature 1.0; no judging. Flags mark a
     message that opens with a narrator preamble or a speaker label, or that names its own
     persona &mdash; the things a real user would not type. One sample per arm: read the
     messages, not the tallies.</p>
</div>
<div class="tabs">{tabs}</div>
{panels}
<script>
  document.querySelectorAll(".tab").forEach(function (tab) {{
    tab.addEventListener("click", function () {{
      document.querySelectorAll(".tab").forEach(function (t) {{ t.classList.remove("is-active"); }});
      document.querySelectorAll(".panel").forEach(function (p) {{ p.classList.remove("is-active"); }});
      tab.classList.add("is-active");
      document.getElementById(tab.dataset.arm).classList.add("is-active");
    }});
  }});
</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True,
                        help="'<label>:<run_dir>', repeatable; first arm shows by default")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    arms = []
    for spec in args.arm:
        label, _, path = spec.rpartition(":")
        arms.append((label, Path(path)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(arms), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
