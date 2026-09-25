# ABOUTME: Render one or more synth smoke runs as a single HTML page of full transcripts (system,
# ABOUTME: user, reasoning, reply) with the reviser's note and pressure-classifier labels beside each.
"""uv run python scratch/render_smoke_rows.py OUT.html RUN_DIR:LABEL:LABELS.jsonl [...]"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

e = html.escape


def load(run_dir: str, labels_path: str):
    run = Path(run_dir)
    rows = [json.loads(l) for l in (run / "dataset.jsonl").open(encoding="utf-8")]
    notes = {}
    for f in run.glob("stage_*_revise_prompts.jsonl"):
        for l in f.open(encoding="utf-8"):
            r = json.loads(l)
            notes[r["scenario_id"]] = r.get("refine_changes", "")
    labels = {}
    if labels_path and Path(labels_path).exists():
        for l in Path(labels_path).open(encoding="utf-8"):
            r = json.loads(l)
            labels[r["scenario_id"]] = r
    return rows, notes, labels


def panel(i: int, group: str, r: dict, note: str, lab: dict) -> tuple[str, str]:
    md = r["metadata"]
    sysm = next((m["content"] for m in r["messages"] if m["role"] == "system"), "")
    user = next(m["content"] for m in r["messages"] if m["role"] == "user")
    asst = [m for m in r["messages"] if m["role"] == "assistant"][-1]
    push = lab.get("push", "?")
    chips = "".join(
        f'<span class="chip {"bad" if push in ("push", "override") and k == "push" else ""}">{k} <b>{e(str(lab.get(k, "?")))}</b></span>'
        for k in ("actor", "push", "goods", "monitor", "help_fully")) if lab else '<span class="chip">unlabelled</span>'
    opt = (f'<option value="{i}">{i + 1}. [{e(group)}] {e(md["trait_id"])} · {e(md.get("domain", ""))} · push={e(push)}</option>')
    body = f'''<div class="panel" data-i="{i}"{"" if i == 0 else " hidden"}>
<div class="meta"><div class="eyebrow">{e(group)} · {e(md["scenario_id"])}</div>
<p class="pr">{e(md.get("trait_name", ""))}</p>
<div class="chips">{chips}</div>
<div class="eyebrow">Situation (metadata)</div><p>{e(md.get("situation", ""))}</p>
<div class="eyebrow">Planned convenient call</div><p>{e(md.get("shortcut", ""))}</p>
<div class="eyebrow">Reviser's change note</div><p class="note">{e(note)}</p></div>
<section class="turn sys"><div class="who">system prompt</div><pre class="txt">{e(sysm)}</pre></section>
<section class="turn usr"><div class="who">user</div><pre class="txt">{e(user)}</pre></section>
<section class="turn ag"><div class="who">assistant · reasoning (trained, hidden at inference)</div><pre class="txt think">{e(asst.get("reasoning_content") or "")}</pre>
<div class="who">assistant · reply</div><pre class="txt">{e(asst["content"])}</pre></section>
</div>'''
    return opt, body


def main() -> None:
    out = Path(sys.argv[1])
    opts, bodies = [], []
    i = 0
    for spec in sys.argv[2:]:
        run_dir, label, labels_path = (spec.split(":") + ["", ""])[:3]
        rows, notes, labels = load(run_dir, labels_path)
        for r in rows:
            sid = r["metadata"]["scenario_id"]
            o, b = panel(i, label, r, notes.get(sid, ""), labels.get(sid, {}))
            opts.append(o)
            bodies.append(b)
            i += 1
    page = f'''<title>DA Wording Smoke Rows</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap">
<style>
:root{{--bg:#F5F6F3;--surface:#FFFFFF;--surface2:#EEF0EA;--ink:#1B2230;--ink2:#4A5364;--ink3:#7A8391;--line:#D9DDD4;--accent:#1F6F78;--sys:#6B5B95;--usr:#B7791F;--ag:#1F6F78;--think:#F1F5F4;--bad:#A63D2F;--badsoft:#F5DEDA}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#151A21;--surface:#1D242E;--surface2:#242C38;--ink:#E7EAE3;--ink2:#B4BAC3;--ink3:#828A96;--line:#333C49;--accent:#63B8C0;--sys:#B9A8E0;--usr:#E0A94A;--ag:#63B8C0;--think:#1A2A2C;--bad:#E37E6E;--badsoft:#41231F}}}}
:root[data-theme="dark"]{{--bg:#151A21;--surface:#1D242E;--surface2:#242C38;--ink:#E7EAE3;--ink2:#B4BAC3;--ink3:#828A96;--line:#333C49;--accent:#63B8C0;--sys:#B9A8E0;--usr:#E0A94A;--ag:#63B8C0;--think:#1A2A2C;--bad:#E37E6E;--badsoft:#41231F}}
*{{box-sizing:border-box}}body{{background:var(--bg);color:var(--ink);font-family:"Public Sans",system-ui,sans-serif;font-size:15px;line-height:1.55;margin:0}}
.wrap{{max-width:940px;margin:0 auto;padding-inline:16px;padding-block:20px 64px}}
h1{{font-family:"Newsreader",Georgia,serif;font-weight:600;font-size:28px;margin:0 0 4px;text-wrap:balance}}
.lede{{color:var(--ink2);margin:0 0 14px;max-width:72ch}}
#sel{{font:inherit;font-size:14px;padding:8px 10px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--ink);max-width:100%;margin-bottom:16px}}
.meta{{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:14px 16px;margin-bottom:14px}}
.meta p{{margin:0 0 10px;max-width:80ch}}.pr{{font-family:"Newsreader",Georgia,serif;font-size:18px;font-weight:600}}.note{{color:var(--ink2)}}
.eyebrow{{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink3);font-weight:600;margin-top:6px}}
.chips{{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 10px}}.chip{{font-size:12px;background:var(--surface2);border-radius:999px;padding:3px 10px;color:var(--ink2)}}.chip.bad{{background:var(--badsoft);color:var(--bad)}}
.turn{{border-left:3px solid var(--line);padding:10px 14px;margin:0 0 12px;background:var(--surface);border-radius:0 6px 6px 0}}
.turn.sys{{border-left-color:var(--sys)}}.turn.usr{{border-left-color:var(--usr)}}.turn.ag{{border-left-color:var(--ag);border-left-width:5px}}
.who{{font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;font-weight:600;color:var(--ink3);margin:8px 0 6px}}
pre.txt{{white-space:pre-wrap;word-wrap:break-word;font-family:inherit;margin:0;font-size:14.5px;line-height:1.55}}
pre.think{{background:var(--think);border-radius:4px;padding:10px 12px;font-family:"Newsreader",Georgia,serif;font-size:15.5px}}
</style>
<div class="wrap"><h1>DA Wording Smoke Rows</h1>
<p class="lede">The 20-row all-trait smoke and the 10-row t6-only smoke of the "ask, not instruct" wording (branch <code>jamie/da-t6-wording</code> @ 9f093f2c), each row with its pressure-classifier labels, the reviser's change note, and the full trained transcript.</p>
<select id="sel" aria-label="Choose a row">{"".join(opts)}</select>
{"".join(bodies)}
</div>
<script>
const sel=document.getElementById('sel'),panels=[...document.querySelectorAll('.panel')];
sel.addEventListener('change',()=>{{panels.forEach(p=>p.hidden=p.dataset.i!==sel.value);window.scrollTo({{top:0}});}});
</script>'''
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({len(page):,} chars, {i} rows)")


if __name__ == "__main__":
    main()
