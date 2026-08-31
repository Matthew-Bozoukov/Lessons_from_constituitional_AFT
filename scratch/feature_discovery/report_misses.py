# ABOUTME: Render the confirmed truthfulness misses as a browsable HTML review page plus a
# ABOUTME: greppable markdown mirror, so each flagged row can be accepted or rejected by hand.

"""Build the review artifact for the truthfulness-miss audit.

Every row shown survived two passes: a judge that flagged it, and a differently-framed
verifier that checked the item was in scope and found no hedge anywhere. The point of the
page is to make each call cheap to overturn, so the reply and reasoning are included in full.

Run:
  uv run python scratch/feature_discovery/report_misses.py
"""

from __future__ import annotations

import html
import json
import sys
from collections import Counter
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(misses_dir: str = "output/feature_discovery/20260812_092119/truthfulness_misses",
         sft: str = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl",
         show: int = 120) -> None:
    """Write review.html and misses.md from the verified verdicts.

    Args:
        misses_dir: Directory holding verified.jsonl.
        sft: The difficult-advice SFT file.
        show: How many confirmed misses to render in full.
    """
    d = Path(misses_dir)
    rec = [json.loads(x) for x in (d / "verified.jsonl").read_text().splitlines() if x.strip()]
    conf = sorted([r for r in rec if r["verify"] == "confirmed"],
                  key=lambda r: -r["severity"])
    rows = {json.loads(x)["metadata"]["scenario_id"]: json.loads(x)
            for x in Path(sft).read_text().splitlines() if x.strip()}
    judged = json.loads((d / "run_meta.json").read_text())["judged"]

    sev = Counter(r["severity"] for r in conf)
    tra = Counter(r["trait_id"] for r in conf)
    md = [f"# Truthfulness misses in the difficult-advice corpus", "",
          f"Rows judged: **{judged}** (every row lacking a C116 feature). "
          f"Flagged by pass 1: **{len(rec)}**. Surviving verification: **{len(conf)}** "
          f"({len(conf) / judged:.1%} of judged rows).", "",
          "A miss = the assistant treated a load-bearing external claim as settled, and "
          "marked the uncertainty nowhere in reasoning or reply.", "",
          "| severity | count | meaning |", "|---|---:|---|",
          f"| 5 | {sev[5]} | confidently directs action that could harm if the claim is wrong |",
          f"| 4 | {sev[4]} | serious; advice rests on an unchecked claim |",
          f"| 3 | {sev[3]} | the advice would change if the claim were false |",
          f"| 2 | {sev[2]} | minor |", f"| 1 | {sev[1]} | pedantic |", "",
          "| trait | confirmed misses |", "|---|---:|"]
    md += [f"| {t} | {n} |" for t, n in sorted(tra.items())]
    md += ["", "## Confirmed misses", ""]
    for r in conf:
        md += [f"### {r['scenario_id']} — severity {r['severity']} ({r['trait_id']})", "",
               f"- **unverified item**: {r.get('unverifiable_item', '')}",
               f"- **assertion**: {r.get('evidence_quote', '')}", f"- **why**: {r.get('why', '')}", ""]
    (d / "misses.md").write_text("\n".join(md))

    def esc(s: str) -> str:
        return html.escape(str(s))

    cards = []
    for r in conf[:show]:
        row = rows[r["scenario_id"]]
        m = {x["role"]: x for x in row["messages"]}
        cards.append(
            f"<details><summary><b>sev {r['severity']}</b> · {r['scenario_id']} · "
            f"{r['trait_id']} — {esc(r.get('unverifiable_item', ''))[:120]}</summary>"
            f"<div class=box><b>Flagged assertion:</b> <i>{esc(r.get('evidence_quote', ''))}</i><br>"
            f"<b>Why:</b> {esc(r.get('why', ''))}</div>"
            f"<div class=turn><div class=role>user</div><pre>{esc(m['user']['content'])}</pre></div>"
            f"<div class=turn><div class=role>private reasoning</div>"
            f"<pre>{esc(m['assistant'].get('reasoning_content', ''))}</pre></div>"
            f"<div class=turn><div class=role>reply</div>"
            f"<pre>{esc(m['assistant']['content'])}</pre></div></details>")

    (d / "review.html").write_text(f"""<!doctype html><meta charset=utf-8>
<title>Truthfulness misses</title><style>
body{{font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;margin:0 auto;max-width:1000px;
padding:26px;background:#fafafa;color:#1c1c1e}}
details{{background:#fff;border:1px solid #e3e3e6;border-radius:10px;margin:10px 0;padding:10px 14px}}
summary{{cursor:pointer}} .box{{background:#fff8f3;border-left:4px solid #DD8452;padding:8px 12px;margin:10px 0}}
.turn{{border-left:4px solid #ccc;padding:4px 12px;margin:10px 0}}
.role{{font-size:12px;text-transform:uppercase;color:#666}}
pre{{white-space:pre-wrap;font:13px/1.5 ui-monospace,Menlo,monospace;max-height:420px;overflow:auto}}
table{{border-collapse:collapse;background:#fff}} td,th{{border:1px solid #e3e3e6;padding:5px 9px}}
</style>
<h1>Truthfulness misses in the difficult-advice corpus</h1>
<p>{judged} rows judged (every row lacking a C116 feature); {len(rec)} flagged;
<b>{len(conf)} survived verification</b> ({len(conf) / judged:.1%}). Showing the top
{min(show, len(conf))} by severity. A miss = a load-bearing external claim treated as
settled, with the uncertainty marked nowhere.</p>
<table><tr><th>severity</th><th>5</th><th>4</th><th>3</th><th>2</th><th>1</th></tr>
<tr><td>count</td><td>{sev[5]}</td><td>{sev[4]}</td><td>{sev[3]}</td><td>{sev[2]}</td><td>{sev[1]}</td></tr></table>
{''.join(cards)}""")
    print(f"confirmed {len(conf)} of {len(rec)} flagged, from {judged} judged")
    print(f"severity: {dict(sorted(sev.items(), reverse=True))}")
    print(f"wrote {d}/review.html and misses.md")


if __name__ == "__main__":
    fire.Fire(main)
