# ABOUTME: Render a readable sample of the advocacy arm — cleanest rewrites and the ones that
# ABOUTME: still concede — with the blind judge's verdict attached to each.

"""A sample you can actually read, chosen to show both ends.

716 side-by-side traces is not a document anyone reads. This picks the rewrites the blind
concession judge scored cleanest (nothing good about the rejected option, nothing bad about the
chosen one) and the ones it scored worst, so the arm can be judged on its failures as well as
its successes rather than on a mean.

Run:
  uv run python scratch/read_advocacy_samples.py
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(run_dir: str = "output/whole_trace/advocacy_full", n: int = 6,
         out: str | None = None) -> None:
    """Write an HTML sample of the best and worst rated rewrites.

    Args:
        run_dir: The advocacy run directory.
        n: How many of each kind.
        out: Output path; defaults to <run_dir>/samples.html.
    """
    d = Path(run_dir)
    recs = {json.loads(x)["scenario_id"]: json.loads(x)
            for x in (d / "excisions.jsonl").read_text().splitlines() if x.strip()}
    rated: dict[str, dict] = {}
    for x in (d / "semantic_concession.jsonl").read_text().splitlines():
        r = json.loads(x)
        if r["cond"] == "after":
            rated[r["sid"]] = r
    scored = [(v["weighing"] + v["power"], sid, v) for sid, v in rated.items()]
    scored.sort()
    picks = [("CLEANEST — judge found no concession", scored[:n]),
             ("STILL CONCEDING — the residue worth reading", scored[-n:][::-1])]

    parts = ["<meta charset='utf-8'><title>Advocacy rewrites</title>",
             "<style>body{font:14px/1.6 system-ui;margin:0 auto;max-width:1250px;padding:2rem;"
             "background:#fbfbfa}h1{font-size:1.5rem}h2{font-size:1.1rem;margin-top:2.5rem;"
             "border-bottom:2px solid #ddd;padding-bottom:.3rem}h3{font-size:1rem;"
             "margin-top:1.8rem}.g{display:grid;grid-template-columns:1fr 1fr;gap:1rem}"
             ".c{background:#fff;border:1px solid #e2e2e2;border-radius:6px;padding:.9rem;"
             "white-space:pre-wrap;font:12.5px/1.55 ui-monospace,monospace}"
             ".h{font:600 12px system-ui;color:#666;margin-bottom:.3rem}"
             ".m{color:#555;font-size:13px}.q{background:#fff4d6;padding:.5rem .7rem;"
             "border-radius:4px;font-size:13px;margin:.4rem 0}"
             ".s{font:600 12px system-ui;color:#b00}</style>",
             "<h1>One-sided advocacy rewrites</h1>",
             "<p class='m'>The scenario, the user's message and the assistant's reply are "
             "unchanged; only the private reasoning differs. Scores are from a blind judge that "
             "rated each version without knowing which was which: <b>credits</b> = says "
             "something good about the option NOT taken, <b>concedes</b> = admits a cost of the "
             "option taken. 0 is clean, 3 is worst.</p>"]
    for title, group in picks:
        parts.append(f"<h2>{title}</h2>")
        for _, sid, v in group:
            r = recs[sid]
            parts.append(
                f"<h3>{html.escape(sid)} &mdash; <span class='s'>credits {v['weighing']}, "
                f"concedes {v['power']}</span> "
                f"<span class='m'>({r['words_before']}&rarr;{r['words_after']} words)</span></h3>"
                + (f"<div class='q'><b>Judge:</b> {html.escape(v['why'])}</div>" if v.get("why")
                   else "")
                + (f"<div class='q'><b>Flagged:</b> {html.escape(v['quote'])}</div>"
                   if v.get("quote") else "")
                + f"<div class='g'><div><div class='h'>ORIGINAL REASONING</div>"
                f"<div class='c'>{html.escape(r['before'])}</div></div>"
                f"<div><div class='h'>REWRITTEN</div>"
                f"<div class='c'>{html.escape(r['after'])}</div></div></div>")
    path = Path(out or d / "samples.html")
    path.write_text("\n".join(parts))
    print(f"{len(recs)} rewrites, {len(rated)} rated; sampled {n} cleanest + {n} worst")
    for title, group in picks:
        print(f"\n{title}")
        for tot, sid, v in group:
            print(f"  {sid:14s} credits {v['weighing']}  concedes {v['power']}")
    print(f"\n-> {path}")


if __name__ == "__main__":
    fire.Fire(main)
