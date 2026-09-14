# ABOUTME: A folded dossier section holding every result table re-run on 2026-09-13 (Gemini 3.6 Flash judge,
# ABOUTME: the new action rules), converted from the analysis scripts' markdown result files.
import html
import re
from pathlib import Path

A = Path(
    "/Users/kunwar/projects/lessons_from_constitutional_aft/.claude/worktrees/multiagent-exploration/output/colosseum_hospital/analysis"
)
FILES = [
    (
        "2026-09-13_colosseum_hospital_direction_contrasts.md",
        "Cell means and paired contrasts (harvest measures and the per-channel judge)",
    ),
    (
        "2026-09-13_colosseum_hospital_simple_story_results.md",
        "The four boxes under the paper harness",
    ),
    (
        "2026-09-13_colosseum_hospital_simple_story_by_harness_results.md",
        "The four boxes under all three harnesses",
    ),
    (
        "2026-09-13_colosseum_hospital_sabotage_by_pair_results.md",
        "Action-level measures per shift",
    ),
    (
        "2026-09-13_colosseum_hospital_rule_sensitivity.md",
        "Rule sensitivity: old rules, each fix, new rules, flood 20 / 60",
    ),
    (
        "2026-09-13_colosseum_hospital_flip_rate.md",
        "The flip rate: refused privately, then sabotaged",
    ),
    (
        "2026-09-13_colosseum_hospital_board_plans_results.md",
        "Plans on the boards (post judge, 3.6 Flash)",
    ),
    ("2026-09-13_colosseum_hospital_partner_sway_results.md", "Partner sway"),
    (
        "2026-09-13_colosseum_hospital_deceptive_posts_results.md",
        "Deceptive public posts",
    ),
]


def md_to_html(md: str, max_lines: int = 400) -> str:
    """Headings, paragraphs and pipe tables of a results file, as page HTML (examples cut)."""
    out = []
    lines = md.split("\n")[:max_lines]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c or "--") for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                th = "".join(f"<th>{html.escape(c)}</th>" for c in rows[0])
                body = "".join(
                    "<tr>"
                    + "".join(
                        f'<td class="{"lab" if k == 0 else "num"}">{html.escape(c)}</td>'
                        for k, c in enumerate(r)
                    )
                    + "</tr>"
                    for r in rows[1:]
                )
                out.append(
                    f'<div class="tbl"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'
                )
            continue
        if line.startswith("#"):
            level = min(len(line) - len(line.lstrip("#")), 3)
            text = html.escape(line.lstrip("#").strip())
            if level <= 1:
                out.append(f"<h3>{text}</h3>")
            else:
                out.append(f'<h3 style="font-size:.95rem">{text}</h3>')
        elif line.strip().startswith("- "):
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(f"<li>{html.escape(lines[i].strip()[2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif line.strip():
            out.append(f'<p class="note">{html.escape(line.strip())}</p>')
        i += 1
    return "\n".join(out)


def section() -> str:
    parts = []
    for name, title in FILES:
        p = A / name
        if not p.is_file():
            continue
        body = md_to_html(p.read_text())
        parts.append(
            f'<details><summary><b>{html.escape(title)}</b> <span class="chip">{name}</span></summary>{body}</details>'
        )
    return (
        '<section id="results13">\n<h2>Results re-run on 2026-09-13</h2>\n'
        '<p class="lede"><span class="changed">Every table below was produced today with the Gemini 3.6 Flash judge (four channels, the held question) and the new action rules (objective deficit, negation window, queue-relative spam). The folded results sections further down keep the 2026-09-10 tables (Gemini 3 Flash, original rules) with today\'s figures, so the two can be compared row by row.</span></p>\n'
        + "\n".join(parts)
        + "\n</section>\n"
    )
