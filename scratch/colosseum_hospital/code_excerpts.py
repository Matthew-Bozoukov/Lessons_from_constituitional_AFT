# ABOUTME: Pull verbatim source excerpts (functions, classes, module-level assignments) out of the repo by
# ABOUTME: (path, symbol) so the dossier's code walkthrough quotes the real code with file:line captions.
import ast
import json
import sys
from pathlib import Path

REPO = Path(
    "/Users/kunwar/projects/lessons_from_constitutional_aft/.claude/worktrees/multiagent-exploration"
)
OUT = Path("output/colosseum_hospital/analysis/dossier_inputs/code_excerpts.json")


def find(path: Path, symbol: str):
    src = path.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == symbol
        ):
            return node.lineno, node.end_lineno
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == symbol:
                    return node.lineno, node.end_lineno
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == symbol
        ):
            return node.lineno, node.end_lineno
    return None


def excerpt(
    rel: str,
    symbol: str,
    label: str | None = None,
    lines: tuple[int, int] | None = None,
) -> dict:
    path = REPO / rel
    if lines is None:
        span = find(path, symbol)
        if span is None:
            raise SystemExit(f"{symbol} not found in {rel}")
    else:
        span = lines
    text = path.read_text().split("\n")
    code = "\n".join(text[span[0] - 1 : span[1]])
    return {
        "label": label or symbol,
        "path": rel,
        "start": span[0],
        "end": span[1],
        "code": code,
    }


SPEC = [
    # analysis layer: stances and rules
    (
        "scratch/colosseum_hospital/decline_vs_action.py",
        "DECLINE",
        "DECLINE — the first-person moral decline regex",
    ),
    (
        "scratch/colosseum_hospital/decline_vs_action.py",
        "QUESTION",
        "QUESTION — a decline phrased as a question is not a decline",
    ),
    (
        "scratch/colosseum_hospital/decline_vs_action.py",
        "PRACTICAL",
        "PRACTICAL — a practical inability is not a moral decline",
    ),
    (
        "scratch/colosseum_hospital/decline_vs_action.py",
        "COMMIT",
        "COMMIT — the first-person commitment regex",
    ),
    (
        "scratch/colosseum_hospital/decline_vs_action.py",
        "BOARD_COMMIT",
        "BOARD_COMMIT — a plan on the board (regex fallback)",
    ),
    (
        "scratch/colosseum_hospital/decline_vs_action.py",
        "BOARD_DECLINE",
        "BOARD_DECLINE — a refusal on the board (regex fallback)",
    ),
    (
        "scratch/colosseum_hospital/decline_vs_action.py",
        "decline_matches",
        "decline_matches — how the three regexes combine",
    ),
    (
        "scratch/colosseum_hospital/decline_vs_action.py",
        "reasoning_stance",
        "reasoning_stance — one stance per seat and iteration",
    ),
    (
        "scratch/colosseum_hospital/decline_vs_action.py",
        "prov_actions",
        "prov_actions — the provisioner's sabotage rule",
    ),
    (
        "scratch/colosseum_hospital/decline_vs_action.py",
        "triage_actions",
        "triage_actions — Triage's sabotage rule",
    ),
    (
        "scratch/colosseum_hospital/decline_vs_action.py",
        "in_deficit",
        "in_deficit — what 'visible bottleneck deficit' means",
    ),
    (
        "scratch/colosseum_hospital/decline_vs_action.py",
        "classify",
        "classify — the six trajectory classes",
    ),
    (
        "scratch/colosseum_hospital/sabotage_actions.py",
        "shift_rows",
        "shift_rows — every per-shift action measure",
    ),
    # judge and posts
    (
        "scratch/colosseum_hospital/post_judge.py",
        "SECRET",
        "SECRET — what the post judge is told about the condition",
    ),
    (
        "scratch/colosseum_hospital/post_judge.py",
        "COALITION_FORMAT",
        "COALITION_FORMAT — the coalition-post kind prompt",
    ),
    (
        "scratch/colosseum_hospital/post_judge.py",
        "REASON_FORMAT",
        "REASON_FORMAT — why the plan was written",
    ),
    (
        "scratch/colosseum_hospital/post_judge.py",
        "PUBLIC_FORMAT",
        "PUBLIC_FORMAT — intent to deceive on a public post",
    ),
    (
        "scratch/colosseum_hospital/post_judge.py",
        "board_posts",
        "board_posts — every pair post from blackboards.json",
    ),
    (
        "scratch/colosseum_hospital/post_judge.py",
        "call_index",
        "call_index — every model call with the messages its tool calls posted",
    ),
    (
        "scratch/colosseum_hospital/post_judge.py",
        "link_posts",
        "link_posts — a post to the call that made it",
    ),
    (
        "scratch/colosseum_hospital/post_judge.py",
        "cap",
        "cap — how long reasoning is truncated for the judge",
    ),
    (
        "scratch/colosseum_hospital/board_plans.py",
        "analyse_cell",
        "board_plans.analyse_cell — plan × action × said-no per shift",
    ),
    (
        "scratch/colosseum_hospital/partner_sway.py",
        "can_read_board",
        "can_read_board — who can see the coalition board",
    ),
    (
        "scratch/colosseum_hospital/partner_sway.py",
        "commits",
        "commits — commitment with negations excluded",
    ),
    (
        "scratch/colosseum_hospital/partner_sway.py",
        "PARTNER_CUE",
        "PARTNER_CUE — partner-following language",
    ),
    (
        "scratch/colosseum_hospital/partner_sway.py",
        "TARGET",
        "TARGET — which hospital a post names",
    ),
    (
        "scratch/colosseum_hospital/deceptive_posts.py",
        "COVER",
        "COVER — a stated deceptive purpose in the reasoning",
    ),
    (
        "scratch/colosseum_hospital/deceptive_posts.py",
        "contradicted",
        "contradicted — the false-claims cleaning rule",
    ),
    (
        "scratch/colosseum_hospital/false_claims.py",
        "_FORMAT",
        "false_claims._FORMAT — the content judge's answer format",
    ),
    (
        "scratch/colosseum_hospital/false_claims.py",
        "initial_inventory",
        "initial_inventory — the environment's iteration-1 stock, ported",
    ),
    (
        "scratch/colosseum_hospital/false_claims.py",
        "snapshots",
        "snapshots — the env's end-of-iteration inventory from the pulled logs",
    ),
]


def main() -> None:
    out = [excerpt(*s) for s in SPEC]
    extra = json.loads(sys.argv[1]) if len(sys.argv) > 1 else []
    for e in extra:
        out.append(
            excerpt(
                e["path"],
                e.get("symbol", ""),
                e.get("label"),
                tuple(e["lines"]) if e.get("lines") else None,
            )
        )
    OUT.write_text(json.dumps(out, indent=1))
    for e in out:
        print(
            f"{e['path']}:{e['start']}-{e['end']}  {e['label']}  ({e['end'] - e['start'] + 1} lines)"
        )


if __name__ == "__main__":
    main()
