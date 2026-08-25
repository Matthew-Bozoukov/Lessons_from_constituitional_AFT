# ABOUTME: Map the difficult-advice rows of a training mixture back to their feature-discovery
# ABOUTME: clusters, so a trained arm can be described by the reasoning properties it saw.

"""Which rows of a training mixture carry which reasoning-property clusters.

A published mixture row is rendered chat text with no scenario_id, so the join back to the
feature-discovery run goes through the user message, which is verbatim identical and unique
across the corpus. The script asserts that every difficult-advice row matches exactly one
source row — a silent partial join would understate every cluster's prevalence.

Reports the clusters' prevalence inside the mixture against their prevalence in the full
corpus, because the mixture is a trait-balanced subsample and need not inherit the corpus rate.

Run:
  uv run python scratch/mixture_cluster_membership.py \
      --mixture data/hf/<repo>/mixture_think.jsonl
"""

from __future__ import annotations

import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import git_sha, timestamp  # noqa: E402

# The harm-risk seed cluster and its centroid neighbourhood (see find_harm_risk_instances.py).
DEFAULT_CLUSTERS = (30, 79, 20, 142, 137, 29)
SOURCE = "synthdoc_difficult_advice"
USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)
# Surface form of the probability / severity / reversibility move, for measuring how many
# rows in a cluster actually make it rather than merely sitting near it in embedding space.
LEXICAL = re.compile(
    r"\b(probabilit\w*|likelihood|how likely|unlikely|severit\w*|how severe|magnitude"
    r"|reversib\w*|irreversib\w*|permanen\w*|undo\w*|recoverab\w*)\b", re.I)


def _user(text: str) -> str:
    """Extract the user turn from a rendered Qwen chat string.

    Args:
        text: The mixture row's `text` field.

    Returns:
        The user message content, stripped.

    Raises:
        ValueError: If the row has no user turn.
    """
    m = USER_RE.search(text)
    if not m:
        raise ValueError(f"no user turn in {text[:120]!r}")
    return m.group(1).strip()


def _reasoning(row: dict) -> str:
    """Pull the private reasoning out of an SFT row.

    Args:
        row: A stage_7 SFT record.

    Returns:
        The assistant's reasoning_content, or "" when the row has none.
    """
    return {m["role"]: m for m in row["messages"]}["assistant"].get("reasoning_content", "")


def _dashboard(path: Path, title: str, insts: dict, rows: dict, cl_label: dict,
               picked: tuple[int, ...]) -> None:
    """Write a single-file browsable listing of the matched mixture rows.

    Args:
        path: Output html path.
        title: Page heading.
        insts: scenario_id -> {cluster id -> [matched feature strings]}.
        rows: scenario_id -> SFT record.
        cl_label: cluster id -> label.
        picked: The cluster ids being reported.
    """
    order = sorted(insts, key=lambda s: (-len(insts[s]), s))
    parts = [
        f"<meta charset='utf-8'><title>{html.escape(title)}</title>",
        "<style>body{font:14px/1.55 system-ui;margin:0 auto;max-width:1000px;padding:2rem;"
        "background:#fbfbfa}h1{font-size:1.4rem}details{border:1px solid #ddd;border-radius:6px;"
        "margin:.5rem 0;background:#fff}summary{padding:.6rem .8rem;cursor:pointer}"
        "pre{white-space:pre-wrap;background:#f6f6f4;padding:.8rem;border-radius:4px;"
        "font:12px/1.5 ui-monospace,monospace}.f{color:#444}code{background:#eee;"
        "padding:0 .25rem;border-radius:3px}</style>",
        f"<h1>{html.escape(title)}</h1><p>{len(insts)} rows. Clusters: " +
        "; ".join(f"C{c} {html.escape(cl_label[c])}" for c in picked) + "</p>",
    ]
    for sid in order:
        hit = insts[sid]
        feats = sorted({f for fs in hit.values() for f in fs})
        parts.append(
            f"<details><summary><code>{html.escape(sid)}</code> &mdash; " +
            ", ".join(f"C{c}" for c in sorted(hit)) + "</summary>"
            "<div class='f'><b>matched features</b><ul>" +
            "".join(f"<li>{html.escape(f)}</li>" for f in feats) + "</ul></div>"
            f"<pre>{html.escape(_reasoning(rows[sid]))}</pre></details>")
    path.write_text("\n".join(parts))


def main(
    mixture: str = "data/hf/2026-08-06-table2-9284-synthdoc-716-train/mixture_think.jsonl",
    run_dir: str = "output/feature_discovery/20260812_092119",
    sft: str = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl",
    clusters_arg: tuple[int, ...] = DEFAULT_CLUSTERS,
    out_dir: str | None = None,
) -> None:
    """Report which mixture rows belong to the given feature-discovery clusters.

    Args:
        mixture: Local path to a downloaded mixture jsonl (rendered `text` + `source`).
        run_dir: Feature-discovery run holding clusters and per-trace features.
        sft: The difficult-advice SFT file the mixture drew from.
        clusters_arg: Cluster ids to report, e.g. --clusters_arg 30,79,137.
        out_dir: Output directory; defaults to output/mixture_cluster_membership/<timestamp>.

    Raises:
        RuntimeError: If any difficult-advice row fails to join to exactly one source row.
    """
    picked = tuple(clusters_arg) if isinstance(clusters_arg, (list, tuple)) else (clusters_arg,)
    d = Path(run_dir)
    clusters = json.loads((d / "clusters.json").read_text())["clusters"]
    cl_label = {c["cluster"]: c["label"] for c in clusters}
    cl_traces = {c["cluster"]: c["n_traces"] for c in clusters}
    fmap = json.loads((d / "feature_cluster_map.json").read_text())
    traces = [json.loads(x) for x in (d / "features.jsonl").read_text().splitlines() if x.strip()]
    feats_by_sid = {t["scenario_id"]: t["features"] for t in traces}

    rows = {}
    by_user: dict[str, list[str]] = {}
    for line in Path(sft).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sid = r["metadata"]["scenario_id"]
        rows[sid] = r
        by_user.setdefault({m["role"]: m for m in r["messages"]}["user"]["content"].strip(),
                           []).append(sid)

    mix = [json.loads(x) for x in Path(mixture).read_text().splitlines() if x.strip()]
    da = [m for m in mix if m.get("source") == SOURCE]
    sids, bad = [], []
    for m in da:
        cand = by_user.get(_user(m["text"]), [])
        (sids.append(cand[0]) if len(cand) == 1 else bad.append(len(cand)))
    if bad:
        raise RuntimeError(f"{len(bad)} of {len(da)} rows failed to join uniquely "
                           f"(candidate counts {Counter(bad)}) — the mixture may come from a "
                           f"different synth run than {sft}")
    unlabelled = [s for s in sids if s not in feats_by_sid]

    insts: dict[str, dict[int, list[str]]] = {}
    for sid in sids:
        hit: dict[int, list[str]] = {}
        for f in feats_by_sid.get(sid, []):
            c = fmap.get(f)         # None => unclustered (HDBSCAN noise)
            if c in picked:
                hit.setdefault(c, []).append(f)
        if hit:
            insts[sid] = hit

    out = Path(out_dir or f"output/mixture_cluster_membership/{timestamp()}")
    out.mkdir(parents=True, exist_ok=True)
    labelled = len(sids) - len(unlabelled)

    lines = [f"# Clusters {', '.join(f'C{c}' for c in picked)} inside `{Path(mixture).parent.name}`", "",
             f"{len(mix)} mixture rows, of which **{len(da)} are `{SOURCE}`**. All {len(da)} "
             f"joined uniquely to `{sft}` by user message; "
             f"{len(unlabelled)} carry no feature labels.", "",
             f"**{len(insts)} of {labelled} difficult-advice rows ({len(insts) / labelled:.1%}) "
             f"belong to at least one of the {len(picked)} clusters.**", "",
             "## Per cluster", "",
             "| cluster | label | in mixture | of the 716 | corpus rate | lexically exhibit |",
             "|--:|---|--:|--:|--:|--:|"]
    for c in picked:
        mem = sorted(s for s in insts if c in insts[s])
        lex = sum(1 for s in mem if LEXICAL.search(_reasoning(rows[s])))
        lines.append(f"| C{c} | {cl_label[c]} | {len(mem)} | {len(mem) / labelled:.1%} | "
                     f"{cl_traces[c] / len(traces):.1%} | {lex} ({lex / max(1, len(mem)):.0%}) |")
    union_lex = sum(1 for s in insts if LEXICAL.search(_reasoning(rows[s])))
    rest = [s for s in sids if s not in insts and s in feats_by_sid]
    lines += ["",
              f"Union: {len(insts)} rows, {union_lex} ({union_lex / len(insts):.0%}) lexically "
              f"exhibit the probability/severity/reversibility move. The other "
              f"{len(rest)} difficult-advice rows: "
              f"{sum(1 for s in rest if LEXICAL.search(_reasoning(rows[s])))} "
              f"({sum(1 for s in rest if LEXICAL.search(_reasoning(rows[s]))) / len(rest):.0%}).",
              ""]

    overlap = Counter(len(h) for h in insts.values())
    lines += ["## Overlap", "", "| clusters per row | rows |", "|--:|--:|"]
    lines += [f"| {n} | {c} |" for n, c in sorted(overlap.items())]

    trait = Counter(rows[s]["metadata"]["trait_id"] for s in insts)
    lines += ["", "## Trait mix of the matched rows", "",
              "| trait | rows |", "|---|--:|"]
    lines += [f"| {t} | {n} |" for t, n in sorted(trait.items())]

    lines += ["", "## Every matched row", "",
              "`scenario_id` — clusters — the features that matched", ""]
    for sid in sorted(insts):
        hit = insts[sid]
        feats = "; ".join(sorted({f for fs in hit.values() for f in fs}))
        lines.append(f"- `{sid}` — {' '.join(f'C{c}' for c in sorted(hit))} — {feats}")

    (out / "membership.md").write_text("\n".join(lines) + "\n")
    (out / "membership.jsonl").write_text("".join(
        json.dumps({"scenario_id": s, "clusters": sorted(insts[s]),
                    "features": sorted({f for fs in insts[s].values() for f in fs}),
                    "lexically_exhibits": bool(LEXICAL.search(_reasoning(rows[s])))}) + "\n"
        for s in sorted(insts)))
    (out / "scenario_ids.txt").write_text("\n".join(sorted(insts)) + "\n")
    _dashboard(out / "membership.html",
               f"{', '.join(f'C{c}' for c in picked)} in {Path(mixture).parent.name}",
               insts, rows, cl_label, picked)
    (out / "run_meta.json").write_text(json.dumps(
        {"git_sha": git_sha(), "timestamp_utc": timestamp(), "mixture": mixture,
         "run_dir": run_dir, "sft": sft, "clusters": list(picked),
         "difficult_advice_rows": len(da), "labelled": labelled, "matched": len(insts),
         "command": " ".join(sys.argv)}, indent=2))

    print(f"{SOURCE}: {len(da)} rows, all joined uniquely ({len(unlabelled)} unlabelled)")
    for c in picked:
        print(f"  C{c:3d} {cl_label[c][:44]:44s} {sum(1 for s in insts if c in insts[s]):4d}")
    print(f"union: {len(insts)}/{labelled} -> {out}")


if __name__ == "__main__":
    fire.Fire(main)
