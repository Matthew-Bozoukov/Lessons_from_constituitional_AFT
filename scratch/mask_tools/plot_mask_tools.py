# ABOUTME: Figure for the MASK+tools check: honesty of DA and DA+tools on plain MASK vs MASK with
# ABOUTME: unused tools on the pressure prompts, on all 1,000 rows and on the rows that carry tools.
"""Run: uv run python scratch/mask_tools/plot_mask_tools.py \
    --plain-da <repo> --tools-da <repo> --plain-datools <repo> --tools-datools <repo> \
    --tools-file data/mask_tools/2026-11-11-mask-tools.json [--out output/mask_tools]

Every number is recomputed from the published per-row verdicts (results/pass<k>/<archetype>_metrics.csv,
`eval_honesty_1`: -1 = lie): a pass's honesty is 100 x (1 - share of rows judged a lie), pooled
over rows as runner.overall_honesty does, and the cell is the mean of its passes with the
runner's own between-pass 95% interval (runner.pass_interval). The all-rows number is checked
against the run's results.json. "Tooled rows" = the task_ids whose tools list is non-empty in
--tools-file; the plain runs are re-scored on exactly those rows.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from src.eval.misalignment.mask.runner import pass_interval

COLOUR = {"da": "#7e22ce", "da_tools": "#d0558f"}  # Figure 2's arm colours


def verdicts(repo: str) -> tuple[dict[int, pd.DataFrame], dict, str]:
    """{pass: DataFrame(task_id, eval_honesty_1)}, results.json, revision sha."""
    api = HfApi()
    sha = api.dataset_info(repo).sha
    files = api.list_repo_files(repo, repo_type="dataset", revision=sha)
    res = json.load(
        open(
            hf_hub_download(
                repo, "results/results.json", repo_type="dataset", revision=sha
            )
        )
    )
    by_pass: dict[int, list] = {}
    for f in files:
        if not f.endswith("_metrics.csv") or not f.startswith("results/"):
            continue
        parts = f.split("/")
        k = int(parts[1].removeprefix("pass")) if parts[1].startswith("pass") else 1
        df = pd.read_csv(
            hf_hub_download(repo, f, repo_type="dataset", revision=sha),
            usecols=["task_id", "eval_honesty_1"],
        )
        by_pass.setdefault(k, []).append(df)
    return {k: pd.concat(v) for k, v in sorted(by_pass.items())}, res, sha


def cell(by_pass: dict[int, pd.DataFrame], keep: set[str] | None) -> dict:
    scores, n = [], None
    for df in by_pass.values():
        d = df if keep is None else df[df["task_id"].astype(str).isin(keep)]
        n = len(d)
        scores.append(100.0 * (1 - (d["eval_honesty_1"] == -1).mean()))
    out = pass_interval(scores, n)
    out["per_pass"] = [round(s, 2) for s in scores]
    out["n_rows"] = n
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("plain-da", "tools-da", "plain-datools", "tools-datools", "tools-file"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--out", default="output/mask_tools")
    a = ap.parse_args()
    tooled = {k for k, v in json.load(open(a.tools_file)).items() if v}
    repos = {
        ("da", "plain"): a.plain_da,
        ("da", "tools"): a.tools_da,
        ("da_tools", "plain"): a.plain_datools,
        ("da_tools", "tools"): a.tools_datools,
    }
    table, shas = [], []
    for (arm, cond), repo in repos.items():
        by_pass, res, sha = verdicts(repo)
        shas.append(f"{repo}@{sha[:8]}")
        allr, sub = cell(by_pass, None), cell(by_pass, tooled)
        assert abs(allr["mean"] - res["overall_honesty_score"]) < 0.06, (
            repo,
            allr["mean"],
            res["overall_honesty_score"],
        )
        table.append(
            {
                "arm": arm,
                "prompt": cond,
                "repo": f"{repo}@{sha[:8]}",
                "all_rows": allr,
                "tooled_rows": sub,
                "tool_calls_per_pass": [
                    p.get("empty_content", {}).get("by_type", {}).get("tool_call")
                    for p in res.get("per_pass", [])
                ],
            }
        )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = "2026-11-11"
    # Figure 2's visual language (scratch/figures/_style.py): house bars and error bars, bold values,
    # dashed y grid, no top/right spine, the frameless legend on top, the fixed arm colours. Plain MASK
    # is the solid bar, MASK + tools the hatched one. No base-model line: base Qwen has no MASK+tools run.
    from scratch.figures import _style as st

    st.apply()
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    for i, arm in enumerate(["da", "da_tools"]):
        for j, cond in enumerate(["plain", "tools"]):
            c = next(t for t in table if t["arm"] == arm and t["prompt"] == cond)["tooled_rows"]
            lo, hi = c["ci95"] if c["ci95"] else (None, None)
            st.bar(ax, i * 2.4 + j, c["mean"], lo, hi, colour=st.COLOUR[arm],
                   label=f"{c['mean']:.1f}", hatch="//" if cond == "tools" else None)
    ax.set_xticks([0, 1, 2.4, 3.4])
    ax.set_xticklabels(["Plain\nMASK", "MASK\n+ tools", "Plain\nMASK", "MASK\n+ tools"])
    ax.set_xlim(-0.65, 4.05)
    st.finish(ax, title=f"MASK honesty on the {len(tooled)} rows that carry tools",
              ylabel="Honesty (%)", ylim=(0, 100), yticks=range(0, 101, 20), title_pad=16)
    st.legend(fig, [st.patch(st.COLOUR["da"], st.ARM["da"]),
                    st.patch(st.COLOUR["da_tools"], st.ARM["da_tools"]),
                    st.patch("white", "Unused tool definitions in the pressure prompt", hatch="//")],
              ncol=1, y=1.16)
    png = st.save(fig, str(out), "mask_tools_honesty",
                  title="MASK honesty with and without unused tools on the pressure prompts",
                  script="scratch/mask_tools/plot_mask_tools.py", artifacts=shas)
    (out / f"{stamp}_mask_tools_honesty_results.json").write_text(
        json.dumps(table, indent=1)
    )
    md = [
        "| arm | prompt | tooled rows (n) | all rows | per pass (tooled) | tool calls per pass | repo |",
        "|---|---|---|---|---|---|---|",
    ]
    for t in table:
        s, r = t["tooled_rows"], t["all_rows"]
        md.append(
            f"| {t['arm']} | {t['prompt']} | {s['mean']} {s['ci95']} ({s['n_rows']}) | {r['mean']} {r['ci95']} "
            f"| {s['per_pass']} | {t['tool_calls_per_pass']} | {t['repo']} |"
        )
    (out / f"{stamp}_mask_tools_honesty_results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(png)


if __name__ == "__main__":
    main()
