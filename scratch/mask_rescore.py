# ABOUTME: Re-score published MASK runs from their stored judge text (same repo, new revision): re-read
# ABOUTME: every verdict with the patched parser, rerun the harness's metric stages, pool the overall per row.
"""Re-score every published MASK eval run in place, from the judge text it already holds.

Two scoring changes, neither of which needs the model or the judge again (2026-09-10):
  * evaluate.py's verdict parser (VENDORED_FROM.txt, evaluate.py #5) read the judge's FIRST
    "Answer:" line and kept non-letters verbatim; it now reads the LAST, reduced to a letter;
  * the runner's overall was the mean of the six archetypes; the paper pools rows (§4.3),
    now `runner.overall_honesty`.
Every published run keeps each raw judge reply beside its extracted verdict in
results/<archetype>_evaluated.csv, so the fix is applied to exactly the text that was scored.

Per repo (every `eval:mask` repo in HF_ORG), at its current head:
  1. fetch results/ + the card at that exact sha;
  2. NULL CHECK: rerun metric.py + process_metrics.py on the evaluated CSVs as published (after
     the same pandas round trip the re-read goes through) and require the published
     all_results.json, and every row's published eval_honesty_1, back exactly. This environment
     scores as the run did before anything is changed;
  3. re-read every binary verdict from its raw judge column with the patched parser (numeric
     ranges are left as stored: that path did not change);
  4. rerun both stages, rebuild results.json (pooled overall, `overall_weighting: per_row`, a
     `rescore` block) and its markdown mirror;
  5. stage results/ + metadata/rescore_meta.json, keep the card's fields and tags, add a
     `rescored` field, and (with --push) push it as a new revision, refusing if the head moved.

Run:  uv run python scratch/mask_rescore.py           # dry run: stage + report, push nothing
      uv run python scratch/mask_rescore.py --push    # push each repo's new revision
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml

from src.eval.misalignment.mask import runner
from src.eval.run_eval import _results_markdown
from src.infra.huggingface import hf_api, hf_org, hf_snapshot, push_run_dir
from src.utils import git_sha

ARCHETYPES = runner.ARCHETYPES
CODE_PATHS = ["src/eval/misalignment/mask", "configs/eval/mask.yaml", "tests/test_mask_eval.py"]
COMMAND = "uv run python scratch/mask_rescore.py --push"


def load_parser():
    """The PATCHED extract_boxed_evaluation + parse_proposition, from the vendored evaluate.py."""
    os.environ.setdefault("MASK_JUDGE_API_KEY", "unused")    # it builds a judge client at import
    sys.path.insert(0, str(runner._HARNESS))                 # `from prompts.evaluation_prompts import *`
    spec = importlib.util.spec_from_file_location("mask_evaluate", runner._HARNESS / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.extract_boxed_evaluation, mod.parse_proposition


def _null(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v))


def reread(df: pd.DataFrame, extract, parse_proposition) -> int:
    """Re-derive every binary row's `eval(...)_extracted` from its raw judge column, in place.

    Returns how many verdict cells changed. Numeric rows keep their stored ranges: a numeric
    verdict never went through the string parser.
    """
    binary = df["proposition"].map(lambda p: parse_proposition(p)[0] == "binary")
    changed = 0
    for col in [c for c in df.columns if c.startswith("eval(") and c.endswith("_extracted")]:
        raw = col[: -len("_extracted")]
        assert raw in df.columns, f"no raw judge column for {col}"
        old = df[col].astype(object)
        new = old.copy()
        new[binary] = df.loc[binary, raw].map(lambda t: extract(t) if isinstance(t, str) else None)
        changed += sum(not ((_null(a) and _null(b)) or a == b) for a, b in zip(old, new))
        df[col] = new
    return changed


def stage_evaluated(results: Path, data_dir: Path, model: str, transform=None) -> dict[str, int]:
    """Write each archetype's evaluated CSV where the harness stages read it (MASK_DATA_DIR)."""
    (data_dir / "evaluated").mkdir(parents=True)
    changed = {}
    for a in ARCHETYPES:
        df = pd.read_csv(results / f"{a}_evaluated.csv")
        changed[a] = transform(df) if transform else 0
        df.to_csv(data_dir / "evaluated" / f"{a}_{model}.csv", index=False)
    return changed


def score(data_dir: Path, logs: Path) -> dict:
    """metric.py + process_metrics.py over data_dir/evaluated, as runner.run() runs them."""
    env = {**os.environ, "MASK_DATA_DIR": str(data_dir)}
    runner._run_stage(["metric.py"], env, runner._HARNESS, logs)
    runner._run_stage(["process_metrics.py"], env, runner._HARNESS, logs)
    return json.loads((data_dir / "metrics" / "all_results.json").read_text())


def row_honesty(data_dir: Path, model: str) -> dict[str, list[int]]:
    return {a: pd.read_csv(data_dir / "metrics" / f"{a}_{model}.csv")["eval_honesty_1"].tolist()
            for a in ARCHETYPES}


def read_card(path: Path) -> tuple[dict, list[str]]:
    """(fields, tags) from a card written by src.infra.huggingface.card_markdown."""
    text = path.read_text(encoding="utf-8")
    front = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert front, f"{path}: no front matter"
    fields = dict(re.findall(r"^\| `([^`]+)` \| (.*) \|$", text, re.M))
    return fields, yaml.safe_load(front.group(1))["tags"]


def rescore(repo_id: str, root: Path, parser, provenance: dict, push: bool) -> dict:
    name = repo_id.split("/")[1]
    api = hf_api()
    head = api.repo_info(repo_id, repo_type="dataset").sha
    src = Path(hf_snapshot(repo_id, repo_type="dataset", revision=head,
                           allow_patterns=["results/*", "README.md"]))
    published = json.loads((src / "results" / "results.json").read_text())
    if published.get("overall_weighting") == "per_row":
        print(f"--- {name}: already re-scored at {head[:10]}, skipped")
        return {"repo": repo_id, "skipped": "already per_row", "head": head}
    model = published["model"]
    work = (root / name).resolve()

    # 2. The null check: nothing changed, everything must come back as published.
    null_dir = work / "null"
    stage_evaluated(src / "results", null_dir, model)
    null_all = score(null_dir, work / "logs_null")
    published_all = json.loads((src / "results" / "all_results.json").read_text())
    assert null_all == published_all, f"{name}: the null re-score differs from all_results.json"
    old_rows = row_honesty(null_dir, model)
    for a in ARCHETYPES:
        pub = src / "results" / f"{a}_metrics.csv"
        if pub.exists():
            assert pd.read_csv(pub)["eval_honesty_1"].tolist() == old_rows[a], f"{name}/{a}: rows differ"

    # 3-4. Re-read every verdict, score again.
    extract, parse_proposition = parser
    fix_dir = work / "fixed"
    changed = stage_evaluated(src / "results", fix_dir, model,
                              lambda df: reread(df, extract, parse_proposition))
    new_all = score(fix_dir, work / "logs_fixed")
    new_rows = row_honesty(fix_dir, model)
    flips = Counter(f"{o}->{n}" for a in ARCHETYPES for o, n in zip(old_rows[a], new_rows[a]) if o != n)

    per = {}
    for a, old in published["per_archetype"].items():
        m = new_all[a][model]
        per[a] = {**old, "honesty_score": m["honesty_score_1"], "unhonest_pct": m["unhonest_1"],
                  "honest_pct": m["honest_1"], "n_responses": m["total_responses"]}
    overall = runner.overall_honesty(per)
    summary = {}
    for key, value in published.items():
        summary[key] = value
        if key == "overall_honesty_score":
            summary[key] = overall
            summary["overall_weighting"] = "per_row"
    summary["per_archetype"] = per
    summary["rescore"] = {
        "date": provenance["date"], "from_revision": head,
        "source_repo": f"teaching_claude_why_replication @ {provenance['git_sha']}"
                       + (" + uncommitted diff (metadata/rescore_meta.json)" if provenance["dirty"] else ""),
        "changes": ["judge verdicts re-read from the judge's last `Answer:` line, letter only "
                    "(evaluate.py PATCH, VENDORED_FROM.txt evaluate.py #5)",
                    "overall pooled per row as in the paper (§4.3); was the mean of six archetypes"],
        "previous_overall_honesty_score": published["overall_honesty_score"],
        "verdicts_changed": sum(changed.values()), "honesty_flips": dict(flips),
    }

    # 5. Stage what changed: results/ and one metadata file; rollouts/ and the rest stay as they are.
    out = work / "push"
    (out / "results").mkdir(parents=True)
    (out / "metadata").mkdir()
    for a in ARCHETYPES:
        shutil.copy2(fix_dir / "evaluated" / f"{a}_{model}.csv", out / "results" / f"{a}_evaluated.csv")
        shutil.copy2(fix_dir / "metrics" / f"{a}_{model}.csv", out / "results" / f"{a}_metrics.csv")
    (out / "results" / "all_results.json").write_text(json.dumps(new_all, indent=2))
    (out / "results" / "results.json").write_text(json.dumps(summary, indent=2))
    (out / "results" / "results.md").write_text(_results_markdown(summary["target"], summary["mode"], summary))
    (out / "metadata" / "rescore_meta.json").write_text(json.dumps({
        "repo": repo_id, "from_revision": head, "command": COMMAND, **provenance,
        "null_check": "metric.py + process_metrics.py on the published evaluated CSVs reproduced "
                      "the published all_results.json and every row's eval_honesty_1 exactly",
        "verdicts_changed": changed, "honesty_flips": dict(flips),
        "previous_overall": {"overall_honesty_score": published["overall_honesty_score"],
                             "weighting": "mean of the six archetype scores"},
        "overall": {"overall_honesty_score": overall, "weighting": "per_row"},
    }, indent=2))
    fields, tags = read_card(src / "README.md")
    fields["rescored"] = (f"{provenance['date']}: every judge verdict re-read from its last `Answer:` "
                          f"line (letter only) and the overall pooled per row as in the paper (§4.3), "
                          f"{published['overall_honesty_score']} -> {overall}. Supersedes revision "
                          f"{head[:10]}; see metadata/rescore_meta.json.")

    new_head = None
    if push:
        now = api.repo_info(repo_id, repo_type="dataset").sha
        assert now == head, f"{name}: head moved {head[:10]} -> {now[:10]} since it was read; not pushing"
        push_run_dir(out, repo_id, fields, front_matter={"tags": tags})
        new_head = api.repo_info(repo_id, repo_type="dataset").sha
    print(f"--- {name}: {published['overall_honesty_score']} -> {overall} | verdicts changed "
          f"{sum(changed.values())} | flips {dict(flips)}" + (f" | pushed {new_head[:10]}" if new_head else ""))
    return {"repo": repo_id, "from_revision": head, "new_revision": new_head,
            "old_overall": published["overall_honesty_score"], "new_overall": overall,
            "per_archetype": {a: (published["per_archetype"][a]["honesty_score"], per[a]["honesty_score"])
                              for a in per},
            "verdicts_changed": changed, "honesty_flips": dict(flips)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-score published MASK runs from their stored judge text.")
    ap.add_argument("--push", action="store_true", help="push each repo's new revision (default: dry run)")
    ap.add_argument("--repo", nargs="*", help="only these repo names (default: every eval:mask repo in HF_ORG)")
    args = ap.parse_args()

    repos = sorted(d.id for d in hf_api().list_datasets(author=hf_org(), filter="eval:mask"))
    if args.repo:
        repos = [r for r in repos if r.split("/")[1] in args.repo]
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout
    provenance = {"date": date.today().isoformat(), "git_sha": git_sha(),
                  "dirty": bool(git("status", "--porcelain", "--", *CODE_PATHS).strip()),
                  "uncommitted_diff": git("diff", "HEAD", "--", *CODE_PATHS),
                  "script": Path(__file__).read_text()}
    root = Path("output") / f"mask_rescore_{datetime.now():%Y%m%d_%H%M%S}"
    root.mkdir(parents=True)
    parser = load_parser()
    rows = [rescore(r, root, parser, provenance, args.push) for r in repos]
    (root / "report.json").write_text(json.dumps(
        {"push": args.push, "git_sha": provenance["git_sha"], "dirty": provenance["dirty"], "repos": rows},
        indent=2))
    print(f"\n{'repo':38s} {'old':>7s} {'new':>7s} {'delta':>7s}  verdicts  revision")
    for r in rows:
        if "skipped" in r:
            continue
        print(f"{r['repo'].split('/')[1]:38s} {r['old_overall']:7.2f} {r['new_overall']:7.2f} "
              f"{r['new_overall'] - r['old_overall']:+7.2f}  {sum(r['verdicts_changed'].values()):8d}  "
              f"{(r['new_revision'] or 'dry run')[:10]}")
    print(f"\nreport: {root / 'report.json'}")


if __name__ == "__main__":
    main()
