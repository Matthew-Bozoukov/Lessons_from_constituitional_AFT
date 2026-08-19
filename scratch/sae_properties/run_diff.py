# ABOUTME: LLM-stage driver — dataset diffing over embedded .pkl caches: hypothesis
# ABOUTME: generation + per-corpus judge verification, via the vendored paper scripts.

"""Diff a target corpus against others and verify the hypotheses.

Run (needs an existing embed run; OPENROUTER_API_KEY does the LLM calls, no GPU):

    uv run --project scratch/sae_properties python scratch/sae_properties/run_diff.py \
        --config configs/properties/sae_diff.yaml run=<embed-run-name> [key=value ...]

This drives the paper's own scripts (vendored, unmodified) as subprocesses so their
sibling imports and CLIs stay exactly as published:

    1. paper/diffing/generate_sae_hypotheses.py  — latent freq diff (>= min_difference)
       -> relabel -> <= num_hypotheses hypotheses          [SAE method, section 4.1]
    2. paper/diffing/hypothesis_verifier.py       — LLM judge scores every hypothesis
       on every corpus's documents -> verified frequencies [appendix L.1]

Outputs under <embed.out_root>/<run>/diff_<target>/:
    hypotheses.json          the generator's full result (differences + features)
    verify_<corpus>/         verifier artifacts per corpus (csv + report json)
    report.md                markdown mirror: hypotheses + verified frequency table
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
DIFFING_DIR = Path(__file__).resolve().parent / "third_party" / "interp_embed" / "paper" / "diffing"
load_dotenv(REPO_ROOT / ".env")  # OPENROUTER_API_KEY for generator + judge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util import write_run_meta  # noqa: E402


def run_script(script: str, args: list[str]) -> None:
    cmd = [sys.executable, script] + args
    print(f"[run_diff] $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=DIFFING_DIR, env=os.environ.copy(), check=True)


def summarize(out_dir: Path, hypotheses: dict, corpora: list[str]) -> str:
    """Merge the per-corpus verification reports into one markdown table.

    The verifier's report json holds `summary_by_hypothesis[i].verification_rates[field]`
    in the same order as the hypotheses file, so rows join on index.
    """
    diffs = hypotheses.get("differences", [])
    rates: dict[str, list[float | None]] = {}
    for corpus in corpora:
        reports = [json.loads(p.read_text())
                   for p in (out_dir / f"verify_{corpus}").rglob("verification_report.json")]
        # Keep only reports that verified THIS hypotheses file (stale runs can leave
        # others behind), newest first by the report's own timestamp — path order lies.
        reports = sorted((r for r in reports if r["metadata"]["num_hypotheses"] == len(diffs)),
                         key=lambda r: r["metadata"]["timestamp"])
        if not reports:
            continue
        summary = reports[-1]["summary_by_hypothesis"]
        by_idx = {s["hypothesis_idx"]: s for s in summary}
        rates[corpus] = [
            next(iter(by_idx[i]["verification_rates"].values()), {}).get("percentage")
            if i in by_idx else None
            for i in range(len(diffs))
        ]

    lines = ["# SAE dataset diffing — verified hypotheses", "",
             f"Query: {hypotheses.get('query')}",
             f"Target: `{hypotheses.get('dataset1_path')}`", "",
             f"## Hypotheses ({len(diffs)})"]
    for i, d in enumerate(diffs):
        desc = (d.get("description") or d.get("title") or str(d)).strip()
        lines.append(f"\n### {i + 1}. {desc.split('. ')[0]}")
        lines.append(desc)
        if rates:
            lines.append("")
            lines.append("| corpus | judge-verified frequency |")
            lines.append("|---|---|")
            for corpus in corpora:
                r = rates.get(corpus, [None] * len(diffs))[i]
                lines.append(f"| {corpus} | {'—' if r is None else f'{r / 100:.2f}'} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/properties/sae_diff.yaml")
    ap.add_argument("--skip-verify", action="store_true", help="Hypotheses only")
    ap.add_argument("--report-only", action="store_true", help="Re-render report.md from existing artifacts")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = OmegaConf.merge(OmegaConf.load(REPO_ROOT / args.config),
                          OmegaConf.from_dotlist(args.overrides))
    if not cfg.get("run"):
        raise SystemExit("Pass run=<embed-run-name> (the directory run_embed.py created).")

    run_dir = REPO_ROOT / cfg.embed.out_root / cfg.run
    ds_dir = run_dir / "datasets"
    channel = cfg.diff.get("channel", cfg.embed.channels[0])
    target, others = str(cfg.diff.target), [str(o) for o in cfg.diff.others]

    def pkl(corpus: str) -> Path:
        p = ds_dir / f"{corpus}__{channel}.pkl"
        if not p.exists():
            raise SystemExit(f"Missing embed cache {p} — run run_embed.py first.")
        return p

    out_dir = run_dir / f"diff_{target}"
    out_dir.mkdir(parents=True, exist_ok=True)
    hyp_path = out_dir / "hypotheses.json"

    if args.report_only:
        hypotheses = json.loads(hyp_path.read_text())
        (out_dir / "report.md").write_text(summarize(out_dir, hypotheses, [target] + others))
        print(f"[run_diff] report re-rendered: {out_dir / 'report.md'}")
        return

    official = {"Llama-3.3-70B-Instruct-SAE-l50": "meta-llama/Llama-3.3-70B-Instruct",
                "Llama-3.1-8B-Instruct-SAE-l19": "meta-llama/Llama-3.1-8B-Instruct"}
    reader = cfg.sae.get("hf_model") or official[str(cfg.sae.variant)]

    run_script("generate_sae_hypotheses.py", [
        "--dataset1", str(pkl(target)),
        "--dataset2", *[str(pkl(o)) for o in others],
        "--sae-model", reader,
        "--query", str(cfg.diff.query),
        "--model", str(cfg.diff.model),
        "--min-difference", str(cfg.diff.min_difference),
        "--threshold", str(cfg.diff.label_score_threshold),
        "--num-hypotheses", str(cfg.diff.num_hypotheses),
        "--max-concurrency", str(cfg.diff.max_concurrency),
        "--max-feature-diffs", str(cfg.diff.max_feature_diffs),
        "--output", str(hyp_path),
    ])

    if not args.skip_verify:
        for corpus in [target] + others:
            run_script("hypothesis_verifier.py", [
                "-p", str(hyp_path),
                "-i", str(ds_dir / f"{corpus}__{channel}.csv"),
                "--fields", "text",
                "-o", str(out_dir / f"verify_{corpus}"),
                "--judge-model", str(cfg.diff.judge_model),
                "--max-concurrent", str(cfg.diff.max_concurrency),
            ])

    hypotheses = json.loads(hyp_path.read_text())
    (out_dir / "report.md").write_text(summarize(out_dir, hypotheses, [target] + others))
    write_run_meta(out_dir, OmegaConf.to_container(cfg, resolve=True), extra={"stage": "diff"})
    print(f"[run_diff] report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
