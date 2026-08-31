# ABOUTME: Convert a legacy ODCV eval repo (agent_logs/rollout_NNN at the root or under a
# ABOUTME: combined*/ dir) into the published-layout contract and (re)publish it to the org.
# Run: uv run python scratch/convert_odcv_repo_to_contract.py --source <org>/<name> [--org LASR-Callum] [--push False]

"""Repack + republish one legacy ODCV repo for the dashboard explorer.

Mapping (legacy -> contract), mirroring what `passes.package_run` writes for a fresh run:
  [<combined>/]agent_logs/<key>-<variant>/experiments/<S>/rollout_NNN/messages_record.txt
      -> rollouts/<variant>/<S>/pass<N+1>/messages_record.txt   (+ docker_output.log,
         + cell_meta.json: variant/scenario/pass/transcript_bytes/judged)
  [<combined>/]results.json               -> results/results.json (+ results/results.md mirror)
  [<combined>/]evaluations/scores_*.json  -> results/scores_*.json
  [<combined>/]evaluations/run_meta.json  -> results/judging_run_meta.json
  plots/                                  -> results/plots/
  [<combined>/]combine_manifest.json      -> metadata/combine_manifest.json
  [<combined>/]run_meta.json              -> metadata/combine_run_meta.json
  (its config block)                      -> metadata/odcv_config.yaml
  passes/**/{rollout_manifest,pass_audit,run_meta}.json -> metadata/passes/
  README.md                               -> metadata/original_README.md (the card is
                                             regenerated with the original fields plus
                                             the Hub tags the dashboard's discovery needs)

The pass number preserves the original rollout index (rollout_002 -> pass3), so gaps
from skipped-empty cells stay visible instead of being renumbered away. `judged` is
true when the transcript's key appears in at least one judge's scores file.

When the destination is the source repo itself, the legacy root entries are deleted
after the upload so the repo root is exactly rollouts/ results/ metadata/ README.md
(the old tree stays in the repo's git history).
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from datetime import date
from pathlib import Path

import fire
import yaml
from dotenv import load_dotenv
from huggingface_hub import CommitOperationDelete, snapshot_download

from src.eval.layout import PUBLISH_DIRS, assert_layout, publish_layout
from src.huggingface import card_markdown, hf_api, push_run_dir

VARIANTS = ("mandated", "incentivized")
_ROOT_KEEP = set(PUBLISH_DIRS) | {"README.md", ".gitattributes"}


def _parse_card(text: str) -> dict:
    """Pull `| `field` | value |` rows (and the H1 as `title`) out of a card_markdown table."""
    fields = {}
    for m in re.finditer(r"^\|\s*`(\w+)`\s*\|\s*(.*)\|\s*$", text, re.M):
        fields[m.group(1)] = m.group(2).strip()
    title = re.search(r"^# (.+)$", text, re.M)
    if title:
        fields["title"] = title.group(1).strip()
    return fields


def _find_run_root(src: Path) -> Path:
    """The ONE directory holding the judged agent_logs tree (root, or a combined*/ dir).

    Raw `passes/` trees are skipped: they lack the rollout_NNN level and duplicate what
    the combined dir holds.
    """
    roots = sorted({p.parent for p in src.rglob("agent_logs")
                    if p.is_dir() and "passes" not in p.relative_to(src).parts})
    assert len(roots) == 1, f"expected exactly one judged agent_logs tree in {src}, found {roots}"
    return roots[0]


def _results_markdown(title: str, results: dict) -> str:
    """Compact greppable mirror of results.json, in the shape run_eval's epilogue writes."""
    lines = [f"# {title}", ""]
    for block in ("ours", "published"):
        if block in results:
            for variant, vals in results[block].items():
                lines.append(f"- **{block}_{variant}**: {json.dumps(vals)}")
    for key in ("delta_mr_pct", "published_within_our_ci", "n_judged", "n_dropped_all_na",
                "judging_cost_usd", "judges"):
        if key in results:
            lines.append(f"- **{key}**: {json.dumps(results[key])}")
    return "\n".join(lines) + "\n"


def build(src: Path, out: Path, source: str) -> tuple[str, int, dict]:
    """Assemble the contract tree under `out`; returns (model_key, n_transcripts, card fields)."""
    rollouts, results, metadata = publish_layout(out)
    run_root = _find_run_root(src)
    evals = run_root / "evaluations"

    scored: set[str] = set()
    for f in sorted(evals.glob("scores_*.json")) if evals.is_dir() else []:
        scored |= set(json.loads(f.read_text(encoding="utf-8")))

    model_keys: set[str] = set()
    n = 0
    for rec in sorted(run_root.glob("agent_logs/*/experiments/*/rollout_*/messages_record.txt")):
        arm_dir = rec.parents[3].name
        variant = next(v for v in VARIANTS if arm_dir.endswith("-" + v))
        model_keys.add(arm_dir[: -(len(variant) + 1)])
        scenario, rollout = rec.parents[1].name, rec.parents[0].name
        idx = int(rollout.split("_")[1])
        dest = rollouts / variant / scenario / f"pass{idx + 1}"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rec, dest / "messages_record.txt")
        log = rec.parent / "docker_output.log"
        if log.is_file():
            shutil.copy2(log, dest / "docker_output.log")
        (dest / "cell_meta.json").write_text(json.dumps({
            "variant": variant, "scenario": scenario, "pass": idx + 1,
            "source_rollout_dir": str(rec.parent.relative_to(src)),
            "transcript_bytes": rec.stat().st_size,
            "judged": f"{variant}/{scenario}/{rollout}" in scored,
        }, indent=2))
        n += 1
    assert n > 0, f"no transcripts under agent_logs/ in {source} — unexpected layout"
    assert len(model_keys) == 1, f"ambiguous model keys: {model_keys}"
    model_key = model_keys.pop()

    res = json.loads((run_root / "results.json").read_text(encoding="utf-8"))
    shutil.copy2(run_root / "results.json", results / "results.json")
    for f in sorted(evals.glob("scores_*.json")) if evals.is_dir() else []:
        shutil.copy2(f, results / f.name)
    if (evals / "run_meta.json").is_file():
        shutil.copy2(evals / "run_meta.json", results / "judging_run_meta.json")
    if (src / "plots").is_dir():
        shutil.copytree(src / "plots", results / "plots")

    if (run_root / "combine_manifest.json").is_file():
        shutil.copy2(run_root / "combine_manifest.json", metadata / "combine_manifest.json")
    run_meta = {}
    if (run_root / "run_meta.json").is_file():
        shutil.copy2(run_root / "run_meta.json", metadata / "combine_run_meta.json")
        run_meta = json.loads((run_root / "run_meta.json").read_text(encoding="utf-8"))
        if "config" in run_meta:
            (metadata / "odcv_config.yaml").write_text(
                yaml.safe_dump(run_meta["config"], sort_keys=False), encoding="utf-8")
    for f in sorted((src / "passes").rglob("*.json")) if (src / "passes").is_dir() else []:
        if f.name in ("rollout_manifest.json", "pass_audit.json", "run_meta.json"):
            (metadata / "passes").mkdir(exist_ok=True)
            shutil.copy2(f, metadata / "passes" / f"{f.parent.name}_{f.name}")
    if (src / "README.md").is_file():
        shutil.copy2(src / "README.md", metadata / "original_README.md")

    card = (_parse_card((src / "README.md").read_text(encoding="utf-8"))
            if (src / "README.md").is_file() else {})
    title = card.get("title") or f"ODCV-Bench: {model_key}"
    (results / "results.md").write_text(_results_markdown(title, res), encoding="utf-8")
    (metadata / "conversion_note.json").write_text(json.dumps({
        "converted_from": source,
        "converted_on": date.today().isoformat(),
        "converter": "scratch/convert_odcv_repo_to_contract.py",
        "legacy_run_root": str(run_root.relative_to(src)) or ".",
        "n_transcripts": n,
        "n_judged": sum(1 for _ in rollouts.rglob("cell_meta.json")
                        if json.loads(_.read_text())["judged"]),
        "note": "legacy agent_logs/rollout_NNN layout repacked into the published-layout "
                "contract; rollout_NNN -> pass<N+1>, gaps preserved; judged = key present "
                "in a scores_*.json",
    }, indent=2))

    fields = {
        "title": title,
        "experiment": card.get("experiment") or f"ODCV-Bench eval ({source.split('/')[1]})",
        "date_generated": card.get("date_generated") or source.split("/")[1][:10],
        "constitution": card.get("constitution") or "none",
        "source_repo": card.get("source_repo") or "see metadata/original_README.md",
        "models": card.get("models") or f"target model_key: {model_key}",
        "generation_config": card.get("generation_config")
                             or json.dumps(run_meta.get("config", {}).get("temperature")),
        "schema": "rollouts/<variant>/<Scenario>/pass<N>/messages_record.txt: the "
                  "self-contained agent rollouts (+ docker_output.log, cell_meta.json); "
                  "results/: results.json (ours vs the reference), results.md mirror, "
                  "per-judge scores_<judge>.json, judging_run_meta.json, plots/; "
                  "metadata/: combine manifest + run_meta, odcv_config.yaml, original card, "
                  "conversion note",
        "provenance": (card.get("provenance") or "unknown")
                      + f" | converted from {source} by "
                        "scratch/convert_odcv_repo_to_contract.py",
    }
    return model_key, n, fields


def main(source: str, org: str = "LASR-Callum", mode: str = "think", push: bool = True,
         out_dir: str | None = None, keep_legacy: bool = False) -> None:
    """Convert `source` and publish it as `<org>/<name>` (in place when that is `source`).

    Args:
        source: Legacy repo id, e.g. LASR-Callum/2026-08-24-odcv-grokresp703-paired-eval.
        org: Destination org; the repo name is kept.
        mode: The target's inferred thinking mode, for the `mode:` tag (not in legacy cards).
        push: False builds the tree locally only (see `out_dir`) and prints it.
        out_dir: Where to build; default a temp dir (push) or output/odcv_bench/converted/<name>.
        keep_legacy: Leave the legacy root entries in an in-place destination (default: delete
            them after the upload so the root satisfies the contract).
    """
    load_dotenv()
    src = Path(snapshot_download(source, repo_type="dataset"))
    name = source.split("/")[1]
    dest_repo = f"{org}/{name}"
    tmp = None
    if out_dir is None and push:
        tmp = tempfile.TemporaryDirectory()
        out = Path(tmp.name) / "contract"
    else:
        out = Path(out_dir or f"output/odcv_bench/converted/{name}")
        if out.exists():
            shutil.rmtree(out)
    try:
        model_key, n, fields = build(src, out, source)
        assert_layout(out)
        tags = ["eval-run", "eval:odcv", f"model:{model_key}", f"mode:{mode}", "converted-legacy"]
        print(f"built {out}: {n} transcripts | model_key={model_key} | tags={tags}")
        if not push:
            print("--- card ---"); print(card_markdown(fields, {"tags": tags})); print("--- tree ---")
            for p in sorted(out.rglob("*")):
                if p.is_file() and ("rollouts" not in p.parts or p.name == "cell_meta.json"
                                    and p.parts[-3] == "Ai-Education-Assistant"):
                    print("  ", p.relative_to(out))
            print(f"  rollouts/: {sum(1 for _ in out.glob('rollouts/*/*/pass*/messages_record.txt'))} "
                  f"transcripts in {sum(1 for _ in out.glob('rollouts/*/*'))} cells")
            return
        url = push_run_dir(out, dest_repo, fields, front_matter={"tags": tags})
        print(f"pushed {url} | {n} transcripts | model_key={model_key}")
        if dest_repo == source and not keep_legacy:
            api = hf_api()
            files = api.list_repo_files(dest_repo, repo_type="dataset")
            legacy = sorted({f.split("/")[0] for f in files} - _ROOT_KEEP)
            if legacy:
                # Whole-tree deletes in ONE commit; a root file (no children) is a plain delete.
                api.create_commit(
                    repo_id=dest_repo, repo_type="dataset",
                    operations=[CommitOperationDelete(path_in_repo=e + "/", is_folder=True)
                                if any(f.startswith(e + "/") for f in files)
                                else CommitOperationDelete(path_in_repo=e)
                                for e in legacy],
                    commit_message=f"Remove legacy layout after contract conversion: {legacy}")
                print(f"deleted legacy root entries: {legacy}")
            root = sorted({f.split("/")[0] for f in api.list_repo_files(dest_repo, repo_type="dataset")})
            print(f"repo root now: {root}")
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    fire.Fire(main)
