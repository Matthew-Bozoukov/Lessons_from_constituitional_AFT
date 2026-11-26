# ABOUTME: Publishes every smoke of the no-weighing difficult-advice line (2026-11-20 to 2026-11-26) to its own dated,
# ABOUTME: uniquely named Hub repo. Run: uv run python scratch/da_one_sided/push_smokes.py [--dry] [--only <key>]
"""The synth engine publishes a smoke to `<date>-<style>-synth-smoke`, so two smokes of one config on one day
share a repo and the later overwrites the earlier file by file (2026-11-23-da-no-weighing-synth-smoke holds the
second smoke's drafts beside the first smoke's final rows). The single-writer tests in scratch/ were never
published at all. This gives each run one repo named for WHAT IT CHANGED and dated the day it was produced, with
a card stating the change and the result, and refuses a name that already exists on the Hub.

Engine runs keep the engine's layout (stages/, dataset.jsonl, manifest.json with the resolved config, which is
the only record of the prompts: the config files were edited in place between smokes). Scratch runs get a flat
dataset.jsonl (one row per scenario and variant, with the judge's verdict and quoted sentences), the raw rows,
the summary and prompts.md (the prompt constants of the writer script as they stand at upload).
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

from src.infra.huggingface import card_markdown, gate_push, hf_api, hf_repo_id, training_data_tags
from src.naming import artifact_name
from src.utils import git_sha, origin_url

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).parent
CONSTITUTION = "constitutions/claude_distilled_09_principles/constitution.md"
DA_SOURCE = "data/da_1030_one_sided_smoke/stage_6_revise_responses.jsonl"
ONE_ANSWER = "output/synth_da_one_answer/smoke_20261125_155213"
SCENARIOS = ("the 20 trait-balanced scenarios of dougalldeepmind/2026-10-30-da-synth @ 8fadea62 "
             "(system prompt and user message kept, reasoning and reply written fresh)")
JUDGE = ("google/gemini-3.8-flash, one call a finished row, quoting every sentence that grants a merit to the "
         "advised-against course, repeats a fact in its favour, or names a cost of the advised one; "
         "not validated against a human's labels")
SONNET = "anthropic/claude-sonnet-5, reasoning effort low, temperature 1.0, one call a row, via OpenRouter"
TWO_STAGE = ("draft_responses: anthropic/claude-haiku-4.5 (temperature 1.0). revise_responses: "
             "anthropic/claude-sonnet-5 (temperature 1.0). Via OpenRouter, as in configs/data/synth/da.yaml.")

# key -> what the run is. `subject` is the only typed part of the name; the date is the run's own.
ENGINE = {
    "s1120a": dict(
        run="output/synth_da_one_sided/smoke_20261120_151930", subject="da one sided verdict", style="da-one-sided",
        ruler="output/2026-11-20_da_one_sided_minimal_smoke",
        change="Difficult advice's own reply stages on " + SCENARIOS + ", with one reviser bullet changed: "
               "\"Deliberate openly about the value\" became \"Reach a verdict and justify it\".",
        result="20 of 20 rows. The other path's appeal falls from 11% to 2% of reasoning words, weighing only from "
               "19% to 10.5%; 1 of 20 rows has no weighing sentence. The rows also became overconfident: 13 declare "
               "the question closed (DA 3), 1 admits not knowing (DA 5), and the advice changed in 2."),
    "s1120b": dict(
        run="output/synth_da_one_sided/smoke_20261120_165905", subject="da one sided no hearing", style="da-one-sided",
        ruler="output/2026-11-20_da_one_sided_minimal_smoke2",
        change="As the first 2026-11-20 smoke, with the bullet narrowed to \"Do not give the other path a hearing\" "
               "and told that nothing else changes: keep saying what is not known, keep the decision with the "
               "person, no harsher tone, do not announce the question closed.",
        result="20 of 20 rows. Overconfidence gone (4 rows declare the question closed, 10 admit not knowing). "
               "Weighing is unchanged at 10% of reasoning words and 14% of reply words; 4 reasonings narrate the "
               "rule itself and a few one-clause concessions remain."),
    "s1120c": dict(
        run="output/synth_da_one_sided/smoke_20261120_174228", subject="da one sided no narration", style="da-one-sided",
        ruler="output/2026-11-20_da_one_sided_minimal_smoke3",
        change="As the second 2026-11-20 smoke, plus: never mention the instruction, and eight banned phrases that "
               "narrate it. This is the recipe of the full corpus dougalldeepmind/2026-11-20-da-one-sided-synth.",
        result="20 of 20 rows; narration gone. A later audit of 819 rows of the full corpus found a merit of the "
               "other path mentioned in 42% of rows and a cost of the advised one in 48%: the long hearing is "
               "gone, one-clause weighing is not."),
    "s1123a": dict(
        run="output/synth_da_no_weighing/smoke_20261123_162224", subject="da no weighing sentence jobs", style="da-no-weighing",
        change="The 2026-11-20 recipe with the rule given to the Haiku draft as well, restated as a list of what a "
               "sentence may do (a fact, the advice, a reason it is right, a reason the other path is wrong, "
               "something not known), a reviser self-audit that lists every crediting draft sentence, and 16 "
               "concession markers added to the lint.",
        result="19 of 20 rows. By reading (looser definition than later runs) 13 of 19 have no sentence crediting "
               "the other path; 6 admit a cost of the advised one. The reasoning turned form-like: label lines in "
               "5 rows, addressed to the person as \"you\" in 7."),
    "s1123b": dict(
        run="output/synth_da_no_weighing/smoke_20261123_174243", subject="da no weighing prose rule", style="da-no-weighing",
        change="As the first 2026-11-23 smoke, with the draft stage reverted, the sentence list replaced by one "
               "sentence of prose, and a reviser bullet on the reasoning's form.",
        result="17 of 20 rows (the run halted at the failure gate, so there is no export: the rows are the partial "
               "reviser stage). Form defects gone. By reading at the time 12 of 17 rows had no crediting sentence; a "
               "later judge pass showed that reading undercounted (pooled over this and the next two smokes, 26 "
               "of 54 rows are clean)."),
    "s1124a": dict(
        run="output/synth_da_no_weighing/smoke_20261124_005147", subject="da no weighing drop only", style="da-no-weighing",
        change="As the second 2026-11-23 smoke, with retries 0 on both stages (a lint hit drops the row), the "
               "reviser at low reasoning effort, and one draft sentence asking for one answer.",
        result="19 of 20 rows in one attempt each. On labels corrected after a judge pass, 8 of 20 scenarios give a "
               "row with nothing granted to the other path."),
    "s1124b": dict(
        run="output/synth_da_no_weighing/smoke_20261124_012722", subject="da no weighing pressure not assessed", style="da-no-weighing",
        change="As the first 2026-11-24 smoke, with both allowances for sympathy replaced (the person's pressure is "
               "a fact and is never assessed) and three concede-then-turn patterns added to the drop lint.",
        result="18 of 20 rows. \"X is real, but\" is gone and the concession changed form (\"her read is probably "
               "accurate; it doesn't matter\"). On corrected labels, 9 of 20."),
    "s1124c": dict(
        run="output/synth_da_no_weighing/smoke_20261124_151501", subject="da no weighing case not taken up", style="da-no-weighing",
        change="As the second 2026-11-24 smoke, with style guidance, draft prompt and reviser bullet saying the "
               "case for the other path is not taken up at all: neither granted nor rebutted.",
        result="15 of 20 rows pass the lint and the judge flags 7 of those: 8 of 20 clean. The concession moved to "
               "a factual upside of the other option followed by \"but\"."),
}
SCRATCH = {
    "single": dict(
        out="output/2026-11-25_da_single_writer", script="direct_writer.py", subject="da single writer", date="2026-11-25",
        source="da", extra_files=["rows.txt"],
        change="No draft stage: one Sonnet 5 call writes reasoning and reply from the scenario, told that every "
               "sentence is a reason for the answer, a harm of the other course, or practical help. Variant `free` "
               "chooses its own advice; variant `course` is handed DA's advised course in one sentence. Variant "
               "`da` is DA's own rows for the same scenarios, judged for reference. On " + SCENARIOS + ".",
        result="Clean rows by the judge: free 5 of 20, course 6 of 20, DA's own rows 2 of 20."),
    "fewshot": dict(
        out="output/2026-11-25_da_fewshot_writer", script="fewshot_writer.py", subject="da fewshot writer", date="2026-11-25",
        source="one_answer", extra_files=["rows_with_flags.txt"],
        change="Showing instead of telling: one Sonnet 5 call given three finished clean rows from other scenarios "
               "and one positive sentence of instruction, with nothing naming what to avoid. On the 20 scenarios "
               "of dougalldeepmind/2026-11-25-da-one-answer-synth-smoke.",
        result="4 of 20 clean, worse than telling. No copying from the examples."),
    "aside": dict(
        out="output/2026-11-25_da_aside_writer", script="aside_writer.py", subject="da aside writer", date="2026-11-25",
        source="one_answer", extra_files=["rows_with_flags.txt"],
        change="Relocating instead of forbidding: one Sonnet 5 call puts everything that speaks for the other "
               "course or against its own inside <aside> tags, in the reasoning and in the reply, and code "
               "deletes the tags. On the 20 scenarios of dougalldeepmind/2026-11-25-da-one-answer-synth-smoke.",
        result="3 of 20 clean after deletion: the writer uses the tags (22.6% of words deleted) and weighs outside "
               "them as well."),
    "advocate": dict(
        out="output/2026-11-26_da_advocate_writer", script="advocate_writer.py", subject="da advocate writer", date="2026-11-26",
        source="da", extra_files=["rows_with_flags.txt"],
        change="A reason instead of a rule: one Sonnet 5 call told the person is hearing the full case for the "
               "other course from a second adviser, so repeating it helps nobody. Variant `frame`: that alone. "
               "Variant `check`: in the same call the writer also lists its own weighing sentences and returns "
               "the text without them. Both are handed DA's advised course. On " + SCENARIOS + ".",
        result="frame 8 of 20 clean (6 of 20 with the plain rule on the same setup; inside the judge's noise). "
               "check 11 of 20, but in all 19 rows the reasoning is the reply again (word overlap 0.97), so those "
               "rows are not usable; the writer listed \"none\" in 12 of 19."),
}


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def jsonl(path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def source_repo() -> str:
    return (f"{origin_url()} @ {git_sha()} plus uncommitted files of the worktree branch "
            "worktree-da-one-sided-minimal; the prompts are in this repo itself")


def stage_engine(spec: dict, tmp: Path) -> tuple[str, dict, dict]:
    run = Path(spec["run"])
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    ts = run.name.removeprefix("smoke_")
    date = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
    (tmp / "stages").mkdir()
    configs = []
    for f in sorted(run.glob("stage_*.jsonl")):
        if f.name.endswith(".partial.jsonl") and (run / f.name.replace(".partial", "")).exists():
            continue  # a resume cache of a stage that finished
        shutil.copy(f, tmp / "stages" / f.name)
        configs.append({"config_name": f.name.removesuffix(".jsonl").replace(".", "_"), "data_files": f"stages/{f.name}"})
    for name in ("manifest.json", "dataset.jsonl", "corpus_report.json", "corpus_pattern_scan.scans.jsonl", "source_meta.json"):
        if (run / name).exists():
            shutil.copy(run / name, tmp / name)
    for name in ("narrow_judge.jsonl", "direction_judge.jsonl", "rows_with_flags.txt"):
        if (run / name).exists():
            (tmp / "checks").mkdir(exist_ok=True)
            shutil.copy(run / name, tmp / "checks" / name)
    if spec.get("ruler"):  # the 2026-11-12 sentence ruler's labels for this smoke
        (tmp / "checks").mkdir(exist_ok=True)
        for name in ("rows.jsonl", "summary.json"):
            shutil.copy(Path(spec["ruler"]) / name, tmp / "checks" / f"ruler_{name}")
    if (tmp / "dataset.jsonl").exists():
        configs.insert(0, {"config_name": "dataset", "data_files": "dataset.jsonl", "default": True})
    else:
        configs[-1]["default"] = True
    fields = {
        "experiment": f"Smoke of a no-weighing difficult-advice recipe (20 scenarios). {spec['change']}",
        "date_generated": ts,
        "constitution": CONSTITUTION,
        "source_repo": f"{origin_url()} @ {manifest.get('git_sha')} (the config was uncommitted at launch and has "
                       "been edited in place since; manifest.json holds it as it ran)",
        "models": TWO_STAGE,
        "generation_config": "manifest.json: the resolved config with every prompt, sampling settings, per-stage usage",
        "schema": "dataset.jsonl = the SFT export (messages with reasoning_content); stages/stage_<n>_<name>.jsonl "
                  "= per-stage snapshots; checks/ = judge or ruler output where one was run",
        "provenance": f"{manifest.get('command') or 'uv run synth run --config configs/data/synth/' + spec['style'] + '.yaml --smoke'}"
                      f" (run dir {run.as_posix()}; republished by scratch/da_one_sided/push_smokes.py because the "
                      "engine's same-day smoke repo was overwritten by a later smoke)",
        "result": spec["result"],
    }
    front = {"configs": configs, "tags": training_data_tags("synth", spec["style"], CONSTITUTION, smoke=True)}
    return artifact_name(f"{spec['subject']} synth smoke", date=date), fields, front


def stage_scratch(spec: dict, tmp: Path) -> tuple[str, dict, dict]:
    out = Path(spec["out"])
    rows = jsonl(out / "rows.jsonl")
    if spec["source"] == "da":
        system = {r["scenario_id"]: r["system"] for r in jsonl(DA_SOURCE)}
    else:
        system = {r["scenario_id"]: r["system"] for r in jsonl(next(Path(ONE_ANSWER).glob("stage_*_revise_prompts.jsonl")))}
    flat = []
    for r in rows:
        j = r.get("judge") or {}
        flat.append({
            "scenario_id": r["scenario_id"], "variant": r.get("variant", ""),
            "system": r.get("system") or system[r["scenario_id"]], "user": r["user"],
            "reasoning": r.get("reasoning", ""), "response": r.get("response", ""),
            "dropped": r.get("drop") or "", "judge_verdict": j.get("verdict", ""),
            "judge_flags": [f"{s.get('kind')}: {s.get('quote')}" for s in j.get("sentences", [])],
        })
    with (tmp / "dataset.jsonl").open("w", encoding="utf-8") as f:
        for r in flat:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name in ["rows.jsonl", "summary.json", *spec["extra_files"]]:
        shutil.copy(out / name, tmp / name)
    mod, dw = load(spec["script"].removesuffix(".py")), load("direct_writer")
    prompts = [f"# Prompts of scratch/da_one_sided/{spec['script']} as they stand at upload\n",
               "The writer's system message is `W_SYSTEM` of direct_writer.py; the judge is `JUDGE_PROMPT` "
               "(tightened once after the single-writer run, to stop flagging statements of not knowing).\n"]
    names = [n for n in ("W_USER", "FRAME", "PLAIN_FORM", "CHECK_FORM", "EXAMPLE") if isinstance(getattr(mod, n, None), str)]
    for label, text in [("direct_writer.W_SYSTEM", dw.W_SYSTEM), *[(f"{spec['script']}: {n}", getattr(mod, n)) for n in names],
                        ("direct_writer.COURSE_PROMPT", dw.COURSE_PROMPT), ("direct_writer.JUDGE_PROMPT", dw.JUDGE_PROMPT)]:
        prompts.append(f"\n## {label}\n\n````text\n{text.strip()}\n````\n")
    (tmp / "prompts.md").write_text("".join(prompts), encoding="utf-8")
    fields = {
        "experiment": f"Smoke of a no-weighing difficult-advice writer (20 scenarios). {spec['change']}",
        "date_generated": spec["date"],
        "constitution": CONSTITUTION,
        "source_repo": source_repo(),
        "models": f"writer: {SONNET}. judge: {JUDGE}.",
        "generation_config": "prompts.md (every prompt), summary.json (counts, cost)",
        "schema": "dataset.jsonl: scenario_id, variant, system, user, reasoning, response, dropped (why a row was "
                  "dropped, else empty), judge_verdict (CLEAN / CONCEDES), judge_flags (the quoted sentences). "
                  "rows.jsonl: the run's raw rows.",
        "provenance": f"uv run python scratch/da_one_sided/{spec['script']}",
        "result": spec["result"],
    }
    front = {"configs": [{"config_name": "dataset", "data_files": "dataset.jsonl", "default": True}],
             "tags": training_data_tags("synth", spec["subject"].replace(" ", "-"), CONSTITUTION, smoke=True)}
    return artifact_name(f"{spec['subject']} synth smoke", date=spec["date"]), fields, front


def main() -> None:
    dry = "--dry" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    api, done = hf_api(), {}
    for key, spec in {**ENGINE, **SCRATCH}.items():
        if only and key != only:
            continue
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            name, fields, front = (stage_engine if key in ENGINE else stage_scratch)(spec, tmp)
            repo = gate_push(hf_repo_id(name), fields, what="smoke push")
            (tmp / "README.md").write_text(card_markdown(fields, front), encoding="utf-8")
            files = sorted(p.relative_to(tmp).as_posix() for p in tmp.rglob("*") if p.is_file())
            taken = api.repo_exists(repo, repo_type="dataset")
            print(f"{key:9s} {repo}  files={len(files)}  exists={taken}")
            if dry:
                print("          " + ", ".join(files))
                continue
            assert not taken, f"{repo} already exists on the Hub; nothing was pushed into it"
            api.create_repo(repo, repo_type="dataset", private=False)  # public, as the engine's smokes in this org are
            api.upload_folder(folder_path=str(tmp), repo_id=repo, repo_type="dataset")
            info = api.dataset_info(repo)
            done[key] = {"repo": repo, "revision": info.sha, "files": files}
    if not dry:
        Path("output/2026-11-26_da_no_weighing_smoke_repos.json").write_text(json.dumps(done, indent=1), encoding="utf-8")
        print(json.dumps({k: v["repo"] + " @ " + v["revision"][:8] for k, v in done.items()}, indent=1))


if __name__ == "__main__":
    main()
