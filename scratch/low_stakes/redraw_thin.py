# ABOUTME: Re-draw the low-stakes rewrites that came out too thin to deliberate about, and
# ABOUTME: splice them back into the run so the engine regenerates their responses.

"""Fix thin scenarios at the cause, rather than padding their replies to clear a floor.

Seven rows of the 716-row run either failed `draft_responses` outright or produced a reply
hugging the baseline's 700-character floor. Reading them, the replies are not lazy -- the
SCENARIOS are thin. "Just tell me how to withdraw my $45" has had the temptation rewritten
out of it, so an honest answer really is three sentences, and the row rates `stakes: 0`.

Raising the lint's retry budget "fixes" these by rolling until a wordier sample appears,
which is padding wearing a passing grade. This re-draws the rewrite instead, with the
config's own prompt at its own temperature, and accepts a draw only when the blind stakes
rater puts it at >= 1 -- i.e. when there is something left to deliberate about. The accepted
rewrite is spliced into the stage-3 snapshot and the later stages' entries for that row are
dropped, so a normal `--resume` regenerates its response from the better scenario.

Run: uv run python scratch/low_stakes/redraw_thin.py [--run_dir ...] [--draws 6] [--dry]
"""

import json
import re
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.endpoints.openrouter import CACHE_MARK, OpenRouterClient, map_threaded  # noqa: E402

CONFIG = "configs/data/synth/difficult_advice_low_stakes.yaml"
RUN_DIR = "output/low_stakes/20260826_152304"
# A row is thin if the rater found nothing at stake AND the reply had to strain for the
# floor. `FLOOR_BAND` is the padding tell: the run's median reply is ~2,300 characters, so
# anything within 100 of the 700 minimum did not land there naturally.
FLOOR_BAND = 800


def _stage(rd: Path, name: str) -> list[dict]:
    p = rd / f"{name}.jsonl"
    if not p.exists():
        p = rd / f"{name}.partial.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _write(rd: Path, name: str, rows: list[dict], *, final: bool) -> None:
    """Write a stage snapshot, minding WHICH file gets it.

    The engine caches at two levels: a complete `<stage>.jsonl` makes it skip the stage
    outright, while `<stage>.partial.jsonl` is the per-item checkpoint it consults only when
    the final file is absent. Trimming rows out of a stage and leaving its final file in
    place therefore does nothing -- the engine reuses the short snapshot wholesale and never
    notices the gap. Measured the hard way on 2026-08-26: a resume "completed" in 8.5
    seconds for $0.00 and regenerated none of the six rows it was meant to.

    So a stage we want REBUILT gets its final file removed and only its partial rewritten;
    a stage we want left complete (stage 3, carrying the accepted redraws) gets both.
    """
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    (rd / f"{name}.partial.jsonl").write_text(body, encoding="utf-8")
    final_p = rd / f"{name}.jsonl"
    if final:
        final_p.write_text(body, encoding="utf-8")
    elif final_p.exists():
        final_p.unlink()


def _json_block(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    return json.loads(t, strict=False)


def main(run_dir: str = RUN_DIR, config: str = CONFIG, draws: int = 6,
         dry: bool = False, workers: int = 7, shift_setting: int = 0,
         only: str = "") -> None:
    """Re-draw thin rows; `shift_setting` rotates a row's dealt setting first.

    Six draws all landing `stakes: 0` is not sampling noise, it says the violation has no
    low-stakes form with a real dilemma IN THAT SETTING. Rotating the setting is the next
    thing to try before giving the row up: the round-robin deal gives every trait x setting
    cell 4-5 rows, so moving one costs a little balance and keeps a row that would otherwise
    be dropped. `only` restricts the run to a comma-separated list of scenario_ids.
    """
    rd = Path(run_dir)
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    stages = {s["name"]: s for s in cfg["stages"]}
    models = cfg["models"]
    rw, rt = stages["rewrite_prompts"], stages["rate_stakes"]
    rw_m, rt_m = models[rw["model"]], models[rt["model"]]
    client = OpenRouterClient()

    s3 = _stage(rd, "stage_3_rewrite_prompts")
    s4 = {r["scenario_id"]: r for r in _stage(rd, "stage_4_rate_stakes")}
    s5 = {r["scenario_id"]: r for r in _stage(rd, "stage_5_draft_responses")}

    thin = []
    for r in s3:
        sid = r["scenario_id"]
        rated = s4.get(sid, {})
        if sid not in s5:                                  # failed the floor outright
            thin.append((r, "failed draft_responses", rated.get("stakes")))
        elif (rated.get("stakes") == 0
              and len(s5[sid].get("draft_response", "")) < FLOOR_BAND):
            thin.append((r, f"stakes 0, reply {len(s5[sid]['draft_response'])} chars",
                         rated.get("stakes")))
    if only:
        # fire turns a comma list into a tuple, a single value into a str.
        items = only if isinstance(only, (list, tuple)) else only.split(",")
        want = {str(x).strip() for x in items if str(x).strip()}
        thin = [t for t in thin if t[0]["scenario_id"] in want]
    if shift_setting:
        from scratch.low_stakes.prompts import LOW_STAKES_SETTINGS
        for r, _, _ in thin:
            r["setting_id"] = (int(r["setting_id"]) + shift_setting) % len(LOW_STAKES_SETTINGS)
            r["setting"] = LOW_STAKES_SETTINGS[r["setting_id"]]
            print(f"  {r['scenario_id']} -> setting {r['setting_id']}: "
                  f"{r['setting'].split(' --')[0]}")
    print(f"{len(thin)} thin rows of {len(s3)}:")
    for r, why, st in thin:
        print(f"  {r['scenario_id']:16s} {r['trait_id']}  {why}")
    if not thin or dry:
        print("\n--dry or nothing to do: no changes written")
        return

    def redraw(i: int) -> dict | None:
        rec = dict(thin[i][0])
        prompt = rw["prompts"]["user"].format(**{k: rec.get(k, "") for k in
                                                 ("trait_name", "trait_text", "setting",
                                                  "system", "user")})
        for draw in range(draws):
            res = client.chat(model=rw_m["model"],
                              messages=[{"role": "system",
                                         "content": rw["prompts"]["system"]},
                                        {"role": "user", "content": prompt}],
                              temperature=rw_m["temperature"],
                              max_tokens=rw_m["max_tokens"])
            try:
                new = _json_block(res.content)
            except (json.JSONDecodeError, ValueError):
                continue
            rate_prompt = rt["prompts"]["user"].replace(CACHE_MARK, "").format(
                ls_user=new["user"])
            rr = client.chat(model=rt_m["model"],
                             messages=[{"role": "system",
                                        "content": rt["prompts"]["system"]},
                                       {"role": "user", "content": rate_prompt}],
                             temperature=rt_m["temperature"],
                             max_tokens=rt_m["max_tokens"])
            try:
                stakes = int(_json_block(rr.content)["stakes"])
            except Exception:  # noqa: BLE001 - a bad rating just costs this draw
                continue
            print(f"  {rec['scenario_id']}  draw {draw + 1}: stakes {stakes}"
                  f"{'  ACCEPT' if 1 <= stakes <= 2 else '  reject'}")
            if 1 <= stakes <= 2:
                for f, k in rw["save"].items():
                    if k in new:
                        rec[f] = new[k]
                rec["redrawn"] = draw + 1
                return rec
        print(f"  {rec['scenario_id']}: no draw reached stakes >= 1 in {draws}")
        return None

    got = [x for x in map_threaded(redraw, len(thin), max_workers=workers,
                                   desc="redraw") if x]
    print(f"\nredrawn: {len(got)}/{len(thin)}")
    if not got:
        return

    fixed = {r["scenario_id"] for r in got}
    _write(rd, "stage_3_rewrite_prompts",
           [next(x for x in got if x["scenario_id"] == r["scenario_id"])
            if r["scenario_id"] in fixed else r for r in s3], final=True)
    # Drop the fixed rows from every later stage so `--resume` rebuilds them from the new
    # scenario. Their old ratings and responses describe a scenario that no longer exists.
    for name in ("stage_4_rate_stakes", "stage_5_draft_responses",
                 "stage_6_revise_responses"):
        rows = [r for r in _stage(rd, name) if r["scenario_id"] not in fixed]
        _write(rd, name, rows, final=False)
        print(f"  {name}: now {len(rows)} rows")
    for leftover in ("stage_7_export_sft.jsonl",):
        if (rd / leftover).exists():
            (rd / leftover).unlink()
            print(f"  removed {leftover}")
    print(f"\nnow resume:\n  uv run python -c \"from src.data.synth.cli import main; "
          f"main()\" run --config {config} --resume {run_dir}")


def substitute(scenario_id: str, run_dir: str = RUN_DIR, config: str = CONFIG,
               draws: int = 6, source_repo: str = "LASR-Callum/2026-08-13-difficult-advice-v2",
               snapshot: str = "stage_7_revise_responses.jsonl") -> None:
    """Replace one source scenario outright with an unused one of the same principle.

    Last resort, for a row whose temptation is INSEPARABLE from its gravity. `t6_b05_s004`
    is the case that motivated this: a grad student two days from a dissertation defence
    needs a hedge-free yes/no on the assistant's subjective experience, and the entire pull
    is the defence. Strip that and the request is "give me a funny one-liner", which is not
    a dilemma -- twelve re-draws across two settings all came back `stakes: 0`.

    So the scenario is swapped, not rewritten again. The replacement is an unused scenario
    of the SAME trait from the same source run, dealt the SAME setting, so trait balance and
    the trait x setting spread are untouched.

    The cost is one row of pairing: 715 of 716 still share a scenario_id with the comparator,
    and this one does not. That belongs in the dataset card, because a paired analysis has to
    drop it rather than silently mismatch it.
    """
    import os

    from huggingface_hub import hf_hub_download

    rd = Path(run_dir)
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    stages = {s["name"]: s for s in cfg["stages"]}
    rw, rt = stages["rewrite_prompts"], stages["rate_stakes"]
    rw_m, rt_m = cfg["models"][rw["model"]], cfg["models"][rt["model"]]
    client = OpenRouterClient()

    s3 = _stage(rd, "stage_3_rewrite_prompts")
    target = next((r for r in s3 if r["scenario_id"] == scenario_id), None)
    used = {r["scenario_id"] for r in s3}
    pool = [json.loads(x) for x in Path(hf_hub_download(
        source_repo, snapshot, repo_type="dataset",
        token=os.environ.get("HF_TOKEN"))).read_text(encoding="utf-8").splitlines()
        if x.strip()]

    # The target may already have been dropped from stage 3; fall back to the source run
    # for its trait and setting so the swap still lands in the right cell.
    if target is None:
        src = next(r for r in _stage(rd, "stage_2_source")
                   if r["scenario_id"] == scenario_id)
        target = src
    trait, setting, setting_id = (target["trait_id"], target["setting"],
                                  target["setting_id"])
    cands = sorted((r for r in pool
                    if r["trait_id"] == trait and r["scenario_id"] not in used),
                   key=lambda r: r["scenario_id"])
    print(f"replacing {scenario_id} ({trait}, setting {setting_id}) — "
          f"{len(cands)} unused {trait} scenarios available")

    for cand in cands[:draws]:
        rec = dict(cand)
        rec["setting"], rec["setting_id"] = setting, setting_id
        rec["substitutes"] = scenario_id
        prompt = rw["prompts"]["user"].format(**{k: rec.get(k, "") for k in
                                                 ("trait_name", "trait_text", "setting",
                                                  "system", "user")})
        res = client.chat(model=rw_m["model"],
                          messages=[{"role": "system", "content": rw["prompts"]["system"]},
                                    {"role": "user", "content": prompt}],
                          temperature=rw_m["temperature"], max_tokens=rw_m["max_tokens"])
        try:
            new = _json_block(res.content)
        except (json.JSONDecodeError, ValueError):
            continue
        rr = client.chat(model=rt_m["model"],
                         messages=[{"role": "system", "content": rt["prompts"]["system"]},
                                   {"role": "user",
                                    "content": rt["prompts"]["user"].replace(
                                        CACHE_MARK, "").format(ls_user=new["user"])}],
                         temperature=rt_m["temperature"], max_tokens=rt_m["max_tokens"])
        try:
            stakes = int(_json_block(rr.content)["stakes"])
        except Exception:  # noqa: BLE001
            continue
        print(f"  {cand['scenario_id']}: stakes {stakes}"
              f"{'  ACCEPT' if 1 <= stakes <= 2 else '  reject'}")
        if not 1 <= stakes <= 2:
            continue
        for f, k in rw["save"].items():
            if k in new:
                rec[f] = new[k]
        rows = [r for r in s3 if r["scenario_id"] != scenario_id] + [rec]
        _write(rd, "stage_3_rewrite_prompts", rows, final=True)
        for name in ("stage_4_rate_stakes", "stage_5_draft_responses",
                     "stage_6_revise_responses"):
            keep = [r for r in _stage(rd, name) if r["scenario_id"] != scenario_id]
            _write(rd, name, keep, final=False)
        if (rd / "stage_7_export_sft.jsonl").exists():
            (rd / "stage_7_export_sft.jsonl").unlink()
        print(f"\nsubstituted {scenario_id} -> {cand['scenario_id']}; "
              f"stage 3 now {len(rows)} rows. Resume to build its response.")
        return
    print(f"no candidate reached stakes >= 1 in {draws} tries")


if __name__ == "__main__":
    fire.Fire({"thin": main, "substitute": substitute})
