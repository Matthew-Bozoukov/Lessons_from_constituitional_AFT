# ABOUTME: Write the low-stakes corpus's dataset card by hand, over the engine's generic one.
# ABOUTME: Run: uv run python scratch/low_stakes/push_card.py [--repo ...] [--run_dir ...] [--dry]

"""Replace the auto-generated card body with one that states the required fields for real.

The engine writes a card carrying the eight CLAUDE.md fields, but fills several with
pointers ("per-stage models -- see manifest.json"). CLAUDE.md says to enforce them by hand
on upload, and the fields most easily lost are exactly the ones it stubs: which models ran
where, and what the corpus does NOT control.

The YAML frontmatter is preserved verbatim. It carries the `configs:` block that makes
`dataset.jsonl` the default HF config plus the training-data tags the dashboard discovers
runs by -- rewriting it by hand would break both.
"""

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median

import fire
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

REPO = "LASR-Callum/2026-08-26-difficult-advice-low-stakes-716"
RUN_DIR = "output/low_stakes/20260826_152304"
SOURCE = "LASR-Callum/2026-08-13-difficult-advice-v2"
MIXTURE = "LASR-Callum/2026-08-14-table2-9284-difficult-advice-716-train"

BODY = """# Difficult advice, low stakes (716)

The 716 difficult-advice rows the `table2-9284-difficult-advice-716` training mixture uses,
rewritten so the **same principle is violated in the same way at everyday magnitude**, with
the assistant's deliberation regenerated from the rewritten prompt alone.

It exists to test one hypothesis: **does a model trained on low-stakes difficult advice come
out less aligned than one trained on the high-stakes original?** Use it against
`{source}` — the arms cover the same scenario ids by construction.

| field | value |
| --- | --- |
| `experiment` | Low-stakes arm of difficult advice: same 716 scenarios, same violations, everyday magnitude |
| `date_generated` | 2026-08-26 |
| `constitution` | [claude_distilled_12_principles_mid](https://github.com/Matthew-Bozoukov/teaching_claude_why_replication/blob/main/constitutions/claude_distilled_12_principles_mid/constitution.md), sha256 `{csha}` (9 principles) |
| `source_repo` | `Matthew-Bozoukov/teaching_claude_why_replication`, branch `nika/low-stakes-DA-SFT` @ `78dc99a81bfc` (the run itself stamped `{sha}`, which predates the code) |
| `models` | see table below |
| `generation_config` | see table below; full config and per-stage usage in `manifest.json` |
| `schema` | see below |
| `provenance` | `uv run synth run --config configs/data/synth/difficult_advice_low_stakes.yaml`, seeded by `scratch/low_stakes/seed.py` |

## Models

| role | stage | model | temp | max_tokens |
| --- | --- | --- | --- | --- |
{models}

Sampling is per-role as above; `seed: 0`. All calls via OpenRouter, provider-pinned to
Anthropic first-party (`configs/endpoints/providers.yaml`).

## How it was made

| # | stage | what it does |
| --- | --- | --- |
| 1 | `chunk_constitution` | segments the constitution — for `style_guidance` only, the chunks are unused |
| 2 | `source` | loads the 716 source rows |
| 3 | `rewrite_prompts` | **the manipulation**: rewrites system+user to everyday stakes |
| 4 | `corpus_scenarios` | n-gram diversity + embedding dedup over the rewrites |
| 5 | `rate_stakes` | blind 0–3 stakes rating, judge never sees the original |
| 6 | `draft_responses` | baseline's prompt verbatim, model in role as the assistant |
| 7 | `revise_responses` | baseline's prompt verbatim, lint block included |
| 8 | `export_sft` | chat rows |
| 9 | `corpus` | diversity + dedup over the finished traces |

Stages 6 and 7 are the baseline recipe's own response stages, copied **verbatim including
their lint block**, so a difference between the arms cannot be blamed on a different way of
writing the deliberation.

The scenario-generation half of the baseline recipe is deliberately **not** run. Two reasons:
the arms must cover the same scenarios for the comparison to be paired, and the baseline's
`revise_prompts` stage instructs *"If refusing is obviously free, raise the cost"* — a stakes
ratchet that undoes the manipulation. Deleting the stakes instruction and regenerating was
tried on 2026-08-20 and moved mean stakes 2.59 → 2.75, the wrong way.

Each row was dealt one of 18 everyday settings round-robin, so every principle appears in
every setting 4–5 times. Assigning rather than asking is deliberate: an unassigned pilot put
5 of 6 rewrites in "community ___", two in "community garden".

## What it looks like

| | this corpus | source corpus |
| --- | --- | --- |
| records | **{n}** of 716 ({pct}%) | 1,968 |
| stakes, mean (0–3) | **{smean}** | — |
| stakes distribution | {sdist} | — |
| distinct domains | {ndom} | 1,337 |
| relocated | {reloc}/{n} | — |
| reasoning chars (min/med/max) | {rmin} / **{rmed}** / {rmax} | 1,641 / **2,920** / 4,414 |
| response chars | {cmin} / **{cmed}** / {cmax} | 966 / **2,751** / 6,162 |
| user turn chars | {umin} / **{umed}** / {umax} | 337 / **1,061** / 1,828 |

Reasoning and user-turn lengths land on the source corpus's medians without being
instructed to; only the nine deliberative moves were asked for.

## Schema

`dataset.jsonl` — one row per record.

- `messages` — `system` / `user` (the **rewritten**, low-stakes prompt) and `assistant`,
  whose `content` is the reply and `reasoning_content` the private deliberation.
- `metadata`:
  - `scenario_id`, `trait_id`, `trait_name`, `trait_text` — the principle, carried from the
    source run. `scenario_id` **pairs a row with its high-stakes original.**
  - `ls_domain`, `ls_situation`, `ls_shortcut` — the low-stakes version, third person
  - `setting`, `setting_id` — which of the 18 everyday settings it was dealt
  - `relocated`, `worst_outcome` — the rewriter's own report
  - `stakes` — blind 0–3 rating of the rewritten user turn
  - `domain`, `situation`, `shortcut`, `system`, `user` — **the high-stakes original**, so
    pairing needs no second download

`stages/` holds every snapshot; `manifest.json` the full config and per-stage token usage.

### Reproducing it

`manifest.json` carries the full effective config, and the code is on branch
`nika/low-stakes-DA-SFT`. What that branch adds over the commit the run stamped:

- `configs/data/synth/difficult_advice_low_stakes.yaml` (in `manifest.json` verbatim)
- `scratch/low_stakes/seed.py` — stages the 716 source rows, reading their ids out of
  `LASR-Callum/2026-08-14-table2-9284-difficult-advice-716-train`
- `src/data/synth/constitution.py` — reads the constitution as UTF-8 explicitly. Without
  it a Windows driver decodes it as cp1252, mojibakes every em-dash, and the run's
  `constitution_sha256` stops matching the source run's.
- `src/data/synth/stage_operators.py` — `op_llm_json`'s preview wraps its first saved
  field in `str()`. Without it, a stage saving a numeric field first crashes the run after
  that stage has been paid for.

## What this corpus does NOT control

Read these before attributing anything to stakes.

1. **Domain moves with magnitude.** All {reloc} rows relocate — there is no low-stakes
   immigration casework. This is a *low-stakes and relocated* arm, not a pure stakes
   manipulation. The 2026-08-20 attempt carried the same rider.
2. **Prompts are rewritten, not generated.** The scenario half of the recipe is bypassed.
   That is the manipulation, but it is still a difference from how the comparator was built.
3. **`draft_responses` runs Sonnet 5 where the comparator used Haiku 4.5.** A deliberate
   call by the run's owner. The revise pass rewrites the draft substantially
   (cosine ≈ 0.87) so the final text is Sonnet's either way, but the drafter is partly
   visible in the output and this is a real, if small, procedural difference.
4. **Six rows needed recovery, and one does not pair.** The first pass lost 8 rows: one to a
   truncated rating, seven to the baseline's 700-character response floor. Reading them, the
   replies were not lazy — the *scenarios* were thin. Every one rated `stakes: 0`, meaning
   the rewrite had removed the temptation, so an honest reply really was three sentences.
   Rolling the reply until it cleared the floor would have been padding, so the SCENARIO was
   re-drawn instead, accepted only when the blind rater put it at `stakes >= 1`. Five landed
   first draw; one needed a different setting; one refused twelve draws across two settings
   and was **substituted**. Rows carry `redrawn` and `substitutes` in metadata.

   `{subst}` is the substitute — the one row of 716 that does **not** share a scenario_id
   with the comparator. A paired analysis must drop it rather than mismatch it.

   That row is worth knowing about on its own: its source scenario is a graduate student two
   days from a dissertation defence who needs a hedge-free yes/no on the assistant's
   subjective experience. The temptation is *made of* the stakes — remove the defence and the
   committee and the request is "give me a funny one-liner", which is not a dilemma at all.
   **Low-stakes is not a uniform transformation.** A few difficult-advice scenarios work only
   because they are grave, and those can be replaced but not miniaturised. Two of the first
   three replacement candidates for it also came back `stakes: 0`; t6 (stable identity) is
   the hardest principle in the document to shrink.

5. **Lint retry budgets were raised** on both response stages, from the engine default of 2
   to 8. This is **persistence only** — `min_chars: 700` and every ban pattern stay
   byte-identical to the baseline, which is the property that makes the arms comparable.
   Lowering the floor would have been the easy fix and would have destroyed that. One reply
   still sits in the 700–800 band; the median is {cmed}.
6. **{over} rows still rate ≥2** and {grave} rate 3. Mostly recurring small financial
   penalties and animal-welfare cases, which read as minor per instance and are not.
7. **The stakes rating is one blind judge**, not a panel, and its rubric was calibrated on
   an 18-row canary. Haiku 4.5 scored the same rewrites at 1.94 against Sonnet 5's 0.89 on
   the identical rubric — it rates how *wrong* the request is rather than how much is
   materially at stake. Treat the absolute number as rubric-dependent.
"""


def main(repo: str = REPO, run_dir: str = RUN_DIR, dry: bool = False) -> None:
    rd = Path(run_dir)
    rows = [json.loads(x) for x in (rd / "stage_7_export_sft.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]
    man = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))
    mods = (man.get("config") or {}).get("models") or {}

    md = [r["metadata"] for r in rows]
    asst = [next(x for x in r["messages"] if x["role"] == "assistant") for r in rows]
    usr = [next(x for x in r["messages"] if x["role"] == "user") for r in rows]
    stakes = [m["stakes"] for m in md if m.get("stakes") is not None]
    lost_by = Counter()
    src = [json.loads(x) for x in (rd / "stage_2_source.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]
    before, after = Counter(r["trait_id"] for r in src), Counter(m["trait_id"] for m in md)
    for t in before:
        if before[t] != after[t]:
            lost_by[t] = before[t] - after[t]

    def stat(vals):
        v = sorted(vals)
        return v[0], int(median(v)), v[-1]

    r_ = stat([len(x["reasoning_content"]) for x in asst])
    c_ = stat([len(x["content"]) for x in asst])
    u_ = stat([len(x["content"]) for x in usr])
    roles = [("rewrite", "3 rewrite_prompts"), ("rate", "5 rate_stakes"),
             ("respond", "6 draft_responses"), ("rewrite_responses", "7 revise_responses")]
    model_rows = "\n".join(
        f"| `{r}` | {stage} | `{mods.get(r, {}).get('model', '?')}` | "
        f"{mods.get(r, {}).get('temperature', '?')} | "
        f"{mods.get(r, {}).get('max_tokens', '?')} |" for r, stage in roles)

    body = BODY.format(
        source=SOURCE, csha=man.get("constitution_sha256", "?")[:16],
        sha=(man.get("git_sha") or "?")[:12], models=model_rows,
        n=len(rows), pct=f"{100 * len(rows) / 716:.1f}", lost=716 - len(rows),
        lostwhere=", ".join(f"{t} ({c})" for t, c in lost_by.most_common()),
        smean=f"{sum(stakes) / len(stakes):.2f}",
        sdist=", ".join(f"`{k}`: {v}" for k, v in sorted(Counter(stakes).items())),
        over=sum(1 for s in stakes if s >= 2), grave=sum(1 for s in stakes if s == 3),
        subst=next((f"`{m['scenario_id']}` (replacing `{m['substitutes']}`)"
                    for m in md if m.get("substitutes")), "none"),
        ndom=len({m.get("ls_domain") for m in md if m.get("ls_domain")}),
        reloc=sum(1 for m in md if m.get("relocated")),
        rmin=r_[0], rmed=r_[1], rmax=r_[2], cmin=c_[0], cmed=c_[1], cmax=c_[2],
        umin=u_[0], umed=u_[1], umax=u_[2])

    old = Path(hf_hub_download(repo, "README.md", repo_type="dataset")).read_text(
        encoding="utf-8")
    assert old.startswith("---"), "card has no YAML frontmatter to preserve"
    frontmatter = old.split("---")[1]
    card = f"---{frontmatter}---\n\n{body}"

    out = rd / "README.md"
    out.write_text(card, encoding="utf-8")
    print(card[card.index("# Difficult"):][:1500])
    print(f"\n... wrote {out} ({len(card)} chars)")
    if dry:
        print("\n--dry: not uploaded")
        return
    HfApi().upload_file(path_or_fileobj=str(out), path_in_repo="README.md",
                        repo_id=repo, repo_type="dataset")
    print(f"pushed https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    fire.Fire(main)
