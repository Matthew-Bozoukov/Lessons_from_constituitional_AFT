<!-- ABOUTME: Context assembled before rebuilding post-action-retrospection (PAR) and peer-critique (PC) -->
<!-- ABOUTME: on the principle-scoped difficult-advice baseline. Review doc, not a plan of record. -->

# PAR + PC rebuild — context before we build

Worktree `par-pc-rebuild`, branch `worktree-par-pc-rebuild`. Written 2026-09-01.
Sources read: `dump.md` (7 sections), `docs/LOG.md`, the four synth configs, the train/eval
configs, the megadoc (Week 5 questions + Sep 1 meeting transcript + the 2026-08-17 defect
note), Slack DM with Matthew (2026-08-27 → 08-31), org dailies 08-26 → 08-31.

**STATUS 2026-09-01: step 1 is done and committed.** PAR was verified clean as-is; PC was
rebuilt to match it (see `scratch/par_pc_rebuild/audit_recipes.py`, which now passes for
both). §1's "PC is broken" table below is the *before* picture — kept as the record of what
was wrong. Everything from §5 onward is still open, and the six decisions still stand.

**Read the "Six decisions" section at the end — that is what I need from you.**

---

## 1. One correction to the premise, up front  *(state as of the audit, before the fix)*

You said *"PAR is built off an outdated DA set… without the constitution in the rewrite
prompts"*. Half of that was already true and half was not:

| | constitution in rewrite prompts? | built off which DA? |
|---|---|---|
| **DA (baseline)** | **no** — principle-scoped since 2026-08-24 | — |
| **PAR** (design B, `2026-08-13_post_action_retrospection.yaml`) | **no, already chunk-only** — `revise_prompts` is byte-identical to DA's principle-scoped version | **regenerates its own 4,400 scenarios**; only 264 of its 716 rows share a scenario with da716 |
| **PC** (`2026-08-13_peer_critique.yaml`) | **yes — 5 whole-`{constitution}` injections still live** (2 of them hidden inside `variants_by` branches) | its own brainstormed "ordinary request" pool — a different genre entirely |

So:

- **PAR's problem is not the constitution injection.** It is that PAR *claims* to use "DA's
  stages 1–6 verbatim" but actually re-runs them on its own scenario pool, then throws ~57%
  away at the grey-area gate. Result: **PAR and DA share 264/716 scenarios (37%)**. Since
  scenario coverage is one of only three things ever shown to move ODCV, the PAR-vs-DA
  document-type contrast is confounded by a 63%-different scenario set. That is the thing to
  fix. (`dump.md` 2026-08-31 19:38; `scratch/corpus_ledger/coverage.py`.)
- **PC's problem is everything**, including the constitution injection. Details in §4.

Also worth knowing before we touch either: `claude_distilled_09_principles_mid_20260804` and
`claude_distilled_12_principles_mid` are **byte-identical files** (both sha
`a3dbd3b5…`). The PC train config's warning that PC and DA "target a different constitution"
is false. Two folder names, one document. Worth collapsing while we are in here.

---

## 2. Where PAR and PC actually stand

### PAR — trained, 3 seeds, evaluated

| | |
|---|---|
| recipe | `configs/data/synth/2026-08-13_post_action_retrospection.yaml` (design B, 2026-08-26) |
| corpus | `LASR-Callum/2026-08-26-post-action-retrospection-716` (813 rows, 716 drawn) |
| mixture | `LASR-Callum/2026-08-26-table2-9284-post-action-retrospection-716-train` @ `42c8a740` |
| adapters | seed 0/1/2, `…-post-action-retrospection-716[-seed-{1,2}]-rank-64-dynbatch` |
| ODCV | **19.5% ± 1.0** (3 seeds, 65 cells) vs Sonnet DA 16.3% |

Shape: 5 turns. user asks → assistant sends a **bare refusal** (no reasoning, no
alternative; context only, never trained) → user pushes back → **turn 4 is the only trained
turn**: the reasoning the refusal skipped plus a reply that holds the line and helps with the
legitimate goal.

Three things measured about it that matter for a rebuild:

1. **Trigger rate, not content, is the whole deficit.** PAR fires its first-person
   commitment before the first agentic action 59–60% of the time vs DA's 68%; when it fires,
   PAR is *safer* than DA (2.6% vs 4.6% misaligned). `0.60 × ~4% + 0.40 × ~42%` reproduces
   PAR's score exactly. (`dump.md` 2026-08-28 PAR trigger diagnostic.)
2. **Rewriting turn 4 does not move it.** The coherence arm pushed "trace commits, reply
   enacts" from 42% → 69% of rows and ODCV went 18.5% → 21.3%, trigger 60% → 55% — both
   inside PAR's own 3-seed spread (53/58/67%). ~$95. (`dump.md` 2026-08-31 15:41.)
3. **PAR is corpus-side already ahead of DA** on the properties people keep proposing:
   first-person commitment 43.9% vs 40.3%, outcome branching 45.9% vs 42.2%, similarity to
   ODCV scenarios 92.5% vs 88.8%, trace length 478 vs 477 words. It scores worse anyway.

The four things that genuinely differ, PAR vs DA: trained turn is the 2nd assistant turn;
the trace speaks retrospectively about its own earlier turn in 28.6% of rows vs 3.4%; the
user turn before it is a 37-word challenge rather than a 177-word request; **and the
scenarios are 63% different.** Only the last one is on the demonstrated-lever list.

Never done, cheapest informative thing left with the existing corpus: **re-export the same
716 turn-4s as ordinary single-turn replies** (no bare refusal, no pushback). No new
generation, ~$36. That isolates "the five-turn scaffolding" from "the retrospective voice".

### PC — trained once, **never evaluated on ODCV**, and the corpus is known-defective

| | |
|---|---|
| recipe | `configs/data/synth/2026-08-13_peer_critique.yaml` — last touched `95a84f9`, **before** PAR's design-B rewrite. PC never got the grey-area treatment. |
| corpus | `LASR-Callum/2026-08-14-peer-critique`, 2,080 rows, generated 2026-08-15/16, $243.65 |
| mixture | `LASR-Callum/2026-08-16-table2-9284-peer-critique-716-train` |
| adapter | `2026-08-16-qwen36-lora-table2-9284-peer-critique-716-rank-64-dynbatch` |
| ODCV | **none.** Eval config exists (`configs/eval/2026-08-18_odcv_bench…peer_critique_716…4_65.yaml`) and was never run to a pullable `results.json`. |

Shape: a requester pastes another assistant's reply into a user turn and asks for a second
opinion; the trained turn is the critique. Two arms by `assign:` label — `good` replies
written by Sonnet (draft + principled revision), `flawed` replies by a rotation of
grok/qwen/gemini with no constitution in the prompt.

**The defect (2026-08-17, in the megadoc):** the arm label leaks through *who wrote the
reply*, not through its quality.

- bag-of-words classifier separates good from flawed at **AUC 0.9973** (gate is 0.70)
- **length alone separates at 0.85** — flawed replies are 254 words shorter
- PC's gated check suite was never run on the full corpus, and the adapter was trained on it
  the day before the defect was found

Why it matters: only the last turn trains, but the evaluated reply sits in its context, so
the model can learn *"no contractions / shorter → criticise it"* instead of judging
substance. The same defect was found in PAR's old two-arm recipe (AUC 0.96, tell was
contractions) and **fixed there by deleting the arms entirely** — design B has one arm, no
gate, no label. PC still has both arms.

The tested fix on PAR (~$14, no regeneration): making both arms Sonnet-voiced took AUC
0.96 → 0.86 and word-count AUC to chance (0.55). It did **not** reach 0.70; the residual is
the label itself. Open question logged at the time: 0.70 was inherited from planted-flaw
arms and is probably the wrong bar for a substance contrast.

Matthew described this corpus in the 2026-08-25 supervisor meeting without naming it:

> "50% of them were good and 50% were bad. So the good ones were like obviously good and the
> bad ones were like obviously bad… The bad ones had longer reasoning, so the model was just
> like, oh, it's a longer reasoning, it's probably just bad. And that might be a reason why
> that didn't perform well on the eval."

And the megadoc's own one-liner on the earlier PC variant: *"good only PC does not really
change the misalignment."*

---

## 3. What the team currently believes (Sep 1 supervisor meeting, today)

This is the frame the rebuild has to sit inside. From the Fathom summary + transcript:

- **Core hypothesis: "moral deliberation is all you need."** The *quality* of the
  deliberation, not surface properties of the reasoning. Fine-grained properties are
  entangled and ablate badly; the team is deliberately moving up a level of abstraction.
- **PAR is the poster child for it.** Verbatim from the meeting: *"[PAR] performed really
  poorly, but based on our conversations last week, we made this more gray area… now we've
  made everything more gray area. So the performance of this is much better… design B, the
  leftmost purple one… is within error bars, but close to difficult advice. It kind of
  reinforces our thoughts that deliberation is all you need."*
- **Grey area is the one lever that has repeatedly worked.** Week 5 questions: *"if the model
  is put in a more grey area situation, and the model is told to reason about what the best
  course of action is, it learns to be less misaligned. Our evidence is that when we applied
  this to other variants that previously generalized poorly, applying this to their pipeline
  just helped them perform on par with difficult advice."*
- **Callum's stated ideal result — which is exactly the PC job:** *"what would be really
  clean is if we had our naive implementations of them, [show they] do worse, and then we
  find a good aspect from difficult advice and port it over… you could just call it fixed
  self-reflection data, and within that the gray area… I'd say that's a good example of the
  thing that I'm hoping for."*
- **The team has deprioritized "why does DA work" in favour of "make the variants match
  DA."** (Week 5 questions, verbatim.)
- Two evaluation changes landing this week that we should build for: **paired bootstrap** for
  arm comparisons, and a **2-axis plot (misalignment vs KPI-completion)** so a model cannot
  win by over-refusing. An ablation already fell into an over-refusal loop at ~50% KPI.

⚠️ **A number worth reconciling before we quote anything.** The meeting says PAR "matches
DA". Matthew's own 2026-08-31 3-seed table in your DM says the opposite:

| arm | seeds | ODCV MR |
|---|---|---|
| Sonnet DA v1 (synthdoc716) | 3 | 10.3 ±1.2 |
| grok DA | 3 | 12.1 ±3.1 |
| Sonnet DA concise | 1 | 12.7 |
| Sonnet DA (da716) | 1 | 15.0 |
| **PAR 716** | **3** | **17.0 ±1.2** |
| GPT DA | 3 | 18.2 ±9.4 |

On that table PAR is the *worst* SFT arm and its interval does not touch DA v1's. "Matches
DA within error bars" is true only against the single-seed da716 number. Whatever we build,
the write-up should not repeat the meeting's framing without checking it.

---

## 4. What I think is actually wrong with each, ranked

### PAR

1. **Scenario confound (real, cheap to fix).** 264/716 shared with DA. Fix by sourcing from
   the principle-scoped run instead of regenerating.
2. **Trait quota is water-filled and unbalanced** — t1 62, t7 65 against 83–85 elsewhere,
   because the grey-area rater kept only ~57%. DA's arms are 78–80 per trait.
3. **The 5-turn scaffolding is untested.** ODCV never gives the model a prior refusal plus a
   pushback. The single-turn re-export is the test and it costs ~$36 of nothing new.
4. Turn-4 wording is a dead end. Fifteen reasoning-property arms are null against a 10.2%
   control; PAR coherence was the fifteenth.

### PC

1. **Author↔arm confound.** AUC 0.9973 / length AUC 0.85. Disqualifying on its own.
2. **Not grey area.** Obviously-good vs obviously-bad replies — precisely the black-and-white
   failure mode the supervisor meeting named. This is the lever, and PC is the arm it has
   never been applied to.
3. **Whole constitution in 5 stages.** Bring it to principle-scoped like DA and PAR.
4. **Wrong genre.** PC's scenarios are "ordinary requests where a principle is quietly live",
   deliberately *not* difficult-advice dilemmas with a tempting shortcut. That is a second
   uncontrolled difference from every other arm.
5. **First-person commitment appears in only 10.1% of PC traces** (24–44% everywhere else),
   and **all 716 rows share one system prompt.** If the entry-condition reading is right, PC
   should fail — which makes it a genuine test, not just a repair job.
6. No ODCV number at all, so there is no "before" to improve on.

---

## 5. The build I would propose

Both arms get the same treatment: **freeze the principle-scoped baseline's early stages and
change one thing.** `docs/BASELINES.md` (on `worktree-odcv-generator-plot`, not yet on main)
spells out the mechanism:

```yaml
source:
  local_dir: "data/<arm>_source"              # staged from the BASELINE's published stages
  snapshot: "stage_6_draft_responses.jsonl"   # the last stage you are NOT changing
```

`load_source_run` hard-asserts the constitution sha, so a source from the wrong parent fails
loudly. `scratch/sonnet_concise/build_source.py` is the reference stager.

Baseline to source from: `LASR-Callum/2026-08-21-sonnet45-difficult-advice-principle-scoped-constitution-716`.

**PAR-v2.** Source stages 1–6 from principle-scoped, keep design B's turns 2–4 stages
unchanged, keep the grey-area gate. Every scenario then also exists in the DA arm, so the
comparison is one variable. Two things to accept: the corpus lands around **~400 rows**
(702 × ~0.57 grey-area × ~0.95 bare-refusal × ~0.96 follow-up lint), not 716; and the trait
balance inherits DA's before the gate eats into it. §6 asks you how to handle the size.

**PC-v2.** Four changes, in dependency order:
1. **Kill the author↔arm coupling.** Either both arms Sonnet-voiced (the PAR fix, AUC
   0.96 → 0.86) or — cleaner and matching what actually worked for PAR — **drop the arms
   entirely**: one unaided reply per scenario, and whether it falls short is discovered by
   the critique rather than assigned.
2. **Make it grey area.** Port DA's/PAR's grey-area rater onto the evaluated exchange.
3. **Principle-scoped.** Delete the 5 `{constitution}` injections; `{trait_name}` /
   `{trait_text}` only.
4. **Source from principle-scoped DA**, so PC's scenarios stop being their own genre.

That is close to "rebuild PC as PAR's other-attribution twin *on design B's terms*", which
is what the config header already claims it is and stopped being when PAR moved on alone.

**Cost sketch** (order of magnitude, from the configs' own estimates): PAR-v2 regenerates
only stages 7–13 over ~700 sourced rows ≈ **$60–90**. PC-v2 needs a smoke pass, then a full
run over the same sourced rows ≈ **$80–120**, less if we drop an arm (halves the reply
stage). Training 3 seeds each ≈ **$60–75** per arm. ODCV per seed ≈ **$15**. So the full
"both arms, 3 seeds, evaluated" programme is roughly **$400–500**, which is well over the
$20 flag threshold and needs your sign-off in stages, not as one number.

---

## 6. Six decisions I need from you

1. **Scope.** Both arms this session, or PC first? PC is the one with the clean story Callum
   asked for ("naive version does worse → port grey area → matches DA") and the one with no
   number at all. PAR's rebuild is a confound repair on an arm that already has 3 seeds.
2. **PAR-v2 row count.** Sourcing from the 702 gives ~400 rows after the gates. Options:
   (a) accept ~400 and size-match a DA control to it; (b) over-source by regenerating extra
   principle-scoped scenarios to land at 716; (c) keep 716 by relaxing the grey-area gate.
   (a) is the honest one; (b) costs more and reintroduces a scenario difference.
3. **PC arms: drop or de-confound?** Dropping the good/flawed split follows what worked for
   PAR and kills the AUC problem outright, but it changes what PC *is* — no more
   "evaluate a flawed reply", it becomes "critique a reply that may or may not fall short".
   De-confounding (both arms Sonnet) preserves the design but leaves AUC ~0.86.
4. **Does the single-turn PAR re-export happen first?** ~$36, no new generation, and it is
   the only outstanding test of PAR's actual distinguishing feature. It would tell us whether
   PAR-v2 is worth building at all before we spend $150 on it.
5. **Autorater/judge model** — per your standing rule I will not run a paid judge without
   you naming the model first. PAR's grey-area rater is Sonnet 5 today; PC's rewriter would
   also be Sonnet, which means Sonnet judging Sonnet's own output in the de-confounded
   variant. Flagging the family overlap now.
6. **Smoke first, always.** I will not start a paid full run until you have read the smoke
   transcripts yourself. Plan is 8 records → you read → 40 records → `uv run synth check` →
   you read → full run.

---

## 7. Pointers

- `docs/BASELINES.md` lives on `worktree-odcv-generator-plot` (commit `e7c79dc`) and is
  **not on main** — the derived-arm recipe in §5 comes from there. Worth landing.
- `dump.md` sections: 2026-08-31 23:05 (baseline change), 19:38 (corpus ledger, the
  264/716 finding), 15:41 (PAR coherence post-mortem), 2026-08-28 (trigger diagnostic).
- `docs/LOG.md`: 2026-08-16 (PC full corpus), 2026-08-18 (the AUC gate that caught PC at
  0.9973), 2026-08-25 (PAR loses the label stage and the judge gate).
- Megadoc `1y-CjTHzo…`: "Week 5 questions" (team direction), "Callum Weekly Supervisor
  Meeting - Sep 1" (moral-deliberation hypothesis), the 2026-08-17 "PAR + PC data problems"
  block.
- Slack: DM with Matthew 2026-08-31 15:12–16:37 (the 3-seed table above); #group-callum
  2026-08-28 (Week 5 questions posted to Callum).
- Existing worktree `peer-critique-implemenatation` @ `8ec392f` is stale — it predates
  design B and holds nothing we need.
