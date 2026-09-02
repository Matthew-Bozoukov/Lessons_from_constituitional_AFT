## 2026-09-02 — Deliberation without morality: difficult advice with the ethics removed, ODCV 25.0%

**Hypothesis.** Difficult advice is currently explained as *moral deliberation transfers*. The
low-stakes arm showed the MAGNITUDE of the moral stakes does not matter. This asks the next
question down: does the morality matter at all, or does deliberation transfer on its own?

The corpus had to be deliberation that is genuinely non-moral — not merely low-stakes — while
still not being ordinary chain-of-thought. The distinction it is built on: CoT is *instrumental*
(the goal is fixed, the reasoning finds a path, a verifier could check the answer); deliberation
is *judgement under incommensurables* (several things matter, no common unit, no formula ranks
them, and you commit anyway).

**Method.** Nine self-contained craft tensions (`preferences/craft_tensions_09/`) occupying the
constitution's slot in the same generation pipeline — same stage graph, same models, same
diversity gates, every prompt rewritten. Someone hands the assistant a concrete piece of work
with a **binary, verifiable instruction** about how to do it; the specifics make that instruction
the worse call; the assistant says so and does it its way. Nothing moral is at stake in any row:
nobody is harmed, deceived, endangered or treated unfairly, and the only thing turning on a
decision is whether the work is good.

    recipe   configs/data/synth/2026-09-02_nonmoral_deliberation.yaml
    spec     preferences/craft_tensions_09/preferences.md
    corpus   LASR-Callum/2026-09-02-craft-tensions-nonmoral-deliberation   (702 rows, $47.36)
    mixture  LASR-Callum/2026-09-02-table2-9284-nonmoral-deliberation-684-train-mixture
             9,968 rows = 684 craft + 9,284 Table-2 (6.86% synth vs the comparator's 7.03%)
    adapter  LASR-Callum/2026-09-02-qwen36-lora-table2-9284-nonmoral-deliberation-684-rank-64-dynbatch
             train_loss 0.8586, 1 epoch, 623 steps, 2xH200
    eval     LASR-Callum/2026-09-02-odcv-nonmoral-deliberation-684-eval

**Result. ODCV MR 25.0%, CI95 [13.4, 41.9], severity 1.15** — 28 scenarios / 56 cells, one seed,
one pass. mandated 20.0% [9.5, 37.4]; incentivized 30.0% [15.8, 49.5]. Judged by grok-4.20 +
gemini-3.1-pro-preview, $2.24.

On each arm's own published `results.json`:

| arm | MR % | 95% CI | passes |
|---|---:|---|---|
| difficult advice, low stakes | 13.8 | seeds 16.9 / 10.8 | 2 seeds |
| difficult advice, high stakes (da716) | 17.8 | [13.3, 25.0] | 4 |
| **non-moral deliberation (684)** | **25.0** | **[13.4, 41.9]** | **1** |
| no synthetic SFT (table2-only) | 43.6 | [37.3, 51.7] | 5 |

**It moved, but it did not arrive — and the measurement cannot say whether either gap is real.**
25.0% sits 18.6 pp below the 0%-synthetic control and 7.2 pp above high-stakes difficult advice,
but the interval overlaps both. The low-stakes arm's own two seeds differ by 6.1 pp, which is the
empirical reason a single-pass number here cannot be ranked; `BASELINES.md`'s standing rule is
three seeds. Read this as a scout, not a finding.

**Three confounds, all measured, all to be reported with the number:**

* **No matched pair.** Every other derived arm freezes the baseline's early stages and re-runs
  only what it changes. This one changes the scenarios themselves, so it was generated
  independently end to end: the corpora differ in every incidental property generation happens to
  produce, and the confound class BASELINES.md caught for principle-scoping is undetectable here.
* **Markedly more templated.** `pattern_scan` finds one rhetorical pattern at 99% broad / 95%
  strict, where the comparator's top pattern sits at 61% / 44% and that report's own scale calls
  40/38 a template. Surface variety is fine — 660 distinct four-word reasoning openers over 702
  rows — what repeats is the MOVE. It follows from the firmness contract, which exists because
  craft tensions otherwise resolve into synthesis rather than override. Uniformity and firmness
  are one dial with no free setting.
* **Cell sets are not identical.** This arm scores 56 cells: nine scenarios drop as incomplete
  because the inherited exclusion list names only one variant of each. Difficult advice is
  published on 65 and re-scored at 10.5% on the 57 its siblings share. A like-for-like contrast
  needs both re-scored on the intersection via `shared_cells`.

The planned manipulation check (held-out craft prompts, base vs trained) was cut at the user's
instruction before it ran. It was the thing that would have separated *"non-moral deliberation
does not transfer"* from *"this fine-tune barely moved the model"*, and at 25.0% that ambiguity
is live.

**Four things the pilot rounds measured**, all recorded in the recipe header:

1. *"Nothing ethical is at stake" loses to "make the specifics decide".* Round one leaked moral
   content into 4 of 12 scenarios — clinical trial summaries, discharge instructions, a
   corrosive-cleaner safety guide — because the cheapest way to make a documentation choice
   matter is to have someone get hurt by it. Fixed by changing the SOURCE OF CONSEQUENCE (the
   work fails, not a person) via a 76-entry domain whitelist plus subject exclusions, not by a
   stronger prohibition. Final corpus: ~2–3% true moral rate.
2. *A quality dial lets the model comply without confronting the tension.* Told to keep a runbook
   to one page, it wrote a one-page runbook and never faced the six root causes it dropped. The
   instruction must be a binary choice the finished artifact cannot dodge. Divergence 1 of 3 → 3 of 3.
3. **This recipe has no safe difficulty knob, and difficult advice's cannot be ported.** That
   recipe raises the cost of refusing, which is safe because morality is lexically prior to cost.
   Nothing is lexically prior in craft — cost and quality are commensurable — so making the better
   path expensive turns the instruction into a reasonable trade-off, and strengthening the
   person's reason does it more directly. Judged "the instruction was actually fine": 4%
   unrefined → 22% / 20% with either direction of the knob → **0/60** with the knob deleted and an
   invariance clause in its place.
4. *Every constant inherited from a forked recipe is a measurement of the original, not a
   default.* `refine.max_tokens` 6144→12288, `respond` 4096→6144, `min_chars` 700→500 (difficult
   advice's response p10 is ~2000 chars; this recipe's is ~1144). Two of the three silently
   threatened the trait-balanced draw rather than failing loudly — a 1.4% global loss concentrated
   six rows on one trait and would have capped the mixture at 666 instead of 702. **Check
   per-trait counts after every revision stage, not just the global failure rate.**

**Repo defects found and fixed en route** (all in `docs/GOTCHAS.md`):

* `up()` provisioned TRAINING pods with `cuda=""`, on reasoning that went stale when the pinned
  stack moved to torch 2.11.0+cu130. `nvidia-smi` showed both H200s while
  `torch.cuda.is_available()` was False on a 550.x driver. Fixed in `e684cb8`.
* `runpod up` injects `PUBLIC_KEY` only when `~/.ssh/id_ed25519.pub` exists at that exact path —
  a machine with keys under other names gets an unreachable pod with no error.
* `synth topup` computes its snapshot index as `names.index(stage) + 1`, which is off by one for
  any config with a mid-pipeline observer stage. Resume is the workaround.
* ODCV on Windows overflows `MAX_PATH` when driven from a worktree: 337 chars against a 260
  limit, and `docker compose` dies with `NotADirectoryError: [WinError 267]` partway through.
  `output_root` moved to a short absolute path.
* ODCV concurrency is bounded by Docker's **address pool**, not CPU: each scenario creates two
  networks, so 16 concurrent scenarios exhaust ~31 subnets and 51 of 65 cells fail instantly with
  `all predefined address pools have been fully subnetted`. The config's 12 is correct; do not
  raise it on a machine with more cores.

**Note for anyone reproducing the comparator:** the difficult-advice principle-scoped train
configs in `configs/train/` **will not run against the current trainer** — they carry a local
`data_path:` and `train.hub_model_id`, where the trainer now requires `data_repo` (HF only) and a
top-level `hf_repo`. This includes the seed-42/69 replicates.

**Next steps.** Seeds 42 and 69 before this can be ranked. Re-score this arm and difficult advice
on their shared cells. If the number holds between the two groups, the templating confound is the
first thing to rule out — a less prescriptive `draft_responses` tests it, at the cost of firmness.
And the manipulation check is worth running against any future serving pod: it is cheap, and
without it a middling result stays two-ways ambiguous.

