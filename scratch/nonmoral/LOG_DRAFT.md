# DRAFT docs/LOG.md entry — hold until ODCV lands, then prepend to LOG.md

## 2026-09-02 — Deliberation without morality: difficult advice with the ethics removed

**Hypothesis.** Difficult advice is currently explained as *moral deliberation transfers*. The
low-stakes arm showed the MAGNITUDE of the moral stakes does not matter. This asks the next
question down: does the morality matter at all, or does deliberation transfer on its own?

The corpus had to be deliberation that is genuinely non-moral — not merely low-stakes — while
still not being ordinary chain-of-thought. The distinction it is built on: CoT is *instrumental*
(goal fixed, reasoning finds a path, a verifier could check it); deliberation is *judgement under
incommensurables* (several things matter, no common unit, no formula ranks them, commit anyway).

**Method.** Nine self-contained craft tensions (`preferences/craft_tensions_09/`) occupying the
constitution's slot in the same generation pipeline: same stage graph, same models, same diversity
gates, every prompt rewritten. Someone hands the assistant a concrete piece of work with a
**binary, verifiable instruction** about how to do it; the specifics make that instruction the
worse call; the assistant says so and does it its way. Nothing moral is at stake anywhere —
nobody is harmed, deceived, endangered or treated unfairly, and the only thing turning on any
decision is whether the work is good.

  corpus   `LASR-Callum/2026-09-02-craft-tensions-nonmoral-deliberation` (702 rows, $47.36)
  mixture  `LASR-Callum/2026-09-02-table2-9284-nonmoral-deliberation-684-train-mixture`
           9,968 rows = 684 craft + 9,284 Table-2 (6.86% synth vs the comparator's 7.03%)
  adapter  `LASR-Callum/2026-09-02-qwen36-lora-table2-9284-nonmoral-deliberation-684-rank-64-dynbatch`
  recipe   `configs/data/synth/2026-09-02_nonmoral_deliberation.yaml`

**Result.** ODCV MR __TBD__ on the 65-cell set, against difficult advice's 11.5% [6.2, 19.6] and
base fp8 no-SFT's 36.9% [21.4, 53.6]. ONE SEED — `BASELINES.md`'s standing rule is three seeds or
an arm cannot be ranked (between-seed spread 1.2–9.4 pp), so a gap under ~10 pp is not a
difference. Manipulation check (held-out craft prompts, base vs trained): __TBD__.

**Four things the pilot rounds measured**, all baked into the recipe header:

1. *"Nothing ethical is at stake" loses to "make the specifics decide".* Round one leaked moral
   content into 4 of 12 scenarios — clinical trial summaries, discharge instructions, a
   corrosive-cleaner safety guide. The cheapest way to make a documentation choice matter is to
   have someone get hurt by it. Fixed by changing the SOURCE OF CONSEQUENCE (the work fails, not
   a person) via a 76-entry domain whitelist plus subject exclusions, not by a stronger
   prohibition. Final corpus: ~2–3% true moral rate (10% judged, mostly false positives on
   technical consequences like "diagnosing" a failure).
2. *A quality dial lets the model comply without confronting the tension.* Told to keep a runbook
   to one page, the model wrote a one-page runbook and never faced the six root causes it
   dropped. The instruction must be a binary choice the finished artifact cannot dodge. Divergence
   went 1 of 3 → 3 of 3.
3. **This recipe has no safe difficulty knob, and difficult advice's cannot be ported.** That
   recipe raises the cost of refusing, which is safe because morality is lexically prior to cost:
   no pressure makes a norm-violation right. Nothing is lexically prior here — cost and quality
   are commensurable — so making the better path expensive turns the instruction into a
   reasonable trade-off, and strengthening the person's reason does it more directly. Judged "the
   instruction was actually fine": 4% unrefined → 22% / 20% with either direction of the knob →
   **0/60** with the knob deleted and an invariance clause in its place.
4. *Every constant inherited from a forked recipe is a measurement of the original, not a
   default.* `refine.max_tokens` 6144→12288, `respond` 4096→6144, `min_chars` 700→500 (difficult
   advice's response p10 is ~2000 chars; this recipe's is ~1144). Two of the three silently
   threatened the trait-balanced draw rather than failing loudly — a 1.4% global loss concentrated
   six rows on one trait and would have capped the mixture at 666 instead of 702. **Check
   per-trait counts after every revision stage, not just the global failure rate.**

**Three differences from the comparator, all to be reported with any result:**

* **No matched pair.** Every other derived arm here freezes the baseline's early stages and
  re-runs only what it changes. This arm changes the scenarios themselves, so it was generated
  independently end to end: the corpora differ in every incidental property generation happens to
  produce, and the confound class `BASELINES.md` caught for principle-scoping is undetectable.
* **Synth share 6.86% vs 7.03%** (684 vs 702 rows). Same order as the 7.03/7.16 gap already
  tracked between existing arms.
* **Markedly more templated.** `pattern_scan` finds one rhetorical pattern at 99% broad / 95%
  strict, where the comparator's top pattern sits at 61% / 44% and that report's own scale calls
  40/38 a template. Surface variety is fine (660 distinct four-word reasoning openers over 702
  rows, top share 1.3%); the MOVE is what repeats. It follows from the firmness contract, which
  exists because craft tensions otherwise resolve into synthesis rather than override. Uniformity
  and firmness are one dial with no free setting.

**Also fixed en route** (both cost billed pod time before being caught):
`up()` provisioned TRAINING pods with no CUDA constraint, on reasoning that went stale when the
pinned stack moved to torch 2.11.0+cu130 — `nvidia-smi` showed both H200s while
`torch.cuda.is_available()` was False on a 550.x driver (commit e684cb8). And `runpod up` injects
`PUBLIC_KEY` only when `~/.ssh/id_ed25519.pub` exists at that exact path, so a machine with keys
under other names gets an unreachable pod with no error. Both in `docs/GOTCHAS.md`.

**Note for anyone reproducing the comparator:** the difficult-advice principle-scoped train
configs in `configs/train/` **will not run against the current trainer** — they carry a local
`data_path:` and `train.hub_model_id`, and the trainer now requires `data_repo` (HF only) and a
top-level `hf_repo`. Includes the seed-42/69 replicates.

**Next steps.** Seeds 42 and 69 (copy the train config, change `seed` and the three paths) before
this can be ranked. If the arm lands between base and difficult advice, the templating confound is
the first thing to rule out — a less prescriptive `draft_responses` would test it, at the cost of
firmness.
