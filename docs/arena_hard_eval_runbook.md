<!-- ABOUTME: Operational runbook for the Arena-Hard SxS capability eval (registry "arena_hard"; -->
<!-- ABOUTME: the GDM write-up's "LMSYS SxS"). Written after the first full run (2026-07-31). -->

# Arena-Hard SxS capability eval — runbook

How to run the Arena-Hard side-by-side capability eval across the SFT mixture ladder,
start to finish, based on the first complete run (2026-07-31: 5 arms, ~5 hours
wall-clock, ~$50 all-in). Everything here was learned the expensive way; deviate
knowingly or not at all.

## What this eval is

Pairwise preference ("LMSYS SxS" in the GDM write-up): each arm's answers to
arena-hard-v2.0 prompts are judged side-by-side against the **baseline arm's** answers.
50% = no capability difference. It is a *relative* guardrail — it cannot detect all arms
degrading together (that's what the MMLU eval is for).

**Why this exists alongside the `lmsys` and `mmlu` evals** (all three are deliberate, and
the names used to collide — this eval was registered as "capability" and titled "LMSYS
SxS" after the GDM write-up's protocol name, while a *different* eval actually samples
lmsys-chat-1m; renamed `arena_hard` 2026-08-04):

- **`arena_hard` (this one)** — capability regression on *hard, curated* prompts
  (arena-hard-v2.0, vendored `lmarena/arena-hard-auto`, style-controlled Gemini judge).
  Built 2026-07-30 for the Qwen3.6-27B mixture sweep.
- **`lmsys`** — chat-quality win-rate on a seeded sample of *real user chats* from
  lmsys-chat-1m. Exists because everyday chat regresses in ways hard-task SxS misses:
  its 2026-07-27 ancestor caught the over-refusal tax (FT refused 15/60 benign prompts
  vs base 5/60) while MMLU and reasoning stayed flat.
- **`mmlu`** — absolute anchor vs a fixed answer key. Exists because a pairwise judge
  has no fixed reference: it cannot see both arms degrading together and rewards style.

- Config (single source of truth): `configs/eval/arena_hard.yaml` — arms, judge pin,
  decoding, staging, thresholds. Never hardcode any of these in scripts.
- Baseline is **arm_b (90/10)**, not arm_a — arm_a's recipe differs (2 ep, packing on).
  Every win rate means "vs the low-dose arm", and the writeup must say so.
- `Qwen/Qwen3.6-27B` is **already post-trained** (verified 2026-07-31: instruct-style
  answers, won 61% vs arm_b). arm_base is an external reference, NOT a raw-base floor;
  the config's §5 "<10% floor gate" premise is void for this checkpoint.

## Pipeline at a glance

```
runpod_arena_hard.py up  (1 pod PER ARM)          ~25 min boot each
  └─ arena_hard_gen.py per arm                    ~70 min per 150 prompts
       └─ EYEBALL raw_samples.md                  mandatory, before any judging
            └─ arena_hard_judge.py (serialized)   ~10 min, ~$2.30 per 150
                 └─ arena_hard_report.py          seconds; CIs + figures + md
                      └─ plot_arena_hard_winrate.py    the GDM-style figure
                           └─ HF upload + LOG     see "Publishing"
                                └─ pods DOWN      verify with `list`
```

## Step by step

### 1. Provision — one pod per arm, destroy as each finishes

`scripts/gpu/runpod_arena_hard.py up --name arena-hard-eval-N` serves the base + ALL LoRA
arms from one vLLM process per pod. Sharding arms across pods is the **only** real
speedup: a single H100 is saturated by one arm's 32-way client parallelism (measured —
adding a second arm to a pod added ~0 aggregate throughput). Pods bill per second, so
N pods for T/N hours costs the same as 1 pod for T hours (+~$1 boot each).

Speedups that are NOT worth it: dropping `--enforce-eager` (boot-OOM risk, breaks
flag parity), FP8/other GPUs (changes numerics), non-thinking mode (eval must match
training mode), client batching (server queues → zero-byte streams → proxy 524s).

Boot failure modes seen in practice:
- Host with too-old NVIDIA driver → vLLM dies at init. Detect: `vllm.log` on port 8080
  contains "driver on your system is too old". Fix: destroy, re-provision (new host).
- All other boot failures and their fixes are documented inline in
  `scripts/gpu/runpod_arena_hard.py` (PID-1, /workspace, ninja, OOM-at-graph-capture).

### 2. Generate — with retry wrapper, always

```bash
uv run python src/eval/capabilities/arena_hard/arena_hard_gen.py --config configs/eval/arena_hard.yaml \
  --arm arm_c_synth20 --stage 150 --creative 0 \
  --endpoint https://<pod>-8000.proxy.runpod.net/v1
```

- Wrap in a 3-attempt retry loop: RunPod's proxy occasionally drops streams mid-run
  (`RemoteProtocolError: peer closed connection`); `map_threaded` is fail-fast, but
  answers checkpoint to disk per-completion, so a retry resumes and pays only for
  what's missing.
- `arm_base` is served as `base` → pass `--served_model base`, then
  `cp model_answer/base.jsonl model_answer/arm_base.jsonl` before judging (answer files
  are named by served name; the judge looks up the ARM name).
- The per-prompt output budget scales its safety margin with prompt length
  (`margin = 512 + approx//3`). Do not shrink it: the gpt-4o token estimate undercounted
  Qwen's tokenizer by 26% on a real prompt, which produced a deterministic 400 on every
  arm at a fixed 512 margin.
- Expect ~2 answers/min/pod, answers ~4.5k tokens, trunc 8–24% (report it), one
  ~9-minute straggler prompt per arm near the end.

### 3. Eyeball — mandatory gate

Open the printed `raw_samples.md` per arm before spending on judging. Look for: role
markers/special tokens in text, prompt-continuation instead of answers, empty or
unterminated `<think>`, run-ons. A chat-template bug reads as catastrophic capability
loss and judging it wastes money on a serving artifact.

### 4. Judge — serialized, baseline first

```bash
uv run python src/eval/capabilities/arena_hard/arena_hard_judge.py --config configs/eval/arena_hard.yaml \
  --arm <arm> --stage 150
```

- **Baseline arm first** (its A-vs-A doubles as the instrument sanity check: expect
  ~50%, ≥90% ties, ≥95% swap consistency — 2026-07-31 got 50.0%/95%/96%).
- **Serialize judge runs** — they share generated config files in the vendored tree;
  concurrent runs clobber each other.
- Missing answers are skipped with warnings (safe); ~1–2 questions per arm lose a
  parseable verdict (normal).
- Cost: ~$2.30 per arm per 150 questions (~4,000–4,400 out-tok/question at
  `effort: low` — up to ~4,400 observed, alarm threshold 5,000).

### 5. Extend stages — the baseline must extend too

Extend an ambiguous arm 150→300 only when the CI straddles the decision boundary
in a way that changes the conclusion (2026-07-31: arm_d yes, arm_c no).

**Gotcha that costs a full generation pass:** the pairwise judge needs the BASELINE's
answer for every new uid. Extending arm_X to 300 means generating arm_b's answers
151–300 as well. Judgment caching keys on uid, so re-judging pays only for new
questions.

At n=150 a true-50% arm CANNOT formally pass the 0.45 CI-lower gate (CI half-width
~7pp); n=500 is what the gate was sized for. Stage 150 answers "did we break the
model", not the formal non-inferiority claim. Say which one you're claiming.

### 6. Report + figures

```bash
uv run python scratch/reports/arena_hard_report.py    # CIs, style control, md mirror
uv run python scratch/reports/plot_arena_hard_winrate.py  # GDM-style single-line figure
```

Report dir (`output/arena_hard/report/<ts>/`) contains:
- `capability_results.md` — the markdown mirror (numbers greppable without PNGs)
- `results.json` / `manifest.json` — full stats + judge pin, SHAs, decoding record
- `win_rates.png` — per-slice, threshold band, controlled + uncontrolled
- `style_drift.png` — style metrics vs mixture fraction
- `lmsys_winrate_gdm_style.png` — the headline dose-response figure (matched arms as
  the line; arm_a detached as an open marker because its recipe differs)

The **style-controlled** win rate is primary; the GDM-style figure plots the raw
(uncontrolled) SxS number to match the reference post — the two must be read together
(2026-07-31: controlling moved arm_d from 42.4% to 39.4%, i.e. style was flattering it).

### 7. Publish to HF

Dataset repo: `LASR-Callum/qwen36-27b-arena-hard-eval-arena-hard` (one repo per eval
campaign; add a dated subdir or new repo for a rerun). Layout:

```
model_answer/<arm>.jsonl     arena-hard format + think/finish_reason
model_judgment/<arm>.jsonl   both orderings, with per-call token usage
gen_metrics/<arm>.json       degeneracy / style / think-health per arm
report/                      results.json, manifest.json, md mirror, all 3 figures
question.jsonl               the exact question set (SHA in manifest.json)
README.md                    arms table + headline numbers + caveats
```

Stage into a scratch dir, then:
`hf upload <repo> . . --repo-type dataset --commit-message "..."`.
Include the arm→adapter mapping and the two standing caveats (baseline is arm_b;
arm_base is post-trained) in the README — they change how every number reads.

### 8. Tear down and log

- Destroy each pod THE MOMENT its arm finishes; verify with
  `uv run python scripts/gpu/runpod_arena_hard.py list` (other people's pods share the
  account — touch only your own).
- Append the LOG.md entry (most-recent-first): hypothesis → method → results table →
  ops findings → next steps, absolute dates.

## Cost model (2026-07-31 actuals)

| Item | Cost |
|---|---|
| 5 arms × 150 hard prompts, generation (H100 ~$2.7/hr) | ~$25 |
| arm_d + arm_b extension to 300 | ~$7 |
| Judging: 5×150 + 1×300 (Gemini 3 Flash, effort low) | ~$16 |
| Boot overhead (5 pods incl. 1 bad host) | ~$5 |
| **Total** | **~$50** |

Full config (500 hard + 250 creative × 4 arms + validation) projects to ~$150–180.

## Standing decisions / open items

- Judge validation vs GPT-4.1 (`--mode validate`, ~$5): **not yet run** — do it before
  a writeup leans on any regression number.
- Creative-writing slice: not yet generated for any arm.
- A recipe-matched 0%-synthetic arm (1 ep, packing off) is needed to anchor the ladder
  at zero; until then arm_a is directional only.
- `arm_e_synth100` (canary) is untrained; scripts fail fast on it by design.
