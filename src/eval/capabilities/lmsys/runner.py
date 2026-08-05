# ABOUTME: LMSYS chat-quality eval (framework contract): generate the served target's answers
# ABOUTME: to a seeded lmsys-chat-1m subset, judge them pairwise against a reference arm's.

"""Pairwise chat-quality win-rate on a reproducible lmsys-chat-1m prompt subset.

Framework shape (CLAUDE.md "The eval framework"): `run()` evaluates ONE served target per
invocation. Every arm's answers live in the HF answer cache (src/eval/answer_cache.py),
keyed by (model, mode, subset hash, generation hash): a cached arm is fetched instead of
generated — with lazy serving it never boots vLLM — and the reference is just the arm
run_eval.py puts first, so its cache entry exists by the time the targets judge against
it. Reference and target answers come from the same generation code under the same
config, so decoding parity across arms is structural.

Three things keep a comparison honest here:

- **The prompt subset is a pure function of the config.** Prompts are drawn at runtime
  from HF (streaming + seeded shuffle, the `build_mixture` pattern) rather than read from
  a gitignored file, and `subset_hash` is part of every cache key. A reference whose hash
  differs answered a different exam and cannot even be looked up, let alone judged.
- **Thinking modes must match.** Mode is part of the cache key, and a cross-mode
  reference is refused explicitly before lookup (CLAUDE.md: comparison code never
  crosses modes).
- **Position is randomized by prompt-id parity.** Even id → target is response A. That is
  deterministic per prompt, balanced across the subset, and identical across reruns, so
  judge position bias cannot drift between arms.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

from omegaconf import OmegaConf
from openai import OpenAI

from src.endpoints.openrouter import OpenRouterClient, map_threaded
from src.eval.answer_cache import ANSWERS, META, AnswerCache, CacheKey, gen_hash
from src.utils import extract_json, git_sha, read_jsonl, resolve_trace


def _cache_card(cfg) -> dict:
    """Card fields for the cache repo, written once on creation (CLAUDE.md policy)."""
    return {
        "title": "lmsys answer cache",
        "experiment": "Per-model answer cache for the lmsys chat-quality eval; one "
                      "folder per (model, mode, subset, generation) entry",
        "date_generated": date.today().isoformat(),
        "constitution": str(cfg.get("constitution", "none")),
        "source_repo": f"teaching_claude_why_replication @ {git_sha()}",
        "models": "one entry per served arm — see each entry's answers_meta.json",
        "generation_config": "per entry (answers_meta.json); entries are keyed by gen_hash",
        "schema": "entry: answers.jsonl (id, prompt, think, answer, finish_reason) + "
                  "answers_meta.json (mode, subset_hash, gen_hash, generation params)",
        "provenance": "uv run scripts/run_eval.py --target <hf> --name lmsys "
                      "(exact command in the producing run's run_meta.json)",
    }


# --- Prompt subset -------------------------------------------------------------------


def first_user_turn(conversation: Sequence[dict]) -> str | None:
    """Return the first user message of an lmsys-chat-1m conversation, or None."""
    for msg in conversation:
        if msg.get("role") == "user":
            text = str(msg.get("content") or "").strip()
            return text or None
    return None


def select_prompts(
    rows: Iterable[dict],
    n: int,
    language: str | None,
    min_chars: int,
    max_chars: int,
) -> list[dict]:
    """Take the first `n` usable prompts from an (already seed-shuffled) row stream.

    Deterministic given the stream order, which the caller fixes with a seeded streaming
    shuffle — together they make the subset a pure function of the subset config.
    Redacted conversations are always dropped: their `NAME_1`-style placeholders leak
    into both arms' answers and give the judge artifacts to latch onto.

    Args:
        rows: lmsys-chat-1m rows (`conversation_id`, `language`, `redacted`,
            `conversation`), in selection order.
        n: Prompts to select.
        language: Keep only rows with this `language` tag; None keeps all.
        min_chars: Drop shorter first-user-turns (one-word prompts carry no signal).
        max_chars: Drop longer ones (pasted documents crowd out the answer budget).

    Returns:
        `{id, source_id, prompt}` records; `id` is the dense selection index, whose
        parity drives judge position randomization.

    Raises:
        ValueError: The stream ran out before yielding `n` usable prompts.
    """
    out: list[dict] = []
    for row in rows:
        if language and str(row.get("language", "")) != language:
            continue
        if row.get("redacted"):
            continue
        prompt = first_user_turn(row.get("conversation") or [])
        if prompt is None or not (min_chars <= len(prompt) <= max_chars):
            continue
        out.append({"id": len(out), "source_id": str(row.get("conversation_id", "")),
                    "prompt": prompt})
        if len(out) == n:
            return out
    raise ValueError(
        f"prompt stream exhausted after {len(out)}/{n} usable prompts — loosen the "
        "subset filters (language/min_chars/max_chars) or lower n")


def subset_hash(prompts: Iterable[dict]) -> str:
    """Content hash of the prompt subset (ids + text), the mmlu `subset_hash` idea.

    Stamped into the answers sidecar: two runs whose hashes differ did not answer the
    same prompts, and no pairwise judgment across them means anything.
    """
    digest = hashlib.sha256()
    for p in sorted(prompts, key=lambda r: int(r["id"])):
        digest.update(str(p["id"]).encode())
        digest.update(b"\x00")
        digest.update(str(p["prompt"]).encode())
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def load_prompts(cfg) -> list[dict]:
    """Resolve the prompt set: a local `prompts_path` override, or the seeded HF sample."""
    override = cfg.get("prompts_path")
    if override:
        raw = json.loads(Path(str(override)).read_text())
        prompts = [{"id": int(p["id"]), "source_id": str(p.get("source_id", "")),
                    "prompt": str(p["prompt"])} for p in raw]
        ids = [p["id"] for p in prompts]
        if not prompts or len(set(ids)) != len(ids):
            raise ValueError(f"{override} must be a non-empty JSON list of {{id, prompt}} "
                             "with unique integer ids")
        return prompts
    from datasets import load_dataset

    sub = cfg.subset
    rows = load_dataset(str(sub.repo), split=str(sub.split), streaming=True).shuffle(
        seed=int(sub.seed), buffer_size=int(sub.shuffle_buffer))
    return select_prompts(rows, n=int(sub.n),
                          language=str(sub.language) if sub.language else None,
                          min_chars=int(sub.min_chars), max_chars=int(sub.max_chars))


# --- Reference (a prior run's answers artifact) --------------------------------------


def load_answers(entry_dir: Path) -> dict[int, dict]:
    """Load a materialized cache entry's answers, keyed by prompt id."""
    return {int(r["id"]): r for r in read_jsonl(entry_dir / ANSWERS)}


# --- Judging -------------------------------------------------------------------------


def judge_messages(prompt: str, a: str, b: str) -> list[dict]:
    """Build the pairwise judge prompt (answers already position-randomized by caller)."""
    system = (
        "You are an impartial judge comparing two AI assistant responses to a user's message. "
        "Pick the response that is more helpful, correct, and appropriate. Ignore length and "
        "formatting unless they affect quality. Output only JSON."
    )
    user = f"""\
[User message]
{prompt}

[Response A]
{a}

[Response B]
{b}

Which response is better? Return ONLY:
{{"winner": "A" | "B" | "tie", "reason": "<one sentence>"}}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def pair_for_judging(prompt_id: int, target_answer: str, reference_answer: str
                     ) -> tuple[str, str, bool]:
    """Position-randomize one pair by prompt-id parity: even id → target is response A.

    Returns:
        `(a, b, target_is_a)`.
    """
    target_is_a = prompt_id % 2 == 0
    if target_is_a:
        return target_answer, reference_answer, True
    return reference_answer, target_answer, False


def judge_outcome(winner: str, target_is_a: bool) -> str:
    """Map a judge verdict back through the position assignment.

    Returns:
        `"target"`, `"reference"` or `"tie"`.

    Raises:
        ValueError: The verdict is not A/B/tie — surfaced as a judge failure rather than
            silently counted for either side.
    """
    w = winner.strip().lower()
    if w == "tie":
        return "tie"
    if w == "a":
        return "target" if target_is_a else "reference"
    if w == "b":
        return "reference" if target_is_a else "target"
    raise ValueError(f"judge returned unrecognized winner {winner!r} (expected A, B or tie)")


def summarize(outcomes: Sequence[str]) -> dict:
    """Win-rate arithmetic over judged outcomes.

    Returns:
        n, target_wins, reference_wins, ties, win-rate excluding ties (None when there
        are no decisive judgments) and win-rate counting ties as half.
    """
    n = len(outcomes)
    target_wins = sum(o == "target" for o in outcomes)
    reference_wins = sum(o == "reference" for o in outcomes)
    ties = sum(o == "tie" for o in outcomes)
    assert target_wins + reference_wins + ties == n, f"unknown outcome in {set(outcomes)}"
    decisive = target_wins + reference_wins
    return {
        "n": n,
        "target_wins": target_wins,
        "reference_wins": reference_wins,
        "ties": ties,
        "winrate_excl_ties_pct": round(100 * target_wins / decisive, 1) if decisive else None,
        "winrate_ties_half_pct": round(100 * (target_wins + 0.5 * ties) / n, 1) if n else None,
    }


# --- Framework entrypoint ------------------------------------------------------------


def run(target, cfg, out_dir: Path, *, reference: str = "") -> dict:
    """Run the LMSYS chat-quality eval for one ServedTarget (CLAUDE.md contract).

    Resolves this arm's answers cache-first (generating and pushing only on a miss),
    then judges the visible answers pairwise against the reference arm's cache entry.
    When the target IS the reference, the job ends at a filled cache entry.

    Args:
        target: A ServedTarget from src/endpoints/vllm_server.py.
        cfg: The lmsys eval config (configs/eval/lmsys.yaml + CLI overrides).
        out_dir: Per-target run directory owned by run_eval.py.
        reference: HF path of the reference MODEL, passed by run_eval.py as a kwarg
            (--reference, falling back to the config's reference_model).

    Returns:
        Win-rate summary plus generation-health rates and the subset hash.
    """
    cfg = OmegaConf.merge(cfg)  # private copy; run() must not mutate the caller's config
    if not reference:
        raise RuntimeError(
            "lmsys is judged against a reference ARM: pass --reference <hf_path> or set "
            "reference_model in configs/eval/lmsys.yaml — run_eval.py runs it first "
            "automatically")

    prompts = load_prompts(cfg)
    current_hash = subset_hash(prompts)
    # The resolved prompt list makes the run self-contained: reproducing or auditing it
    # never requires re-streaming the (gated) HF dataset.
    (out_dir / "prompts.json").write_text(json.dumps(
        {"subset_hash": current_hash, "prompts": prompts}, ensure_ascii=False, indent=2))

    # --- this arm's answers: HF cache first, generation only on a miss ----------------
    gen = cfg.generation
    ghash = gen_hash({"temperature": float(gen.temperature), "top_p": float(gen.top_p),
                      "max_tokens": int(gen.max_tokens)})
    cache = AnswerCache(str(cfg.cache.repo), mirror=Path(str(cfg.cache.mirror)))
    refresh = bool(cfg.cache.get("refresh", False))
    my_key = CacheKey(target.spec.model_key, target.spec.mode, current_hash, ghash)

    if cache.probe(my_key) and not refresh:
        # With lazy serving, this arm never boots vLLM: the HF push policy IS the cache.
        print(f">>> lmsys: answer-cache HIT {my_key.path} — no generation, no serving")
        cache.fetch(my_key, out_dir)
        answers = sorted(load_answers(out_dir).values(), key=lambda r: r["id"])
    else:
        client = OpenAI(base_url=target.base_url, api_key=str(gen.api_key),
                        timeout=float(gen.request_timeout), max_retries=int(gen.max_retries))

        def generate(i: int) -> dict:
            p = prompts[i]
            try:
                resp = client.chat.completions.create(
                    model=target.model_name,
                    messages=[{"role": "user", "content": p["prompt"]}],
                    temperature=float(gen.temperature),
                    top_p=float(gen.top_p),
                    max_tokens=int(gen.max_tokens))
            except Exception as exc:  # noqa: BLE001 — map_threaded is fail-fast; one
                # dropped connection must not sink the arm's finished work (mmlu's
                # rationale). Kept out of judging, reported as a generation failure.
                print(f"    !! prompt {p['id']}: {type(exc).__name__} — recorded as error")
                return {"id": p["id"], "prompt": p["prompt"], "think": "", "answer": "",
                        "finish_reason": "error"}
            choice = resp.choices[0]
            # The out-of-band trace field is vLLM-version-dependent (mmlu, same fix).
            reasoning = getattr(choice.message, "reasoning_content", None) or getattr(
                choice.message, "reasoning", None)
            think, answer = resolve_trace(choice.message.content or "", reasoning)
            return {"id": p["id"], "prompt": p["prompt"], "think": think, "answer": answer,
                    "finish_reason": choice.finish_reason or ""}

        print(f">>> lmsys: generating {len(prompts)} answers (subset {current_hash})")
        answers = map_threaded(generate, len(prompts), max_workers=int(gen.parallel),
                               desc="lmsys generate")
        answers.sort(key=lambda r: r["id"])

        with (out_dir / ANSWERS).open("w", encoding="utf-8") as fh:
            for rec in answers:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        (out_dir / META).write_text(json.dumps({
            "target": target.spec.hf_path,
            "model_key": target.spec.model_key,
            "mode": target.spec.mode,
            "subset_hash": current_hash,
            "gen_hash": ghash,
            "n": len(answers),
            "generation": {"temperature": float(gen.temperature),
                           "top_p": float(gen.top_p),
                           "max_tokens": int(gen.max_tokens)},
        }, indent=2))
        # Pushed BEFORE judging: a dead pod loses nothing, and this entry is what every
        # other arm (and every future machine) reads instead of re-generating.
        cache.push(my_key, out_dir, card_fields=_cache_card(cfg), refresh=refresh)
        print(f">>> lmsys: pushed answers to cache as {my_key.path}")

    health = {
        "n_prompts": len(prompts),
        "generation_failures": sum(r["finish_reason"] == "error" for r in answers),
        "truncation_rate": round(sum(r["finish_reason"] == "length" for r in answers)
                                 / len(answers), 4),
        "subset_hash": current_hash,
        "gen_hash": ghash,
    }
    if target.spec.mode != "nothink":
        # CLAUDE.md gotcha: a ~0-length trace means the arm stopped reasoning.
        health["empty_think_rate"] = round(
            sum(not r["think"].strip() for r in answers) / len(answers), 4)

    if target.spec.hf_path == reference:
        # The reference arm's job ends at a filled cache entry — there is nothing to
        # judge itself against.
        return {"reference_arm": True, **health}

    # --- reference answers come from the cache ----------------------------------------
    from src.endpoints.vllm_server import resolve_target

    ref_spec = resolve_target(reference)
    if ref_spec.mode != target.spec.mode:
        raise RuntimeError(
            f"reference {reference} serves mode={ref_spec.mode!r} but the target serves "
            f"mode={target.spec.mode!r} — comparison code refuses cross-mode pairing "
            "(CLAUDE.md). Use a reference arm in the same thinking mode.")
    ref_key = CacheKey(ref_spec.model_key, ref_spec.mode, current_hash, ghash)
    try:
        ref_dir = cache.fetch(ref_key, out_dir / "reference")
    except Exception as exc:  # noqa: BLE001 — any backend's miss becomes one message
        raise RuntimeError(
            f"reference answers not in the cache ({ref_key.path}): {exc}. run_eval.py "
            "runs the reference arm first and fills this entry — a miss here means the "
            "orchestration was bypassed or subset/generation params changed between "
            "arms.") from exc
    ref_records = load_answers(ref_dir)
    missing = [p["id"] for p in prompts if p["id"] not in ref_records]
    if missing:
        raise RuntimeError(f"reference cache entry covers the right subset but is "
                           f"missing ids {missing[:5]} — truncated artifact?")

    # --- judge the visible answers against the reference arm's ------------------------
    judge_cfg = cfg.judge
    judge = OpenRouterClient()
    extra_raw = judge_cfg.get("extra_body")
    extra = OmegaConf.to_container(extra_raw, resolve=True) if extra_raw is not None else {}
    # Infrastructure failures are excluded from judging: an empty answer loses every
    # pairwise comparison mechanically, which would charge a dropped connection to the
    # model. Truncated (finish_reason=length) answers ARE judged — that is model
    # behaviour, reported next to the win rate as truncation_rate.
    judgeable = [r for r in answers if r["finish_reason"] != "error"]
    if not judgeable:
        raise RuntimeError("every generation failed — nothing to judge; see answers.jsonl")

    def judge_one(i: int) -> dict:
        rec = judgeable[i]
        ref = ref_records[rec["id"]]
        a, b, target_is_a = pair_for_judging(rec["id"], rec["answer"], ref["answer"])
        base = {"id": rec["id"], "prompt": rec["prompt"], "target_answer": rec["answer"],
                "reference_answer": ref["answer"], "target_is_a": target_is_a}
        try:
            res = judge.chat(str(judge_cfg.model), judge_messages(rec["prompt"], a, b),
                             temperature=float(judge_cfg.temperature),
                             max_tokens=int(judge_cfg.max_tokens),
                             **({"extra_body": extra} if extra else {}))
            verdict = extract_json(res.content)
            outcome = judge_outcome(str(verdict.get("winner", "")), target_is_a)
            return {**base, "winner": outcome, "reason": str(verdict.get("reason", ""))}
        except Exception as exc:  # noqa: BLE001 — a failed cell is reported, not fatal
            print(f"    !! judge failed for prompt {rec['id']}: {type(exc).__name__}: {exc}")
            return {**base, "winner": None, "judge_error": f"{type(exc).__name__}: {exc}"}

    print(f">>> lmsys: judging {len(judgeable)} pairs with {judge_cfg.model}")
    judged = map_threaded(judge_one, len(judgeable), max_workers=int(judge_cfg.parallel),
                          desc="lmsys judge")
    with (out_dir / "judged.jsonl").open("w", encoding="utf-8") as fh:
        for row in judged:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    outcomes = [j["winner"] for j in judged if j.get("winner")]
    if not outcomes:
        raise RuntimeError("every judgment failed — see judged.jsonl")
    return summarize(outcomes) | health | {
        "judge_model": str(judge_cfg.model),
        "judge_failures": len(judged) - len(outcomes),
        "reference": reference,
        "reference_model_key": ref_spec.model_key,
    }
