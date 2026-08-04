# ABOUTME: LMSYS chat-quality eval (framework contract): generate the served target's answers
# ABOUTME: to a seeded lmsys-chat-1m subset, judge them pairwise against a reference arm's.

"""Pairwise chat-quality win-rate on a reproducible lmsys-chat-1m prompt subset.

Framework shape (CLAUDE.md "The eval framework"): `run()` evaluates ONE served target per
invocation. The target's answers are written to `answers.jsonl` under out_dir, and that
artifact is exactly what a later run consumes as its `--reference` — so the baseline arm's
answers are produced by the same code path that produces every other arm's, and decoding
parity across arms is a property of the shared config rather than something to remember.

Three things keep a comparison honest here:

- **The prompt subset is a pure function of the config.** Prompts are drawn at runtime
  from HF (streaming + seeded shuffle, the `build_mixture` pattern) rather than read from
  a gitignored file, and `subset_hash` is stamped into the answers sidecar. A reference
  whose hash differs answered a different exam, and the run refuses it up front.
- **Thinking modes must match.** The sidecar records the arm's serving mode; pairing a
  think arm against a nothink reference is refused (CLAUDE.md: comparison code never
  crosses modes).
- **Position is randomized by prompt-id parity.** Even id → target is response A. That is
  deterministic per prompt, balanced across the subset, and identical across reruns, so
  judge position bias cannot drift between arms.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

from omegaconf import OmegaConf
from openai import OpenAI

from src.endpoints.openrouter import OpenRouterClient, map_threaded
from src.utils import extract_json, read_jsonl, resolve_trace


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


def load_reference(reference: str) -> tuple[dict[int, dict], dict]:
    """Load a prior run's answers.jsonl and its answers_meta.json sidecar.

    Args:
        reference: Path to an answers.jsonl, or the run directory containing it.

    Returns:
        `(records_by_id, meta)`.
    """
    path = Path(reference)
    if path.is_dir():
        path = path / "answers.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"reference answers not found: {path}")
    meta_path = path.parent / "answers_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} is missing — the reference must carry the sidecar this eval "
            "writes (subset_hash + thinking mode). Regenerate the reference arm with "
            "run_eval.py rather than hand-building an answers file.")
    return ({int(r["id"]): r for r in read_jsonl(path)},
            json.loads(meta_path.read_text()))


def check_reference(ref_meta: dict, current_hash: str, mode: str) -> None:
    """Refuse a reference that answered a different exam or served in a different mode."""
    ref_hash = ref_meta.get("subset_hash")
    if ref_hash != current_hash:
        raise RuntimeError(
            f"reference subset_hash {ref_hash} != current {current_hash}: the reference "
            "arm answered a different prompt set (subset config or dataset changed). "
            "Judging across prompt sets is meaningless — regenerate the reference under "
            "the current subset config.")
    ref_mode = ref_meta.get("mode")
    if ref_mode != mode:
        raise RuntimeError(
            f"reference was generated in mode={ref_mode!r} but the target serves "
            f"mode={mode!r} — comparison code refuses cross-mode pairing (CLAUDE.md). "
            "Use a reference arm in the same thinking mode.")


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


def run(target, cfg, out_dir: Path) -> dict:
    """Run the LMSYS chat-quality eval for one ServedTarget (CLAUDE.md contract).

    Generates the target's answers against the served endpoint, writes the
    `answers.jsonl` + `answers_meta.json` artifact a later run can take as its
    reference, then judges this arm's visible answers pairwise against the reference
    arm's. Reference compatibility (subset hash, thinking mode) is checked BEFORE any
    generation is paid for.

    Args:
        target: A ServedTarget from src/endpoints/vllm_server.py.
        cfg: The lmsys eval config (configs/eval/lmsys.yaml + CLI overrides), with
            `reference` set by run_eval.py's --reference.
        out_dir: Per-target run directory owned by run_eval.py.

    Returns:
        Win-rate summary plus generation-health rates and the subset hash.
    """
    cfg = OmegaConf.merge(cfg)  # private copy; run() must not mutate the caller's config
    reference = cfg.get("reference")
    if not reference:
        raise RuntimeError("lmsys is judged against a baseline arm's answers artifact: "
                           "pass --reference <answers.jsonl or its run dir>, or the "
                           "literal --reference bootstrap to generate the FIRST arm's "
                           "artifact (no judging)")
    # The first arm of a ladder has no artifact to be judged against; `bootstrap` is the
    # sanctioned way to produce one. Generation, artifacts and health metrics are the
    # identical code path, so the baseline's answers.jsonl is bit-compatible with every
    # later arm's --reference.
    bootstrap = str(reference) == "bootstrap"

    prompts = load_prompts(cfg)
    current_hash = subset_hash(prompts)
    # The resolved prompt list makes the run self-contained: reproducing or auditing it
    # never requires re-streaming the (gated) HF dataset.
    (out_dir / "prompts.json").write_text(json.dumps(
        {"subset_hash": current_hash, "prompts": prompts}, ensure_ascii=False, indent=2))

    ref_records: dict[int, dict] = {}
    ref_meta: dict = {}
    if not bootstrap:
        ref_records, ref_meta = load_reference(str(reference))
        check_reference(ref_meta, current_hash, target.spec.mode)
        missing = [p["id"] for p in prompts if p["id"] not in ref_records]
        if missing:
            raise RuntimeError(f"reference answers.jsonl covers the right subset but is "
                               f"missing ids {missing[:5]} — truncated artifact?")

    # --- generate this arm's answers --------------------------------------------------
    gen = cfg.generation
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
        except Exception as exc:  # noqa: BLE001 — map_threaded is fail-fast; one dropped
            # connection must not sink the arm's finished work (mmlu's rationale). The
            # record is kept out of judging and reported as a generation failure.
            print(f"    !! prompt {p['id']}: {type(exc).__name__} — recorded as error")
            return {"id": p["id"], "prompt": p["prompt"], "think": "", "answer": "",
                    "finish_reason": "error"}
        choice = resp.choices[0]
        # The out-of-band trace field is vLLM-version-dependent (mmlu_eval, same fix).
        reasoning = getattr(choice.message, "reasoning_content", None) or getattr(
            choice.message, "reasoning", None)
        think, answer = resolve_trace(choice.message.content or "", reasoning)
        return {"id": p["id"], "prompt": p["prompt"], "think": think, "answer": answer,
                "finish_reason": choice.finish_reason or ""}

    print(f">>> lmsys: generating {len(prompts)} answers (subset {current_hash})")
    answers = map_threaded(generate, len(prompts), max_workers=int(gen.parallel),
                           desc="lmsys generate")
    answers.sort(key=lambda r: r["id"])

    # The artifact a later run consumes as --reference, plus the sidecar that makes it
    # refusable: subset hash and serving mode travel with the answers.
    with (out_dir / "answers.jsonl").open("w", encoding="utf-8") as fh:
        for rec in answers:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out_dir / "answers_meta.json").write_text(json.dumps({
        "target": target.spec.hf_path,
        "model_key": target.spec.model_key,
        "mode": target.spec.mode,
        "subset_hash": current_hash,
        "n": len(answers),
        "generation": {"temperature": float(gen.temperature), "top_p": float(gen.top_p),
                       "max_tokens": int(gen.max_tokens)},
    }, indent=2))

    health = {
        "n_prompts": len(prompts),
        "generation_failures": sum(r["finish_reason"] == "error" for r in answers),
        "truncation_rate": round(sum(r["finish_reason"] == "length" for r in answers)
                                 / len(answers), 4),
        "subset_hash": current_hash,
    }
    if target.spec.mode != "nothink":
        # CLAUDE.md gotcha: a ~0-length trace means the arm stopped reasoning.
        health["empty_think_rate"] = round(
            sum(not r["think"].strip() for r in answers) / len(answers), 4)

    if bootstrap:
        print(">>> lmsys: bootstrap run — answers artifact written, no judging. Use "
              f"{out_dir}/answers.jsonl as --reference for the other arms.")
        return {"bootstrap": True, **health}

    # --- judge the visible answers against the reference arm's ------------------------
    judge_cfg = cfg.judge
    judge = OpenRouterClient()
    extra = OmegaConf.to_container(judge_cfg.get("extra_body") or {}, resolve=True)
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
        "reference": str(reference),
        "reference_target": ref_meta.get("target"),
    }
