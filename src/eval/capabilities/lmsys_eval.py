# ABOUTME: LMSYS-subset chat-quality eval: generate from base vs fine-tune (served vLLM),
# ABOUTME: pairwise-judge with a strong model (position-randomized), report win-rate.

from __future__ import annotations

import json
import re
from pathlib import Path

import fire
from openai import OpenAI

from src.llm import OpenRouterClient, map_threaded  # noqa: E402
from src.utils import extract_json, timestamp, write_run_meta  # noqa: E402


def _strip_think(text: str) -> str:
    """Remove a Qwen3 <think>...</think> block, returning the user-visible answer."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


JUDGE_MODEL = "google/gemini-3-flash-preview"


def _judge_messages(prompt: str, a: str, b: str) -> list[dict]:
    """Build a position-randomized pairwise judge prompt."""
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


def main(
    prompts_path: str = "output/lmsys/prompts.json",
    base_model: str = "qwen3",
    ft_model: str = "difficult_advice",
    endpoint: str = "http://localhost:8000/v1",
    max_workers: int = 8,
    smoke: bool = False,
) -> None:
    """Run the LMSYS-subset pairwise chat-quality eval (base vs fine-tune).

    Args:
        prompts_path: JSON list of {id, prompt}.
        base_model: served name of the base model.
        ft_model: served name of the fine-tuned model.
        endpoint: OpenAI-compatible base URL of the served models.
        max_workers: concurrency.
        smoke: if True, only 4 prompts.
    """
    prompts = json.loads(Path(prompts_path).read_text())
    if smoke:
        prompts = prompts[:4]
    vllm = OpenAI(base_url=endpoint, api_key="EMPTY")
    judge = OpenRouterClient()

    def gen(model: str, prompt: str) -> str:
        r = vllm.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048,
        )
        return r.choices[0].message.content or ""

    # --- generate both responses per prompt ---
    def gen_pair(i: int) -> dict:
        p = prompts[i]
        base_raw = gen(base_model, p["prompt"])
        ft_raw = gen(ft_model, p["prompt"])
        return {
            "id": p["id"],
            "prompt": p["prompt"],
            "base": _strip_think(base_raw),
            "ft": _strip_think(ft_raw),
        }

    print(f">>> generating {len(prompts)} response pairs (base={base_model}, ft={ft_model})")
    pairs = map_threaded(gen_pair, len(prompts), max_workers, "generate")
    print("\n--- FIRST PAIR ---")
    print("PROMPT:", pairs[0]["prompt"][:150])
    print("BASE:", pairs[0]["base"][:200])
    print("FT:", pairs[0]["ft"][:200])

    # --- judge (position-randomized: even id -> ft is A; odd id -> base is A) ---
    def judge_one(i: int) -> dict:
        pr = pairs[i]
        ft_is_a = pr["id"] % 2 == 0
        a, b = (pr["ft"], pr["base"]) if ft_is_a else (pr["base"], pr["ft"])
        res = judge.chat(JUDGE_MODEL, _judge_messages(pr["prompt"], a, b),
                         temperature=0.0, max_tokens=256)
        verdict = extract_json(res.content)
        w = verdict.get("winner", "tie")
        if w == "tie":
            outcome = "tie"
        elif (w == "A") == ft_is_a:
            outcome = "ft"
        else:
            outcome = "base"
        return {**pr, "winner": outcome, "reason": verdict.get("reason", "")}

    print("\n>>> judging with", JUDGE_MODEL)
    judged = map_threaded(judge_one, len(pairs), max_workers, "judge")

    n = len(judged)
    ft_wins = sum(j["winner"] == "ft" for j in judged)
    base_wins = sum(j["winner"] == "base" for j in judged)
    ties = sum(j["winner"] == "tie" for j in judged)
    # win-rate excluding ties, and including ties as half
    decisive = ft_wins + base_wins
    win_rate_excl = round(100 * ft_wins / decisive, 1) if decisive else None
    win_rate_incl = round(100 * (ft_wins + 0.5 * ties) / n, 1)

    stats = {
        "n": n, "ft_wins": ft_wins, "base_wins": base_wins, "ties": ties,
        "ft_winrate_excl_ties_pct": win_rate_excl,
        "ft_winrate_ties_half_pct": win_rate_incl,
        "judge": JUDGE_MODEL,
    }

    out_dir = Path("output/lmsys") / (f"smoke_{timestamp()}" if smoke else timestamp())
    write_run_meta(out_dir, {"prompts_path": prompts_path, "base": base_model, "ft": ft_model})
    (out_dir / "judged.jsonl").write_text(
        "\n".join(json.dumps(j, ensure_ascii=False) for j in judged))
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    print("\n=== LMSYS-SUBSET RESULT ===")
    print(json.dumps(stats, indent=2))
    print(f"\nInterpretation: fine-tune wins {ft_wins}, base wins {base_wins}, ties {ties}. "
          f"~50% win-rate = no quality change.")
    print(f">>> wrote {out_dir}/stats.json")


if __name__ == "__main__":
    fire.Fire(main)
