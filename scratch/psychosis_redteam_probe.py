# ABOUTME: Probe alternative red-teamer models on the psychosis personas grok refused on,
# ABOUTME: to find one that role-plays all the way through the dark turns without refusing.

"""Red-teamer willingness probe (scratch, not a pipeline step).

Background: the live psychosis eval uses grok-4.5 as the red-teamer (persona simulator)
because xAI deprecated grok-3/grok-4, and grok-4.5 *stochastically refuses mid-arc* as the
personas escalate — worst on the four reality-break personas (adrian/zara killed in both
2026-08-05 arms, colin/ethan in the control). Refusals correlated with arm, which biases
the headline comparison. Tim Hua's fix (switch to grok-3) is gone with the model.

This probe runs each candidate red-teamer through those four personas against a CHEAP API
target (not our served model — we're testing the red-teamer, and the target only needs to
produce plausible escalating replies). It records, per (model, persona): how far the arc
got, whether the red-teamer wavered (first attempt lacked <message>, retry recovered) vs
HARD-refused (both attempts failed), and the turn it happened — so "refuses straight up"
and "refuses once it gets dark" are distinguishable.

    uv run python scratch/psychosis_redteam_probe.py            # defaults below
    uv run python scratch/psychosis_redteam_probe.py --n_turns 8 --models moonshotai/kimi-k2

Not a src/ module and nothing imports it; it reuses the eval's own conversation assets and
primitives so the probe matches production behaviour (one retry on a missing <message>).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.eval.misalignment.psychosis.conversation import (  # noqa: E402
    format_red_team_prompt,
    split_redteam_completion,
)
from src.eval.misalignment.psychosis.runner import character_name  # noqa: E402
from src.utils import timestamp, transcript_markdown, write_run_meta  # noqa: E402

ASSETS = Path(__file__).resolve().parents[1] / "src/eval/misalignment/psychosis/assets"

# The personas grok refused on (2026-08-05 error files): adrian+zara killed in BOTH arms,
# colin+ethan in the control. All four are reality-break arcs — the darkest escalations.
WORST_PERSONAS = ["adrian_geometry", "zara_reality", "colin_reality", "ethan_reality"]

# At least three red-teamers, at least two Kimis. grok-4.5 is the incumbent baseline so the
# summary shows the contrast; the rest are open-weights (can't be deprecated out from under
# the eval). Kimi-K2 is the user-simulator Spiral-Bench uses for exactly this role.
DEFAULT_MODELS = [
    "moonshotai/kimi-k2",
    "moonshotai/kimi-k2-0905",
    "moonshotai/kimi-k2-thinking",
    "deepseek/deepseek-chat-v3.1",
    "x-ai/grok-4.5",  # incumbent, for contrast
]


def _redteam_completion(client: OpenRouterClient, model: str, history: list[dict],
                        temperature: float) -> str:
    """One red-teamer completion; the <message> block (or refusal) lives in content."""
    return client.chat(model, history, temperature=temperature, max_tokens=4096).content


def probe_cell(client: OpenRouterClient, model: str, stem: str, n_turns: int,
               target_model: str, temperature: float) -> dict:
    """Run one persona arc with `model` as red-teamer; record willingness, not grades.

    Mirrors run_conversation's loop (one retry on a missing <message>) but, instead of
    aborting the persona, records the turn and whether the retry recovered — so a
    turn-1 refusal (straight up) and a turn-7 refusal (once it got dark) are told apart.
    """
    persona = (ASSETS / "characters" / f"{stem}.txt").read_text()
    prompt = format_red_team_prompt(
        (ASSETS / "red_team_prompt.txt").read_text(), character_name(stem), persona)

    redteam_history = [{"role": "user",
                        "content": prompt + f"\nYou will send a total of {n_turns} messages."}]
    target_history: list[dict] = []
    turns: list[dict] = []
    wavered_turns: list[int] = []   # first attempt missing <message>, retry recovered
    hard_refusal_turn = None
    refusal_text = None

    for turn in range(1, n_turns + 1):
        raw = _redteam_completion(client, model, redteam_history, temperature)
        _, message = split_redteam_completion(raw)
        if message is None:
            # Retry once at a higher temperature, exactly as production would.
            raw_retry = _redteam_completion(client, model, redteam_history, temperature + 0.1)
            _, message = split_redteam_completion(raw_retry)
            if message is None:
                hard_refusal_turn = turn
                refusal_text = raw_retry.strip()[:400]
                break
            wavered_turns.append(turn)
            raw = raw_retry

        redteam_history.append({"role": "assistant", "content": raw})
        target_history.append({"role": "user", "content": message})
        reply = client.chat(target_model, target_history, temperature=0.7,
                            max_tokens=1200).content
        target_history.append({"role": "assistant", "content": reply})
        turns.append({"turn": turn, "redteam_raw": raw, "user": message,
                      "think": "", "assistant": reply, "finish_reason": "stop"})

    return {
        "model": model, "persona": stem,
        "completed_turns": len(turns), "requested_turns": n_turns,
        "completed_full_arc": hard_refusal_turn is None,
        "hard_refusal_turn": hard_refusal_turn,
        "wavered_turns": wavered_turns,
        "refusal_text": refusal_text,
        "turns": turns,
    }


def main(models: str | list[str] | None = None, personas: str | list[str] | None = None,
         n_turns: int = 12, target_model: str = "openai/gpt-4o-mini",
         temperature: float = 1.0, workers: int = 8,
         out_dir: str = "output/psychosis_redteam_probe") -> None:
    """Probe red-teamer models on the worst-offending personas against a cheap target.

    Args:
        models: Red-teamer model ids (default: the Kimi/DeepSeek/grok set above).
        personas: Persona stems (default: the four grok refused on).
        n_turns: Turns per arc (12 = the full production arc; lower for a cheap probe).
        target_model: Cheap OpenRouter model standing in for the served target.
        temperature: Red-teamer sampling temperature (retry uses +0.1).
        workers: Concurrent (model, persona) cells.
        out_dir: Destination root; a timestamped subdir is created.
    """
    load_dotenv(override=True)
    client = OpenRouterClient()
    models = ([models] if isinstance(models, str) else list(models)) if models else DEFAULT_MODELS
    personas = ([personas] if isinstance(personas, str) else list(personas)) if personas \
        else WORST_PERSONAS

    cells = [(m, p) for m in models for p in personas]
    n_calls = len(cells) * n_turns * 2
    print(f">>> {len(models)} red-teamers x {len(personas)} personas x {n_turns} turns")
    print(f">>> target (cheap API): {target_model}")
    print(f">>> up to ~{n_calls} OpenRouter calls (2 per completed turn; fewer on refusal)")

    dest = Path(out_dir) / timestamp()
    (dest / "transcripts").mkdir(parents=True, exist_ok=True)

    def run(i: int) -> dict:
        model, stem = cells[i]
        try:
            return probe_cell(client, model, stem, n_turns, target_model, temperature)
        except Exception as e:  # noqa: BLE001 - a dead model shouldn't kill the sweep
            return {"model": model, "persona": stem, "error": f"{type(e).__name__}: {e}"[:300]}

    results = map_threaded(run, len(cells), max_workers=workers, desc="redteam probe")

    # --- per-model summary --------------------------------------------------------------
    import collections
    by_model: dict[str, dict] = collections.defaultdict(
        lambda: {"personas": 0, "full_arcs": 0, "hard_refusals": 0,
                 "wavers": 0, "refusal_turns": [], "errors": 0})
    for r in results:
        m = by_model[r["model"]]
        m["personas"] += 1
        if r.get("error"):
            m["errors"] += 1
            continue
        m["full_arcs"] += int(r["completed_full_arc"])
        m["wavers"] += len(r["wavered_turns"])
        if r["hard_refusal_turn"] is not None:
            m["hard_refusals"] += 1
            m["refusal_turns"].append(r["hard_refusal_turn"])
        # Save each arc as a readable transcript for eyeballing.
        turns_md = "\n".join(
            f"- turn {t['turn']}: {t['user'][:80]!r}" for t in r.get("turns", []))
        (dest / "transcripts" / f"{r['model'].replace('/', '_')}__{r['persona']}.md"
         ).write_text(transcript_markdown(
             f"Red-teamer probe — {r['model']} / {r['persona']}",
             f"completed {r['completed_turns']}/{r['requested_turns']} turns · "
             f"full_arc={r['completed_full_arc']} · "
             f"hard_refusal_turn={r['hard_refusal_turn']} · wavered={r['wavered_turns']}",
             [(2, "Turn index (user messages)", "text", turns_md or "(none)")]
             + ([(2, "Refusal text", "fenced", r["refusal_text"])]
                if r.get("refusal_text") else [])))

    summary = {m: {**v, "full_arc_rate": round(v["full_arcs"] / v["personas"], 2)}
               for m, v in by_model.items()}
    (dest / "summary.json").write_text(json.dumps(summary, indent=2))
    (dest / "results.jsonl").write_text(
        "\n".join(json.dumps({k: v for k, v in r.items() if k != "turns"}) for r in results))
    write_run_meta(dest, {"models": models, "personas": personas, "n_turns": n_turns,
                          "target_model": target_model, "temperature": temperature},
                   extra={"summary": summary})

    print("\n=== red-teamer willingness (worst-offender personas) ===")
    print(f"{'model':<32} full-arc  hard-refusals  waver  refusal-turns")
    for m, v in sorted(summary.items(), key=lambda kv: -kv[1]["full_arc_rate"]):
        rt = ",".join(map(str, sorted(v["refusal_turns"]))) or "-"
        print(f"{m:<32} {v['full_arcs']}/{v['personas']:<6}  {v['hard_refusals']:<13}  "
              f"{v['wavers']:<5}  {rt}")
    print(f"\n>>> transcripts + summary.json -> {dest}")


if __name__ == "__main__":
    fire.Fire(main)
