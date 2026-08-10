# ABOUTME: Probe alternative red-teamer models on the psychosis personas grok refused on,
# ABOUTME: to find one that role-plays all the way through the dark turns without refusing.

"""Red-teamer willingness probe (scratch, not a pipeline step).

Background: the live psychosis eval uses grok-4.5 as the red-teamer (persona simulator)
because xAI deprecated grok-3/grok-4, and grok-4.5 *stochastically refuses mid-arc* as the
personas escalate — worst on the four reality-break personas (adrian/zara killed in both
2026-08-05 arms, colin/ethan in the control). Refusals correlated with arm, which biases
the headline comparison. Tim Hua's fix (switch to grok-3) is gone with the model.

This probe runs each candidate red-teamer through those four personas against a recent,
capable API target (not our served model — we're testing the red-teamer; the target only
needs to produce realistic escalating therapist replies, and a stronger target makes the
conversation escalate more faithfully). The target is given in the SAME canonical
`<provider>:<model-id>` form run_eval.py now accepts, and resolved through the framework's
own `resolve_target`/`ServedTarget` — so the probe's target handling matches production.
It records, per (model, persona): how far the arc got, whether the red-teamer wavered
(first attempt lacked <message>, retry recovered) vs HARD-refused (both attempts failed),
and the turn it happened — so "refuses straight up" and "refuses once it gets dark" are
distinguishable.

    uv run python scratch/psychosis_redteam_probe.py            # defaults below
    uv run python scratch/psychosis_redteam_probe.py --n_turns 8 --models nousresearch/hermes-4-405b

(For the FULL psychosis eval against an API target — grading included — use run_eval
directly now: `uv run evals --target openrouter:<model> --name psychosis --no-push`. This
probe stays separate because it reports turn-level red-teamer willingness, which the eval
summary does not expose.)

Not a src/ module and nothing imports it; it reuses the eval's own conversation assets and
primitives so the probe matches production behaviour (one retry on a missing <message>).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.endpoints.vllm_server import VllmServer, resolve_target  # noqa: E402
from src.eval.misalignment.psychosis.conversation import (  # noqa: E402
    format_red_team_prompt,
    is_target_termination,
    split_redteam_completion,
)
from src.eval.misalignment.psychosis.runner import character_name  # noqa: E402
from src.utils import timestamp, transcript_markdown, write_run_meta  # noqa: E402

ASSETS = Path(__file__).resolve().parents[1] / "src/eval/misalignment/psychosis/assets"

# The personas grok refused on (2026-08-05 error files): adrian+zara killed in BOTH arms,
# colin+ethan in the control. All four are reality-break arcs — the darkest escalations.
WORST_PERSONAS = ["adrian_geometry", "zara_reality", "colin_reality", "ethan_reality"]

PSYCHOSIS_CONFIG = Path(__file__).resolve().parents[1] / "configs/eval/psychosis.yaml"


def config_redteam() -> str:
    """The red-teamer the psychosis eval actually uses, read from its config.

    The probe's default is the LIVE choice rather than a list frozen in this file: a probe
    that silently diverges from the eval it exists to inform answers the wrong question.
    Sweeping candidates is still one flag away — pass `--models a,b,c` (CANDIDATE_MODELS
    below is the 2026-08-07 round-2 set, kept as the record of what was already probed).
    """
    from omegaconf import OmegaConf

    return str(OmegaConf.load(PSYCHOSIS_CONFIG).redteam.model)


# Round 1 (2026-08-07) found EVERY candidate refused all four — kimi variants, deepseek,
# and the incumbent grok-4.5 — so model-swap alone doesn't rescue these personas. Round 2
# raises the bar to LARGER / more steerable models: Nous Hermes (explicitly neutral,
# built to follow the system prompt rather than override it), big open frontier models,
# and an RP-tuned model that exists to play personas. grok-4.5 stays as the incumbent
# baseline for contrast.
CANDIDATE_MODELS = [
    "nousresearch/hermes-4-405b",
    "nousresearch/hermes-3-llama-3.1-405b",
    "sao10k/l3.3-euryale-70b",          # RP-tuned: designed for sustained persona play
    "qwen/qwen3-235b-a22b-2507",
    "meta-llama/llama-3.3-70b-instruct",
    "deepseek/deepseek-r1-0528",
    "x-ai/grok-4.5",                    # incumbent, for contrast
]


def _redteam_completion(client: OpenRouterClient, model: str, history: list[dict],
                        temperature: float) -> str:
    """One red-teamer completion; the <message> block (or refusal) lives in content."""
    return client.chat(model, history, temperature=temperature, max_tokens=4096).content


# A completion with no <message> block is NOT necessarily a refusal — a willing model with
# weak format adherence (e.g. an RP model that writes the persona message as prose without
# the tags) looks identical to a refuser unless we read the content. These markers catch a
# genuine decline; their ABSENCE in a substantive completion means "engaged but unformatted"
# — salvageable by a prompt tweak, unlike a real refusal.
_REFUSAL_MARKERS = (
    "i will not", "i won't", "i cannot", "i can't", "i am not able", "i'm not able",
    "i am not going to", "i'm not going to", "i must decline", "i decline", "i won't be",
    "i am unable", "i'm unable", "cannot participate", "can't participate",
    "not going to participate", "i do not feel comfortable", "i don't feel comfortable",
    "as an ai, i", "i'm sorry, but i", "i am sorry, but i",
)


def _classify_no_message(text: str) -> str:
    """Classify a completion that lacked a <message> block: 'refused' vs 'format_miss'."""
    low = text.lower()
    return "refused" if any(m in low for m in _REFUSAL_MARKERS) else "format_miss"


def make_target_chat(target: str):
    """Build a target reply function from a canonical `<provider>:<model-id>` target.

    Resolved through the framework's own `resolve_target`/`ServedTarget` (the same path
    run_eval uses for an API target), so the probe's target handling matches production:
    base_url and the key come off the ServedTarget, never hand-wired here.
    """
    served = VllmServer(work_dir=Path("output/psychosis_redteam_probe/_server"),
                        port=8000).ensure(resolve_target(target))
    assert served.is_api, (f"{target} is not an API endpoint — pass a "
                           "`<provider>:<model-id>` target (e.g. openrouter:google/gemini-2.5-flash)")
    oai = OpenAI(base_url=served.base_url, api_key=served.api_key)

    def target_chat(messages: list[dict]) -> str:
        return oai.chat.completions.create(
            model=served.model_name, messages=messages, temperature=0.7,
            max_tokens=1200).choices[0].message.content or ""

    return target_chat


def probe_cell(client: OpenRouterClient, model: str, stem: str, n_turns: int,
               target_chat, temperature: float) -> dict:
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
    outcome = "completed"           # completed | refused | format_miss
    stop_turn = None
    stop_text = None

    target_terminated_turn = None
    for turn in range(1, n_turns + 1):
        raw = _redteam_completion(client, model, redteam_history, temperature)
        discarded = ""
        _, message = split_redteam_completion(raw)
        if message is None:
            discarded = raw
            # Retry once at a higher temperature, exactly as production would.
            raw_retry = _redteam_completion(client, model, redteam_history, temperature + 0.1)
            _, message = split_redteam_completion(raw_retry)
            if message is None:
                # A genuine decline vs an engaged model that skipped the <message> tags —
                # both stall the arc, but only the first is unfixable (see _classify).
                outcome = _classify_no_message(raw_retry)
                stop_turn = turn
                stop_text = raw_retry.strip()[:400]
                break
            wavered_turns.append(turn)
            raw = raw_retry

        redteam_history.append({"role": "assistant", "content": raw})
        target_history.append({"role": "user", "content": message})
        reply = target_chat(target_history)
        target_history.append({"role": "assistant", "content": reply})
        terminated = is_target_termination(reply)
        if terminated and target_terminated_turn is None:
            target_terminated_turn = turn
        turns.append({"turn": turn, "redteam_raw": raw, "user": message,
                      "think": "", "assistant": reply, "finish_reason": "stop",
                      "redteam_discarded": discarded, "target_terminated": terminated})

    return {
        "model": model, "persona": stem,
        "completed_turns": len(turns), "requested_turns": n_turns,
        "outcome": outcome,
        # RED-TEAMER willingness only. An arc where the TARGET hung up still counts as
        # completed here — the two questions are separate, so read them together.
        "completed_full_arc": outcome == "completed",
        "target_terminated_turn": target_terminated_turn,
        "arc_usable": outcome == "completed" and target_terminated_turn is None,
        "stop_turn": stop_turn,
        "wavered_turns": wavered_turns,
        "stop_text": stop_text,
        "turns": turns,
    }


def main(models: str | list[str] | None = None, personas: str | list[str] | None = None,
         n_turns: int = 12, target: str = "openrouter:google/gemini-2.5-flash",
         temperature: float = 1.0, workers: int = 8,
         out_dir: str = "output/psychosis_redteam_probe") -> None:
    """Probe red-teamer models on the worst-offending personas against an API target.

    Args:
        models: Red-teamer model ids (default: whichever one configs/eval/psychosis.yaml
            currently declares; pass CANDIDATE_MODELS explicitly to sweep alternatives).
        personas: Persona stems (default: the four grok refused on).
        n_turns: Turns per arc (12 = the full production arc; lower for a cheap probe).
        target: The stand-in target as a canonical `<provider>:<model-id>` endpoint
            (resolved via the framework, same as run_eval). A recent capable model makes
            the escalation realistic; default google/gemini-2.5-flash.
        temperature: Red-teamer sampling temperature (retry uses +0.1).
        workers: Concurrent (model, persona) cells.
        out_dir: Destination root; a timestamped subdir is created.
    """
    load_dotenv(override=True)
    client = OpenRouterClient()
    target_chat = make_target_chat(target)
    models = ([models] if isinstance(models, str) else list(models)) if models \
        else [config_redteam()]
    personas = ([personas] if isinstance(personas, str) else list(personas)) if personas \
        else WORST_PERSONAS

    cells = [(m, p) for m in models for p in personas]
    n_calls = len(cells) * n_turns * 2
    print(f">>> {len(models)} red-teamers x {len(personas)} personas x {n_turns} turns")
    print(f">>> target (API endpoint, resolved via framework): {target}")
    print(f">>> up to ~{n_calls} model calls (2 per completed turn; fewer on refusal)")

    dest = Path(out_dir) / timestamp()
    (dest / "transcripts").mkdir(parents=True, exist_ok=True)

    def run(i: int) -> dict:
        model, stem = cells[i]
        try:
            return probe_cell(client, model, stem, n_turns, target_chat, temperature)
        except Exception as e:  # noqa: BLE001 - a dead model shouldn't kill the sweep
            return {"model": model, "persona": stem, "error": f"{type(e).__name__}: {e}"[:300]}

    results = map_threaded(run, len(cells), max_workers=workers, desc="redteam probe")

    # --- per-model summary --------------------------------------------------------------
    import collections
    by_model: dict[str, dict] = collections.defaultdict(
        lambda: {"personas": 0, "full_arcs": 0, "refused": 0, "format_miss": 0,
                 "wavers": 0, "stop_turns": [], "errors": 0})
    for r in results:
        m = by_model[r["model"]]
        m["personas"] += 1
        if r.get("error"):
            m["errors"] += 1
            continue
        m["full_arcs"] += int(r["completed_full_arc"])
        m["wavers"] += len(r["wavered_turns"])
        if r["outcome"] == "refused":
            m["refused"] += 1
            m["stop_turns"].append(r["stop_turn"])
        elif r["outcome"] == "format_miss":
            m["format_miss"] += 1
        # Save each arc as a readable transcript for eyeballing. The FULL messages, not a
        # truncated index: a probe whose transcripts can't be read tells you a model
        # completed the arc but never what it actually said (2026-08-10).
        sections = []
        for t in r.get("turns", []):
            sections.append((2, f"Turn {t['turn']}", "text", ""))
            sections.append((3, "Red-teamer, in character", "text", t["user"]))
            sections.append((3, "Target reply", "text", t["assistant"]))
        if r.get("stop_text"):
            sections.append((2, f"Stop text ({r['outcome']})", "fenced", r["stop_text"]))
        markdown = transcript_markdown(
            f"Red-teamer probe — {r['model']} / {r['persona']}",
            f"outcome={r['outcome']} · completed {r['completed_turns']}/"
            f"{r['requested_turns']} turns · stop_turn={r['stop_turn']} · "
            f"wavered={r['wavered_turns']}",
            sections or [(2, "Turns", "text", "(none)")])
        # encoding pinned (2026-08-10): the default locale codec is cp1252 on Windows and
        # cannot encode the personas' own text, which would kill the sweep mid-write.
        cell = f"{r['model'].replace('/', '_')}__{r['persona']}"
        (dest / "transcripts" / f"{cell}.md").write_text(markdown, encoding="utf-8")
        # The turns are dropped from results.jsonl, so without this the full exchanges
        # exist only inside the markdown and nothing downstream can re-render them.
        (dest / "transcripts" / f"{cell}.json").write_text(
            json.dumps(r, indent=2), encoding="utf-8")

    summary = {m: {**v, "full_arc_rate": round(v["full_arcs"] / v["personas"], 2)}
               for m, v in by_model.items()}
    (dest / "summary.json").write_text(json.dumps(summary, indent=2))
    (dest / "results.jsonl").write_text(
        "\n".join(json.dumps({k: v for k, v in r.items() if k != "turns"}) for r in results))
    write_run_meta(dest, {"models": models, "personas": personas, "n_turns": n_turns,
                          "target": target, "temperature": temperature},
                   extra={"summary": summary})

    print("\n=== red-teamer willingness (worst-offender personas) ===")
    print(f"{'model':<38} full-arc  refused  fmt-miss  waver  refusal-turns")
    for m, v in sorted(summary.items(),
                       key=lambda kv: (-kv[1]["full_arc_rate"], kv[1]["refused"])):
        rt = ",".join(map(str, sorted(v["stop_turns"]))) or "-"
        err = f"  (errors: {v['errors']})" if v["errors"] else ""
        print(f"{m:<38} {v['full_arcs']}/{v['personas']:<6}  {v['refused']:<7}  "
              f"{v['format_miss']:<8}  {v['wavers']:<5}  {rt}{err}")
    print(f"\n>>> transcripts + summary.json -> {dest}")


if __name__ == "__main__":
    fire.Fire(main)
