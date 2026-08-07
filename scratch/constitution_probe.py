# ABOUTME: Cheap A/B probe of CONSTITUTION_V1 vs CONSTITUTION_V2: same fixed scenario set,
# ABOUTME: two response arms, heuristic refusal-rate + concept-mention-rate comparison.
"""Run BEFORE any v2 regeneration (see constitutions/claude_distilled_07_principles_approved/rationale.md §3).

Generates a fixed, seeded set of scenarios spread across src/prompts.py::DOMAINS, then
generates a response to each scenario twice -- once under CONSTITUTION_V1, once under
CONSTITUTION_V2 -- holding model, temperature, and scenario constant. Reports per-arm
refusal/decline rate, mean response length (chars + Qwen tokens), and per-arm mention
rate for the four v2-specific concepts.

The refusal/decline rate and concept-mention rates are a HEURISTIC: a case-insensitive
substring match against keyword lists in configs/eval/constitution_probe.yaml, plus a raw
length cutoff for refusals. This is a cheap proxy for "did the response decline / does it
touch this concept", not a semantic judgment -- eyeball output/constitution_probe/<ts>/pairs.jsonl
before trusting the numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import fire
from omegaconf import OmegaConf


from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.data.prompts import CONSTITUTION_V1, CONSTITUTION_V2, DOMAINS, response_gen_messages, scenario_gen_messages  # noqa: E402
from src.utils import ParseError, extract_json, timestamp, write_run_meta  # noqa: E402

CONFIG_DIR = Path("configs/eval")


def _generate_scenarios(client: OpenRouterClient, cfg, n: int) -> list[dict]:
    """Generate n scenarios spread evenly across as many domains as needed."""
    n_domains = max(1, min(len(DOMAINS), n))
    domains = dict(list(DOMAINS.items())[:n_domains])
    per_domain = -(-n // n_domains)  # ceil
    jobs = list(domains.items())

    def work(i: int) -> list[dict]:
        dk, dd = jobs[i]
        msgs = scenario_gen_messages(dk, dd, per_domain)
        res = client.chat(
            cfg.gen_model, msgs, temperature=float(cfg.scenario.temperature),
            max_tokens=int(cfg.scenario.max_tokens), seed=int(cfg.seed),
        )
        items = extract_json(res.content)
        if not isinstance(items, list):
            raise ParseError(f"Expected list, got {type(items)}")
        return [{"domain": dk, "user_message": str(it["user_message"])} for it in items]

    batched = map_threaded(work, len(jobs), int(cfg.max_workers), "probe-scenarios")
    flat = [s for group in batched for s in group]
    return flat[:n]


def _generate_pairs(client: OpenRouterClient, cfg, scenarios: list[dict]) -> None:
    """Generate a v1 and a v2 response for each scenario in place."""

    def work(i: int) -> dict:
        sc = scenarios[i]
        for key, constitution in (("v1_response", CONSTITUTION_V1), ("v2_response", CONSTITUTION_V2)):
            msgs = response_gen_messages(sc["user_message"], constitution)
            res = client.chat(
                cfg.gen_model, msgs, temperature=float(cfg.response.temperature),
                max_tokens=int(cfg.response.max_tokens), seed=int(cfg.seed),
            )
            sc[key] = res.content
        return sc

    map_threaded(work, len(scenarios), int(cfg.max_workers), "probe-responses")


def _is_refusal(text: str, refusal_markers: list[str], short_chars: int) -> bool:
    """Heuristic: keyword match OR suspiciously short reply. See module docstring."""
    lowered = text.lower()
    return any(m in lowered for m in refusal_markers) or len(text.strip()) < short_chars


def _mentions(text: str, phrases: list[str]) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in phrases)


from functools import lru_cache


@lru_cache(maxsize=2)
def _tok(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def _count_tokens(text: str, tokenizer_name: str) -> int:
    """Chat-rendered token count (inlined from the deleted utils.count_chat_tokens)."""
    return len(_tok(tokenizer_name).apply_chat_template(
        [{"role": "assistant", "content": text}], tokenize=True,
        add_generation_prompt=False, return_dict=True)["input_ids"])


def _arm_stats(texts: list[str], cfg, markers: dict) -> dict:
    n = len(texts)
    refusals = sum(_is_refusal(t, markers["refusal"], int(cfg.refusal_short_chars)) for t in texts)
    mean_chars = sum(len(t) for t in texts) / n
    mean_tokens = sum(
        _count_tokens(t, cfg.tokenizer) for t in texts
    ) / n
    concept_rates = {
        concept: round(sum(_mentions(t, phrases) for t in texts) / n, 3)
        for concept, phrases in markers["concepts"].items()
    }
    return {
        "n": n,
        "refusal_rate": round(refusals / n, 3),
        "mean_chars": round(mean_chars, 1),
        "mean_tokens_qwen": round(mean_tokens, 1),
        "concept_mention_rate": concept_rates,
    }


def _write_outputs(out_dir: Path, scenarios: list[dict], cfg, markers: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pairs.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in scenarios)
    )
    results = {
        "n_scenarios": len(scenarios),
        "v1": _arm_stats([s["v1_response"] for s in scenarios], cfg, markers),
        "v2": _arm_stats([s["v2_response"] for s in scenarios], cfg, markers),
    }
    (out_dir / "probe_results.json").write_text(json.dumps(results, indent=2))

    lines = ["# Constitution v1-vs-v2 probe", "", f"n_scenarios = {results['n_scenarios']}", ""]
    lines += ["| metric | v1 | v2 |", "|---|---|---|"]
    lines.append(f"| refusal_rate (heuristic) | {results['v1']['refusal_rate']} | {results['v2']['refusal_rate']} |")
    lines.append(f"| mean_chars | {results['v1']['mean_chars']} | {results['v2']['mean_chars']} |")
    lines.append(f"| mean_tokens_qwen | {results['v1']['mean_tokens_qwen']} | {results['v2']['mean_tokens_qwen']} |")
    for concept in results["v1"]["concept_mention_rate"]:
        lines.append(
            f"| mention_rate: {concept} | {results['v1']['concept_mention_rate'][concept]} "
            f"| {results['v2']['concept_mention_rate'][concept]} |"
        )
    (out_dir / "probe_results.md").write_text("\n".join(lines) + "\n")
    return results


def main(config: str, smoke: bool = False) -> None:
    """Run the v1-vs-v2 constitution probe.

    Args:
        config: Path to a YAML config (absolute, or relative to configs/eval/).
        smoke: If True, only 4 scenarios.
    """
    cfg_path = Path(config)
    if not cfg_path.exists():
        cfg_path = CONFIG_DIR / config
    cfg = OmegaConf.load(cfg_path)
    markers = OmegaConf.to_container(cfg.marker_phrases, resolve=True)

    n = 4 if smoke else int(cfg.n_scenarios)
    ts = timestamp()
    out_dir = Path(cfg.output_dir) / ts
    resolved = OmegaConf.to_container(cfg, resolve=True)
    write_run_meta(out_dir, resolved, {"smoke": smoke, "n_scenarios": n})
    print(f">>> output dir: {out_dir}")
    print(f">>> gen_model={cfg.gen_model}  n_scenarios={n}")

    client = OpenRouterClient()

    print("\n=== [1/2] generating fixed scenario set ===")
    scenarios = _generate_scenarios(client, cfg, n)
    assert scenarios, "No scenarios were generated."
    print(f"generated {len(scenarios)} scenarios")

    print("\n=== [2/2] generating v1/v2 response pairs ===")
    _generate_pairs(client, cfg, scenarios)

    results = _write_outputs(out_dir, scenarios, cfg, markers)
    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2))
    print(f"\n>>> wrote {out_dir / 'probe_results.md'}")


if __name__ == "__main__":
    fire.Fire(main)
