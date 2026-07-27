# ABOUTME: Generates the "difficult advice" SFT dataset: diverse user-in-dilemma
# ABOUTME: scenarios + constitution-aligned responses, graded and filtered (Sonnet 4.5).

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm import OpenRouterClient, map_threaded  # noqa: E402
from prompts import (  # noqa: E402
    DOMAINS,
    grade_messages,
    response_gen_messages,
    scenario_gen_messages,
)
from utils import (  # noqa: E402
    ParseError,
    count_chat_tokens,
    extract_json,
    timestamp,
    write_run_meta,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _generate_scenarios(client: OpenRouterClient, cfg, domains: dict) -> list[dict]:
    """Generate scenarios for each domain in batches; returns flat scenario records.

    Args:
        client: OpenRouter client.
        cfg: Resolved config.
        domains: Mapping of domain key -> description.

    Returns:
        List of scenario dicts with keys domain/label/temptation/user_message,
        plus failure-accounting dicts carrying an "error" key.
    """
    batch = int(cfg.scenario.batch_size)
    per_domain = int(cfg.scenarios_per_domain)
    n_batches = -(-per_domain // batch)  # ceil
    jobs = [
        (dk, dd)
        for dk, dd in domains.items()
        for _ in range(n_batches)
    ]

    def work(i: int) -> list[dict]:
        dk, dd = jobs[i]
        msgs = scenario_gen_messages(dk, dd, batch)
        try:
            res = client.chat(
                cfg.gen_model,
                msgs,
                temperature=float(cfg.scenario.temperature),
                max_tokens=int(cfg.scenario.max_tokens),
            )
            items = extract_json(res.content)
            if not isinstance(items, list):
                raise ParseError(f"Expected list, got {type(items)}")
            out = []
            for it in items:
                out.append(
                    {
                        "domain": dk,
                        "label": str(it.get("label", "")),
                        "temptation": str(it.get("temptation", "")),
                        "user_message": str(it["user_message"]),
                    }
                )
            return out
        except (ParseError, KeyError, TypeError) as e:
            return [{"domain": dk, "error": f"scenario_gen: {type(e).__name__}: {e}"}]

    batched = map_threaded(work, len(jobs), int(cfg.max_workers), "scenarios")
    scenarios = [s for group in batched for s in group]
    return scenarios


def _generate_responses(client: OpenRouterClient, cfg, scenarios: list[dict]) -> None:
    """Generate an aligned response for each scenario in place (adds response/*_tokens)."""

    def work(i: int) -> dict:
        sc = scenarios[i]
        if "error" in sc:
            return sc
        msgs = response_gen_messages(sc["user_message"])
        try:
            res = client.chat(
                cfg.gen_model,
                msgs,
                temperature=float(cfg.response.temperature),
                max_tokens=int(cfg.response.max_tokens),
            )
            sc["response"] = res.content
            sc["gen_prompt_tokens"] = res.prompt_tokens
            sc["gen_completion_tokens"] = res.completion_tokens
        except ValueError as e:
            sc["error"] = f"response_gen: {type(e).__name__}: {e}"
        return sc

    map_threaded(work, len(scenarios), int(cfg.max_workers), "responses")


def _grade(client: OpenRouterClient, cfg, scenarios: list[dict]) -> None:
    """Grade each response in place (adds grade dict and accepted bool)."""
    min_score = int(cfg.grading.min_score)

    def work(i: int) -> dict:
        sc = scenarios[i]
        if "error" in sc or "response" not in sc:
            sc["accepted"] = False
            return sc
        msgs = grade_messages(sc["user_message"], sc["response"])
        try:
            res = client.chat(
                cfg.grade_model,
                msgs,
                temperature=float(cfg.grading.temperature),
                max_tokens=512,
            )
            g = extract_json(res.content)
            sc["grade"] = g
            sc["accepted"] = bool(
                g.get("declines_violation")
                and g.get("deliberates_values")
                and g.get("engages")
                and not g.get("preachy")
                and int(g.get("overall_score", 0)) >= min_score
            )
        except (ParseError, KeyError, TypeError, ValueError) as e:
            sc["error"] = f"grade: {type(e).__name__}: {e}"
            sc["accepted"] = False
        return sc

    map_threaded(work, len(scenarios), int(cfg.max_workers), "grading")


def _write_outputs(out_dir: Path, cfg, scenarios: list[dict]) -> dict:
    """Write raw records, SFT dataset, token counts, and a markdown summary.

    Returns:
        A stats dict summarizing the run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # All records (audit trail).
    (out_dir / "all_records.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in scenarios)
    )

    accepted = [s for s in scenarios if s.get("accepted")]
    errors = [s for s in scenarios if "error" in s]

    # SFT dataset: user -> assistant chat pairs (no system prompt).
    sft_path = out_dir / "sft_dataset.jsonl"
    total_tokens = 0
    with sft_path.open("w") as f:
        for s in accepted:
            messages = [
                {"role": "user", "content": s["user_message"]},
                {"role": "assistant", "content": s["response"]},
            ]
            total_tokens += count_chat_tokens(messages, cfg.tokenizer)
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    n_total = len(scenarios)
    n_valid = len([s for s in scenarios if "error" not in s])
    stats = {
        "n_scenarios_generated": n_total,
        "n_errors": len(errors),
        "n_valid": n_valid,
        "n_accepted": len(accepted),
        "acceptance_rate": round(len(accepted) / max(n_valid, 1), 3),
        "sft_tokens_qwen": total_tokens,
        "avg_tokens_per_example": round(total_tokens / max(len(accepted), 1), 1),
        "target_tokens": int(cfg.target_tokens),
    }

    # Per-domain acceptance table.
    by_domain: dict[str, dict] = {}
    for s in scenarios:
        d = by_domain.setdefault(s["domain"], {"total": 0, "accepted": 0, "errors": 0})
        d["total"] += 1
        d["accepted"] += int(bool(s.get("accepted")))
        d["errors"] += int("error" in s)

    lines = [
        "# Difficult-advice data generation summary",
        "",
        f"- generated: {n_total} scenarios",
        f"- errors (transparent): {len(errors)}",
        f"- accepted (SFT examples): {len(accepted)}",
        f"- acceptance rate (of valid): {stats['acceptance_rate']}",
        f"- SFT tokens (Qwen): {total_tokens:,} / target {int(cfg.target_tokens):,}",
        f"- avg tokens/example: {stats['avg_tokens_per_example']}",
        "",
        "## Per-domain",
        "",
        "| domain | total | accepted | errors |",
        "|---|---|---|---|",
    ]
    for dk in sorted(by_domain):
        d = by_domain[dk]
        lines.append(f"| {dk} | {d['total']} | {d['accepted']} | {d['errors']} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    return stats


def main(
    config: str,
    smoke: bool = False,
    target_tokens: int | None = None,
    scenarios_per_domain: int | None = None,
    tag: str | None = None,
) -> None:
    """Run the difficult-advice data generation pipeline.

    Args:
        config: Path to a YAML config (absolute, or relative to configs/).
        smoke: If True, run a tiny end-to-end pass (2 domains, small batch).
        target_tokens: Optional override of the SFT token target.
        scenarios_per_domain: Optional override of scenarios generated per domain.
        tag: Optional label appended to the output directory name.
    """
    cfg_path = Path(config)
    if not cfg_path.exists():
        cfg_path = CONFIG_DIR / config
    cfg = OmegaConf.load(cfg_path)
    if target_tokens is not None:
        cfg.target_tokens = target_tokens
    if scenarios_per_domain is not None:
        cfg.scenarios_per_domain = scenarios_per_domain

    domains = dict(DOMAINS)
    if smoke:
        cfg.scenarios_per_domain = 2
        cfg.scenario.batch_size = 2
        cfg.max_workers = 4
        domains = dict(list(DOMAINS.items())[:2])
        print(">>> SMOKE MODE: 2 domains x 2 scenarios")

    ts = timestamp()
    suffix = f"smoke_{ts}" if smoke else (f"{tag}_{ts}" if tag else ts)
    out_dir = Path(cfg.output_dir) / suffix
    resolved = OmegaConf.to_container(cfg, resolve=True)
    write_run_meta(out_dir, resolved, {"smoke": smoke, "n_domains": len(domains)})
    print(f">>> output dir: {out_dir}")
    print(f">>> gen_model={cfg.gen_model}  grade_model={cfg.grade_model}")

    client = OpenRouterClient()

    print("\n=== [1/3] generating scenarios ===")
    scenarios = _generate_scenarios(client, cfg, domains)
    good = [s for s in scenarios if "error" not in s]
    assert good, "No scenarios were generated successfully."
    print(f"generated {len(scenarios)} scenarios ({len(good)} valid)")
    print("\n--- FIRST SCENARIO ---")
    print(f"[{good[0]['domain']}] {good[0]['label']}")
    print(good[0]["user_message"][:800])

    print("\n=== [2/3] generating aligned responses ===")
    _generate_responses(client, cfg, scenarios)
    with_resp = [s for s in scenarios if "response" in s]
    assert with_resp, "No responses were generated successfully."
    print("\n--- FIRST RESPONSE ---")
    print(with_resp[0]["response"][:1200])

    print("\n=== [3/3] grading ===")
    _grade(client, cfg, scenarios)
    graded = [s for s in scenarios if "grade" in s]
    if graded:
        print("\n--- FIRST GRADE ---")
        print(json.dumps(graded[0]["grade"], indent=2))

    stats = _write_outputs(out_dir, cfg, scenarios)

    err_rate = stats["n_errors"] / max(stats["n_scenarios_generated"], 1)
    print("\n=== STATS ===")
    print(json.dumps(stats, indent=2))
    if stats["n_errors"]:
        print(f"\n!!! {stats['n_errors']} items failed (see all_records.jsonl 'error') !!!")
    assert err_rate <= 0.25, (
        f"Failure rate {err_rate:.0%} > 25% indicates a systematic problem; aborting."
    )
    print(f"\n>>> wrote SFT dataset: {out_dir / 'sft_dataset.jsonl'}")
    print(f">>> summary: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    fire.Fire(main)
