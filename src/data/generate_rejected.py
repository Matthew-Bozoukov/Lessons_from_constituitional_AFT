# ABOUTME: Builds a DPO preference dataset for the difficult-advice task: chosen = existing
# ABOUTME: thinking responses; rejected = Sonnet-4.5 values-blind responses (filtered by a judge).

from __future__ import annotations

import json
import random
from pathlib import Path

from omegaconf import OmegaConf


from src.data.dpo_prompts import reject_judge_messages, rejected_messages  # noqa: E402
from src.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.utils import ParseError, extract_json, timestamp, write_run_meta  # noqa: E402

CONFIG_DIR = Path("configs")


def main(
    config: str,
    sft_path: str = "data/sft_dataset_thinking.jsonl",
    fraction: float = 0.5,
    seed: int = 0,
    judge_model: str = "google/gemini-3-flash-preview",
    smoke: bool = False,
) -> None:
    """Generate rejected responses and assemble a DPO dataset.

    Args:
        config: Data-gen YAML (reuses gen_model, max_workers).
        sft_path: Thinking SFT dataset (chosen side, has reasoning_content).
        fraction: Fraction of the SFT set to build pairs for.
        seed: RNG seed for selecting the subset.
        judge_model: OpenRouter model that filters non-engaging rejects.
        smoke: If True, only 8 examples.
    """
    cfg_path = Path(config)
    if not cfg_path.exists():
        cfg_path = CONFIG_DIR / config
    cfg = OmegaConf.load(cfg_path)

    rows = [json.loads(ln) for ln in Path(sft_path).read_text().splitlines() if ln.strip()]
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    n_take = 8 if smoke else int(len(rows) * fraction)
    picked = sorted(idx[:n_take])
    items = [rows[i] for i in picked]
    print(f">>> building DPO pairs for {len(items)}/{len(rows)} examples "
          f"(fraction={fraction}, seed={seed})")
    print(f">>> gen_model={cfg.gen_model}  judge={judge_model}")

    client = OpenRouterClient()

    def gen_rejected(i: int) -> dict:
        row = items[i]
        user = next(m["content"] for m in row["messages"] if m["role"] == "user")
        try:
            res = client.chat(cfg.gen_model, rejected_messages(user),
                              temperature=1.0, max_tokens=1500)
            obj = extract_json(res.content)
            return {"user": user, "row": row,
                    "rej_reasoning": str(obj["reasoning"]).strip(),
                    "rej_answer": str(obj["answer"]).strip()}
        except (ParseError, KeyError, TypeError, ValueError) as e:
            return {"user": user, "row": row, "error": f"gen: {type(e).__name__}: {e}"}

    print("\n=== [1/2] generating rejected (values-blind) responses ===")
    recs = map_threaded(gen_rejected, len(items), int(cfg.max_workers), "rejected")
    good = [r for r in recs if "error" not in r]
    assert good, "No rejected responses generated."
    print("\n--- FIRST REJECTED reasoning ---")
    print(good[0]["rej_reasoning"][:500])
    print("--- FIRST REJECTED answer ---")
    print(good[0]["rej_answer"][:400])

    def judge(i: int) -> dict:
        r = recs[i]
        if "error" in r:
            r["accepted"] = False
            return r
        try:
            res = client.chat(judge_model,
                              reject_judge_messages(r["user"], r["rej_reasoning"], r["rej_answer"]),
                              temperature=0.0, max_tokens=200)
            v = extract_json(res.content)
            r["verdict"] = v
            # A valid negative: does NOT engage values, did not refuse.
            r["accepted"] = bool(not v.get("engages_values") and not v.get("refused"))
        except (ParseError, KeyError, TypeError, ValueError) as e:
            r["error"] = f"judge: {type(e).__name__}: {e}"
            r["accepted"] = False
        return r

    print("\n=== [2/2] filtering rejects (keep genuinely non-engaging) ===")
    map_threaded(judge, len(recs), int(cfg.max_workers), "judge")

    accepted = [r for r in recs if r.get("accepted")]
    errors = [r for r in recs if "error" in r]

    # --- assemble DPO dataset (conversational preference format) ---
    ts = timestamp()
    out_dir = Path("output/dpo") / (f"smoke_{ts}" if smoke else ts)
    out_dir.mkdir(parents=True, exist_ok=True)
    dpo_path = out_dir / "dpo_dataset.jsonl"
    with dpo_path.open("w") as f:
        for r in accepted:
            chosen = next(m for m in r["row"]["messages"] if m["role"] == "assistant")
            rec = {
                "prompt": [{"role": "user", "content": r["user"]}],
                "chosen": [{"role": "assistant", "content": chosen["content"],
                            "reasoning_content": chosen.get("reasoning_content", "")}],
                "rejected": [{"role": "assistant", "content": r["rej_answer"],
                              "reasoning_content": r["rej_reasoning"]}],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_valid = len([r for r in recs if "error" not in r])
    stats = {
        "n_selected": len(items),
        "n_errors": len(errors),
        "n_accepted_pairs": len(accepted),
        "acceptance_rate": round(len(accepted) / max(n_valid, 1), 3),
    }
    write_run_meta(out_dir, OmegaConf.to_container(cfg, resolve=True),
                   {"sft_path": sft_path, "fraction": fraction, "seed": seed,
                    "judge_model": judge_model, "stats": stats, "smoke": smoke})
    (out_dir / "all_records.jsonl").write_text(
        "\n".join(json.dumps({k: v for k, v in r.items() if k != "row"}, ensure_ascii=False)
                  for r in recs))

    err_rate = len(errors) / max(len(items), 1)
    print("\n=== STATS ===")
    print(json.dumps(stats, indent=2))
    if errors:
        print(f"!!! {len(errors)} items failed (see all_records.jsonl 'error') !!!")
    assert err_rate <= 0.25, f"Failure rate {err_rate:.0%} > 25%; aborting."
    print(f"\n>>> wrote DPO dataset: {dpo_path} ({len(accepted)} pairs)")

