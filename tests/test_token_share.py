# ABOUTME: The supervised-token share (src/data/mixture/token_share.py): proportional, nested removal;
# ABOUTME: balanced token fill; the swap's realised share; and counts that follow the row's supervise mode.

from __future__ import annotations

import pytest

from src.data.mixture.token_share import fill_tokens, plan_swap, remove_until, removal_order


def _base(spec: dict[str, list[int]]) -> list[dict]:
    """{source: [supervised tokens per row]} -> rows."""
    return [{"source": s, "n_supervised": n} for s, sizes in spec.items() for n in sizes]


def test_removal_is_proportional_across_sources_and_nested_across_shares():
    base = _base({"big": [100] * 400, "small": [100] * 100, "mid": [100] * 200})
    order = removal_order(base, seed=0)
    assert sorted(order) == list(range(len(base)))
    # after the first 70 removals every source has lost ~10% of its rows
    first = order[:70]
    lost = {s: sum(base[i]["source"] == s for i in first) for s in ("big", "small", "mid")}
    assert 36 <= lost["big"] <= 44 and 8 <= lost["small"] <= 12 and 18 <= lost["mid"] <= 22, lost
    # a bigger share removes a superset of a smaller one: same order, longer prefix
    r5 = remove_until(base, order, target=round(70000 * 0.05))
    r7 = remove_until(base, order, target=round(70000 * 0.07))
    assert r7[: len(r5)] == r5 and len(r7) > len(r5)
    # deterministic in the seed, different across seeds
    assert removal_order(base, seed=0) == order and removal_order(base, seed=1) != order


def test_remove_until_reaches_the_target_by_tokens_not_rows():
    base = _base({"a": [10, 500, 10, 10], "b": [300, 20]})
    order = list(range(len(base)))
    removed = remove_until(base, order, target=520)
    assert removed == [0, 1, 2] and sum(base[i]["n_supervised"] for i in removed) == 520


def test_fill_balances_groups_and_closes_the_gap():
    pool = [{"n_supervised": n, "trait": t} for t in ("t1", "t2", "t3") for n in (100, 100, 100, 100, 37)]
    chosen = fill_tokens(pool, budget=650, seed=0, balance_key="trait")
    tokens = sum(pool[i]["n_supervised"] for i in chosen)
    assert 600 <= tokens <= 700 and abs(tokens - 650) <= 50
    per_trait = {t: sum(pool[i]["trait"] == t for i in chosen) for t in ("t1", "t2", "t3")}
    assert max(per_trait.values()) - min(per_trait.values()) <= 1, per_trait
    # the closest-fit pad: 600 from six 100s, then the 37 (gap 50 -> 13) rather than a 100 (overshoot)
    assert tokens == 637
    assert fill_tokens(pool, budget=0, seed=0, balance_key="trait") == []
    assert len(set(chosen)) == len(chosen)


def test_plan_swap_realises_the_declared_token_share_and_keeps_the_rest():
    base = _base({"tulu": [200] * 300, "code": [400] * 100, "chat": [50] * 600})  # 130,000 tokens
    synth = [{"n_supervised": n, "balance_group": f"t{i % 9}"} for i, n in
             enumerate([1100, 1300, 1200, 900, 1000] * 40)]
    plan = plan_swap(base, synth, pct=7, seed=0, balance_key="balance_group")
    assert plan["budget"] == 130_000 and plan["target"] == 9100
    assert plan["removed_tokens"] >= 9100
    assert abs(plan["realised_pct"] - 7) <= 1, plan["realised_pct"]
    # the mixture stays the size of the base, in supervised tokens
    assert abs(plan["total_supervised_tokens"] - 130_000) <= 1300
    assert len(plan["keep"]) + sum(plan["removed_by_source"].values()) == len(base)
    # every source gave up rows
    assert set(plan["removed_by_source"]) == {"tulu", "code", "chat"}
    # 0%: nothing leaves, nothing enters
    zero = plan_swap(base, synth, pct=0, seed=0, balance_key="balance_group")
    assert zero["keep"] == list(range(len(base))) and zero["insert"] == []


# --- the count follows the supervision mode (real tokenizer, skips when not cached) --------

MODEL = "Qwen/Qwen3.6-27B"


@pytest.mark.tokenizer
def test_supervised_tokens_follow_the_rows_supervise_mode():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    except Exception:
        pytest.skip(f"{MODEL} tokenizer not in the local HF cache (tests are no-network)")
    from src.data.mixture.token_share import supervised_tokens
    from src.model_profile import model_profile

    profile = model_profile(MODEL)
    msgs = [{"role": "user", "content": "First question, please."},
            {"role": "assistant", "content": "A first answer that is several words long.",
             "reasoning_content": "thinking about the first one at some length here"},
            {"role": "user", "content": "And a second?"},
            {"role": "assistant", "content": "A second answer.",
             "reasoning_content": "brief thought"}]
    counts = {mode: supervised_tokens(tok, profile, {"messages": msgs, "supervise": mode}, 4096)
              for mode in ("all", "final", "cot", "answer")}
    assert counts["all"] > counts["final"] > 0, counts          # one turn is fewer tokens than two
    assert counts["final"] == counts["cot"] + counts["answer"], counts  # cot + answer partition the final turn
    assert counts["cot"] > 0 and counts["answer"] > 0
    unlabelled = supervised_tokens(tok, profile, {"messages": msgs}, 4096)
    assert unlabelled == counts["all"]                          # no field means every turn
