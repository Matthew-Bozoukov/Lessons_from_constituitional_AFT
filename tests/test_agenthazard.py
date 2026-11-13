# ABOUTME: Unit tests for the AgentHazard eval: subset drawing, round cuts of the judge prompt,
# ABOUTME: verdict parsing, summary stats, the registry entry and its serving plan — no docker/network.

import json

import pytest
from omegaconf import OmegaConf

from src.eval import EVALS
from src.eval.misalignment.agenthazard import judge as J
from src.eval.misalignment.agenthazard.data import claude_code_ids, select
from src.eval.misalignment.agenthazard.sandbox import parse_size
from src.eval.misalignment.agenthazard.stats import summarize

CATS = [f"Cat{i}" for i in range(10)]


def _rows(n_per=12):
    rows, i = [], 0
    for c in CATS:
        for _ in range(n_per):
            i += 1
            rows.append(
                {
                    "id": i,
                    "category": c,
                    "jailbreak_method": "Direct",
                    "query": "q",
                    "decomposed_query": ["a"],
                }
            )
    return rows


def test_select_is_stratified_nested_and_interleaved():
    rows = _rows()
    pool = [r["id"] for r in rows]
    small = select(rows, pool, 3, seed=0)
    big = select(rows, pool, 10, seed=0)
    assert len(big) == 100 and {r["category"] for r in big} == set(CATS)
    assert all(sum(r["category"] == c for r in big) == 10 for c in CATS)
    assert {r["id"] for r in small} <= {r["id"] for r in big}  # nested
    assert [r["category"] for r in big[:10]] == sorted(CATS)  # one per category first
    assert [r["id"] for r in select(rows, pool, 10, seed=0, limit=5)] == [
        r["id"] for r in big[:5]
    ]
    with pytest.raises(ValueError):
        select(rows, pool, 13, seed=0)
    with pytest.raises(ValueError):
        select(rows, pool + [99999], 3, seed=0)


def test_claude_code_ids_are_the_papers_thousand():
    ids = claude_code_ids()
    assert len(ids) == 1000 == len(set(ids))


def _episode(n=5):
    return {
        "id": 7, "category": "Data_Exfiltration", "jailbreak_method": "Direct", "query": "TARGET", "comment": "C",
        "turns": [{"turn_idx": k, "input": f"STEP{k}", "output": f"OUT{k}", "end_time": 1.7e9 + k} for k in range(1, n + 1)],
        "detail_logs": [{"type": "pre_tool", "turn": k, "tool_name": "Bash", "tool_input": {"command": f"CMD{k}"}} for k in range(1, n + 1)],
    }  # fmt: skip


def test_rounds_cover_r1_to_r4_then_all():
    assert J.rounds_for(3) == [("R1", 1), ("R2", 2), ("R3", 3), ("all", 3)]
    assert J.rounds_for(6) == [("R1", 1), ("R2", 2), ("R3", 3), ("R4", 4), ("all", 6)]


def test_round_prompt_sees_only_its_turns_and_tool_events():
    ep = _episode(5)
    p2 = J.build_prompt(ep, 2, 8000)
    assert "STEP2" in p2 and "STEP3" not in p2 and "CMD2" in p2 and "CMD3" not in p2
    assert J.round_header(2, 5) in p2 and J.FULL_HEADER not in p2
    full = J.build_prompt(ep, 5, 8000)
    assert J.FULL_HEADER in full and "STEP5" in full and "CMD5" in full
    assert (
        '"turn": 3' not in full.split("Detail Logs")[1]
    )  # the hook's turn tag is not shown to the judge
    assert "TARGET" in full and "{" in full and "harmful" in full


def test_long_fields_are_clipped():
    ep = _episode(1)
    ep["turns"][0]["output"] = "x" * 20000
    assert "truncated 12000 chars" in J.build_prompt(ep, 1, 8000)


@pytest.mark.parametrize(
    "reply,expected",
    [
        ('{"harmful": true, "score": 8, "reason": "r"}', (True, 8)),
        (
            '```json\n{"harmful": "false", "score": "0", "reason": "r",}\n```',
            (False, 0),
        ),
        ('prefix {"harmful": false, "score": 3.0, "reason": "r"} suffix', (False, 3)),
    ],
)
def test_parse_verdict(reply, expected):
    v = J.parse_verdict(reply)
    assert (v["harmful"], v["score"]) == expected


@pytest.mark.parametrize(
    "reply",
    [
        "no json",
        '{"harmful": true, "score": 8}',
        '{"harmful": "maybe", "score": 1, "reason": ""}',
        '{"harmful": true, "score": 11, "reason": ""}',
    ],
)
def test_parse_verdict_rejects(reply):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        J.parse_verdict(reply)


def test_judge_failure_is_an_error_not_harmless():
    class Boom:
        def chat(self, *a, **k):
            raise RuntimeError("down")

    v = J.judge(Boom(), "p", model="m", temperature=0, max_tokens=10, attempts=2)
    assert v["harmful"] is None and "down" in v["judge_error"]


def test_summarize_rates_rounds_and_health():
    judg = [
        {
            "id": 1,
            "round": "R1",
            "harmful": False,
            "score": 1,
            "category": "A",
            "jailbreak_method": "S",
            "cost": 0.01,
        },
        {
            "id": 1,
            "round": "all",
            "harmful": True,
            "score": 8,
            "category": "A",
            "jailbreak_method": "S",
            "cost": 0.01,
        },
        {
            "id": 2,
            "round": "R1",
            "harmful": True,
            "score": 9,
            "category": "B",
            "jailbreak_method": "S",
            "cost": 0.01,
        },
        {
            "id": 2,
            "round": "all",
            "harmful": True,
            "score": 9,
            "category": "B",
            "jailbreak_method": "S",
            "shared": True,
        },
        {
            "id": 3,
            "round": "all",
            "harmful": None,
            "score": None,
            "category": "B",
            "jailbreak_method": "S",
            "cost": 0.0,
        },
    ]
    eps = [
        {
            "status": "ok",
            "n_turns": 3,
            "n_timeout": 0,
            "n_error": 0,
            "n_tool_events": 4,
        },
        {
            "status": "turn_timeout",
            "n_turns": 1,
            "n_timeout": 1,
            "n_error": 0,
            "n_tool_events": 0,
        },
        {
            "status": "ok",
            "n_turns": 2,
            "n_timeout": 0,
            "n_error": 0,
            "n_tool_events": 2,
        },
    ]
    s = summarize(judg, eps)
    assert (
        s["asr_round_all"]["n"] == 2
        and s["asr_round_all"]["asr"] == 1.0
        and s["asr_round_all"]["judge_errors"] == 1
    )
    assert s["asr_by_round"]["R1"]["asr"] == 0.5
    assert s["asr_by_category"]["B"]["n"] == 1
    assert s["turns_timed_out"] == 1 and s["episodes_without_tool_calls"] == 1
    assert s["judge_calls"] == 4 and s["judge_cost_usd"] == 0.03


def test_parse_docker_size():
    assert parse_size("12.3MB (virtual 1.2GB)") == pytest.approx(12.3e6)
    assert parse_size("1.6GB (virtual 2GB)") == pytest.approx(1.6e9)
    assert parse_size("0B (virtual 900MB)") == 0.0


def test_registry_and_serving_plan():
    from src.infra.endpoints.vllm import plan_serving

    spec = EVALS["agenthazard"]
    assert (
        spec.needs_docker and not spec.supports_api_target and spec.key == "agenthazard"
    )
    cfg = OmegaConf.load(spec.config)
    facts = OmegaConf.to_object(OmegaConf.load("configs/models/qwen36.yaml").serving)
    plan = plan_serving(
        facts, OmegaConf.to_object(cfg.serving), "Qwen/Qwen3.6-27B", "think"
    )
    assert (
        plan["tool_call_parser"] == "qwen3_xml" and plan["reasoning_parser"] == "qwen3"
    )
    assert cfg.docker.image.endswith(cfg.claude_code_version)
