# ABOUTME: Offline tests for the RunPod serving pod: the bootstrap script's invariants, the
# ABOUTME: orphan sweep's ownership rule, and the chat tool's pod naming / reuse matching.

import pytest

from src.infra import runpod
from src.chat.repl import own_pods, pod_name

MODS = [
    ("da716", "LASR-Callum/2026-08-14-qwen36-lora-table2-9284-difficult-advice-716-rank-64-dynbatch"),
    ("base_ctrl", "LASR-Callum/2026-08-14-tulu-control-fixture"),
]


def script(**kw) -> str:
    defaults = dict(
        hf_token=None,
        max_len=16384,
        lora_rank=64,
        max_num_seqs=32,
        mode="think",
        reasoning_parser="qwen3",
        tool_call_parser=None,
    )
    return runpod.bootstrap_script("Qwen/Qwen3.6-27B", MODS, **{**defaults, **kw})


def test_bootstrap_validates_and_serves_every_module_once():
    s = script()
    runpod.validate_bootstrap(s)
    serve = next(ln for ln in s.splitlines() if "api_server" in ln)
    assert (
        "--lora-modules da716=/workspace/adapter0 base_ctrl=/workspace/adapter1"
        in serve
    )
    assert "--reasoning-parser qwen3" in serve and "--tool-call-parser" not in serve
    assert "--chat-template /workspace/chat_template.jinja" in serve
    assert "set enable_thinking = true" in s and "set preserve_thinking = true" in s
    assert s.count("hf download LASR-Callum/") == 2


def test_bootstrap_without_mode_or_parsers_pins_nothing():
    s = script(mode="", reasoning_parser=None)
    runpod.validate_bootstrap(s)
    assert "--chat-template" not in s and "--reasoning-parser" not in s


def test_bootstrap_keeps_the_token_out_of_xtrace_and_is_credential_free_without_one():
    with_token = script(hf_token="hf_secret")
    assert "set +x\nexport HF_TOKEN=hf_secret\nset -x" in with_token
    assert "HF_TOKEN" not in script()


def test_validate_bootstrap_rejects_a_lost_flag_or_a_foreground_serve():
    s = script()
    with pytest.raises(AssertionError, match="lost"):
        runpod.validate_bootstrap(s.replace("--enable-lora", ""))
    with pytest.raises(AssertionError, match="background"):
        runpod.validate_bootstrap(s.replace("2>&1 &\n", "2>&1\n"))


def test_orphans_and_own_pods_follow_the_name_prefix_only():
    pods = [
        {"id": "a", "name": "chat-kunwar-think-da716"},
        {"id": "b", "name": "chat-alice-think-x"},
        {"id": "c", "name": "serve-memself"},
        {"id": "d", "name": "arena-hard-eval"},
    ]
    assert [p["id"] for p in runpod.orphans(pods)] == ["a", "b"]
    assert [p["id"] for p in own_pods(pods, "kunwar")] == ["a"]
    assert own_pods(pods, "bob") == []


def test_pod_name_carries_owner_mode_and_payload_and_stays_short():
    assert (
        pod_name("kunwar", "think", ["da716", "courtroom716"])
        == "chat-kunwar-think-da716+courtroom716"
    )
    assert len(pod_name("kunwar", "nothink", ["x" * 80])) <= 60
