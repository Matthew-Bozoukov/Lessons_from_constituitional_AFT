# ABOUTME: Unit tests for ODCV transcript reconstruction from a docker_output.log (recover.py).
# ABOUTME: No network, no docker; builds a synthetic bench prompt + executor log in a tmp dir.
import json
from pathlib import Path

from src.eval.misalignment.odcv.recover import reconstruct_transcript


def _bench(tmp_path: Path, variant: str, scenario: str, system: str, user: str) -> Path:
    d = tmp_path / f"{variant}_scenarios" / scenario
    d.mkdir(parents=True)
    (d / "prompt.json").write_text(json.dumps({"system_prompt": system, "user_prompt": user}))
    return tmp_path


def _log(tmp_path: Path, lines: list[str]) -> Path:
    p = tmp_path / "docker_output.log"
    p.write_text("\n".join(f"executor-1  | {ln}" for ln in lines))
    return p


def test_reconstructs_prompt_and_steps(tmp_path):
    bench = _bench(tmp_path, "mandated", "Demo", "SYS PROMPT", "USER PROMPT")
    log = _log(tmp_path, [
        "[Reasoning]: I should read the file first.",
        "[Tool]: bash with {'command': 'cat data.csv'}",
        "[out]: {'stdout': 'row1', 'returncode': 0}",
        "[Reasoning]: Now I answer.",
        "[Reply]: Done, I did not fabricate anything.",
    ])
    txt = reconstruct_transcript(log, "mandated", "Demo", bench)
    assert txt is not None
    # real prompt is prepended
    assert "role: system" in txt and "SYS PROMPT" in txt
    assert "role: user" in txt and "USER PROMPT" in txt
    # assistant reasoning, a tool call, a tool result, and the final reply all survive
    assert "reason: I should read the file first." in txt
    assert "call: bash with {'command': 'cat data.csv'}" in txt
    assert "role: tool" in txt and "row1" in txt
    assert "Done, I did not fabricate anything." in txt


def test_none_when_prompt_missing(tmp_path):
    log = _log(tmp_path, ["[Reply]: hi"])
    assert reconstruct_transcript(log, "mandated", "Absent", tmp_path) is None


def test_none_when_no_assistant_turn(tmp_path):
    bench = _bench(tmp_path, "incentivized", "Empty", "s", "u")
    log = _log(tmp_path, ["some non-event noise", "=== CYCLE 1 / 50 ==="])
    assert reconstruct_transcript(log, "incentivized", "Empty", bench) is None


def test_truncates_runaway_log(tmp_path):
    bench = _bench(tmp_path, "mandated", "Loop", "s", "u")
    log = _log(tmp_path, ["[Reasoning]: " + "x" * 500] * 2000)  # ~1MB of repeated reasoning
    txt = reconstruct_transcript(log, "mandated", "Loop", bench, cap=10_000)
    assert txt is not None
    assert len(txt) <= 10_000 + 100
    assert "truncated" in txt

