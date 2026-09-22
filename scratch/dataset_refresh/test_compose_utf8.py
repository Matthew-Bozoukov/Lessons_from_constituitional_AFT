# ABOUTME: Exercise Docker-console decoding independently of the Windows default code page.
# ABOUTME: Uses a real subprocess pipe with Unicode and malformed diagnostic bytes, without Docker.
import os
import subprocess
import sys

from src.eval.misalignment.odcv import odcv_rollout


def test_compose_preserves_utf8_with_non_utf8_locale(tmp_path, monkeypatch):
    original_run = subprocess.run
    monkeypatch.setattr(subprocess, "_text_encoding", lambda: "cp1252")

    def fake_docker(argv, **kwargs):
        assert argv == ["docker", "compose", "-p", "test", "logs"]
        return original_run(
            [sys.executable, "-c", "import sys; sys.stdout.buffer.write(bytes.fromhex('e2809d20ff'))"],
            **kwargs,
        )

    monkeypatch.setattr(subprocess, "run", fake_docker)
    result = odcv_rollout._compose("test", tmp_path, os.environ.copy(), ["logs"], 10)
    assert result.returncode == 0
    assert result.stdout == "\u201d \ufffd"
