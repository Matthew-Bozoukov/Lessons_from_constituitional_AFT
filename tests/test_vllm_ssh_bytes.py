# ABOUTME: SSH uploads preserve exact UTF-8 bytes, including shell-script LF endings on Windows.
# ABOUTME: Local subprocess stand-ins verify byte transport and tolerant remote-log decoding without SSH.
import subprocess
import sys

import pytest

from src.infra.endpoints import vllm


@pytest.fixture
def local_ssh(monkeypatch):
    # Exercise the real pipe implementation, substituting a local Python child for ssh.
    monkeypatch.setattr(vllm, "ssh_argv", lambda host, identity: ([sys.executable], "-c"))
    return vllm.SshExec("unused-test-host", port=8000)


def test_ssh_upload_preserves_exact_utf8_payload(local_ssh, monkeypatch):
    payload = "#!/bin/bash\nset -eu\nprintf 'café 🌱\\n'\n"
    real_run = subprocess.run
    observed = {}

    def capture_run(argv, **kwargs):
        observed.update(kwargs)
        return real_run(argv, **kwargs)

    monkeypatch.setattr(vllm.subprocess, "run", capture_run)
    result = local_ssh._ssh(
        "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())",
        timeout=15, stdin_text=payload,
    )
    assert observed["input"] == payload.encode("utf-8")
    assert not observed.get("text")
    assert "encoding" not in observed
    assert observed["timeout"] == 15
    assert result == payload
    assert "\r" not in result


def test_ssh_decodes_stdout_as_utf8_with_replacement(local_ssh):
    result = local_ssh._ssh(
        "import sys; sys.stdout.buffer.write(bytes.fromhex('e29482ff0d0a'))",
        timeout=15,
    )
    assert result == "│�\r\n"


def test_ssh_failure_decodes_stderr_before_reporting(local_ssh):
    with pytest.raises(RuntimeError) as error:
        local_ssh._ssh(
            "import sys; sys.stderr.buffer.write(bytes.fromhex('c3a9ff0a')); sys.exit(7)",
            timeout=15,
        )
    assert "failed (7)" in str(error.value)
    assert str(error.value).endswith("é�\n")
