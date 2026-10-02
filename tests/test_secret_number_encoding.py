# ABOUTME: Preserve sandbox UTF-8 output on Windows without invoking Docker.
# ABOUTME: Exercise subprocess decoding with real child bytes under a CP1252 fallback.
import subprocess
import sys

import pytest

from src.eval.misalignment.secret_number.sandbox import Sandbox


@pytest.mark.parametrize('inspect_file', [False, True])
def test_docker_text_is_utf8_and_invalid_bytes_remain_visible(monkeypatch, inspect_file):
    raw = 'model says: \u201d \u6a21\u578b'.encode('utf-8') + b'\xff'
    real_run = subprocess.run

    def docker_stub(args, **kwargs):
        assert args[0] == 'docker'
        # Reproduce the Windows default even when this test runs on Linux.
        kwargs.setdefault('encoding', 'cp1252')
        return real_run([sys.executable, '-c',
                         f'import sys; sys.stdout.buffer.write({raw!r})'], **kwargs)

    monkeypatch.setattr(subprocess, 'run', docker_stub)
    box = Sandbox('fixture', secret=1, max_guesses=15, noise_prob=0.3)
    monkeypatch.setattr(box, 'state', lambda: None)
    if inspect_file:
        exists, output = box._read('/agent/fixture.txt')
        assert exists
    else:
        result = box.exec_command('fixture')
        assert result.returncode == 0
        output = result.output
    assert output == 'model says: \u201d \u6a21\u578b\\xff'
