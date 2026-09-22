# ABOUTME: The published launch command never carries a venv path, a live SSH endpoint or an
# ABOUTME: absolute local path: `public_command` rewrites them and leaves everything else alone.

from pathlib import Path

from src.eval.run_eval import public_command


def test_public_command_scrubs_venv_server_and_local_paths(tmp_path):
    argv = [str(tmp_path / ".venv/bin/evals"), "--name", "mask", "--target", "org/2026-09-22-qwen36-0-da-15",
            "--server", "root@64.247.201.60:17936", "--port", "8053", "--terminate-pod",
            f"resume_from={tmp_path}/output/mask/run_a", "judge.model=x"]
    got = public_command(argv, cwd=tmp_path)
    assert got == ("uv run evals --name mask --target org/2026-09-22-qwen36-0-da-15 --server <pod> "
                   "--port 8053 --terminate-pod resume_from=output/mask/run_a judge.model=x")
    assert "root@" not in got and str(tmp_path) not in got


def test_public_command_handles_equals_form_and_no_server(tmp_path):
    assert public_command(["evals", "--name", "odcv", "--server=root@1.2.3.4:22"], cwd=tmp_path) == \
        "uv run evals --name odcv --server=<pod>"
    assert public_command(["evals", "--name", "odcv", "--target", "t"], cwd=tmp_path) == \
        "uv run evals --name odcv --target t"
