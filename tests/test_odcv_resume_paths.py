# ABOUTME: Checks that ODCV resume retains a short caller path for Docker workspaces.
# ABOUTME: Simulates Windows junction resolution and stops before any rollout or provider request.
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from src.eval.misalignment.odcv import odcv_rollout as rollout


def test_resume_preserves_short_alias_instead_of_canonical_path(tmp_path, monkeypatch):
    short = tmp_path / 'short' / 'pass'
    canonical = tmp_path / 'very-long-managed-worktree-path' / 'pass'
    bench = tmp_path / 'bench'
    for path in (short, canonical, bench):
        path.mkdir(parents=True)
    original_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        return canonical if path == short else original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'resolve', resolve)
    monkeypatch.setattr(rollout.OmegaConf, 'load', lambda _: OmegaConf.create({'bench_dir': str(bench)}))
    monkeypatch.setattr(rollout, 'scenario_names', lambda *_: ['example'])
    monkeypatch.setattr(rollout, 'openrouter_usage', lambda: 0.0)
    captured = []

    class StopBeforeExecution(Exception):
        pass

    def capture(out_dir, *_args, **_kwargs):
        captured.append(out_dir)
        raise StopBeforeExecution

    monkeypatch.setattr(rollout, 'write_run_meta', capture)
    with pytest.raises(StopBeforeExecution):
        rollout.main(config='unused.yaml', resume=str(short))
    assert captured == [short]
    assert captured[0] != canonical
