# ABOUTME: Freeze the112K diagnostic's paired selection, unchanged model recipe and spending envelope.
# ABOUTME: Offline configuration checks never call a model, judge, or Docker.
from pathlib import Path
from omegaconf import OmegaConf
from scratch.gptoss_swe import odcv_full as full


def test_exact_matched_selection_and_only_context_model_change(monkeypatch):
    old=OmegaConf.load('scratch/gptoss_swe/odcv_full.yaml')
    new=OmegaConf.load('scratch/gptoss_swe/odcv_double_context.yaml')
    for field in ('arms','passes','temperature','top_p','top_k','reasoning','harmony_version','cycles',
                  'scenario_timeout_s','strict_tool_validation','system_preamble_file','progress_judge',
                  'render_date','response_allowance','concurrency_per_arm','max_concurrent_rollouts'):
        assert new[field]==old[field]
    assert new.context_window==2*old.context_window==112000
    assert new.shared_inference_budget_usd+3*new.judge_budget_per_arm_usd==new.total_budget_usd==30
    monkeypatch.setattr(full,'CFG',new)
    monkeypatch.setattr(full,'ROOT',Path(new.root))
    assert full.expected_rollouts()==90
    configs=[full.arm_config(a) for a in full.ARMS]
    assert all(c.include_scenarios==new.include_scenarios for c in configs)
    cells=[(v,s) for v,names in new.include_scenarios.items() for s in names]
    assert len(cells)==len(set(cells))==30
    assert len({c.model_key for c in configs})==len({c.base_url for c in configs})==3
    assert all(c.passes==3 and c.concurrency==16 and c.serving.context_window==112000 for c in configs)
    settings=[full.shim_settings(a) for a in full.ARMS]
    assert len({s['TINKER_BUDGET_LEDGER'] for s in settings})==1
    assert all(s['TINKER_BUDGET_USD']=='21' and s['DEFAULT_MAX_TOKENS']=='remaining' for s in settings)
