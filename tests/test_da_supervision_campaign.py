# ABOUTME: Checks paired replay selection and paid-run command isolation for the DA campaign.
# ABOUTME: Run: uv run pytest tests/test_da_supervision_campaign.py -q
import shlex
from scratch.da_supervision.build import select_replay
from scratch.da_supervision.owner import train_command


def test_replay_selection_exact_unique_deterministic_and_unchanged():
    rows=[{'source':s,'id':i,'messages':[{'role':'assistant','content':str(i),'reasoning_content':'keep'}]}
          for s,n in [('a',7),('b',3)] for i in range(n)]
    selected,quotas=select_replay(rows,7,0)
    assert len(selected)==7 and quotas=={'a':5,'b':2}
    assert len({i for i,_ in selected})==7
    assert all(row is rows[i] for i,row in selected)
    assert select_replay(rows,7,0)==(selected,quotas)


def test_commands_pin_dataset_and_base_and_smoke_longest_plus_da():
    plan={'constitution':'constitutions/test/constitution.md','base_model_revision':'a'*40}
    arm={'data_repo':'org/data','data_revision':'b'*40,'organism':'org/organism',
         'stats':{'longest_index':9999,'da_index':53}}
    full=shlex.split(train_command(plan,arm))
    smoke=shlex.split(train_command(plan,arm,True))
    assert full[:3]==['/root/.local/bin/uv','run','train']
    assert 'data_revision='+'b'*40 in full and 'base_model_revision='+'a'*40 in full
    assert '--smoke=True' not in full and '--smoke=True' in smoke
    indices=next(x for x in smoke if x.startswith('smoke_indices='))
    assert '9999' in indices and '53' in indices
    assert smoke[:len(full)]==full
    import fire
    captured={}
    def entry(config: str,*overrides: str,smoke: bool=False):
        captured.update(config=config,overrides=overrides,smoke=smoke)
    fire.Fire(entry,command=smoke[3:])
    assert captured['smoke'] is True
    assert indices in captured['overrides']
