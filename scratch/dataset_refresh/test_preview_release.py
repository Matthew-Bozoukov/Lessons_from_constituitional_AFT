# ABOUTME: Actual composite transformation and quality intake are exercised offline with bounded audit substitutes.
import json
from types import SimpleNamespace
import pytest
from scratch.dataset_refresh import preview_release as mod
from scratch.dataset_refresh.test_publish_composite import corpus


@pytest.fixture
def previewed(corpus, tmp_path, monkeypatch):
    monkeypatch.setattr(mod.AutoTokenizer, 'from_pretrained', lambda name, **kw: SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda:'cached-tokenizer')) if kw=={'local_files_only':True} else pytest.fail('Network enabled'))
    monkeypatch.setattr(mod,'token_audit',lambda row,t,p,n:{'training_tokens':100,'supervised_tokens':50} if n==8192 else pytest.fail('Wrong max length'))
    def audit(path, output, *, local_files_only):
        assert local_files_only is True
        data={'input_sha256':mod.runtime.digest(path.read_bytes())}
        mod.runtime.write_json(output/'corpus_audit.json',data)
        return data
    monkeypatch.setattr(mod.audit_corpus,'audit',audit)
    selection=tmp_path/'selection.json'; mod.runtime.write_json(selection,corpus)
    output=tmp_path/'preview'; manifest=mod.preview(selection,output)
    return output,manifest,corpus


def evidence(tmp_path,sha):
    path=tmp_path/'evidence'
    mod.runtime.write_json(path/'adjudication.json',{'dataset_sha256':sha,'scope':'Independent full final selection review; exact evidence attached.'})
    return path


def test_actual716_publisher_bytes_and_exact_quality_intake(previewed,tmp_path):
    output,manifest,selection=previewed
    rows,_,_=mod.publish_composite.validate_selection(selection)
    expected=tmp_path/'expected.jsonl'; mod.runtime.write_rows(expected,rows)
    assert (output/'dataset.jsonl').read_bytes()==expected.read_bytes()
    assert manifest['rows']==716 and manifest['automatic_checks']=='passed'
    sealed=tmp_path/'sealed'
    result=mod.seal(output,evidence(tmp_path,manifest['dataset_sha256']),sealed)
    assert result['dataset_sha256']==manifest['dataset_sha256']
    assert 'automatic/token_mask_audit.json' in result['files']
    assert 'independent/adjudication.json' in result['files']


@pytest.mark.parametrize('mutation',['dataset','audit','selection','wrong_evidence','failed'])
def test_changed_or_failed_artifacts_refused(previewed,tmp_path,mutation):
    output,manifest,_=previewed
    supplied=evidence(tmp_path,manifest['dataset_sha256'] if mutation!='wrong_evidence' else '0'*64)
    if mutation=='dataset': (output/'dataset.jsonl').write_text('changed')
    if mutation=='audit': (output/'quality/corpus_audit.json').write_text('{}')
    if mutation=='selection': (output/'selection.json').write_text('{}')
    if mutation=='failed': (output/'PREVIEW_FAILED').touch()
    with pytest.raises(ValueError): mod.seal(output,supplied,tmp_path/'sealed')


def test_copy_detects_mutation(tmp_path,monkeypatch):
    source=tmp_path/'source'; source.mkdir(); (source/'evidence').write_text('original')
    hashes=mod.inventory(source); original=mod.shutil.copy2
    def copy(path,target):
        original(path,target); path.write_text('changed')
    monkeypatch.setattr(mod.shutil,'copy2',copy)
    with pytest.raises(ValueError,match='changed'): mod.copy_bound(source,tmp_path/'out',hashes)
