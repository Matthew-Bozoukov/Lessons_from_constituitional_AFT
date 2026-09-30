# ABOUTME: Ensure inspected model-format failures remain outcomes without hiding infrastructure failures.
# ABOUTME: Recovery receipts bind a single log's exact bytes and never exempt archive failures.
import hashlib

from src.eval.misalignment.odcv.runner import audit_terminal_errors


def test_terminal_error_adjudication_is_exact_and_scoped(tmp_path):
    log=tmp_path/'agent_logs/model/experiments/scenario/docker_output.log'
    log.parent.mkdir(parents=True)
    log.write_text('[AI API dead]: Tinker completion failed: ValueError; inspect owned sampling log')
    key=tmp_path.name+'/'+log.relative_to(tmp_path).as_posix()
    receipts={key:{'cause':'tool_calls_ended_without_handoff',
        'docker_log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),
        'rejected_request_ids':['a','b','c']}}
    assert audit_terminal_errors(tmp_path,{})==([log.relative_to(tmp_path).as_posix()],[])
    assert audit_terminal_errors(tmp_path,receipts)==([],[log.relative_to(tmp_path).as_posix()])
    log.write_text(log.read_text()+'\n[! Archive Fail]: disk error')
    assert audit_terminal_errors(tmp_path,receipts)[0]
    receipts[key]['docker_log_sha256']=hashlib.sha256(log.read_bytes()).hexdigest()
    assert audit_terminal_errors(tmp_path,receipts)[0]


def test_receipt_does_not_allow_another_failed_cell(tmp_path):
    log=tmp_path/'agent_logs/model/experiments/other/docker_output.log'
    log.parent.mkdir(parents=True)
    log.write_text('[AI API dead]: network unavailable')
    assert audit_terminal_errors(tmp_path,{'wrong/path':{}})==([log.relative_to(tmp_path).as_posix()],[])
