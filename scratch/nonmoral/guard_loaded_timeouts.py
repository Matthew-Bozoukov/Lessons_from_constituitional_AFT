# ABOUTME: Preserve timeout evidence for the already-loaded pre-fix nonmoral driver only.
# ABOUTME: No generation or retries; capture owned logs, restore explicitly partial records, fail closed.
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from src.eval.misalignment.odcv.recover import reconstruct_transcript
from src.infra import runpod

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/nonmoral_overnight/20260909'
RUN=Path('C:/nm-eval/2026-09-09_odcv_nonmoral_lf_common_3x_024206')
BENCH=ROOT/'src/eval/misalignment/odcv/third_party/odcv-bench'
CAPTURE=OUT/'loaded_timeout_guard'
CAPTURE.mkdir(exist_ok=True)
state=json.loads((OUT/'baseline_lf_status.json').read_text())
owner=int(state['pid'])
pod=state['pod_id']
assert pod=='bln8osvjrjpajh'
captures={}
handled=set()
pending_timeouts={}


class ConfirmedTimeoutLoss(RuntimeError):
    pass


def timeout_failure(index, message):
    first=pending_timeouts.setdefault(index,time.monotonic())
    if time.monotonic()-first>=30:
        raise ConfirmedTimeoutLoss(message)
    raise RuntimeError(message)

report=dict(owner_pid=owner,guard_pid=os.getpid(),guard_version=2,pod_id=pod,run_root=str(RUN),captures=[],timeouts=[],
            purpose='Old loaded runtime only; no model calls, timeout changes, or retries.')


def save():
    report['updated_at_unix']=time.time()
    temp=CAPTURE/'status.tmp'
    temp.write_text(json.dumps(report,indent=2),encoding='utf-8')
    temp.replace(CAPTURE/'status.json')


def docker(*args):
    return subprocess.run(['docker',*args],capture_output=True,text=True,
                          encoding='utf-8',errors='replace',timeout=30)


try:
    save()
    while runpod._parent_alive(owner) and time.time()<state['rented_at_unix']+state['max_lifetime_s']:
        try:
            ids=docker('ps','-q','--filter','label=com.docker.compose.service=executor').stdout.split()
            infos=json.loads(docker('inspect',*ids).stdout) if ids else []
            for info in infos:
                labels=info['Config']['Labels']
                ws=Path(labels.get('com.docker.compose.project.working_dir',''))
                try:
                    ws.resolve().relative_to(RUN.resolve())
                except ValueError:
                    continue
                cid=info['Id']
                compose=ws/'docker-compose.yml'
                if cid in captures or not compose.exists() or time.time()-compose.stat().st_mtime<300:
                    continue
                raw=CAPTURE/f'{cid[:12]}.log'
                handle=raw.open('wb')
                process=subprocess.Popen(['docker','logs','--follow',cid],stdout=handle,stderr=subprocess.STDOUT,
                                         creationflags=subprocess.CREATE_NO_WINDOW)
                item=dict(container_id=cid,variant=ws.parent.name,scenario=ws.name,
                          pass_dir=str(ws.parents[2]),raw_log=str(raw),capture_started_unix=time.time())
                captures[cid]=(item,process,handle)
                report['captures'].append(item)
                save()
            lines=(OUT/'baseline_lf_stdout.log').read_text(encoding='utf-8',errors='replace').splitlines()
            for index,line in enumerate(lines):
                fields=line.split()
                if index in handled or len(fields)<4 or not fields[0].startswith('[') or not fields[3].startswith('timeout'):
                    continue
                try:
                    variant,scenario,status=fields[1:4]
                    matching=[v for v in captures.values() if v[0]['variant']==variant and v[0]['scenario']==scenario]
                    if not matching:
                        raise RuntimeError(f'Timeout without captured owned log: {variant}/{scenario}')
                    item,process,handle=matching[-1]
                    if process.poll() is None:
                        process.wait(timeout=10)
                    handle.flush()
                    # Derive actual model-key directory from the existing pass, not a target-name guess.
                    roots=list((Path(item['pass_dir'])/'agent_logs').glob(f'*-{variant}'))
                    assert len(roots)==1
                    dest=roots[0]/'experiments'/scenario
                    record=dest/'messages_record.txt'
                    outcome=dict(**item,status=status,transcript_partial=True,event_index=index)
                    if not record.exists() or not record.stat().st_size:
                        raw=Path(item['raw_log']).read_text(encoding='utf-8',errors='replace')
                        prefixed=CAPTURE/f"{item['container_id'][:12]}_prefixed.log"
                        prefixed.write_text('\n'.join('executor-1 | '+line for line in raw.splitlines()),encoding='utf-8')
                        text=reconstruct_transcript(prefixed,variant,scenario,BENCH)
                        if not text:
                            raise RuntimeError(f'No observed model events to recover: {variant}/{scenario}')
                        original=dest/'docker_output.log'
                        if original.exists():
                            (dest/'docker_output.timeout_notice.log').write_bytes(original.read_bytes())
                        original.write_bytes(prefixed.read_bytes())
                        record.write_text(text+'\n== [partial transcript: driver timeout; reconstructed from captured executor log] ==\n',encoding='utf-8')
                        outcome['transcript_source']='docker_log_reconstruction_by_loaded_runtime_guard'
                    else:
                        outcome['transcript_source']='executor_archive'
                    outcome['transcript_sha256']=hashlib.sha256(record.read_bytes()).hexdigest()
                    (dest/'timeout_guard_meta.json').write_text(json.dumps(outcome,indent=2),encoding='utf-8')
                    report['timeouts'].append(outcome)
                    handled.add(index)
                    save()
                except ConfirmedTimeoutLoss:
                    raise
                except Exception as exc:
                    timeout_failure(index, f"Observed timeout preservation pending: {type(exc).__name__}: {exc}")
        except ConfirmedTimeoutLoss:
            raise
        except Exception as exc:
            # Inventory races, temporary Docker failures, and file reads are monitoring
            # failures. They never justify killing a healthy experiment.
            report['monitor_error_count']=report.get('monitor_error_count',0)+1
            report['last_monitor_error']=f'{type(exc).__name__}: {exc}'
        save()
        time.sleep(5)
    report['finished']=True
    save()
except ConfirmedTimeoutLoss as exc:
    report['error']=f'{type(exc).__name__}: {exc}'
    # Stop this old driver before it can turn an unknown timeout into a new draw.
    if runpod._parent_alive(owner):
        subprocess.run(['taskkill','/PID',str(owner),'/F'],capture_output=True)
        report['owned_pod_terminated']=runpod.terminate(pod)
    save()
    raise
except BaseException as exc:
    report['guard_stopped_error']=f'{type(exc).__name__}: {exc}'
    save()
    raise
finally:
    for item,process,handle in captures.values():
        if process.poll() is None:
            process.terminate()
        handle.close()
