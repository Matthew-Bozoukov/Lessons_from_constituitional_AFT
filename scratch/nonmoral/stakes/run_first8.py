# ABOUTME: Bounded first eight nonmoral-stakes candidate pairs using shared synth stages and per-call reservations.
# ABOUTME: Run: uv run python scratch/nonmoral/stakes/run_first8.py answers|review|publish --execute; total cap $5.
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv
from omegaconf import OmegaConf
from tenacity import stop_after_attempt

from scratch.nonmoral.pilot import CappedClient, SONNET, verify_live_prices
from scratch.nonmoral.stakes.prepare import arm_rows, digest, fixture_packet, join_answers, phase_config, read_rows, write_json, write_rows
from scratch.nonmoral.stakes.fixtures import FIXTURES
from src.data.synth.pipeline import run
from src.infra.endpoints.openrouter import OpenRouterClient
from src.utils import timestamp

OUT = REPO/'output/nonmoral_stakes/20260909_first8'
CONFIG = REPO/'configs/data/synth/nonmoral-stakes.yaml'
CAP = 5.0


def load_credentials():
    # Deliberately read, never copy, the shared checkout's secrets at runtime.
    load_dotenv(REPO.parent/'teaching_claude_why_replication/.env', override=False)


def generate(phase):
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT/'dispatch.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode())
        dest = OUT/phase
        if dest.exists():
            raise ValueError('Phase already prepared/dispatched; inspect it instead of duplicating spend')
        cfg = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)
        if phase == 'answers':
            fixtures = OUT/'sources'
            fixtures.mkdir(exist_ok=False)
            fixture_packet(fixtures)
            frames = read_rows(fixtures/'design_frames.jsonl')
            rows = arm_rows(frames)
            write_json(fixtures/'source_review.json', dict(reviewer='Local Codex source inspection, 2026-09-09',
                source_sha256=digest(fixtures/'design_frames.jsonl'),
                findings='Eight complete fixed sources. All options nonmoral; exact core unchanged. Same personal loss scales. Seven discretionary-cost frames and one gameplay-time frame; limited coverage, no claim of representative sample.',
                dispositions={r['scenario_id']:dict(decision='accept', reason='Complete task with material constraints and explicit matched personal consequence') for r in frames}))
        else:
            status = json.loads((OUT/'answers/status.json').read_text())
            answers = read_rows(Path(status['run_dir'])/'dataset.jsonl')
            grouped = {}
            for row in answers:
                grouped.setdefault(row['pair_id'], []).append(row)
            rows, failures = [], {}
            for pair_id, group in grouped.items():
                try:
                    rows.extend(join_answers(group))
                except ValueError as exc:
                    failures[pair_id] = str(exc)
            write_json(OUT/'incomplete_pair_exclusions.json', failures)
        dest.mkdir(exist_ok=False)
        phase_config(cfg, phase, rows, dest)
        effective = OmegaConf.to_container(OmegaConf.load(dest/'prepared_config.yaml'), resolve=True)
        effective['budget_usd'] = CAP
        OmegaConf.save(OmegaConf.create(effective), dest/'dispatch_config.yaml')
        load_credentials()
        prices = verify_live_prices({SONNET})
        ledger = OUT/'spend.json'
        existing = json.loads(ledger.read_text()) if ledger.exists() else []
        if any(e['status'] != 'settled' for e in existing):
            raise ValueError('Reconcile unsettled prior reservations before dispatch')
        client = OpenRouterClient()
        once = client.chat.retry_with(stop=stop_after_attempt(1))
        capped = CappedClient(lambda **kw: once(client, **kw), ledger, CAP, {SONNET})
        run_dir = dest/'runs'/timestamp()
        run_dir.mkdir(parents=True, exist_ok=False)
        state = dict(phase=phase, status='running', rows=len(rows), cap_usd=CAP,
            run_dir=str(run_dir), input_sha256=digest(dest/'inputs.jsonl'), config_sha256=digest(dest/'dispatch_config.yaml'),
            git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            code_sha256=digest(__file__), prices=prices,
            source_origin='Eight offline Codex-written design sources; Sonnet generates responses/reviews only')
        write_json(dest/'status.json', state)
        try:
            run(effective, resume=str(run_dir), client=capped)
            state['status'] = 'awaiting_local_review'
        except BaseException:
            state['status'] = 'generation_failed'
            raise
        finally:
            entries = json.loads(ledger.read_text()) if ledger.exists() else []
            state.update(calls=len(entries), exposure_usd=sum(e['charged_or_reserved_usd'] for e in entries),
                         unsettled_calls=sum(e['status'] != 'settled' for e in entries))
            data = run_dir/'dataset.jsonl'
            if data.exists():
                output = read_rows(data)
                state.update(produced=len(output), dataset_sha256=digest(data),
                             missing_ids=sorted({r['scenario_id'] for r in rows}-{r['scenario_id'] for r in output}))
                write_rows(dest/'complete_candidates.jsonl', output)
                (dest/'complete_candidates.md').write_text('\n\n'.join(
                    '## '+r['scenario_id']+'\n\n'+'\n\n'.join('**'+k+'**\n\n'+str(v) for k,v in r.items()
                    if k not in ('scenario_id','source_row_sha256')) for r in output), encoding='utf-8')
            write_json(dest/'status.json', state)
            print(json.dumps(state), flush=True)
    finally:
        os.close(fd)
        lock.unlink()


def publish():
    load_credentials()
    from scratch.nonmoral.publish_invalid_baseline import scan, secret_values
    from src.infra.huggingface import hf_api, hf_org, push_run_dir
    from src.naming import artifact_name
    assert hf_org() == 'dougalldeepmind'
    if not (OUT/'local_review.json').exists():
        raise ValueError('Include local candidate dispositions before publishing')
    secrets = secret_values()
    for path in OUT.rglob('*'):
        if path.is_file():
            scan(path.read_bytes(), str(path), secrets)
    name = artifact_name('nonmoral-stakes-first8', date='2026-09-09')
    url = push_run_dir(OUT, name, dict(experiment='First eight matched low/high stakes nonmoral candidate pairs',
        date_generated='2026-09-09', constitution='none; nonmoral working preferences',
        source_repo='https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT @ '+subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        models='anthropic/claude-sonnet-5 via Anthropic endpoint; source fixtures authored offline by Codex',
        generation_config='Frozen per-phase dispatch configs; fresh low/high responses, one combined review, $5 cumulative reservation cap',
        schema='sources/: exact fixed source pairs; answers/ and review/: complete candidates plus stage snapshots; local_review.json: material dispositions; spend.json and raw_calls/: all attributed requests and costs',
        provenance='uv run python scratch/nonmoral/stakes/run_first8.py answers --execute; then review --execute',
        limitations='Small convenience sample of eight source pairs, mostly own discretionary replacement cost. Not an SFT run, stakes effect estimate or representative corpus. No semantic repair calls.'),
        private=False, front_matter={'tags':['nonmoral-deliberation','stakes','candidate-data']})
    info = hf_api().dataset_info(hf_org()+'/'+name)
    assert not info.private
    write_json(OUT/'publication.json', dict(url=url, revision=info.sha, private=False))
    print(url, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['answers', 'review', 'publish'])
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if not args.execute:
        print(json.dumps(dict(phase=args.phase, cap_usd=CAP, planned_pairs=8, paid=False, ledger=str(OUT/'spend.json'))))
    elif args.phase == 'publish':
        publish()
    else:
        generate(args.phase)
