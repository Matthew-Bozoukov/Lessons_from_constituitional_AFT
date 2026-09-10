# ABOUTME: Runs the bounded fresh-data lane using shared tagged stages and the existing cumulative capped client.
# ABOUTME: Freezes configuration and source bytes, preserves raw calls, and writes a complete local review packet.
import argparse
import hashlib
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from scratch.nonmoral.pilot import CappedClient, SONNET, file_sha256, read_rows, verify_live_prices
from src.data.synth.pipeline import build_stages, run
from src.data.synth.stage_runtime import model_cfg
from src.infra.endpoints.openrouter import OpenRouterClient
from src.utils import timestamp

ROOT = Path('output/nonmoral_overnight/20260909/data')


def packet(run_dir, config):
    from transformers import AutoTokenizer
    revision = '6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B', revision=revision, local_files_only=True)
    rows = read_rows(run_dir/'dataset.jsonl') if (run_dir/'dataset.jsonl').exists() else []
    planned = read_rows(Path(config['source']['local_dir'])/'inputs.jsonl')
    lines = ['# Fresh nonmoral validation packet', '', 'Constructed traces; not training-approved. All original 32 IDs remain in the denominator.', '']
    outcomes = []
    for row in rows:
        b, c = (len(tok.encode(row.get(k, ''), add_special_tokens=False)) for k in ('comparative', 'execution'))
        outcomes.append(dict(scenario_id=row['scenario_id'], domain=row['domain'], b_tokens=b, c_tokens=c,
                             b_c_ratio=b/c if c else None, mechanical_length_pass=bool(c and .5 <= b/c <= 2)))
        lines += [f"## {row['scenario_id']} — {row['domain']}", '', '**User**', '', row.get('user', '[missing]'),
                  '', '**B: comparative CoT**', '', row.get('comparative', '[missing]'),
                  '', '**C: construction CoT**', '', row.get('execution', '[missing]'),
                  '', '**Shared full answer**', '', row.get('answer', '[missing]'),
                  '', '**Generator checking notes (not independent validation)**', '', row.get('checks', ''), '']
    missing = sorted({r['scenario_id'] for r in planned} - {r['scenario_id'] for r in rows})
    lines += ['## Missing planned IDs', '', ', '.join(missing) or 'None', '']
    (run_dir/'packet.md').write_text('\n'.join(lines), encoding='utf-8')
    total_b, total_c = sum(r['b_tokens'] for r in outcomes), sum(r['c_tokens'] for r in outcomes)
    summary = dict(planned=len(planned), produced=len(rows), missing_ids=missing, lengths=outcomes,
                   aggregate_b_c_ratio=total_b/total_c if total_c else None,
                   tokenizer_revision=revision, status='awaiting_independent_content_review')
    (run_dir/'local_summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    assert cfg['hf_push'] is False and cfg['batch'] is False and cfg['workers'] == 4
    assert cfg['budget_usd'] == 5 and cfg['total_scenarios'] == 32
    assert Path(cfg['output_dir']).resolve() == (ROOT/'runs').resolve()
    source = Path(cfg['source']['local_dir'])/'inputs.jsonl'
    assert cfg['source']['snapshot'] == 'inputs.jsonl' and file_sha256(source) == cfg['input_sha256']
    inputs = read_rows(source)
    assert len(inputs) == len({r['scenario_id'] for r in inputs}) == 32
    assert len({r['domain'] for r in inputs}) == 8
    repair = cfg.get('phase') == 'targeted_repair'
    expected = ['load_source_run', 'llm_tagged'] if repair else ['load_source_run', 'llm_tagged', 'llm_tagged', 'llm_tagged']
    assert [s['kind'] for s in cfg['stages']] == expected
    if repair:
        stage = cfg['stages'][1]
        assert stage['when'] == {'field': 'needs_revision', 'in': [True]}
        assert stage['save'] == {'comparative': 'reasoning', 'answer': 'response', 'execution': 'construction'}
        assert all('user' in row and 'needs_revision' in row for row in inputs)
        assert all(row.get('repair_attempts', 0) == 0 for row in inputs)
    assert all(not (set(s) & {'lint', 'verify', 'fallback_model'}) for s in cfg['stages'])
    for key in cfg['models']:
        m = model_cfg(cfg, key)
        assert m['model'] == SONNET and m.get('extra_body') == {'reasoning': {'enabled': False}}
    stages = build_stages(cfg)
    print(json.dumps(dict(stages=[s.name for s in stages], planned=32, cap=5, paid=args.execute)), flush=True)
    if not args.execute:
        return
    from dotenv import load_dotenv
    from tenacity import stop_after_attempt
    load_dotenv()
    ROOT.mkdir(parents=True, exist_ok=True)
    marker = ROOT/f'dispatch_{file_sha256(args.config)}.json'
    assert not marker.exists(), 'This frozen batch was already dispatched'
    prices = verify_live_prices({SONNET})
    lock = ROOT/'pilot.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode())
        run_dir = ROOT/'runs'/timestamp()
        run_dir.mkdir(parents=True, exist_ok=False)
        snapshot = dict(config=cfg, config_sha256=file_sha256(args.config), source_sha256=file_sha256(source),
                        verified_prices=prices, run_dir=str(run_dir), status='dispatched')
        with marker.open('x', encoding='utf-8') as handle:
            json.dump(snapshot, handle, indent=2)
        (run_dir/'frozen_config.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
        (ROOT/'status.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
        client = OpenRouterClient()
        single = client.chat.retry_with(stop=stop_after_attempt(1))
        capped = CappedClient(lambda **kw: single(client, **kw), ROOT/'spend.json', 5, {SONNET}, allow_reasoning_off=True)
        completed = False
        try:
            run(cfg, resume=str(run_dir), client=capped)
            completed = True
        finally:
            packet(run_dir, cfg)
            snapshot['status'] = 'finished_awaiting_review' if completed else 'failed_awaiting_review'
            ledger = ROOT/'spend.json'
            snapshot['exposure_usd'] = sum(e['charged_or_reserved_usd'] for e in json.loads(ledger.read_text())) if ledger.exists() else 0
            (ROOT/'status.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
    finally:
        os.close(fd)
        lock.unlink()


if __name__ == '__main__':
    main()
