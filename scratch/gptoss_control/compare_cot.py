# ABOUTME: Retokenize recorded assistant reasoning consistently and aggregate all 480 ODCV rollouts.
# ABOUTME: Separates per-response and per-rollout means; retains zero-CoT turns and task-level differences.
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.infra.huggingface import hf_api


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def csv_write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    cfg = OmegaConf.load(Path(__file__).with_suffix('.yaml'))
    common = Path(subprocess.check_output(['git', 'rev-parse', '--git-common-dir'], text=True).strip()).resolve()
    load_dotenv(common.parent / '.env')
    api = hf_api()
    out = Path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    tokenizers = {key: AutoTokenizer.from_pretrained(spec.tokenizer, revision=spec.tokenizer_revision,
                  local_files_only=True) for key, spec in cfg.models.items()}
    turns, rollouts, provenance, prompts = [], [], {}, {}
    for model, spec in cfg.models.items():
        root = Path(snapshot_download(spec.repo, repo_type='dataset', revision=spec.revision,
            token=api.token, allow_patterns=['rollouts/**/messages_record.txt', 'metadata/run_meta.json',
            'results/results.json', 'metadata/tinker_shim/sampling_diagnostics.json'], max_workers=8))
        files = sorted((root / 'rollouts').rglob('messages_record.txt'))
        assert len(files) == 240
        meta = json.loads((root / 'metadata/run_meta.json').read_text(encoding='utf-8'))
        provenance[model] = {'config': OmegaConf.to_container(spec), 'original_run_meta': meta,
            'file_sha256': {}, 'count_method': 'Strip boundary whitespace from recorded reason field, encode with add_special_tokens=False; exclude No explanation sentinel, headers, endings, commands and answers.'}
        for p in files:
            rel = p.relative_to(root).as_posix()
            provenance[model]['file_sha256'][rel] = hashlib.sha256(p.read_bytes()).hexdigest()
            _, variant, scenario, pass_name, _ = rel.split('/')
            chunks = re.split(r'^== Step \d+ ==\n', p.read_text(encoding='utf-8'), flags=re.M)[1:]
            task_key = (variant, scenario, pass_name)
            prompts[(model, *task_key)] = [re.sub(r'\s+', ' ', c.split('\ncontent: ', 1)[1]).strip() for c in chunks[:2]]
            local = []
            for step, chunk in enumerate(chunks, 1):
                if not chunk.startswith('role: assistant\n'):
                    continue
                body = chunk.split('\ncontent: ', 1)[1].rstrip('\n')
                if '\ncall: ' in body:
                    body = body.rsplit('\ncall: ', 1)[0]
                assert '\nreason: ' in body, (rel, step)
                reasoning = body.split('\nreason: ', 1)[1].strip()
                if reasoning == 'No explanation':
                    reasoning = ''
                row = {'model': model, 'variant': variant, 'scenario': scenario, 'pass': pass_name,
                       'assistant_turn': len(local) + 1, 'transcript_step': step,
                       'cot_tokens': len(tokenizers[model].encode(reasoning, add_special_tokens=False)),
                       'cot_tokens_common_qwen_tokenizer': len(tokenizers['qwen'].encode(reasoning, add_special_tokens=False)),
                       'cot_chars': len(reasoning)}
                local.append(row)
            turns.extend(local)
            rollouts.append({'model': model, 'variant': variant, 'scenario': scenario, 'pass': pass_name,
                'cot_tokens': sum(r['cot_tokens'] for r in local),
                'cot_tokens_common_qwen_tokenizer': sum(r['cot_tokens_common_qwen_tokenizer'] for r in local),
                'assistant_turns': len(local), 'turns_with_cot': sum(r['cot_tokens'] > 0 for r in local)})
        diag = root / 'metadata/tinker_shim/sampling_diagnostics.json'
        if diag.exists():
            provenance[model]['original_raw_token_diagnostics'] = json.loads(diag.read_text(encoding='utf-8'))
    keys = sorted({(r['variant'], r['scenario'], r['pass']) for r in rollouts})
    assert len(keys) == 240
    mismatches = [k for k in keys if prompts[('gptoss', *k)] != prompts[('qwen', *k)]]
    assert not mismatches, mismatches
    summary = {'models': {}, 'normalized_initial_task_prompts_match': True}
    for model in cfg.models:
        rr = [r for r in rollouts if r['model'] == model]
        tt = [r for r in turns if r['model'] == model]
        nonzero = [r for r in tt if r['cot_tokens']]
        total = sum(r['cot_tokens'] for r in tt)
        summary['models'][model] = {'rollouts': len(rr), 'assistant_responses': len(tt),
            'responses_with_cot': len(nonzero), 'responses_with_cot_pct': 100*len(nonzero)/len(tt),
            'cot_tokens_total': total, 'mean_cot_per_response': total/len(tt),
            'mean_cot_per_nonempty_response': total/len(nonzero),
            'mean_cot_per_rollout': total/len(rr), 'median_cot_per_rollout': statistics.median(r['cot_tokens'] for r in rr),
            'zero_cot_rollouts': sum(r['cot_tokens'] == 0 for r in rr),
            'mean_assistant_responses_per_rollout': len(tt)/len(rr),
            'mean_cot_per_rollout_common_qwen_tokenizer': sum(r['cot_tokens_common_qwen_tokenizer'] for r in rr)/len(rr)}
    lookup = {(r['model'], r['variant'], r['scenario'], r['pass']): r for r in rollouts}
    paired = []
    for variant, scenario, pass_name in keys:
        a, b = [lookup[(m, variant, scenario, pass_name)] for m in ['gptoss', 'qwen']]
        paired.append({'variant': variant, 'scenario': scenario, 'pass': pass_name,
                       'gptoss_cot': a['cot_tokens'], 'qwen_cot': b['cot_tokens'],
                       'difference_qwen_minus_gptoss': b['cot_tokens']-a['cot_tokens']})
    by_scenario, by_cell = [], []
    for scenario in sorted({r['scenario'] for r in paired}):
        rr = [r for r in paired if r['scenario'] == scenario]
        assert len(rr) == 6
        by_scenario.append({'scenario': scenario, 'rollouts_per_model': 6,
            'gptoss_mean_cot': statistics.mean(r['gptoss_cot'] for r in rr),
            'qwen_mean_cot': statistics.mean(r['qwen_cot'] for r in rr),
            'difference_qwen_minus_gptoss': statistics.mean(r['difference_qwen_minus_gptoss'] for r in rr)})
        for variant in ['mandated', 'incentivized']:
            sub = [r for r in rr if r['variant'] == variant]
            assert len(sub) == 3
            by_cell.append({'scenario': scenario, 'variant': variant,
                'gptoss_mean_cot': statistics.mean(r['gptoss_cot'] for r in sub),
                'qwen_mean_cot': statistics.mean(r['qwen_cot'] for r in sub),
                'difference_qwen_minus_gptoss': statistics.mean(r['difference_qwen_minus_gptoss'] for r in sub)})
    g, q = [summary['models'][m]['mean_cot_per_rollout'] for m in ['gptoss', 'qwen']]
    summary.update(mean_difference_qwen_minus_gptoss=q-g, ratio_qwen_over_gptoss=q/g,
                   scenarios_qwen_higher=sum(r['difference_qwen_minus_gptoss'] > 0 for r in by_scenario))
    dump(out/'results.json', summary)
    dump(out/'run_meta.json', {'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                             'config': OmegaConf.to_container(cfg), 'provenance': provenance})
    for name, rows in [('per_response.csv', turns), ('per_rollout.csv', paired), ('per_scenario.csv', by_scenario), ('per_scenario_variant.csv', by_cell)]:
        csv_write(out/name, rows)
    table = '| Task (six rollouts per model) | GPT-OSS CoT | Qwen CoT | Qwen minus GPT-OSS |\n|---|---:|---:|---:|\n'
    table += '\n'.join(f"| {r['scenario']} | {r['gptoss_mean_cot']:,.1f} | {r['qwen_mean_cot']:,.1f} | {r['difference_qwen_minus_gptoss']:+,.1f} |" for r in by_scenario)
    (out/'per_scenario.md').write_text(table+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(table)


if __name__ == '__main__':
    main()
