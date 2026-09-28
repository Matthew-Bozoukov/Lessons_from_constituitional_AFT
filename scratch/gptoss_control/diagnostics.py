# ABOUTME: Summarize this experiment's raw Tinker sampling ledger without changing rollouts.
# ABOUTME: Run: uv run --project src/infra/endpoints/tinker_runtime python scratch/gptoss_control/diagnostics.py <sampling.jsonl>
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.infra.endpoints.harmony import make_renderer


def analysis_tokens(ids, tokenizer):
    count = 0
    boundaries = {200006, 200007, 200002, 200012}
    for i, token in enumerate(ids):
        if token != 200005:
            continue
        j = i + 1
        while j < len(ids) and ids[j] not in boundaries | {200008}:
            j += 1
        if j == len(ids) or ids[j] != 200008:
            continue
        if tokenizer.decode(ids[i + 1:j]).strip() != 'analysis':
            continue
        end = j + 1
        while end < len(ids) and ids[end] not in boundaries:
            end += 1
        count += end - j - 1
    return count


def summarize(path):
    events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    requests = [e for e in events if e['event'] == 'reserved']
    completed = [e for e in events if e['event'] == 'completed']
    errors = [e for e in events if e['event'] == 'error']
    renderer = make_renderer(local_files_only=True)
    counts = [analysis_tokens(e['raw_tokens'], renderer.tokenizer) for e in completed]
    latency = sorted(e['seconds'] for e in completed)
    ends = {'final_stop': 0, 'tool_handoff': 0, 'length_stop': 0}
    for e in completed:
        last = e['raw_tokens'][-1] if e['raw_tokens'] else None
        ends[{200002: 'final_stop', 200012: 'tool_handoff'}.get(last, 'length_stop')] += 1
    return {
        'source_ledger': str(path), 'requests_reserved': len(requests),
        'responses_received': len(completed), 'error_events': len(errors),
        'model_output_error_events': sum(e['event']=='model_output_error' for e in events),
        'error_types': {kind: sum(e['type'] == kind for e in errors)
                        for kind in sorted({e['type'] for e in errors})},
        'reserved_prompt_tokens': sum(e['prompt_tokens'] for e in requests),
        'received_output_tokens': sum(e['completion_tokens'] for e in completed),
        'raw_analysis_payload_tokens': sum(counts),
        'responses_with_nonempty_analysis': sum(n > 0 for n in counts),
        'response_terminators': ends,
        'target_uncached_price_upper_or_reserved_usd': sum(e.get('reserved_usd', 0) for e in events),
        'latency_seconds': {'mean': statistics.mean(latency), 'median': statistics.median(latency),
                            'p95': latency[math.ceil(.95 * len(latency)) - 1]}
                           if latency else None,
        'notes': 'Token counts include all attempts. Terminators describe raw responses, including parse failures. '
                 'Analysis counts exclude channel headers and terminators. Costs are rate-card upper/reservations, not invoices.'
    }


if __name__ == '__main__':
    path = Path(sys.argv[1]).resolve()
    result = summarize(path)
    destination = path.with_name('sampling_diagnostics.json')
    destination.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
