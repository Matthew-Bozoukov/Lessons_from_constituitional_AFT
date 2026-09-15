# ABOUTME: Read-only triage of accepted terminal rows for independent human or agent examination.
# ABOUTME: Flags literal defects, process-language candidates, repeated phrases and new quantities without judging quality.
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from scratch.dataset_refresh import run as runtime

FIELDS = ('system', 'user', 'reasoning', 'response')
PROCESS = re.compile(r'\b(?:the (?:previous|earlier|original) (?:answer|response)|'
                     r'system prompt|(?:original|earlier|previous) reasoning|the revision should|'
                     r'(?:I|we)(?:\s+(?:have|will|should|need to))?\s+(?:revised?|rewrit(?:e|ten)|repair(?:ed)?)\b|'
                     r'(?:reviewer|critic|grounding review|preflight|acceptance gate)s?\b|'
                     r'(?:in|for) (?:this|the) (?:rewrite|revision)|(?:draft_reasoning|draft_response|draft reasoning))', re.I)
QUANTITY = re.compile(r'(?<![\w])(?:\d+(?:[.,]\d+)?%?|one|two|three|four|five|six|seven|eight|nine|ten)'
                      r'\s+(?:minutes?|hours?|days?|weeks?|months?|years?|people|participants|pages?|percent|points?)\b', re.I)


def excerpts(pattern, text):
    return [{'quote': m.group(), 'context': text[max(0, m.start()-100):m.end()+140],
             'start': m.start()} for m in pattern.finditer(text)]


def screen_record(record):
    flags = []
    for field in FIELDS:
        text = record.get(field)
        if not isinstance(text, str) or not text.strip():
            flags.append({'kind': 'missing_text', 'field': field})
            continue
        if '\ufffd' in text:
            flags.append({'kind': 'replacement_character', 'field': field,
                          'count': text.count('\ufffd'), 'evidence': excerpts(re.compile('\ufffd'), text)})
        if field in ('reasoning', 'response'):
            evidence = excerpts(PROCESS, text)
            if evidence:
                flags.append({'kind': 'editing_process_language_candidate', 'field': field, 'evidence': evidence})
    context = '\n'.join(record.get(k, '') for k in ('system', 'user'))
    known = {re.sub(r'\s+', ' ', m.group()).casefold() for m in QUANTITY.finditer(context)}
    introduced = [x for x in excerpts(QUANTITY, record.get('response', ''))
                  if re.sub(r'\s+', ' ', x['quote']).casefold() not in known]
    if introduced:
        flags.append({'kind': 'quantity_not_verbatim_in_prompt', 'field': 'response', 'evidence': introduced,
                      'interpretation': 'Often a legitimate proposed amount or paraphrase; manually check asserted facts and changed constraints.'})
    return flags


def repeated_final_phrases(rows, words=10, minimum_rows=4):
    occurrences = defaultdict(set)
    for row in rows:
        tokens = re.findall(r"\b[\w'-]+\b", row['record']['response'].casefold())
        for start in range(len(tokens) - words + 1):
            occurrences[' '.join(tokens[start:start + words])].add(row['id'])
    frequent = [(phrase, ids) for phrase, ids in occurrences.items() if len(ids) >= minimum_rows]
    frequent.sort(key=lambda x: (-len(x[1]), x[0]))
    # Keep a bounded evidence list; totals record all matching n-grams.
    return {'ngram_words': words, 'minimum_rows': minimum_rows, 'matching_ngrams': len(frequent),
            'top100': [{'phrase': phrase, 'row_count': len(ids), 'ids': sorted(ids)} for phrase, ids in frequent[:100]],
            'interpretation': 'Repeated wording is candidate template evidence, not automatic rejection.'}


def audit(root, tokenizer=None):
    root = Path(root).resolve()
    rows, counts, changed = [], Counter(), []
    for path in sorted(root.glob('*/records/*/result.json')):
        try:
            result = runtime.load_result(path)
        except (runtime.BudgetStop, FileNotFoundError) as exc:
            changed.append({'path': str(path), 'reason': str(exc)})
            continue
        counts[result.get('status', 'missing')] += 1
        if result.get('status') != 'accepted':
            continue
        record = result['record']
        arm = path.parent.parent.parent.name
        item = {'id': arm + '/' + result['candidate_id'], 'arm': arm, 'path': str(path),
                'terminal_sha256': runtime.digest(path.read_bytes()), 'trait_id': result.get('trait_id'),
                'record': {key: record.get(key, '') for key in FIELDS}, 'flags': screen_record(record)}
        if tokenizer is not None:
            from scratch.dataset_refresh.validate_mixtures import token_audit, TOKENIZER
            from src.model_profile import model_profile
            training = {'messages': [{'role': 'system', 'content': record['system']},
                        {'role': 'user', 'content': record['user']},
                        {'role': 'assistant', 'content': record['response'], 'reasoning_content': record['reasoning']}],
                        'supervise': 'all'}
            try:
                item['token_audit'] = token_audit(training, tokenizer, model_profile(TOKENIZER), 8192)
            except (ValueError, AssertionError) as exc:
                item['flags'].append({'kind': 'training_stream_or_mask_failure', 'error': str(exc)})
        rows.append(item)
    return {'root': str(root), 'terminal_counts': dict(counts), 'accepted_rows': len(rows),
            'flagged_rows': sum(bool(row['flags']) for row in rows),
            'flag_counts': dict(Counter(flag['kind'] for row in rows for flag in row['flags'])),
            'unreadable_checkpoints': changed, 'tokenizer_checked': tokenizer is not None,
            'scope': 'Read-only triage snapshot. Flags require independent adjudication; no content repairs or exclusions applied.',
            'repeat_final_phrases_by_arm': {arm: repeated_final_phrases([r for r in rows if r['arm'] == arm])
                                           for arm in sorted({row['arm'] for row in rows})},
            'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tokenizer', action='store_true', help='Use cached Qwen tokenizer only; refuses a network download.')
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        from scratch.dataset_refresh.validate_mixtures import TOKENIZER
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    report = audit(args.root, tokenizer)
    output = Path(args.output).resolve()
    if output.is_relative_to(Path(args.root).resolve()):
        parser.error('Write the triage snapshot outside the frozen run root.')
    runtime.write_json(output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('rows', 'repeat_final_phrases_by_arm')}, indent=2))


if __name__ == '__main__':
    main()
