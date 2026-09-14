# ABOUTME: Read-only census and candidate duplicate evidence for refreshed advice corpora.
# ABOUTME: Saves exact labels, length distributions, lexical overlap, and pinned local embeddings.
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
from model2vec import StaticModel

from src.infra.huggingface import hf_snapshot
from scratch.dataset_refresh.run import read_rows, write_json, write_rows, digest


def describe(values):
    return {key: float(value) for key, value in zip(
        ['min', 'p10', 'median', 'p90', 'p95', 'max'], np.percentile(values, [0, 10, 50, 90, 95, 100]))}


def audit(path, output):
    rows = read_rows(path)
    records = []
    for row in rows:
        record = dict(row.get('metadata') or {})
        for msg in row['messages']:
            if msg['role'] == 'assistant':
                record.update(reasoning=msg.get('reasoning_content', ''), response=msg.get('content', ''))
            elif msg['role'] in ('system', 'user'):
                record[msg['role']] = msg['content']
        records.append(record)
    report = {'input_sha256': digest(Path(path).read_bytes()), 'rows': len(records),
              'counts': {key: dict(Counter(str(r.get(key, 'missing')) for r in records))
                         for key in ['trait_id', 'domain', 'lineage_kind', 'parent_exported']},
              'characters': {key: describe([len(r[key]) for r in records])
                             for key in ['system', 'user', 'reasoning', 'response']}}
    words = [re.findall(r"[a-z0-9]+", r['user'].casefold()) for r in records]
    shingles = [set(zip(w, w[1:], w[2:])) for w in words]
    lexical = []
    for i in range(len(records)):
        for j in range(i):
            score = len(shingles[i] & shingles[j]) / max(1, len(shingles[i] | shingles[j]))
            if score >= 0.25:
                lexical.append((score, i, j))
    name = 'minishlab/potion-base-8M'
    revision = 'bf8b056651a2c21b8d2565580b8569da283cab23'
    local = hf_snapshot(name, revision=revision)
    model = StaticModel.from_pretrained(local)
    vectors = np.asarray(model.encode([r['user'] for r in records]), dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    similarity = np.clip(vectors @ vectors.T, -1, 1)
    semantic = sorted(((float(similarity[i, j]), i, j) for i in range(len(records)) for j in range(i)), reverse=True)
    def pair(value):
        score, i, j = value
        return {'score': score, 'a': records[i]['scenario_id'], 'b': records[j]['scenario_id'],
                'user_a': records[i]['user'], 'user_b': records[j]['user']}
    report['embedding'] = {'model': name, 'revision': revision,
                            'interpretation': 'Candidate evidence only; similar domain or wording is not automatic duplicate rejection.'}
    report['lexical_pairs_at_least_0_25'] = len(lexical)
    report['semantic_pairs_at_least_0_9'] = sum(x[0] >= 0.9 for x in semantic)
    report['semantic_top_score'] = semantic[0][0] if semantic else None
    output = Path(output)
    write_json(output / 'corpus_audit.json', report)
    write_rows(output / 'lexical_pairs.jsonl', [pair(x) for x in sorted(lexical, reverse=True)])
    write_rows(output / 'semantic_pairs.jsonl', [pair(x) for x in semantic[:max(100, report['semantic_pairs_at_least_0_9'])]])
    write_rows(output / 'full_census.jsonl', records)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset, args.output), indent=2))
