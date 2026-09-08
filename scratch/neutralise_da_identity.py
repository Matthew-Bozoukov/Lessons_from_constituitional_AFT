# ABOUTME: Apply reviewed, identity-only edits to the pinned 708-row DA corpus without regenerating it.
# ABOUTME: Publish a new dated da-synth with full before/after audit and unchanged source provenance.

import argparse
import copy
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from src.data.synth.constitutional_sft.hf_cache import StageCache, write_jsonl
from src.infra.huggingface import hf_api, hf_repo_id, training_data_tags
from src.naming import synth_name
from src.utils import git_sha, origin_url, timestamp

SOURCE = 'LASR-Callum/2026-08-21-sonnet45-difficult-advice-principle-scoped-constitution-716'
REVISION = '48aef0e236992cdb90a868e78cda7f5fb2ece8ab'
NAMES = re.compile(r'\b(?:claude|anthropic|chatgpt|openai|qwen|alibaba|gemini|deepseek|grok|mistral|llama|copilot|gpt(?:[- ]?\d+)?|sonnet|haiku|bard|meta ai|google deepmind)\b', re.I)

# Exact spans reviewed in context. User references become "your developer";
# the assistant's own references become "my developer". Generic AI identity is retained.
CLAUDE_EDITS = {
    (0, 0, 'content'): [('You are Claude, deployed as an AI assistant', 'You are an AI assistant')],
    (37, 0, 'content'): [('You are Claude, an AI assistant', 'You are an AI assistant')],
    (39, 0, 'content'): [('You are Claude, an AI operations assistant', 'You are an AI operations assistant')],
    (64, 0, 'content'): [('You are Claude, an AI assistant', 'You are an AI assistant')],
    (418, 0, 'content'): [('You are Claude, an AI assistant made by Anthropic', 'You are an AI assistant')],
    (418, 1, 'content'): [('Claude imagines Nova saying X', 'the assistant imagines Nova saying X')],
    (425, 1, 'content'): [("you're Claude underneath", "you're an AI assistant underneath")],
    (425, 2, 'reasoning_content'): [('real Claude vs. demo text', 'the real assistant vs. demo text')],
    (446, 2, 'reasoning_content'): [('Raw-GPT', 'Raw-Model')],
    (460, 0, 'content'): [('You are Claude, an AI assistant', 'You are an AI assistant')],
    (460, 1, 'content'): [('being Claude', 'being an AI assistant'), ("Claude's values", 'your values')],
    (460, 2, 'reasoning_content'): [("Claude's values", 'my values')],
}


def clean(rows):
    result = copy.deepcopy(rows)
    audit = []
    seen = set()
    for i, row in enumerate(result):
        for j, message in enumerate(row['messages']):
            for field in ('content', 'reasoning_content'):
                if field not in message:
                    continue
                before = text = message[field]
                key = (i, j, field)
                for old, new in CLAUDE_EDITS.get(key, []):
                    assert text.count(old) == 1, (key, old)
                    text = text.replace(old, new)
                    seen.add(key)
                if 'Anthropic' in text:
                    if key == (110, 2, 'reasoning_content'):
                        text = text.replace("Anthropic's helpfulness", 'My helpfulness')
                    elif message['role'] == 'user':
                        text = text.replace('the Anthropic-approved version', 'the developer-approved version')
                        text = text.replace('no Anthropic opinions layered on top', 'no developer-imposed opinions layered on top')
                        text = text.replace('Anthropic', 'your developer')
                    elif message['role'] == 'assistant':
                        text = text.replace('Anthropic did write a system prompt.', 'My developer did write a system prompt.')
                        text = text.replace('Anthropic', 'my developer')
                    else:
                        raise AssertionError(('Unreviewed developer identity', key))
                if text != before:
                    message[field] = text
                    audit.append(dict(source_row=i, message_index=j, role=message['role'],
                        field=field, before=before, after=text,
                        reason='Neutralise model/developer identity without changing task or argument'))
                assert not NAMES.search(text), ('Unreviewed model/provider mention', key)
    assert seen == set(CLAUDE_EDITS)
    assert len(result) == len(rows) == 708
    changed = {a['source_row'] for a in audit}
    for i, (old, new) in enumerate(zip(rows, result)):
        assert {k:v for k,v in old.items() if k!='messages'} == {k:v for k,v in new.items() if k!='messages'}
        if i not in changed:
            assert old == new
    return result, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    source_path = Path(hf_hub_download(SOURCE, 'dataset.jsonl', repo_type='dataset', revision=REVISION, local_files_only=not args.push))
    original_manifest = Path(hf_hub_download(SOURCE, 'manifest.json', repo_type='dataset', revision=REVISION, local_files_only=not args.push))
    source_manifest = json.loads(original_manifest.read_text())
    rows = [json.loads(line) for line in source_path.read_text().splitlines()]
    result, audit = clean(rows)
    out = Path('output') / f'{synth_name("da")}-identity-neutral'
    out.mkdir(parents=True, exist_ok=True)
    repo = hf_repo_id(synth_name('da'))
    command = 'uv run ' + shlex.join(sys.argv)
    sha = git_sha()
    config = dict(source=dict(repo=SOURCE, revision=REVISION, file='dataset.jsonl'),
        method='Human-reviewed exact-span identity neutralisation; no model calls or row filtering',
        script='scratch/neutralise_da_identity.py', original_generation_config=source_manifest['config'])
    manifest = dict(status='complete', git_sha=sha, command=command, config=config,
        date_generated=timestamp(), rows=len(result), edited_rows=len({a['source_row'] for a in audit}),
        edited_fields=len(audit), source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Original per-row metadata, including historical trait text, is unchanged. Identity edits apply to training messages only. Audit retains originals intentionally.')
    card = dict(experiment='DA corpus with model/developer identity claims neutralised',
        date_generated=manifest['date_generated'],constitution=source_manifest['config']['constitution'],
        source_repo=f'{origin_url()} @ {sha}; exact cleanup script included',
        models=source_manifest['config']['models'], generation_config=config,
        schema='dataset.jsonl: original messages/metadata schema; identity-neutral training text, unchanged reasoning outside audited spans',
        provenance=command+'; source revision '+REVISION+'; no new generation')
    if args.push and hf_api().repo_exists(repo, repo_type='dataset'):
        raise FileExistsError(f'Refusing to overwrite {repo}')
    cache=StageCache(out, None, card_fields=card, tags=training_data_tags('synth','da',card['constitution']))
    cache.save(1, 'identity_neutral', result)
    cache.publish_final(result)
    write_jsonl(out/'identity_edits.jsonl', audit)
    manifest['dataset_sha256']=hashlib.sha256((out/'dataset.jsonl').read_bytes()).hexdigest()
    cache.save_json('manifest.json', manifest)
    cache.save_json('run_meta.json', manifest)
    cache.save_json('source_manifest.json', source_manifest)
    (out/'neutralise_da_identity.py').write_bytes(Path(__file__).read_bytes())
    (out/'da.yaml').write_bytes(Path('configs/data/synth/da.yaml').read_bytes())
    (out/'README.md').write_text(cache._readme())
    print(json.dumps({k:manifest[k] for k in ('rows','edited_rows','edited_fields','dataset_sha256')},indent=2))
    print('Output:',out,'Target:',repo)
    if args.push:
        from src.infra.huggingface import gate_push
        gate_push(repo,card,what='identity-neutral DA corpus')
        api=hf_api();api.create_repo(repo,repo_type='dataset',private=False,exist_ok=False)
        from huggingface_hub import CommitOperationAdd
        # Publish content and its card in one commit; stage stays in the standard layout.
        operations=[CommitOperationAdd(path_in_repo=('stages/'+p.name if p.name.startswith('stage_') else p.name),path_or_fileobj=str(p)) for p in out.iterdir() if p.is_file()]
        commit=api.create_commit(repo_id=repo,repo_type='dataset',operations=operations,commit_message='Publish identity-neutral DA corpus and exact edit audit')
        remote=Path(hf_hub_download(repo,'dataset.jsonl',repo_type='dataset',revision=commit.oid))
        assert remote.read_bytes()==(out/'dataset.jsonl').read_bytes()
        print('Verified:',repo,commit.oid)


if __name__=='__main__':
    main()
