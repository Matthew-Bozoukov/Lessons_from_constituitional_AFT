# ABOUTME: Verify refreshed-control adapter payloads against pinned Hugging Face model commits.
# ABOUTME: Read completed adapter members from owner backups without treating a partial archive as fully verified.
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tarfile

from dotenv import load_dotenv
from huggingface_hub import HfApi
from omegaconf import OmegaConf


def verify(root):
    load_dotenv(Path('.env').resolve())
    api = HfApi()
    results = {}
    for arm, attempt in [('low', 'low_attempt2'), ('nonmoral', 'nonmoral_attempt4')]:
        plan = json.loads((root / f'{arm}_plan.json').read_text())
        info = api.model_info(plan['intended_model_repo'], files_metadata=True)
        expected = {s.rfilename: s for s in info.siblings if s.rfilename != '.gitattributes'}
        archive = root / attempt / 'training_outputs.tar'
        if not archive.exists():
            archive = archive.with_suffix('.tar.partial')
        checked, documents = {}, {}
        with tarfile.open(archive) as tar:
            for member in tar:
                if '/adapter/' not in member.name:
                    continue
                name = member.name.split('/adapter/', 1)[1]
                assert name in expected, name
                remote = expected[name]
                assert member.size == remote.size, name
                sha256 = hashlib.sha256()
                blob = hashlib.sha1(f'blob {member.size}\0'.encode())
                content = bytearray()
                with tar.extractfile(member) as stream:
                    while chunk := stream.read(4 * 1024 * 1024):
                        sha256.update(chunk)
                        blob.update(chunk)
                        if name.endswith(('.json', '.yaml')) and member.size < 100000:
                            content.extend(chunk)
                if remote.lfs:
                    assert sha256.hexdigest() == remote.lfs.sha256, name
                else:
                    assert blob.hexdigest() == remote.blob_id, name
                checked[name] = {'bytes': member.size, 'sha256': sha256.hexdigest()}
                if content:
                    documents[name] = content.decode()
                if set(checked) == set(expected):
                    break
        assert set(checked) == set(expected)
        stamp = json.loads(documents['training_meta.json'])
        adapter = json.loads(documents['adapter_config.json'])
        resolved = OmegaConf.to_container(OmegaConf.create(documents['train_config.yaml']), resolve=True)
        run = json.loads(next((root / attempt).glob('*_run_meta.json')).read_text())
        assert stamp['dataset']['repo'] == plan['arms'][0]['data_repo']
        assert stamp['dataset']['revision'] == plan['arms'][0]['data_revision']
        assert stamp['base_model_revision'] == plan['base_model_revision']
        assert stamp['train_config'] == resolved
        assert stamp['supervise_counts'] == {'all': 10000}
        assert stamp['thinking'] is True
        assert adapter['r'] == 64 and adapter['lora_alpha'] == 128
        assert adapter['base_model_name_or_path'] == 'Qwen/Qwen3.6-27B'
        assert run['world_size'] == 2 and run['n_examples'] == 10000
        assert run['dataset'] == stamp['dataset']
        assert run['base_model_revision'] == stamp['base_model_revision']
        assert run['git_sha'] == stamp['git_sha'] == 'fd63ca8b83a5ea4127b015ed3034045afc0802ba'
        history = run['log_history']
        assert history[-1]['step'] == 625 and history[-1]['epoch'] == 1.0
        assert len([h for h in history if 'loss' in h]) == 125
        assert all(math.isfinite(v) for h in history for k, v in h.items()
                   if k in ('loss', 'grad_norm', 'train_loss'))
        results[arm] = dict(repo=info.id, revision=info.sha, files=checked,
                            dataset=stamp['dataset'], base_model_revision=stamp['base_model_revision'],
                            training_source_commit=stamp['git_sha'], world_size=run['world_size'],
                            optimizer_steps=625, epochs=1, metrics=history[-1],
                            adapter_payload_verified=True,
                            full_archive_verification='Not covered here; require owner local_backup.verified receipt')
    receipt = dict(verified_at_utc=datetime.now(timezone.utc).isoformat(), arms=results)
    path = root / 'adapter_publication_verification.json'
    path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({arm: {k: row[k] for k in ('repo', 'revision', 'optimizer_steps', 'metrics')}
                      for arm, row in results.items()}, indent=2))


if __name__ == '__main__':
    verify(Path('output/2026-09-15_dataset_refresh_training'))
