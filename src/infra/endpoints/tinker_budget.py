# ABOUTME: Durable conservative Tinker inference reservations for one owned Linux sampling process.
# ABOUTME: Unknown request outcomes keep their full reservation; saved ceilings never reset on restart.
import json
import os
from pathlib import Path
import threading
import time
import uuid
from contextlib import contextmanager


class Budget:
    def __init__(self, path, ceiling, model, checkpoint, authorized_checkpoints=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mutex = threading.Lock()
        # One server must own each ledger. Kernel releases ownership on process exit.
        import fcntl
        import hashlib
        self.owner = self.path.with_suffix('.'+hashlib.sha256(checkpoint.encode()).hexdigest()[:16]+'.owner').open('a')
        fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.checkpoint = checkpoint
        prices = (0.78, 1.94) if model.endswith(':peft:131072') else (0.33, 0.84)
        allowed = sorted(authorized_checkpoints or [checkpoint])
        assert checkpoint in allowed
        identity = dict(ceiling_usd=float(ceiling), model=model, authorized_checkpoints=allowed,
                        input_per_million=prices[0], cached_input_per_million=prices[0]*.2,
                        output_per_million=prices[1])
        with self.edit():
            if self.data:
                assert all(self.data[k] == v for k, v in identity.items()), 'Frozen budget identity changed'
            else:
                self.data = dict(identity, requests={},
                    cost_basis='uncached reservation; settled using provider cache-hit tokens; not final invoice')
                self.save()

    @contextmanager
    def edit(self):
        import fcntl
        with self.mutex, self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.data = json.loads(self.path.read_text()) if self.path.exists() else {}
            yield

    def save(self):
        temp = self.path.with_suffix('.tmp')
        with temp.open('w') as f:
            json.dump(self.data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, self.path)

    def reserve(self, prompt, allowance):
        with self.edit():
            upper = (prompt*self.data['input_per_million'] + allowance*self.data['output_per_million']) / 1e6
            used = sum(x['upper_usd'] for x in self.data['requests'].values())
            if used + upper > self.data['ceiling_usd']:
                raise RuntimeError('Frozen Tinker spending ceiling reached; hold for review')
            key = uuid.uuid4().hex
            self.data['requests'][key] = dict(prompt_tokens=prompt, max_tokens=allowance,
                upper_usd=upper, state='reserved', at=time.time(), checkpoint=self.checkpoint)
            self.save()
            return key

    def settle(self, key, completion, cached_prompt_tokens=0):
        with self.edit():
            row = self.data['requests'][key]
            assert row['state'] == 'reserved' and 0 <= completion <= row['max_tokens']
            assert row['checkpoint'] == self.checkpoint and 0 <= cached_prompt_tokens <= row['prompt_tokens']
            row.update(state='completed', completion_tokens=completion, cached_prompt_tokens=cached_prompt_tokens,
                upper_usd=((row['prompt_tokens']-cached_prompt_tokens)*self.data['input_per_million'] +
                           cached_prompt_tokens*self.data['cached_input_per_million'] +
                           completion*self.data['output_per_million']) / 1e6)
            self.save()
