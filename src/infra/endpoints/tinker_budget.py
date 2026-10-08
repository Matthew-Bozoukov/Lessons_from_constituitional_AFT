# ABOUTME: Durable conservative Tinker inference reservations for one owned Linux sampling process.
# ABOUTME: Unknown request outcomes keep their full reservation; saved ceilings never reset on restart.
import json
import os
from pathlib import Path
import threading
import time
import uuid


class Budget:
    def __init__(self, path, ceiling, model, checkpoint):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mutex = threading.Lock()
        # One server must own each ledger. Kernel releases ownership on process exit.
        import fcntl
        self.owner = self.path.with_suffix('.lock').open('a')
        fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prices = (0.78, 1.94) if model.endswith(':peft:131072') else (0.33, 0.84)
        identity = dict(ceiling_usd=float(ceiling), model=model, checkpoint=checkpoint,
                        input_per_million=prices[0], output_per_million=prices[1])
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            assert all(self.data[k] == v for k, v in identity.items()), 'Frozen budget identity changed'
        else:
            self.data = dict(identity, requests={}, cost_basis='uncached upper bound; not provider invoice')
            self.save()

    def save(self):
        temp = self.path.with_suffix('.tmp')
        with temp.open('w') as f:
            json.dump(self.data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, self.path)

    def reserve(self, prompt, allowance):
        with self.mutex:
            upper = (prompt*self.data['input_per_million'] + allowance*self.data['output_per_million']) / 1e6
            used = sum(x['upper_usd'] for x in self.data['requests'].values())
            if used + upper > self.data['ceiling_usd']:
                raise RuntimeError('Frozen Tinker spending ceiling reached; hold for review')
            key = uuid.uuid4().hex
            self.data['requests'][key] = dict(prompt_tokens=prompt, max_tokens=allowance,
                upper_usd=upper, state='reserved', at=time.time())
            self.save()
            return key

    def settle(self, key, completion):
        with self.mutex:
            row = self.data['requests'][key]
            assert row['state'] == 'reserved' and 0 <= completion <= row['max_tokens']
            row.update(state='completed', completion_tokens=completion,
                upper_usd=(row['prompt_tokens']*self.data['input_per_million'] +
                           completion*self.data['output_per_million']) / 1e6)
            self.save()
