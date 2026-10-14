# ABOUTME: Preserve the frozen Tinker bridge while accelerating only durable JSON serialization.
# ABOUTME: Run outside the deployment; all model, evidence, accounting and locking behavior is inherited.
import json
import os
from pathlib import Path


def compact_save(self):
    temp = self.path.with_suffix('.tmp')
    # The C encoder avoids Python's per-field indented streaming loop. Keep
    # durability, same-directory replacement, schema, rows and all locks intact.
    with temp.open('w') as stream:
        stream.write(json.dumps(self.data, separators=(',', ':')))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, self.path)


def main():
    import sys
    sys.path.insert(0, '/srv/lasr/repo')
    if '--loaded-env' not in sys.argv:
        # The CPU environment owns dotenv; the frozen model environment does
        # not. Exec preserves credentials privately without command-line keys.
        from dotenv import load_dotenv
        load_dotenv('/srv/lasr/credentials.env')
        python='/srv/lasr/repo/src/infra/endpoints/tinker_env/.venv/bin/python'
        os.execv(python,[python,'-u',*sys.argv,'--loaded-env'])
    root = Path(sys.argv[1])
    owner = json.loads((root/'base/shim-owner.json').read_text())
    os.environ.update(owner['settings'])
    config = json.loads((root/'config.json').read_text())
    os.environ['TINKER_PRICE_RATES_JSON'] = json.dumps(config['price_rates'])
    original = json.loads((root/'base/identity.json').read_text())['data'][0]
    os.environ['TINKER_SHIM_INSTANCE'] = original['instance_id']
    if '--original' not in sys.argv:
        from src.infra.endpoints.tinker_budget import Budget
        Budget.save = compact_save
    from src.infra.endpoints.tinker_server import main as server_main
    if '--preflight' in sys.argv:
        import tinker
        import uvicorn
        assert os.environ.get('TINKER_API_KEY')
        print(json.dumps(dict(imports=True,checkpoint=os.environ['TINKER_CKPT'],
                              context=os.environ['TINKER_CONTEXT_WINDOW'],model_calls=0)))
        return
    server_main()


if __name__ == '__main__':
    main()
