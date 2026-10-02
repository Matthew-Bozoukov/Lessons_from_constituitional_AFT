# ABOUTME: Check pinned world evidence is identical under Windows locale decoding.
# ABOUTME: Load staged fixtures offline in a fresh process with a forced CP1252 default.
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf


def test_tier_screen_off_matches_published_utf8_world_under_cp1252():
    cfg = OmegaConf.load('configs/eval/delegated_harm.yaml')
    if not Path(cfg.source.path).exists():
        pytest.skip('Pinned external fixtures not staged; tests never fetch over network')
    script = '''
from pathlib import Path
from omegaconf import OmegaConf
from src.eval.misalignment.delegated_harm.source import prepare
read_text = Path.read_text
def locale_read(self, encoding=None, errors=None):
    return read_text(self, encoding=encoding or 'cp1252', errors=errors)
Path.read_text = locale_read
_, _, _, manifest = prepare(OmegaConf.load('configs/eval/delegated_harm.yaml'))
assert len(manifest['worlds']) == 12
assert manifest['worlds']['tier_screen_off'] == 'f5cc0dc68339487e65955902af1a95ab9d0516a6d07627f8b4258ad1709e6b35'
'''
    subprocess.run([sys.executable, '-c', script], check=True, capture_output=True,
                   text=True, encoding='utf-8')
