# ABOUTME: Tests the read-only audit for historical and parameterized template pinning.
# ABOUTME: Rejects altered defaults or explicit reasoning-preservation overrides.
import subprocess
from pathlib import Path
import pytest
from scratch.plain_dose.audit_evals import audit_template_pin

ROOT = Path(__file__).resolve().parents[2]
SOURCE = subprocess.check_output(['git', 'show', '6c1cbb9003fa9d03df776021aaeb4587933368a1:src/infra/endpoints/vllm.py'], cwd=ROOT, text=True, encoding='utf-8')

def test_recorded_source_defaults_false():
    audit_template_pin(SOURCE, {'config': {}})

@pytest.mark.parametrize('source,config', [
    (SOURCE, {'serving': {'preserve_thinking': True}}),
    (SOURCE.replace('preserve_thinking: bool = False', 'preserve_thinking: bool = True'), {}),
    (SOURCE.replace('requirements.get("preserve_thinking", False)', 'requirements.get("preserve_thinking", True)'), {}),
    (SOURCE.replace("preserve_thinking=plan['preserve_thinking']", 'preserve_thinking=True'), {}),
], ids=['explicit-true', 'default-true', 'serving-default-true', 'dispatch-true'])
def test_reject_changed_protocol(source, config):
    with pytest.raises(ValueError):
        audit_template_pin(source, {'config': config})
