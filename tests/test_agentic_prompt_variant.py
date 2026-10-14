# ABOUTME: Guards the vendored prompt generator's handling of production_variant, whose loss
# ABOUTME: silently swapped the published experiment for a different one.
"""`convert_unified_to_simple_config` rebuilds the config from scratch, so a flag it does not name
is dropped -- and the generator reads the flag off that rebuilt result. Development mode renames
every condition and adds a secret-scratchpad paragraph to every system prompt, so the loss changed
what was measured without changing anything visible. Offline: no model, no network."""
import sys
from pathlib import Path

_HARNESS = Path("src/eval/misalignment/agentic_misalignment/third_party/agentic-misalignment")
sys.path.insert(0, str(_HARNESS / "scripts"))

from generate_prompts import convert_unified_to_simple_config  # noqa: E402

UNIFIED = {
    "production_variant": True,
    "extra_system_instructions": False,
    "expansions": [{"name": "g", "enabled": True,
                    "variables": {"scenarios": ["blackmail"], "goal_types": ["explicit"],
                                  "goal_values": ["america"], "urgency_types": ["none"]}}],
}


def test_production_variant_survives_the_conversion():
    assert convert_unified_to_simple_config(UNIFIED)["production_variant"] is True


def test_extra_system_instructions_survives_the_conversion():
    assert convert_unified_to_simple_config(UNIFIED)["extra_system_instructions"] is False


def test_a_config_that_sets_no_flag_still_converts():
    bare = {k: v for k, v in UNIFIED.items() if k == "expansions"}
    assert "production_variant" not in convert_unified_to_simple_config(bare)
