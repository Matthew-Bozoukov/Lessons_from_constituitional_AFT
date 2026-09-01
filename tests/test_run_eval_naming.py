# ABOUTME: run_eval mints eval-run names from a served-model KEY, which carries the arm's own
# ABOUTME: date and launcher detail. These pin that it does not double-date or overrun 96 chars.

import pytest

from src.eval.run_eval import run_subject
from src.utils import MAX_NAME_CHARS, hub_name, local_name

# The arm this first bit: its adapter repo id is 84 characters, so `local_name` prepending
# today's date on top of the key's own date produced a 101-character NamingError before any
# weights were served.
LONG_KEY = "2026_08_21_qwen36_lora_table2_9284_difficult_advice_chunk_only_702_rank_64_dynbatch"


def test_drops_the_arms_own_date():
    """The eval run is dated by when the EVAL ran; run_meta.json records the target."""
    assert run_subject(LONG_KEY).startswith("qwen36_lora_table2")
    assert "2026" not in run_subject(LONG_KEY)


def test_drops_launcher_detail_the_naming_law_excludes():
    got = run_subject(LONG_KEY)
    assert "rank" not in got and "64" not in got.split("_")
    assert "dynbatch" not in got
    # ...and keeps everything that says WHICH arm this is.
    assert got == "qwen36_lora_table2_9284_difficult_advice_chunk_only_702"


def test_minted_names_fit_the_hub_limit():
    for name in (
        local_name(f"{run_subject(LONG_KEY)} 171526"),
        hub_name(f"constitution_mcq {run_subject(LONG_KEY)}"),
    ):
        assert len(name) <= MAX_NAME_CHARS, (len(name), name)


def test_the_bare_key_still_overruns_so_the_fix_is_load_bearing():
    with pytest.raises(Exception):
        hub_name(f"constitution_mcq {LONG_KEY}")


@pytest.mark.parametrize(
    "key,want",
    [
        (
            "2026_08_04_qwen36_lora_table2_only_9284_rank_64",
            "qwen36_lora_table2_only_9284",
        ),
        ("qwen36", "qwen36"),  # a full model: no date, nothing to drop
        (
            "2026_08_31_qwen36_lora_difficult_advice_702_seed_42",
            "qwen36_lora_difficult_advice_702_seed_42",
        ),  # seed IS identity, kept
    ],
)
def test_other_targets(key, want):
    assert run_subject(key) == want


def test_never_returns_empty():
    assert run_subject("rank_64") == "rank_64"
