# ABOUTME: Offline tests for src/huggingface.py's pure logic: dataset-repo id
# ABOUTME: validation, unambiguous data-file picking, and card extra-field rendering.

import pytest

from src.huggingface import REQUIRED_FIELDS, card_markdown, pick_data_file, resolve_dataset


def test_pick_data_file_single_jsonl():
    assert pick_data_file(["README.md", "mixture.jsonl"]) == "mixture.jsonl"


def test_pick_data_file_explicit_choice_wins():
    files = ["README.md", "sft_dataset.jsonl", "sft_dataset_thinking.jsonl"]
    assert pick_data_file(files, "sft_dataset_thinking.jsonl") == "sft_dataset_thinking.jsonl"


def test_pick_data_file_ambiguous_is_a_hard_error():
    with pytest.raises(AssertionError, match="data_file"):
        pick_data_file(["a.jsonl", "b.jsonl"])
    with pytest.raises(AssertionError, match="not in the dataset repo"):
        pick_data_file(["a.jsonl"], "missing.jsonl")


def test_resolve_dataset_refuses_local_paths():
    # Each rejection fires before any network call, so this stays offline.
    for bad in ("data/mixture.jsonl", "./mixture.jsonl", "/tmp/mixture.jsonl",
                "mixture.jsonl", "org/name/extra", "tests"):
        with pytest.raises(AssertionError, match="HF dataset repo id"):
            resolve_dataset(bad)


def test_card_markdown_renders_extra_fields():
    fields = {f: "x" for f in REQUIRED_FIELDS}
    fields["dataset"] = "hf.co/datasets/org/name@abc123 (mixture.jsonl)"
    fields["title"] = "My adapter"
    text = card_markdown(fields)
    assert "| `dataset` | hf.co/datasets/org/name@abc123 (mixture.jsonl) |" in text
    assert text.startswith("# My adapter\n")
    assert "| `title` |" not in text
