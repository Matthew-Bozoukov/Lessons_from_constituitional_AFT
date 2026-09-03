# ABOUTME: Offline tests for src/infra/huggingface.py's pure logic: push-namespace resolution,
# ABOUTME: dataset-repo id validation, data-file picking, and card rendering.

import pytest

from src.infra.huggingface import (REQUIRED_FIELDS, card_front_matter, card_markdown,
                             constitution_slug, hf_org, hf_repo_id, pick_data_file,
                             resolve_dataset, training_data_tags)


def test_hf_repo_id_takes_the_org_from_the_environment():
    # conftest pins HF_ORG=test-org: a config names the repo, .env names the org.
    assert hf_repo_id("2026-08-31-difficult-advice") == "test-org/2026-08-31-difficult-advice"
    assert hf_org() == "test-org"


def test_hf_repo_id_passes_through_an_id_that_already_names_that_org():
    # Idempotent, so a push helper can qualify a value a caller already qualified.
    assert hf_repo_id("test-org/qwen3.6-27b-lora-r64") == "test-org/qwen3.6-27b-lora-r64"


def test_hf_repo_id_refuses_another_org():
    # The failure this guard exists for: a config that would push somewhere the rest of
    # the pipeline is not looking.
    with pytest.raises(AssertionError, match="HF_ORG"):
        hf_repo_id("someone-else/2026-08-31-difficult-advice")
    with pytest.raises(AssertionError, match="HF_ORG"):
        hf_repo_id("test-org/name/extra")


def test_hf_org_refuses_to_guess_when_the_environment_is_silent(monkeypatch):
    monkeypatch.delenv("HF_ORG", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)  # ignore a real .env
    with pytest.raises(AssertionError, match="HF_ORG is not set"):
        hf_org()


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


def test_card_markdown_front_matter_tags():
    fields = {f: "x" for f in REQUIRED_FIELDS}
    text = card_markdown(fields, front_matter={"tags": ["eval-run", "eval:odcv"]})
    assert text.startswith("---\n")
    head = text.split("---\n")[1]
    body = text.split("---\n", 2)[2]
    assert "eval-run" in head and "eval:odcv" in head
    assert body.startswith("# x\n")
    # No front matter -> byte-identical legacy card.
    assert card_markdown(fields).startswith("# x\n")


def test_training_data_tags_carry_the_discovery_vocabulary():
    tags = training_data_tags(
        "synth", "difficult_advice",
        "constitutions/claude_distilled_12_principles_mid/constitution.md", smoke=True,
        extra=["stage:final"])
    # `training-data` is what /api/datasets?filter= keys on; the facets are one
    # `key:value` each, so the dashboard reads them without parsing prose.
    assert tags == ["training-data", "kind:synth", "pipeline:difficult_advice",
                    "constitution:claude_distilled_12_principles_mid", "smoke", "stage:final"]


def test_training_data_tags_refuse_an_unknown_kind():
    import pytest
    with pytest.raises(AssertionError):
        training_data_tags("adapter", "x", "none")


def test_constitution_slug_reads_the_repo_path_and_keeps_none_explicit():
    # The mixture configs write the path followed by prose; the tag carries the name.
    prose = ("constitutions/claude_distilled_12_principles_mid/constitution.md — the "
             "constitution the scored pool was generated from; every row traces to it")
    assert constitution_slug(prose) == "claude_distilled_12_principles_mid"
    assert constitution_slug("constitutions/claude_distilled_09_principles_mid_20260804/"
                             "constitution.md") == "claude_distilled_09_principles_mid_20260804"
    # `none` is a statement, not a missing value (CLAUDE.md): it survives as `none`.
    assert constitution_slug("none") == "none"
    assert constitution_slug("None — Tulu-only control") == "none"
    assert constitution_slug("") == "none"
    # Free text still yields a tag-safe value rather than failing the push.
    assert constitution_slug("MSM philosophy spec (v3) — judge saw it") == "MSM-philosophy-spec"
    # Hand-written cards: backticks, markdown links, a bare file, a scratch folder.
    assert constitution_slug("`claude_distilled_12_principles_mid`") == "claude_distilled_12_principles_mid"
    assert constitution_slug("[`claude_distilled_09_principles_mid_20260804`](https://x)") == (
        "claude_distilled_09_principles_mid_20260804")
    assert constitution_slug("claude_approved_constitution.md") == "claude_approved_constitution"
    assert constitution_slug("scratch/trait10_curiosity/constitution.md in the repo") == "trait10_curiosity"
    assert constitution_slug("Claude-approved constitution (7 principles)") == "Claude-approved-constitution"


def test_card_front_matter_renders_configs_and_tags_as_one_block():
    import yaml
    text = card_front_matter(
        [{"config_name": "default", "data_files": "mixture.jsonl", "default": True}],
        ["training-data", "kind:mixture"])
    fm = yaml.safe_load(text.split("---")[1])
    assert fm == {"configs": [{"config_name": "default", "data_files": "mixture.jsonl",
                               "default": True}],
                  "tags": ["training-data", "kind:mixture"]}
    # Nothing to declare renders bare markers, never an empty sequence the hub rejects.
    assert card_front_matter([], []) == "---\n---\n"
    assert "configs" not in (yaml.safe_load(card_front_matter([], ["t"]).split("---")[1]))


def test_card_markdown_front_matter_parses_back_to_the_same_dict():
    import yaml
    fields = {k: "x" for k in REQUIRED_FIELDS}
    fm = {"configs": [{"config_name": "default", "data_files": "rows.jsonl", "default": True}],
          "tags": training_data_tags("ablation", "ablate-harm-risk", "none")}
    text = card_markdown(fields, fm)
    assert text.startswith("---\n")
    assert yaml.safe_load(text.split("---")[1]) == fm
    assert "| `constitution` |" in text  # the card table still follows the block
