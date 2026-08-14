# ABOUTME: Offline tests for the synth->mixture HF layout: dataset.jsonl at the root as
# ABOUTME: the default config, stage snapshots under stages/, configs block never lagging.

import yaml

from src.data.synth.hf_cache import StageCache, dataset_card

FIELDS = {"experiment": "x", "date_generated": "2026-08-14", "constitution": "none",
          "source_repo": "r @ sha", "models": "m", "generation_config": "{}",
          "schema": "s", "provenance": "p"}


def _front_matter(text: str) -> dict | None:
    return yaml.safe_load(text.split("---")[1])


def test_dataset_card_declares_default_and_stage_configs():
    text = dataset_card(FIELDS, ["stage_10_b.jsonl", "stage_2_a.jsonl"], True)
    cfgs = _front_matter(text)["configs"]
    # `dataset` first and default (load_dataset(repo) fetches it alone); stages sort
    # numerically, not lexically (stage_10 after stage_2), and live under stages/.
    assert [c["config_name"] for c in cfgs] == ["dataset", "stage_2_a", "stage_10_b"]
    assert cfgs[0] == {"config_name": "dataset", "data_files": "dataset.jsonl",
                       "default": True}
    assert cfgs[1]["data_files"] == "stages/stage_2_a.jsonl"
    assert "| `constitution` |" in text  # the card table still follows


def test_dataset_card_before_any_upload_omits_the_configs_key():
    # An empty `configs:` sequence is invalid front-matter on the hub.
    assert _front_matter(dataset_card(FIELDS, [], False)) is None


class _FakeApi:
    def __init__(self):
        self.commits = []

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        self.commits.append((commit_message, [op.path_in_repo for op in operations]))


def _cache(tmp_path):
    c = StageCache(tmp_path, "org/repo", token="offline", card_fields=FIELDS)
    c._api = _FakeApi()  # short-circuits _hf(): no repo creation, no network
    return c


def test_stage_save_lands_under_stages_with_readme_in_the_same_commit(tmp_path):
    c = _cache(tmp_path)
    c.save(2, "scenarios", [{"a": 1}])
    msg, paths = c._api.commits[0]
    assert paths == ["stages/stage_2_scenarios.jsonl", "README.md"]
    assert "stage 2" in msg
    assert c.load(2, "scenarios") == [{"a": 1}]  # local layout stays flat (resume)


def test_publish_final_is_the_root_dataset_and_default_config(tmp_path):
    c = _cache(tmp_path)
    c.save(2, "scenarios", [{"a": 1}])
    c.publish_final([{"messages": []}])
    _, paths = c._api.commits[-1]
    assert paths == ["dataset.jsonl", "README.md"]
    cfgs = _front_matter(c._readme())["configs"]
    assert [x["config_name"] for x in cfgs] == ["dataset", "stage_2_scenarios"]
    assert cfgs[0]["default"] is True
