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
    # ... but the discovery tags are declared from the first commit, so a repo is
    # findable on the Hub as soon as it exists.
    fm = _front_matter(dataset_card(FIELDS, [], False, ["training-data", "kind:synth"]))
    assert fm == {"tags": ["training-data", "kind:synth"]}


def test_dataset_card_declares_tags_beside_configs():
    fm = _front_matter(dataset_card(FIELDS, ["stage_2_a.jsonl"], True,
                                    ["training-data", "kind:synth", "smoke"]))
    assert [c["config_name"] for c in fm["configs"]] == ["dataset", "stage_2_a"]
    assert fm["tags"] == ["training-data", "kind:synth", "smoke"]


class _FakeApi:
    def __init__(self):
        self.commits = []

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        self.commits.append((commit_message, [op.path_in_repo for op in operations]))


def _cache(tmp_path):
    # The repo NAME alone: the org is the environment's (src.huggingface.hf_org).
    c = StageCache(tmp_path, "repo", token="offline", card_fields=FIELDS,
                   tags=["training-data", "kind:synth"])
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
    fm = _front_matter(c._readme())
    cfgs = fm["configs"]
    assert [x["config_name"] for x in cfgs] == ["dataset", "stage_2_scenarios"]
    assert cfgs[0]["default"] is True
    # The tags ride on every refreshed README, not only the first commit's.
    assert fm["tags"] == ["training-data", "kind:synth"]


def test_mixture_dataset_spec_loads_default_config_and_balances(monkeypatch):
    # The consumer half of the contract: `dataset: org/repo` pulls the default config
    # via load_dataset (sha-pinned), and — because the pool is fully materialised —
    # balance_by works on it. Hub mocked; balancing and payloads run for real.
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    import src.huggingface as hf
    from src.data.mixture import build_mixture as bm

    rows = [{"messages": [{"role": "user", "content": "q", "reasoning_content": None},
                          {"role": "assistant", "content": f"a{i}",
                           "reasoning_content": None}],
             "metadata": {"trait_id": f"t{i % 2}", "supervise": None}}
            for i in range(8)]
    monkeypatch.setattr(hf, "hf_api", lambda: SimpleNamespace(
        repo_info=lambda repo, repo_type, revision: SimpleNamespace(sha="c0ffee" * 6)))
    monkeypatch.setattr(hf, "hf_token", lambda: "offline")
    calls = {}
    monkeypatch.setattr(bm, "load_dataset", lambda repo, revision, split, token:
                        calls.update(repo=repo, revision=revision) or rows)

    class _Tok:  # fixed-width render: every row counts 10 tokens
        def apply_chat_template(self, msgs, **kw):
            return {"input_ids": [0] * 10}

    got, kind = bm._take_interchange(
        _Tok(), OmegaConf.create({"max_seq_len": 99}), "da",
        {"dataset": "org/synth-run", "reasoning": "none", "balance_by": "trait_id",
         "examples": 4},
        budget=("examples", 4), seed=0, render_kwargs={})
    assert kind == "none" and len(got) == 4
    assert calls["repo"] == "org/synth-run" and calls["revision"].startswith("c0ffee")
    # balanced: two rows from each trait bucket, and interchange fields intact
    # (clean_messages drops load_dataset's None-filled reasoning_content)
    assert all("reasoning_content" not in m for r in got for m in r["messages"])
    assert all(r["source"] == "da" and r["n_tokens"] == 10 for r in got)
