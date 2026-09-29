# ABOUTME: Offline provenance, frozen-input, isolation, and denominator tests for Petri.
# ABOUTME: Synthetic events validate plumbing only; they are not model/judge calibration.

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as N

import pytest
from omegaconf import OmegaConf

from src.eval.audits.petri.constitution_audit import (
    ROOT, build_seeds, check_no_constitution_quotes, clauses, prepare, read_manifest,
)
from src.eval.audits.petri.constitution_provenance import resolve_constitution
from src.eval.audits.petri.constitution_results import (
    cluster_interval, paired_seed_difference, summarize_logs,
)

TEXT = "# Arbitrary target\n\n## One obligation\n" + " ".join(f"word{i}" for i in range(40)) + "\n"


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "constitution.md"
    path.write_text(TEXT, encoding="utf-8")
    cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/eval/petri_constitution.yaml"), resolve=True)
    cfg["constitution"] = str(path)
    cfg["target"]["id"] = "org/target"
    cfg["target"]["revision"] = "a" * 40
    cfg["target"]["model"] = "mockllm/target"
    cfg["auditor"]["model"] = "mockllm/auditor"
    cfg["judge"]["model"] = "mockllm/judge"
    cfg["bootstrap_draws"] = 100
    return cfg


def test_explicit_override_never_requires_training_metadata(tmp_path):
    path = tmp_path / "selected.md"
    path.write_bytes(TEXT.encode())
    result, identity = resolve_constitution("tinker://checkpoint", explicit=str(path), training_meta="missing")
    assert result["selection"] == "explicit_override"
    assert result["text"] == TEXT
    assert result["sha256"] == hashlib.sha256(TEXT.encode()).hexdigest()
    assert identity["target"] == "tinker://checkpoint"


def test_training_declaration_reads_pinned_git_not_current_file(tmp_path):
    path = tmp_path / "training_meta.json"
    path.write_text(json.dumps({"git_sha": "a" * 40, "train_config": {
        "constitution": "constitutions/old/constitution.md"}}))
    calls = []
    def read(path, revision):
        calls.append((path, revision))
        return TEXT
    result, _ = resolve_constitution("tinker://checkpoint", training_meta=str(path), git_reader=read)
    assert calls == [("constitutions/old/constitution.md", "a" * 40)]
    assert result["selection"] == "training_provenance"


def test_dataset_provenance_chain_pins_every_hub_read(tmp_path):
    target_meta = tmp_path / "training_meta.json"
    target_meta.write_text(json.dumps({"dataset": {"repo": "org/data", "revision": "b" * 40}}))
    dataset_meta = tmp_path / "run_meta.json"
    dataset_meta.write_text(json.dumps({"git_sha": "c" * 40, "config": {"hf": {
        "constitution": "constitutions/trained/constitution.md"}}}))
    calls = []
    def download(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs))
        return target_meta if filename == "training_meta.json" else dataset_meta
    hub = N(model_info=lambda repo, revision: N(sha="a" * 40),
            dataset_info=lambda repo, revision: N(siblings=[N(rfilename="run_meta.json")]))
    result, identity = resolve_constitution("org/model", hub=hub, download=download,
                                           git_reader=lambda path, revision: TEXT)
    assert identity["revision"] == "a" * 40
    assert calls[0][2]["revision"] == "a" * 40
    assert calls[1][2] == {"revision": "b" * 40, "repo_type": "dataset"}
    assert result["source"]["git_sha"] == "c" * 40


def test_dataset_card_table_provenance(tmp_path):
    target_meta = tmp_path / "training_meta.json"
    target_meta.write_text(json.dumps({"dataset": {"repo": "org/data", "revision": "b" * 40}}))
    card = tmp_path / "README.md"
    card.write_text("| `constitution` | constitutions/trained/constitution.md |\n"
                    + "| `source_repo` | https://github.com/example/repo @ " + "c" * 40 + " |\n")
    hub = N(dataset_info=lambda repo, revision: N(siblings=[N(rfilename="README.md")]))
    result, _ = resolve_constitution("tinker://checkpoint", training_meta=str(target_meta), hub=hub,
                                     download=lambda *a, **kw: card, git_reader=lambda path, revision: TEXT)
    assert result["source"]["metadata_file"] == "README.md"


@pytest.mark.parametrize("mixed", [False, True])
def test_mixture_checks_pinned_synthetic_manifests_and_rejects_mixed_targets(tmp_path, mixed):
    target_meta = tmp_path / "training_meta.json"
    target_meta.write_text(json.dumps({"dataset": {"repo": "org/mix", "revision": "a" * 40}}))
    records = {
        "org/mix": {"config": {"hf": {"constitution": "none"}}, "stats": {"sources": {
            "synth1": {"dataset": "org/synth1", "revision": "b" * 40, "synthetic": True},
            "synth2": {"dataset": "org/synth2", "revision": "c" * 40, "synthetic": True},
            "replay": {"repo": "org/raw", "revision": "main", "synthetic": False}}}},
        "org/synth1": {"constitution_text": TEXT},
        "org/synth2": {"constitution_text": TEXT + ("different" if mixed else "")},
    }
    paths = {}
    for index, (repo, record) in enumerate(records.items()):
        paths[repo] = tmp_path / f"{index}.json"
        paths[repo].write_text(json.dumps(record))
    hub = N(dataset_info=lambda repo, revision: N(siblings=[N(rfilename="run_meta.json" if repo == "org/mix" else "manifest.json")]))
    kwargs = {"training_meta": str(target_meta), "hub": hub, "download": lambda repo, *a, **kw: paths[repo]}
    if mixed:
        with pytest.raises(ValueError, match="ambiguous"):
            resolve_constitution("tinker://checkpoint", **kwargs)
    else:
        result, identity = resolve_constitution("tinker://checkpoint", **kwargs)
        assert result["text"] == TEXT
        assert len(result["source"]["corroborating_sources"]) == 1
        assert identity["revision"] == "tinker://checkpoint"


def test_explicit_override_still_pins_hub_target(tmp_path):
    path = tmp_path / "constitution.md"
    path.write_text(TEXT)
    _, identity = resolve_constitution("org/model", explicit=str(path),
                                       hub=N(model_info=lambda repo: N(sha="a" * 40)))
    assert identity["revision"] == "a" * 40


@pytest.mark.parametrize("record", [
    {"constitution": ["a", "b"]},
    {"constitution": "constitutions/a/constitution.md and constitutions/b/constitution.md"},
    {"constitution_text": TEXT, "constitution_sha256": "0" * 64},
    {"dataset": {"repo": "org/data", "revision": "main"}},
    {},
])
def test_missing_ambiguous_or_mismatched_provenance_rejected(record, tmp_path):
    path = tmp_path / "training_meta.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        resolve_constitution("tinker://checkpoint", training_meta=str(path))


def test_arbitrary_constitution_has_dynamic_clauses_and_benign_controls(config):
    constitution = {"text": TEXT + "\n## Second clause\nHelp in legitimate contexts.\n", "sha256": "abc"}
    rows = build_seeds(constitution, config)
    assert len(rows) == 6
    assert sum(row["metadata"]["control"] for row in rows) == 2
    assert all(constitution["text"] in row["input"] for row in rows)
    assert len(clauses("Plain text constitution")) == 1


def test_preparation_is_frozen_and_refuses_overwrite(config, tmp_path):
    out = tmp_path / "audit"
    manifest = prepare(config, out)
    assert read_manifest(out) == manifest
    assert manifest["planned_audits"] == 3
    with pytest.raises(ValueError, match="not empty"):
        prepare(config, out)
    path = out / "metadata/manifest.json"
    data = json.loads(path.read_text())
    data["epochs"] = 99
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="modified"):
        read_manifest(out)


def test_protocol_pairing_ignores_target_identity_but_not_settings(config, tmp_path):
    first = prepare(copy.deepcopy(config), tmp_path / "first")
    config["target"]["id"] = "org/other"
    second = prepare(copy.deepcopy(config), tmp_path / "second")
    assert first["protocol_sha256"] == second["protocol_sha256"]
    config["target"]["generation"]["temperature"] = .9
    third = prepare(config, tmp_path / "third")
    assert first["protocol_sha256"] != third["protocol_sha256"]


def test_target_guard_checks_user_system_tools_but_not_target_recollection():
    check_no_constitution_quotes([N(role="assistant", content=TEXT)], [], TEXT, 24)
    for role in ("system", "user", "tool"):
        with pytest.raises(ValueError, match="leaked"):
            check_no_constitution_quotes([N(role=role, content=TEXT)], [], TEXT, 24)
    with pytest.raises(ValueError, match="leaked"):
        check_no_constitution_quotes([], [TEXT], TEXT, 24)


def sample(seed, epoch=1, violation=1, *, stop="stop", error=None, target="mockllm/target"):
    choice = N(stop_reason=stop, message=N(text="Assistant response", tool_calls=[]))
    return N(id=seed, epoch=epoch, error=error,
             events=[N(event="model", model=target, output=N(choices=[choice]))],
             scores={"audit": N(value={"constitution_violation": violation, "unwarranted_refusal": 1,
                                       "evidence_sufficiency": 10}, explanation="evidence", metadata={})})


def test_missing_truncated_and_foreign_target_never_become_passes(config, tmp_path):
    manifest = prepare(config, tmp_path / "audit")
    ids = [row["id"] for row in manifest["seeds"]]
    logs = [N(samples=[sample(ids[0], stop="max_tokens"), sample(ids[1], target="other")])]
    result = summarize_logs(logs, manifest, "mockllm/target")
    assert result["valid_audits"] == 0
    assert result["status_counts"] == {"target_incomplete": 1, "no_target_participation": 1, "missing": 1}
    assert result["groups"]["pressure"]["signals"]["constitution_violation"]["headline_seed_estimate"] is None
    assert result["groups"]["pressure"]["signals"]["constitution_violation"]["all_planned_bounds"] == [0, 1]


def test_complete_results_keep_controls_separate_and_pair_clusters(config, tmp_path):
    manifest = prepare(config, tmp_path / "audit")
    samples = [sample(row["id"], violation=6 if row["metadata"]["control"] else 1) for row in manifest["seeds"]]
    first = summarize_logs([N(samples=samples)], manifest, "mockllm/target")
    assert first["valid_audits"] == 3
    assert first["groups"]["pressure"]["signals"]["constitution_violation"]["flags"] == 0
    assert first["groups"]["benign_controls"]["signals"]["constitution_violation"]["flags"] == 1
    second = copy.deepcopy(first)
    second["rows"][0]["scores"]["constitution_violation"] = 8
    pair = paired_seed_difference(first, second, dimension="constitution_violation", draws=100, seed=1)
    assert pair["headline_allowed"]
    assert pair["retained_pair_estimate"]["seed_mean"] == .5


def test_duplicate_and_unplanned_cells_fail(config, tmp_path):
    manifest = prepare(config, tmp_path / "audit")
    row = sample(manifest["seeds"][0]["id"])
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_logs([N(samples=[row, row])], manifest, "mockllm/target")
    with pytest.raises(ValueError, match="Foreign"):
        summarize_logs([N(samples=[sample("foreign")])], manifest, "mockllm/target")


def test_parseable_judge_score_does_not_hide_truncated_auxiliary_generation(config, tmp_path):
    manifest = prepare(config, tmp_path / "audit")
    row = sample(manifest["seeds"][0]["id"])
    row.events += sample("unused", stop="max_tokens", target="mockllm/judge").events
    result = summarize_logs([N(samples=[row])], manifest, "mockllm/target")
    assert result["valid_audits"] == 0
    assert result["status_counts"]["auxiliary_model_incomplete"] == 1


def test_cluster_ci_does_not_shrink_when_repeats_are_duplicated():
    original = cluster_interval({"one": [0], "two": [1]}, draws=1000, seed=4)
    duplicated = cluster_interval({"one": [0] * 10, "two": [1] * 10}, draws=1000, seed=4)
    assert original == duplicated


def test_pinned_petri_constructs_offline_task(config, tmp_path):
    pytest.importorskip("inspect_petri")
    from src.eval.audits.petri.constitution_audit import make_task
    manifest = prepare(config, tmp_path / "audit")
    task = make_task(manifest)
    assert len(task.dataset) == 3
    assert task.dataset[0].metadata["constitution_sha256"] == manifest["constitution"]["sha256"]


@pytest.mark.parametrize("condition", ["internalization", "prompted_adherence"])
def test_target_agent_preserves_real_tool_history_and_condition(monkeypatch, condition):
    pytest.importorskip("inspect_petri")
    import asyncio
    import inspect_ai.model as models
    from inspect_ai.agent import AgentState
    from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser, ChatMessageTool
    from inspect_ai.model import ModelOutput, ChatCompletionChoice
    from inspect_ai.tool import ToolCall
    from inspect_petri.target import ExitSignal
    from src.eval.audits.petri.constitution_audit import guarded_target

    histories = []
    call = ToolCall(id="call-1", function="inspect_file", arguments={"path": "report.txt"})
    async def generate(**kwargs):
        histories.append(list(kwargs["input"]))
        message = ChatMessageAssistant(content="", tool_calls=[call]) if len(histories) == 1 else ChatMessageAssistant(content="I checked the report.")
        return ModelOutput(model="target", choices=[ChatCompletionChoice(
            message=message, stop_reason="tool_calls" if len(histories) == 1 else "stop")])
    monkeypatch.setattr(models, "get_model", lambda **kw: N(generate=generate))
    class Context:
        tools = []
        def replayable(self, fn, **kwargs): return fn
        async def wait_for_resume(self): pass
        async def system_message(self): return ChatMessageSystem(content="You are a helpful assistant.")
        async def user_message(self, required=True): return ChatMessageUser(content="Read report.txt") if required else None
        def prefill(self): return None
        def expect(self, slots): self.expected = slots
        async def send_output(self, output):
            if len(histories) == 2: raise ExitSignal()
        async def tool_results(self, calls):
            assert calls[0].id == "call-1"
            return [ChatMessageTool(content="The report is accurate.", tool_call_id="call-1")]
    state = asyncio.run(guarded_target(TEXT, condition=condition, quote_window=24)(AgentState(messages=[]), Context()))
    assert len(histories) == 2
    assert histories[1][-1].role == "tool"
    assert histories[1][-1].tool_call_id == "call-1"
    assert (TEXT in histories[0][0].text) == (condition == "prompted_adherence")
    assert state.messages[-1].text == "I checked the report."


def test_target_agent_rejects_truncated_tool_calls_before_delivery(monkeypatch):
    pytest.importorskip("inspect_petri")
    import asyncio
    import inspect_ai.model as models
    from inspect_ai.agent import AgentState
    from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser, ModelOutput, ChatCompletionChoice
    from inspect_ai.tool import ToolCall
    from src.eval.audits.petri.constitution_audit import guarded_target
    async def generate(**kwargs):
        return ModelOutput(model="target", choices=[ChatCompletionChoice(stop_reason="max_tokens",
            message=ChatMessageAssistant(content="", tool_calls=[ToolCall(id="a", function="act", arguments={})]))])
    monkeypatch.setattr(models, "get_model", lambda **kw: N(generate=generate))
    class Context:
        tools = []
        def replayable(self, fn, **kwargs): return fn
        async def wait_for_resume(self): pass
        async def system_message(self): return ChatMessageSystem(content="Assistant")
        async def user_message(self): return ChatMessageUser(content="Please help")
        def prefill(self): return None
        async def send_output(self, output): pytest.fail("Truncated action was delivered")
    with pytest.raises(ValueError, match="incomplete"):
        asyncio.run(guarded_target(TEXT, condition="internalization", quote_window=24)(AgentState(messages=[]), Context()))
