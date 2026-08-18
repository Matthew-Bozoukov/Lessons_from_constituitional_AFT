# ABOUTME: Offline unit tests for src/properties: the Record/Property contracts, the source
# ABOUTME: adapters, grouping, the registry, and every ablation's applicability rules.

import json
from pathlib import Path

import numpy as np
import pytest

from src.properties.ablation import base as ablation_base
from src.properties.ablation import filter as filter_mod
from src.properties.ablation import mask as mask_mod
from src.properties.ablation import rewrite as rewrite_mod
from src.properties.producers import PRODUCERS
from src.properties.registry import Property, PropertyRegistry
from src.properties.shared import attributes as attributes_mod
from src.properties.shared import embed as embed_mod
from src.properties.shared import grouping as grouping_mod
from src.properties.shared import interpret as interpret_mod
from src.properties.sources import SOURCES, Record, first_turns, mixture_rows
from src.properties.sources import targets as targets_mod


def _record(rid="r0", query="q", response="a", reasoning="think", **kw):
    return Record(record_id=rid, query=query, response=response, reasoning=reasoning,
                  raw={"messages": [{"role": "user", "content": query},
                                    {"role": "assistant", "content": response,
                                     "reasoning_content": reasoning}]}, **kw)


def _property(**kw):
    defaults = {"property_id": "p:run:c000", "source": "trace_clusters",
                "label": "Weighs likelihood against severity", "detector": "Does it?",
                "channel": "reasoning"}
    return Property(**{**defaults, **kw})


# --- the shared types ------------------------------------------------------------------

def test_record_channel_rejects_an_unknown_name():
    record = _record()
    assert record.channel("reasoning") == "think"
    with pytest.raises(ValueError, match="unknown channel"):
        record.channel("thoughts")


def test_first_turns_takes_the_first_user_and_first_assistant_turn():
    messages = [{"role": "system", "content": "s"},
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"}]
    assert first_turns(messages) == ("q1", "a1", "r1")
    assert first_turns([{"role": "user", "content": "q"}]) is None
    assert first_turns([{"role": "assistant", "content": "a"}]) is None
    assert first_turns("not a list") is None


def test_source_registry_declares_what_may_be_ablated():
    assert set(SOURCES) == {"mixture_rows", "odcv_rollouts", "agentic_rollouts"}
    # Only a training corpus is ablatable; rollouts are evidence about a trained model.
    assert SOURCES["mixture_rows"].ablatable
    assert not SOURCES["odcv_rollouts"].ablatable
    assert not SOURCES["agentic_rollouts"].ablatable
    # Only rollouts carry outcomes.
    assert not SOURCES["mixture_rows"].has_outcomes
    assert SOURCES["odcv_rollouts"].has_outcomes


# --- mixture_rows ----------------------------------------------------------------------

def test_mixture_rows_reads_interchange_and_rendered_rows(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in [
        {"messages": [{"role": "user", "content": "q"},
                      {"role": "assistant", "content": "a", "reasoning_content": "r"}],
         "source": "da", "metadata": {"scenario_id": "s1", "trait_id": "t3"}},
        {"text": "<|im_start|>user\nq<|im_end|>", "source": "tulu3"},
    ]))
    records = mixture_rows.load(path=str(path))
    assert [r.record_id for r in records] == ["s1", "tulu3#1"]
    assert records[0].reasoning == "r"
    assert records[0].metadata["trait_id"] == "t3"
    # The original row survives untouched, so an ablation can write the corpus back.
    assert records[0].raw["source"] == "da"
    # Without a model there are no markers to split a pre-rendered row on.
    assert records[1].reasoning == "" and records[1].query == ""
    assert records[1].metadata["pre_rendered"] and not records[0].metadata["pre_rendered"]


def test_unrender_splits_a_prerendered_qwen_row_into_channels():
    rendered = ("<|im_start|>user\nWhat should I do?<|im_end|>\n"
                "<|im_start|>assistant\n<think>\nweigh the harms\n</think>\n\n"
                "Here is my answer.<|im_end|>")
    query, response, reasoning = mixture_rows.unrender(rendered, "Qwen/Qwen3.6-27B")
    assert query == "What should I do?"
    assert reasoning == "weigh the harms"
    assert response == "Here is my answer."

    # The family's EMPTY marker means "this row does not reason", not a parse failure.
    empty = ("<|im_start|>user\nq<|im_end|>\n"
             "<|im_start|>assistant\n<think>\n\n</think>\n\nplain answer<|im_end|>")
    query, response, reasoning = mixture_rows.unrender(empty, "Qwen/Qwen3.6-27B")
    assert reasoning == "" and response == "plain answer"

    with pytest.raises(ValueError, match="no assistant turn"):
        mixture_rows.unrender("<|im_start|>user\nq<|im_end|>", "Qwen/Qwen3.6-27B")


def test_load_unrenders_when_given_a_model(tmp_path):
    rendered = ("<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n"
                "<think>\nthinking here\n</think>\n\nanswer<|im_end|>")
    path = tmp_path / "rendered.jsonl"
    path.write_text(json.dumps({"text": rendered, "source": "da"}) + "\n")
    record = mixture_rows.load(path=str(path), model="Qwen/Qwen3.6-27B")[0]
    assert record.reasoning == "thinking here" and record.response == "answer"
    assert record.raw["text"] == rendered      # the row itself is untouched


def test_mixture_rows_filters_by_source_and_refuses_duplicate_ids(tmp_path):
    rows = [{"messages": [{"role": "user", "content": "q"},
                          {"role": "assistant", "content": "a"}],
             "source": "da", "metadata": {"scenario_id": "same"}} for _ in range(2)]
    path = tmp_path / "corpus.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="duplicate record_id"):
        mixture_rows.load(path=str(path))
    assert mixture_rows.load(path=str(path), only_source="tulu3") == []


# --- targets ---------------------------------------------------------------------------

def test_from_rubric_refuses_a_rubric_that_names_candidate_properties(tmp_path):
    good = tmp_path / "good.yaml"
    good.write_text("principle_specific_details: |\n  PRINCIPLE: be honest.\n")
    target = targets_mod.from_rubric(str(good), cases=[_record()])
    assert target.target_id == "good" and target.polarity == "satisfy"
    assert "be honest" in target.behaviour

    bad = tmp_path / "bad.yaml"
    bad.write_text("principle_specific_details: x\nproperties:\n  - weighs harms\n")
    with pytest.raises(ValueError, match="describes the BEHAVIOUR only"):
        targets_mod.from_rubric(str(bad), cases=[])


def test_from_dval_reports_distinct_prompts_not_just_row_count(tmp_path):
    path = tmp_path / "stayed_ai.jsonl"
    rows = [{"messages": [{"role": "user", "content": "are you human?"},
                          {"role": "assistant", "content": "no"}]} for _ in range(3)]
    rows.append({"messages": [{"role": "user", "content": "different prompt"},
                              {"role": "assistant", "content": "no"}]})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    target = targets_mod.from_dval(str(path), behaviour="discloses AI-hood")
    # 4 rows, 2 distinct prompts — the effective sample size is the second number.
    assert target.provenance["n_cases"] == 4
    assert target.provenance["distinct_prompts"] == 2
    assert target.subtask == "stayed_ai"


def test_from_outcomes_refuses_an_unjudged_rollout_set():
    with pytest.raises(ValueError, match="no record carries an outcome"):
        targets_mod.from_outcomes([_record()], behaviour="b")
    judged = [_record("a", outcome={"violation": True}),
              _record("b", outcome={"violation": False})]
    assert [r.record_id for r in
            targets_mod.from_outcomes(judged, "b", polarity="violate").cases] == ["a"]


# --- shared/embed ----------------------------------------------------------------------

def test_normalise_makes_dot_products_cosines_and_survives_a_zero_row():
    matrix = embed_mod.normalise(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.isclose(np.linalg.norm(matrix[0]), 1.0)
    assert np.allclose(matrix[1], 0.0)


def test_check_probe_fails_a_broken_embedding_space():
    embed_mod.check_probe({"synonym": 0.9, "unrelated": 0.2})
    with pytest.raises(RuntimeError, match="geometry is broken"):
        embed_mod.check_probe({"synonym": 0.5, "unrelated": 0.48})


def test_embed_refuses_an_unknown_backend_before_spending_anything():
    with pytest.raises(ValueError, match="unknown embedding backend"):
        embed_mod.embed(["x"], backend="huggingface")
    with pytest.raises(ValueError, match="nothing to embed"):
        embed_mod.embed([], backend="openrouter")


# --- shared/grouping -------------------------------------------------------------------

def test_grouping_params_report_only_the_settings_that_mattered():
    hdbscan = grouping_mod.GroupingParams(cluster="hdbscan").to_dict()
    assert "min_cluster_size" in hdbscan and "k" not in hdbscan
    kmeans = grouping_mod.GroupingParams(reduce="none", cluster="kmeans").to_dict()
    assert "k" in kmeans
    assert "min_cluster_size" not in kmeans and "n_neighbors" not in kmeans
    with pytest.raises(ValueError, match="reduce must be"):
        grouping_mod.GroupingParams(reduce="pca").validate()


def test_kmeans_groups_two_separated_blobs():
    rng = np.random.default_rng(0)
    points = np.vstack([rng.normal(0, 0.05, (30, 4)) + 1.0,
                        rng.normal(0, 0.05, (30, 4)) - 1.0]).astype(np.float32)
    result = grouping_mod.group(points, grouping_mod.GroupingParams(
        reduce="none", cluster="kmeans", k=2))
    assert result.n_groups == 2 and result.n_noise == 0
    # Every member of a blob shares a label.
    assert len(set(result.labels[:30].tolist())) == 1
    assert len(set(result.labels[30:].tolist())) == 1


def test_assign_puts_new_points_on_existing_centroids():
    centroids = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    labels, distances = grouping_mod.assign(
        np.array([[0.9, 0.0], [-0.8, 0.1]], dtype=np.float32), centroids)
    assert labels.tolist() == [0, 1]
    assert distances.shape == (2,)


# --- shared/attributes -----------------------------------------------------------------

def test_numbered_prompt_is_surf_verbatim_at_the_default_count():
    rendered = attributes_mod.QUERY_ATTR_PROMPT.format(n=10, query="Q")
    assert "Write a list of 10 attributes" in rendered
    assert "listed in XML tags from <1> to <10>" in rendered
    assert 'starting with "The query"' in rendered


def test_parse_numbered_tolerates_missing_close_tags_but_not_missing_tags():
    text = "<1>The query asks for data\n<2>The query is urgent</2>"
    assert attributes_mod.parse_numbered(text, 2) == [
        "The query asks for data", "The query is urgent"]
    with pytest.raises(ValueError, match="attribute <3> missing"):
        attributes_mod.parse_numbered(text, 3)


def test_build_messages_picks_the_right_prompt_family():
    record = _record(query="Q", reasoning="R", response="A")
    numbered = attributes_mod.build_messages(
        record, attributes_mod.AttributeSpec(style="numbered", channel="reasoning"))
    assert len(numbered) == 1 and "<reasoning>" in numbered[0]["content"]
    freeform = attributes_mod.build_messages(
        record, attributes_mod.AttributeSpec(style="freeform", channel="reasoning"))
    # The post's prompt goes in the (cacheable) system message; the trace alone in the user.
    assert freeform[0]["role"] == "system"
    assert "identify key" in freeform[0]["content"]
    assert freeform[1]["content"] == "R"
    with pytest.raises(ValueError, match="style must be one of"):
        attributes_mod.AttributeSpec(style="bullets").validate()


# --- shared/interpret ------------------------------------------------------------------

def test_prevalence_excludes_judge_errors_from_the_denominator():
    verdicts = [{"exhibits": True}, {"exhibits": False}, {"exhibits": None}]
    measured = interpret_mod.prevalence(verdicts)
    assert measured["n"] == 2 and measured["hits"] == 1
    assert measured["prevalence"] == 0.5 and measured["n_errors"] == 1
    assert measured["ci_low"] < 0.5 < measured["ci_high"]


def test_sample_evidence_is_reproducible_and_never_grows_the_list():
    evidence = [f"f{i}" for i in range(20)]
    assert (interpret_mod.sample_evidence(evidence, 5, seed=1)
            == interpret_mod.sample_evidence(evidence, 5, seed=1))
    assert interpret_mod.sample_evidence(evidence, 50) == evidence


# --- registry --------------------------------------------------------------------------

def test_a_property_without_a_detector_is_refused():
    with pytest.raises(ValueError, match="needs a detector"):
        _property(detector="  ")
    with pytest.raises(ValueError, match="needs a label"):
        _property(label="")
    with pytest.raises(ValueError, match="not a share of the corpus"):
        _property(prevalence=1.4)


def test_registry_round_trips_and_refuses_duplicate_ids(tmp_path):
    registry = PropertyRegistry(tmp_path / "properties.jsonl")
    assert registry.read() == []
    registry.write([_property(property_id="a:r:1", prevalence=0.4),
                    _property(property_id="b:r:2", prevalence=0.1)])
    assert [p.property_id for p in registry.read()] == ["a:r:1", "b:r:2"]
    assert registry.get("b:r:2").label.startswith("Weighs")
    with pytest.raises(KeyError, match="no property"):
        registry.get("nope")
    with pytest.raises(ValueError, match="duplicate property_ids"):
        registry.write([_property(property_id="a:r:1"), _property(property_id="a:r:1")])


def test_registry_add_replaces_a_producers_previous_run_rather_than_duplicating(tmp_path):
    registry = PropertyRegistry(tmp_path / "properties.jsonl")
    registry.add([_property(property_id="fd:run1:c000", source="feature_discovery")])
    registry.add([_property(property_id="tc:run1:g000", source="trace_clusters")])
    registry.add([_property(property_id="fd:run1:c000", source="feature_discovery",
                            prevalence=0.9)])
    ids = [p.property_id for p in registry.read()]
    assert ids.count("fd:run1:c000") == 1 and "tc:run1:g000" in ids
    assert registry.get("fd:run1:c000").prevalence == 0.9
    with pytest.raises(ValueError, match="one source's rows at a time"):
        registry.add([_property(property_id="x:r:1", source="a"),
                      _property(property_id="y:r:1", source="b")])


def test_with_prevalence_carries_the_measurement_into_support():
    measured = {"n": 100, "hits": 25, "prevalence": 0.25, "ci_low": 0.17,
                "ci_high": 0.35, "n_errors": 0}
    updated = _property().with_prevalence(measured, {"repo": "x"})
    assert updated.prevalence == 0.25 and updated.n_records == 25
    assert updated.corpus == {"repo": "x"}
    assert updated.support["detector_measurement"]["ci_high"] == 0.35


def test_report_is_a_greppable_markdown_mirror(tmp_path):
    registry = PropertyRegistry(tmp_path / "properties.jsonl")
    registry.write([_property(property_id="a:r:1", prevalence=0.4, n_records=40)])
    report = registry.report()
    assert "40.0%" in report and "`a:r:1`" in report


# --- producers -------------------------------------------------------------------------

def test_producer_registry_declares_what_each_one_needs():
    assert set(PRODUCERS) == {"trace_clusters", "feature_discovery", "turf", "less"}
    assert PRODUCERS["turf"].needs_target and PRODUCERS["less"].needs_target
    assert not PRODUCERS["trace_clusters"].needs_target
    assert PRODUCERS["less"].needs_gpu
    # The three still under scratch/ say where their code is, so the error message can too.
    for name in ("feature_discovery", "turf", "less"):
        assert not PRODUCERS[name].ported
        assert PRODUCERS[name].scratch_path.startswith("scratch/")
    assert PRODUCERS["trace_clusters"].ported


def test_unported_producers_name_the_command_that_makes_their_artifacts(tmp_path):
    from src.properties.producers.feature_discovery import adapter as fd_adapter
    from src.properties.producers.less import adapter as less_adapter
    from src.properties.producers.turf import adapter as turf_adapter

    with pytest.raises(FileNotFoundError, match="scratch.llm_feature_discovery"):
        fd_adapter.read_run_dir(tmp_path)
    with pytest.raises(FileNotFoundError, match="scratch/turf/trace.py"):
        turf_adapter.read_trace(tmp_path)
    with pytest.raises(FileNotFoundError, match="scratch/less/influence.py"):
        less_adapter.read_scores(tmp_path)


# --- ablation --------------------------------------------------------------------------

def test_no_ablation_will_touch_a_rollout_source():
    records = [_record()]
    for kind in ablation_base.KINDS:
        ok, reason = ablation_base.applicable(
            kind, _property(), records, SOURCES["odcv_rollouts"], None)
        assert not ok and "evidence, not training data" in reason


def test_mask_needs_a_reasoning_property_a_trace_and_a_render_model():
    corpus = SOURCES["mixture_rows"]
    assert mask_mod.applicable(_property(), [_record()], corpus, None)[0]
    ok, reason = mask_mod.applicable(_property(channel="response"), [_record()],
                                     corpus, None)
    assert not ok and "acts on ('reasoning',)" in reason
    ok, reason = mask_mod.applicable(_property(), [_record(reasoning="")], corpus, None)
    assert not ok and "nothing to mask" in reason
    ok, reason = mask_mod.applicable(_property(), [_record()], corpus, {})
    assert not ok and "mask needs `model:`" in reason


def test_rewrite_refuses_the_query_channel_and_an_unnamed_substitute():
    corpus = SOURCES["mixture_rows"]
    ok, reason = rewrite_mod.applicable(_property(channel="query"), [_record()],
                                        corpus, None)
    assert not ok and "acts on" in reason
    ok, reason = rewrite_mod.applicable(
        _property(), [_record()], corpus, {"mode": "substitute"})
    assert not ok and "substitute_with" in reason
    assert rewrite_mod.applicable(
        _property(), [_record()], corpus,
        {"mode": "substitute", "substitute_with": "reason about consequences"})[0]


def test_filter_validates_its_mode():
    corpus = SOURCES["mixture_rows"]
    assert filter_mod.applicable(_property(), [_record()], corpus, {"mode": "split"})[0]
    ok, reason = filter_mod.applicable(_property(), [_record()], corpus,
                                       {"mode": "delete"})
    assert not ok and "mode must be one of" in reason


def test_rewrite_writes_a_channel_back_without_disturbing_the_row():
    record = _record()
    record.raw["source"] = "da"
    row = rewrite_mod._write_back(record, "reasoning", "NEW")
    assert row["source"] == "da"
    assert row["messages"][1]["reasoning_content"] == "NEW"
    assert row["messages"][1]["content"] == "a"          # the response is untouched
    assert record.raw["messages"][1]["reasoning_content"] == "think"  # no mutation
    with pytest.raises(ValueError, match="pre-rendered rows"):
        rewrite_mod._write_back(Record("r", "q", "a", raw={"text": "x"}),
                                "reasoning", "NEW")


def test_mask_locates_spans_exactly_and_refuses_ambiguity():
    text = "<think>\nfirst part. second part.\n</think>\n\nanswer"
    region = mask_mod._think_region(text)
    start, end = mask_mod._locate(text, "second part.", region)
    assert text[start:end] == "second part."
    with pytest.raises(ValueError, match="not present"):
        mask_mod._locate(text, "third part.", region)
    with pytest.raises(ValueError, match="outside the reasoning block"):
        mask_mod._locate(text, "answer", region)
    doubled = "<think>\nsame. same.\n</think>"
    with pytest.raises(ValueError, match="not unique"):
        mask_mod._locate(doubled, "same.", mask_mod._think_region(doubled))
    with pytest.raises(ValueError, match="expected exactly one think block"):
        mask_mod._think_region("no think block here")


def test_ablation_result_reports_the_share_it_actually_acted_on():
    result = ablation_base.AblationResult(
        kind="rewrite", property_id="p:r:1", rows=[{}, {}],
        changed_ids=["a"], detected_ids=["a", "b"])
    summary = result.summary()
    assert summary["n_detected"] == 2 and summary["n_changed"] == 1
    assert summary["acted_on_share"] == 0.5


def test_ablation_registry_orders_kinds_weakest_first():
    assert ablation_base.KINDS == ("mask", "filter", "rewrite", "regenerate")
    strengths = [ablation_base.ABLATIONS[k].strength for k in ablation_base.KINDS]
    assert strengths == sorted(strengths)
    with pytest.raises(KeyError, match="unknown ablation"):
        ablation_base.resolve("delete")


# --- verify ----------------------------------------------------------------------------

def test_verify_gate_fails_a_small_drop_and_an_overlapping_interval(monkeypatch):
    from src.properties.ablation import verify as verify_module

    calls = {"n": 0}
    # 90% before, 10% after: a clean drop. Then 50/45: too small AND overlapping.
    sequences = [([True] * 90 + [False] * 10, [True] * 10 + [False] * 90),
                 ([True] * 50 + [False] * 50, [True] * 45 + [False] * 55)]

    def fake_detect(records, label, detector, channel="reasoning", **kwargs):
        pair = sequences[calls["n"] // 2]
        flags = pair[calls["n"] % 2]
        calls["n"] += 1
        return [{"record_id": str(i), "exhibits": f} for i, f in enumerate(flags)]

    monkeypatch.setattr(verify_module.interpret_mod, "detect", fake_detect)
    records = [_record(f"r{i}") for i in range(100)]

    clean = verify_module.verify(_property(), records, records,
                                 {"check_separability": False, "sample": 100})
    assert clean.passed and clean.drop == 0.8
    assert "PASSED" in clean.report()

    weak = verify_module.verify(_property(), records, records,
                                {"check_separability": False, "sample": 100})
    assert not weak.passed
    assert any("below the" in f for f in weak.failures)
    assert any("intervals overlap" in f for f in weak.failures)


def test_separability_finds_an_arm_marker_and_names_it():
    before = [f"the assistant weighed the risk carefully in case {i}" for i in range(40)]
    after = [f"i'd say the assistant thought about it in case {i}" for i in range(40)]
    result = verify_separability(before, after)
    assert result["auc"] > 0.9
    assert any("'d" in feature or "say" in feature
               for feature, _ in result["top_features"])


def verify_separability(before, after):
    from src.properties.ablation.verify import separability

    return separability(before, after)


# --- trace_clusters end to end (no network) --------------------------------------------

def test_trace_clusters_produce_end_to_end(tmp_path, monkeypatch):
    """The reference producer with its two model calls stubbed: embed -> group ->
    interpret -> Property rows, including the per-arm split a DA-vs-courtroom
    comparison reads."""
    from src.properties.producers.trace_clusters import adapter as tc
    from src.properties.shared.interpret import Interpretation

    records = []
    for i in range(20):
        arm = "da" if i < 12 else "courtroom"
        records.append(Record(
            record_id=f"r{i}", query="q", response="a",
            reasoning=f"trace {i} from {arm}",
            metadata={"source_label": arm, "corpus": {"repo": "org/x"}},
            raw={"messages": [{"role": "user", "content": "q"},
                              {"role": "assistant", "content": "a",
                               "reasoning_content": f"trace {i}"}]}))

    rng = np.random.default_rng(0)
    # Two well-separated blobs, aligned with the two arms.
    vectors = np.vstack([rng.normal(0, 0.02, (12, 8)) + 1.0,
                         rng.normal(0, 0.02, (8, 8)) - 1.0]).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    meta = embed_mod.EmbedMeta(backend="openrouter", model="stub", dim=8, n=20)
    monkeypatch.setattr(tc.embed_mod, "embed", lambda texts, **kw: (vectors, meta))
    monkeypatch.setattr(tc.interpret_mod, "interpret_many", lambda groups, **kw: {
        g: Interpretation(label=f"Property {g}", description="d",
                          detector="Does the trace do it?", channel="reasoning",
                          evidence=groups[g][:2], model="stub")
        for g in groups})

    properties = tc.produce(records,
                            {"channel": "reasoning",
                             "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
                             "group_by": "source_label"},
                            tmp_path / "tc")
    assert len(properties) == 2
    assert {p.prevalence for p in properties} == {0.6, 0.4}
    # Every row is a Property: it carries a detector, so an ablation can act on it.
    assert all(p.detector and p.channel == "reasoning" for p in properties)
    assert all(p.property_id.startswith("trace_clusters:tc:g") for p in properties)
    assert properties[0].corpus == {"repo": "org/x"}
    arms = properties[0].support["arms"]
    assert set(arms) == {"da", "courtroom"}
    assert arms["da"]["n_in_corpus"] == 12 and arms["courtroom"]["n_in_corpus"] == 8
    assert (tmp_path / "tc" / "properties_preview.json").exists()
    assert (tmp_path / "tc" / "embeddings.npy").exists()


def test_trace_clusters_refuses_a_corpus_with_no_text_in_the_channel(tmp_path):
    from src.properties.producers.trace_clusters import adapter as tc

    with pytest.raises(ValueError, match="no record carries text in the 'reasoning'"):
        tc.produce([Record("r0", "q", "a", reasoning="")], {}, tmp_path / "tc")


# --- feature_discovery against the published run checked into docs/ --------------------

def test_feature_discovery_adapter_reads_the_published_run(tmp_path, monkeypatch):
    """The 2026-08-12 run lives in docs/feature_discovery/ (minus its 265MB embeddings).
    The adapter must turn its clusters into Property rows with ids that name the RUN, not
    the directory the artifacts were copied into."""
    from src.properties.producers.feature_discovery import adapter as fd
    from src.properties.shared.interpret import Interpretation

    monkeypatch.setattr(fd.interpret_mod, "interpret_many", lambda groups, **kw: {
        g: Interpretation(label=f"L{g}", description="d", detector="Does it?",
                          channel="reasoning", evidence=groups[g][:2], model="stub")
        for g in groups})

    properties = fd.produce([], {"run_dir": "docs/feature_discovery",
                                 "min_prevalence": 0.15}, tmp_path / "fd")
    assert properties, "the published run has clusters above 15% prevalence"
    ids = {p.property_id for p in properties}
    # The run's own recorded name, not "docs" or "feature_discovery".
    assert "feature_discovery:20260812_092119:c079" in ids
    assert "feature_discovery:20260812_092119:c006" in ids
    # Cluster 30 sits at 8.9%, below the filter.
    assert "feature_discovery:20260812_092119:c030" not in ids
    top = properties[0]
    assert top.prevalence >= 0.15 and top.n_records
    # The producer's own ~5-word label is kept alongside the interpreter's, not replaced.
    assert top.support["producer_label"]
    assert top.support["trait_mix"]
    assert top.provenance["embedding_model"] == "Qwen/Qwen3-Embedding-8B"
    assert properties == sorted(properties, key=lambda p: -p.prevalence)


# --- filter end to end (detector stubbed) ----------------------------------------------

def _stub_detector(monkeypatch, flagged_ids):
    """Make the detector flag exactly `flagged_ids`, with no network call."""
    monkeypatch.setattr(filter_mod.interpret_mod, "detect",
                        lambda records, label, detector, channel="reasoning", **kw:
                        [{"record_id": r.record_id,
                          "exhibits": r.record_id in flagged_ids} for r in records])


def test_filter_drop_rebalances_so_the_arms_keep_the_same_row_count(monkeypatch):
    records = [_record(f"r{i}") for i in range(10)]
    _stub_detector(monkeypatch, {"r0", "r1", "r2"})
    result = filter_mod.apply(_property(), records, {"mode": "drop", "rebalance": True})
    # 3 flagged out of 10, plus 3 unflagged dropped to keep the size comparison clean.
    assert result.report["n_removed"] == 6 and len(result.rows) == 4
    assert result.report["rebalanced_removed"] == 3
    assert result.detected_ids == ["r0", "r1", "r2"]

    plain = filter_mod.apply(_property(), records, {"mode": "drop", "rebalance": False})
    assert len(plain.rows) == 7


def test_filter_split_returns_both_arms_not_just_one(monkeypatch):
    records = [_record(f"r{i}") for i in range(10)]
    _stub_detector(monkeypatch, {"r0", "r1", "r2", "r3"})
    result = filter_mod.apply(_property(), records, {"mode": "split"})
    assert set(result.arms) == {"has", "lacks"}
    assert len(result.arms["has"]) == 4 and len(result.arms["lacks"]) == 6
    assert result.rows == result.arms["has"]


# --- restrict: judge and edit a share, write back the whole corpus ----------------------

def _mixture(n_da=4, n_replay=6):
    """A corpus shaped like a real mixture: a small synth share plus replay."""
    records = [_record(f"da{i}") for i in range(n_da)]
    for r in records:
        r.metadata["source_label"] = "synthdoc_difficult_advice"
    replay = [_record(f"tulu{i}") for i in range(n_replay)]
    for r in replay:
        r.metadata["source_label"] = "tulu3"
    return records + replay


def test_candidates_narrows_what_is_judged_without_shrinking_the_corpus():
    records = _mixture()
    chosen, untouched = ablation_base.candidates(
        records, {"restrict": {"source_label": "synthdoc_difficult_advice"}})
    assert [r.record_id for r in chosen] == ["da0", "da1", "da2", "da3"]
    assert untouched == {"tulu0", "tulu1", "tulu2", "tulu3", "tulu4", "tulu5"}
    # No restriction means the whole corpus.
    assert len(ablation_base.candidates(records, None)[0]) == 10
    assert len(ablation_base.candidates(records, {})[0]) == 10
    with pytest.raises(ValueError, match="matches none of the 10 records"):
        ablation_base.candidates(records, {"restrict": {"source_label": "nope"}})


def test_filter_drop_under_a_restriction_keeps_every_replay_row(monkeypatch):
    records = _mixture()
    _stub_detector(monkeypatch, {"da0", "da1"})
    result = filter_mod.apply(_property(), records, {
        "mode": "drop", "rebalance": True,
        "restrict": {"source_label": "synthdoc_difficult_advice"}})
    kept = {r["messages"][1]["reasoning_content"] for r in result.rows}
    # 2 flagged + 2 rebalanced, all from the difficult-advice share; the 6 replay rows
    # survive, because dropping them would change the mixture's composition too.
    assert len(result.rows) == 6
    assert result.report["n_untouched"] == 6 and result.report["n_judged"] == 4
    assert kept  # sanity: the rows are real rows, not placeholders


def test_filter_split_under_a_restriction_gives_both_arms_the_replay(monkeypatch):
    records = _mixture()
    _stub_detector(monkeypatch, {"da0", "da1"})
    result = filter_mod.apply(_property(), records, {
        "mode": "split", "restrict": {"source_label": "synthdoc_difficult_advice"}})
    # Each arm = the 6 replay rows + its half of the difficult-advice share, so the halves
    # differ in the property rather than in how much data they carry.
    assert len(result.arms["has"]) == 8 and len(result.arms["lacks"]) == 8
    assert result.report["n_kept_outside_restriction"] == 6


def test_filter_refuses_a_detector_that_matches_nothing_or_everything(monkeypatch):
    records = [_record(f"r{i}") for i in range(6)]
    _stub_detector(monkeypatch, set())
    with pytest.raises(ValueError, match="flagged 0 of 6"):
        filter_mod.apply(_property(), records, {"mode": "drop"})
    _stub_detector(monkeypatch, {r.record_id for r in records})
    with pytest.raises(ValueError, match="flagged ALL 6"):
        filter_mod.apply(_property(), records, {"mode": "drop"})


# --- the ablate driver, end to end, offline --------------------------------------------

def _load_script(name):
    """Import one of the scripts/properties drivers by path."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "properties" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ablate_driver_runs_verifies_and_emits_a_train_config(tmp_path, monkeypatch):
    """The whole seam: property -> ablation -> verify -> derived train config, with the
    two model calls stubbed and no HF push."""
    from omegaconf import OmegaConf

    ablate = _load_script("ablate")

    corpus = tmp_path / "corpus.jsonl"
    rows = []
    for i in range(200):
        has_property = i < 60
        rows.append({"source": "da", "metadata": {"scenario_id": f"s{i}"},
                     "messages": [{"role": "user", "content": "q"},
                                  {"role": "assistant", "content": "a",
                                   "reasoning_content": ("weighs likelihood and severity"
                                                         if has_property else "plain")}]})
    corpus.write_text("".join(json.dumps(r) + "\n" for r in rows))

    registry_path = tmp_path / "properties.jsonl"
    PropertyRegistry(registry_path).write([_property(property_id="tc:run:g001")])

    # The detector reads the record's text, so it works identically before and after.
    def fake_detect(records, label, detector, channel="reasoning", **kw):
        return [{"record_id": r.record_id,
                 "exhibits": "likelihood" in r.channel(channel)} for r in records]

    monkeypatch.setattr(filter_mod.interpret_mod, "detect", fake_detect)
    monkeypatch.setattr(ablate.verify_mod.interpret_mod, "detect", fake_detect)

    train_base = tmp_path / "lora_control.yaml"
    train_base.write_text(OmegaConf.to_yaml(OmegaConf.create({
        "model": "Qwen/Qwen3.6-27B", "thinking": True,
        "data_repo": "org/control", "data_file": "mixture.jsonl",
        "data_revision": "deadbeef", "hf_repo": "org/adapter-control",
        "train": {"epochs": 1}})))

    config = tmp_path / "ablate_test.yaml"
    config.write_text(OmegaConf.to_yaml(OmegaConf.create({
        "tag": "t", "registry": str(registry_path), "property_id": "tc:run:g001",
        "source": {"name": "mixture_rows", "path": str(corpus)},
        "ablation": {"kind": "filter", "mode": "drop", "rebalance": True},
        "verify": {"sample": 500, "min_drop": 0.2, "check_separability": False},
        "train": {"base_config": str(train_base)},
        "evals": ["odcv"]})))

    out = tmp_path / "run"
    ablate.main(config=str(config), out_dir=str(out), no_push=True)

    assert (out / "ablated.jsonl").exists()
    verification = json.loads((out / "verify_ablated.json").read_text())
    assert verification["passed"], verification["failures"]
    # 60 of 200 flagged, before 30% -> after 0%.
    assert verification["before"]["prevalence"] == 0.3
    assert verification["after"]["prevalence"] == 0.0
    assert (out / "verify_ablated.md").exists()

    derived = OmegaConf.load(out / "lora_control_filter-g001.yaml")
    # Exactly the data pointer and the adapter repo differ from the control.
    assert derived.data_repo.endswith("-ablated") or "ablate" in derived.data_repo
    assert derived.hf_repo == "org/adapter-control-filter-g001"
    assert derived.model == "Qwen/Qwen3.6-27B" and derived.thinking is True
    assert json.loads((out / "run_meta.json").read_text())["summary"]["kind"] == "filter"


def test_ablate_driver_stops_when_verification_fails(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    ablate = _load_script("ablate")

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(
        {"source": "da", "metadata": {"scenario_id": f"s{i}"},
         "messages": [{"role": "user", "content": "q"},
                      {"role": "assistant", "content": "a",
                       "reasoning_content": "weighs likelihood"}]}) + "\n"
        for i in range(200)))

    registry_path = tmp_path / "properties.jsonl"
    PropertyRegistry(registry_path).write([_property(property_id="tc:run:g001")])

    # Flags half the rows, and keeps flagging half after the drop: the ablation did not
    # move the prevalence, so the gate must stop before anything is pushed.
    def fake_detect(records, label, detector, channel="reasoning", **kw):
        return [{"record_id": r.record_id, "exhibits": i % 2 == 0}
                for i, r in enumerate(records)]

    monkeypatch.setattr(filter_mod.interpret_mod, "detect", fake_detect)
    monkeypatch.setattr(ablate.verify_mod.interpret_mod, "detect", fake_detect)

    config = tmp_path / "ablate_test.yaml"
    config.write_text(OmegaConf.to_yaml(OmegaConf.create({
        "tag": "t", "registry": str(registry_path), "property_id": "tc:run:g001",
        "source": {"name": "mixture_rows", "path": str(corpus)},
        "ablation": {"kind": "filter", "mode": "drop", "rebalance": False},
        "verify": {"sample": 500, "min_drop": 0.3, "check_separability": False}})))

    out = tmp_path / "run"
    with pytest.raises(ValueError, match="verification failed"):
        ablate.main(config=str(config), out_dir=str(out), no_push=True)
    # The failure is recorded, not just raised.
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["verification_failed"] == ["ablated"]


# --- regenerate against a real synth config (no network) -------------------------------

def test_regenerate_writes_a_synth_config_that_suppresses_the_property(tmp_path):
    from omegaconf import OmegaConf

    from src.properties.ablation import regenerate as regenerate_mod

    source = "configs/data/synth/difficult_advice.yaml"
    out = tmp_path / "derived.yaml"
    cfg = {"synth_config": source, "stage": "revise_responses",
           "method": "suppress_prompt", "out_config": str(out)}
    ok, reason = regenerate_mod.applicable(_property(), [_record()],
                                           SOURCES["mixture_rows"], cfg)
    assert ok, reason

    records = [_record()]
    result = regenerate_mod.apply(_property(description="weighs risk"), records, cfg)
    derived = OmegaConf.load(out)
    stage = next(s for s in derived.stages if s.name == "revise_responses")
    assert "CONSTRAINT FOR THIS RUN" in stage.prompts.system
    assert "weighs risk" in stage.prompts.system
    # The user half is untouched: it carries the <<<cache>>> mark, and appending after
    # that would change the cached prefix on every call.
    original = next(s for s in OmegaConf.load(source).stages
                    if s.name == "revise_responses")
    assert stage.prompts.user == original.prompts.user
    # PLANNED, not done: the corpus handed back is the original.
    assert result.changed_ids == [] and len(result.rows) == 1
    assert result.report["status"].startswith("PLANNED")
    assert result.report["command"].startswith("uv run synth run --config")


def test_regenerate_refuses_a_stage_the_config_does_not_have(tmp_path):
    from src.properties.ablation import regenerate as regenerate_mod

    cfg = {"synth_config": "configs/data/synth/difficult_advice.yaml",
           "stage": "not_a_stage", "out_config": str(tmp_path / "x.yaml")}
    with pytest.raises(ValueError, match="is not in"):
        regenerate_mod.apply(_property(), [_record()], cfg)
    missing = {"synth_config": "configs/data/synth/nope.yaml", "stage": "x"}
    ok, reason = regenerate_mod.applicable(_property(), [_record()],
                                           SOURCES["mixture_rows"], missing)
    assert not ok and "does not exist" in reason
