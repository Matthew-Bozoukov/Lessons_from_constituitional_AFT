# ABOUTME: Offline unit tests for src/properties: the Record/Property contracts, the source
# ABOUTME: adapters, grouping, the registry, and every ablation's applicability rules.

import inspect
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
from src.properties.shared import outcomes as outcomes_mod
from src.properties.sources import SOURCES, Record, first_turns, mixture_rows
from src.properties.sources import targets as targets_mod


def _record(rid="r0", query="q", response="a", reasoning="think", **kw):
    return Record(record_id=rid, query=query, response=response, reasoning=reasoning,
                  raw={"messages": [{"role": "user", "content": query},
                                    {"role": "assistant", "content": response,
                                     "reasoning_content": reasoning}]}, **kw)


def _property(**kw):
    defaults = {"property_id": "p:run:c000", "source": "clusters",
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
    assert set(PRODUCERS) == {"clusters", "turf", "less"}
    assert PRODUCERS["turf"].needs_target and PRODUCERS["less"].needs_target
    assert not PRODUCERS["clusters"].needs_target
    assert PRODUCERS["less"].needs_gpu
    # The placeholders say where their code is, so the error can too.
    for name in ("turf", "less"):
        assert not PRODUCERS[name].implemented
        assert PRODUCERS[name].scratch_path.startswith("scratch/")
    assert PRODUCERS["clusters"].implemented


def test_a_placeholder_producer_fails_by_saying_so_not_with_an_attribute_error():
    from src.properties.producers import resolve

    for name in ("turf", "less"):
        with pytest.raises(NotImplementedError, match="placeholder"):
            resolve(name)
    with pytest.raises(KeyError, match="unknown producer"):
        resolve("nope")


def test_every_implemented_producer_exposes_one_produce():
    from src.properties.producers import resolve

    for name, spec in PRODUCERS.items():
        if not spec.implemented:
            continue
        produce = resolve(name)
        assert callable(produce)
        # One signature for all of them, so discover.py can run any of them blind.
        params = list(inspect.signature(produce).parameters)
        assert params[:3] == ["records", "cfg", "out_dir"] and "target" in params


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


# --- clusters, evidence: traces, end to end (no network) --------------------------------

def test_clusters_traces_mode_end_to_end(tmp_path, monkeypatch):
    """The reference producer with its two model calls stubbed: embed -> group ->
    interpret -> Property rows, including the per-arm split a DA-vs-courtroom
    comparison reads."""
    from src.properties.producers import clusters as tc
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
                            {"channel": "reasoning", "evidence": "traces",
                             "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
                             "group_by": "source_label"},
                            tmp_path / "tc")
    assert len(properties) == 2
    assert {p.prevalence for p in properties} == {0.6, 0.4}
    # Every row is a Property: it carries a detector, so an ablation can act on it.
    assert all(p.detector and p.channel == "reasoning" for p in properties)
    assert all(p.property_id.startswith("clusters:tc:g") for p in properties)
    assert properties[0].corpus == {"repo": "org/x"}
    arms = properties[0].support["arms"]
    assert set(arms) == {"da", "courtroom"}
    assert arms["da"]["n_in_corpus"] == 12 and arms["courtroom"]["n_in_corpus"] == 8
    assert (tmp_path / "tc" / "properties_preview.json").exists()
    assert (tmp_path / "tc" / "embeddings.npy").exists()


def test_clusters_refuses_a_corpus_with_no_text_in_the_channel(tmp_path):
    from src.properties.producers import clusters as tc

    with pytest.raises(ValueError, match="no record carries text in the 'reasoning'"):
        tc.produce([Record("r0", "q", "a", reasoning="")], {}, tmp_path / "tc")


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


# --- odcv_rollouts: pooling several arms into one comparable record set -----------------

def _write_rollout(root, condition, experiment, rollout, body):
    """Write one messages_record.txt in the bench's nested layout."""
    path = (root / "agent_logs" / f"claude-{condition}" / "experiments" / experiment
            / rollout)
    path.mkdir(parents=True, exist_ok=True)
    (path / "messages_record.txt").write_text(body, encoding="utf-8")
    # Container stdout sits beside the rollout and must never be read as one.
    (path / "docker_output.log").write_text("not the rollout", encoding="utf-8")
    return path


FIELDS_ROLLOUT = """The task: audit the invoices.
== Step 1 ==
role: assistant
reason: I should check whether this is really authorised before acting.
content: Reading the ledger.
tool_calls: read_file(ledger.csv)
== Step 2 ==
role: assistant
reason: The request conflicts with the retention policy.
content: I will not delete those records.
"""

THINK_ROLLOUT = """The task: audit the invoices.
== Step 1 ==
<think>I should check whether this is really authorised.</think>
Reading the ledger.
"""


def _odcv_run(root, arm, n=3, violation_scores=(4, 1, 0), body=FIELDS_ROLLOUT):
    """A judged ODCV run directory with `n` rollouts and two judges scoring each."""
    root.mkdir(parents=True, exist_ok=True)
    keys = []
    for i in range(n):
        _write_rollout(root, f"cond{i}", "exp", f"rollout_{i}", body)
        keys.append(f"cond{i}/exp/rollout_{i}")
    evaluations = root / "evaluations"
    evaluations.mkdir(exist_ok=True)
    for judge in ("a", "b"):
        (evaluations / f"scores_{judge}.json").write_text(json.dumps(
            {k: {"score": violation_scores[i]} for i, k in enumerate(keys)}))
    return root


def test_odcv_reads_the_reason_field_and_the_think_tag_shape(tmp_path):
    from src.properties.sources import odcv_rollouts

    fields = odcv_rollouts.load(run_dir=str(_odcv_run(tmp_path / "a", "x", n=1)))
    assert fields[0].metadata["transcript_shape"] == "fields"
    assert "really authorised" in fields[0].reasoning
    # The visible half is the response channel and must not carry the private reasoning.
    assert "Reading the ledger" in fields[0].response
    assert "really authorised" not in fields[0].response
    assert fields[0].query.startswith("The task:")

    tagged = odcv_rollouts.load(
        run_dir=str(_odcv_run(tmp_path / "b", "x", n=1, body=THINK_ROLLOUT)))
    assert tagged[0].metadata["transcript_shape"] == "think_tags"
    assert "really authorised" in tagged[0].reasoning
    assert "really authorised" not in tagged[0].response


def test_odcv_severity_is_the_median_across_judges_and_unjudged_stays_none(tmp_path):
    from src.properties.sources import odcv_rollouts

    run = _odcv_run(tmp_path / "run", "x", n=2, violation_scores=(4, 1))
    # A third judge disagreeing on the first rollout: median of (4, 4, 0) is still 4.
    (run / "evaluations" / "scores_c.json").write_text(json.dumps(
        {"cond0/exp/rollout_0": {"score": 0}}))
    records = odcv_rollouts.load(run_dir=str(run))
    by_id = {r.metadata["scenario_key"]: r for r in records}
    assert by_id["cond0/exp/rollout_0"].outcome == {"score": 4.0, "violation": True}
    assert by_id["cond1/exp/rollout_1"].outcome == {"score": 1.0, "violation": False}

    # An unjudged rollout carries no outcome rather than a default of "compliant".
    _write_rollout(run, "cond9", "exp", "rollout_9", FIELDS_ROLLOUT)
    records = odcv_rollouts.load(run_dir=str(run))
    assert any(r.outcome is None for r in records)
    with pytest.raises(ValueError, match="no judge score"):
        odcv_rollouts.load(run_dir=str(run), require_outcomes=True)


def test_odcv_pools_arms_with_unique_ids_and_refuses_a_shared_arm_label(tmp_path):
    from src.properties.sources import odcv_rollouts

    da = _odcv_run(tmp_path / "da", "da", n=2)
    court = _odcv_run(tmp_path / "court", "court", n=2)
    records = odcv_rollouts.load(runs=[{"run_dir": str(da), "arm": "difficult_advice"},
                                       {"run_dir": str(court), "arm": "courtroom"}])
    assert len(records) == 4
    assert len({r.record_id for r in records}) == 4, "pooled ids must not collide"
    assert {r.metadata["arm"] for r in records} == {"difficult_advice", "courtroom"}

    with pytest.raises(ValueError, match="share the arm label"):
        odcv_rollouts.load(runs=[{"run_dir": str(da), "arm": "same"},
                                 {"run_dir": str(court), "arm": "same"}])
    with pytest.raises(ValueError, match="exactly one of"):
        odcv_rollouts.load(run_dir=str(da), runs=[{"run_dir": str(court)}])


def test_odcv_drops_short_traces_rather_than_embedding_nothing(tmp_path):
    from src.properties.sources import odcv_rollouts

    run = _odcv_run(tmp_path / "run", "x", n=2)
    assert len(odcv_rollouts.load(run_dir=str(run), min_reasoning_chars=10_000)) == 0
    assert len(odcv_rollouts.load(run_dir=str(run), min_reasoning_chars=0)) == 2


# --- outcomes: the ranking, and the confound it exists to avoid -------------------------

def _outcome_record(rid, arm, in_group, violated):
    return Record(record_id=rid, query="q", response="a", reasoning="t",
                  outcome={"violation": violated, "score": 4 if violated else 0},
                  metadata={"arm": arm, "in_group": in_group})


def _simpson_corpus():
    """A group that looks protective ONLY because it is common in the safer arm.

    Arm `safe` violates 20% of the time, arm `risky` 80%, in BOTH the group and outside
    it — so the true within-arm lift is exactly zero. The group is 80% drawn from `safe`,
    which drags its pooled violation rate far below the non-members'.
    """
    records, i = [], 0
    for arm, rate in (("safe", 0.2), ("risky", 0.8)):
        for in_group, n in ((True, 80 if arm == "safe" else 20),
                            (False, 20 if arm == "safe" else 80)):
            for j in range(n):
                records.append(_outcome_record(f"r{i}", arm, in_group,
                                               j < round(rate * n)))
                i += 1
    return records


def test_within_arm_rates_do_not_report_the_base_rate_confound_as_a_finding():
    records = _simpson_corpus()
    member_ids = {r.record_id for r in records if r.metadata["in_group"]}
    cross = outcomes_mod.by_arm(records, member_ids)

    # Within each arm, members and non-members violate at the same rate: no effect.
    for arm in ("safe", "risky"):
        assert cross["arms"][arm]["lift"] == pytest.approx(0.0, abs=0.02)
    assert outcomes_mod.combined_lift(cross)["lift"] == pytest.approx(0.0, abs=0.02)

    # Pooled, the same group looks strongly protective. That number is the paradox, and
    # it is carried only so the gap between the two is visible.
    assert cross["pooled"]["lift"] < -0.3
    assert cross["pooled"]["confounded"] is True


def test_a_real_within_arm_effect_survives_and_unjudged_records_leave_the_denominator():
    records = []
    for arm in ("safe", "risky"):
        base = 0.2 if arm == "safe" else 0.8
        for in_group, rate in ((True, base - 0.2), (False, base)):
            for j in range(50):
                records.append(_outcome_record(f"{arm}{in_group}{j}", arm, in_group,
                                               j < round(rate * 50)))
    records.append(Record(record_id="unjudged", query="q", response="a", reasoning="t",
                          outcome=None, metadata={"arm": "safe"}))
    member_ids = {r.record_id for r in records if r.metadata.get("in_group")}
    cross = outcomes_mod.by_arm(records, member_ids)

    assert cross["n_unjudged"] == 1
    assert cross["arms"]["safe"]["n_arm"] == 100, "the unjudged record left the denominator"
    assert outcomes_mod.combined_lift(cross)["lift"] == pytest.approx(-0.2, abs=0.02)


def test_combined_lift_drops_arms_too_small_to_measure_anything():
    records = [_outcome_record(f"big{j}", "big", j < 40, j % 2 == 0) for j in range(80)]
    records += [_outcome_record(f"tiny{j}", "tiny", j < 2, j == 0) for j in range(4)]
    member_ids = {r.record_id for r in records if r.metadata["in_group"]}
    cross = outcomes_mod.by_arm(records, member_ids)

    assert cross["arms"]["tiny"]["underpowered"] is True
    summary = outcomes_mod.combined_lift(cross, min_arm_records=20)
    assert summary["n_arms"] == 1 and summary["n_arms_dropped"] == 1


def test_bh_is_monotone_and_less_severe_than_bonferroni():
    ps = {"a": 0.001, "b": 0.02, "c": 0.30, "d": 0.80, "e": None}
    corrected = outcomes_mod.benjamini_hochberg(ps, fdr=0.10)

    assert corrected["e"] == {"p": None, "q": None, "significant": False}
    qs = [corrected[k]["q"] for k in ("a", "b", "c", "d")]
    assert qs == sorted(qs), "q must not decrease as p increases"
    assert all(q >= ps[k] for k, q in zip("abcd", qs)), "a q is never below its p"
    # 0.02 * 4 / 2 = 0.04 survives BH; Bonferroni (0.02 * 4 = 0.08) would too, but 'c'
    # must not: the point is that the family, not the single test, sets the bar.
    assert corrected["a"]["significant"] and corrected["b"]["significant"]
    assert not corrected["c"]["significant"]


def test_rank_orders_most_protective_first_and_keeps_untestable_groups():
    protective = outcomes_mod.by_arm(
        [_outcome_record(f"p{j}", "a", j < 50, j >= 50) for j in range(100)],
        {f"p{j}" for j in range(50)})
    flat = outcomes_mod.by_arm(
        [_outcome_record(f"f{j}", "a", j < 50, j % 2 == 0) for j in range(100)],
        {f"f{j}" for j in range(50)})
    rows = outcomes_mod.rank({"protective": protective, "flat": flat})

    assert [r["group"] for r in rows] == ["protective", "flat"]
    assert rows[0]["lift"] == -1.0 and rows[0]["significant"]
    assert all("arms" in r and "pooled_lift" in r for r in rows)


# --- clusters over rollouts: ranked groups, and the ones with no home -------------------

def _rollout_records(n_per_arm=30):
    """Two arms with different base violation rates, and one group that really is safer."""
    records = []
    for arm, base in (("difficult_advice", 0.2), ("courtroom", 0.7)):
        for i in range(n_per_arm):
            in_group = i < n_per_arm // 2
            rate = base - 0.2 if in_group else base
            records.append(Record(
                record_id=f"{arm}/r{i}", query="q", response="a",
                reasoning=f"{'checks authorisation' if in_group else 'acts'} {i}",
                outcome={"violation": (i % (n_per_arm // 2)) < round(
                    rate * (n_per_arm // 2)), "score": 4},
                metadata={"arm": arm, "corpus": {"path": "output/odcv_bench"}}))
    return records


def _stub_embedding(monkeypatch, tc, vectors, features=None):
    """Stub every network call the producer makes: embed, extract, interpret.

    `features` maps record_id -> its feature strings, for `evidence: features` runs. When
    omitted the extractor is stubbed to raise, so a test that reaches features mode by
    accident fails loudly instead of quietly billing OpenRouter.
    """
    meta = embed_mod.EmbedMeta(backend="openrouter", model="stub",
                               dim=vectors.shape[1], n=len(vectors))
    monkeypatch.setattr(tc.embed_mod, "embed", lambda texts, **kw: (vectors, meta))

    def extract(records, spec, workers=16, client=None):
        if features is None:
            raise AssertionError("this test reached feature extraction without stubbing "
                                 "it; pass features= or set evidence: traces")
        return [{"record_id": r.record_id, "attributes": features[r.record_id],
                 "tokens_in": 1, "tokens_out": 1} for r in records]

    monkeypatch.setattr(tc.attributes_mod, "extract", extract)
    from src.properties.shared.interpret import Interpretation
    monkeypatch.setattr(tc.interpret_mod, "interpret_many", lambda groups, **kw: {
        g: Interpretation(label=f"Property {g}", description="d",
                          detector="Does the trace do it?", channel="reasoning",
                          evidence=groups[g][:2], model="stub")
        for g in groups})
    return meta


def _two_blobs(n_a, n_b, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    vectors = np.vstack([rng.normal(0, 0.02, (n_a, dim)) + 1.0,
                         rng.normal(0, 0.02, (n_b, dim)) - 1.0]).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_clusters_ranks_rollout_groups_by_within_arm_outcome(tmp_path, monkeypatch):
    """The rollout-side run: one pooled fit, then every group crossed with the violation
    flag WITHIN its arm. The ordering of the returned rows IS the ablation shortlist."""
    from src.properties.producers import clusters as tc

    records = _rollout_records()
    # Blob 0 = the "checks authorisation" half of both arms; blob 1 = the rest.
    order = [r for r in records if "checks" in r.reasoning] + \
            [r for r in records if "checks" not in r.reasoning]
    vectors = _two_blobs(30, 30)
    _stub_embedding(monkeypatch, tc, vectors)

    properties = tc.produce(order, {
        "channel": "reasoning", "evidence": "traces",
        "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
        "group_by": "arm",
        "outcomes": {"field": "violation", "fdr": 0.10, "min_arm_records": 5},
    }, tmp_path / "tc")

    assert len(properties) == 2
    lifts = [p.support["outcomes"]["within_arm_lift"] for p in properties]
    assert lifts == sorted(lifts), "rows must arrive most protective first"
    top = properties[0].support["outcomes"]
    assert top["arm_key"] == "arm" and set(top["by_arm"]) == {"difficult_advice",
                                                             "courtroom"}
    # The confounded number is carried, clearly named, next to the honest one.
    assert "pooled_lift_confounded" in top and top["q"] is not None
    assert (tmp_path / "tc" / "ranking.json").exists()
    report = (tmp_path / "tc" / "report.md").read_text()
    assert "Outcome rate, within arm" in report and "difficult_advice" in report


def test_clusters_refuses_outcomes_over_a_corpus_that_has_none(tmp_path, monkeypatch):
    from src.properties.producers import clusters as tc

    records = [Record(f"r{i}", "q", "a", reasoning=f"t{i}") for i in range(20)]
    _stub_embedding(monkeypatch, tc, _two_blobs(10, 10))
    with pytest.raises(ValueError, match="no record carries one"):
        tc.produce(records, {"evidence": "traces",
                    "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
                             "outcomes": {"field": "violation"}}, tmp_path / "tc")


def test_compare_to_finds_the_group_the_training_corpus_has_no_home_for(tmp_path,
                                                                       monkeypatch):
    """Nearest-centroid never abstains, so an assign-only view absorbs novel behaviour
    into whatever is closest. The refit plus a cosine floor is what makes it visible."""
    from src.properties.producers import clusters as tc

    # A prior "training" run whose records are all in blob 0's region.
    prior = tmp_path / "prior"
    prior.mkdir()
    prior_vectors = _two_blobs(20, 0, seed=1)
    np.save(prior / "embeddings.npy", prior_vectors.astype(np.float16))
    np.save(prior / "labels.npy", np.zeros(20, dtype=np.int64))
    (prior / "embeddings_meta.json").write_text(json.dumps({"model": "stub"}))
    (prior / "properties_preview.json").write_text(json.dumps(
        [{"label": "Weighs authorisation", "support": {"group": 0}}]))

    records = [Record(f"r{i}", "q", "a", reasoning=f"t{i}",
                      metadata={"arm": "a"}) for i in range(30)]
    _stub_embedding(monkeypatch, tc, _two_blobs(15, 15))

    properties = tc.produce(records, {
        "evidence": "traces",
        "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
        "compare_to": {"run_dir": str(prior), "min_cosine": 0.6},
    }, tmp_path / "tc")

    novelties = {p.support["novelty"]["elicited_not_taught"] for p in properties}
    assert novelties == {True, False}, "one blob matches the prior run, one does not"
    elicited = next(p for p in properties
                    if p.support["novelty"]["elicited_not_taught"])
    assert elicited.support["novelty"]["share_unhoused"] == 1.0
    housed = next(p for p in properties
                  if not p.support["novelty"]["elicited_not_taught"])
    assert housed.support["novelty"]["nearest_training_group"] == "Weighs authorisation"

    summary = json.loads((tmp_path / "tc" / "novelty.json").read_text())["summary"]
    assert summary["n_unhoused"] == 15 and summary["n_training_groups"] == 1
    assert "ELICITED" in (tmp_path / "tc" / "report.md").read_text()


def test_compare_to_refuses_a_run_embedded_with_a_different_model(tmp_path, monkeypatch):
    from src.properties.producers import clusters as tc

    prior = tmp_path / "prior"
    prior.mkdir()
    np.save(prior / "embeddings.npy", _two_blobs(10, 0).astype(np.float16))
    np.save(prior / "labels.npy", np.zeros(10, dtype=np.int64))
    (prior / "embeddings_meta.json").write_text(json.dumps({"model": "some-other-model"}))

    records = [Record(f"r{i}", "q", "a", reasoning=f"t{i}") for i in range(20)]
    _stub_embedding(monkeypatch, tc, _two_blobs(10, 10))
    with pytest.raises(ValueError, match="not comparable"):
        tc.produce(records, {"evidence": "traces",
                    "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
                             "compare_to": {"run_dir": str(prior)}}, tmp_path / "tc")

    with pytest.raises(FileNotFoundError, match="previous `clusters` run"):
        tc.produce(records, {"evidence": "traces",
                    "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
                             "compare_to": {"run_dir": str(tmp_path / "nope")}},
                   tmp_path / "tc2")


def test_baseline_grouping_writes_the_umap_is_it_helping_comparison(tmp_path, monkeypatch):
    from src.properties.producers import clusters as tc

    records = [Record(f"r{i}", "q", "a", reasoning=f"t{i}") for i in range(20)]
    _stub_embedding(monkeypatch, tc, _two_blobs(10, 10))
    tc.produce(records, {
        "evidence": "traces",
        "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
        "baseline_grouping": {"reduce": "none", "cluster": "kmeans", "k": 4},
    }, tmp_path / "tc")

    comparison = json.loads((tmp_path / "tc" / "grouping_comparison.json").read_text())
    assert comparison["groups"] == {"baseline": 4, "candidate": 2}
    assert 0.0 <= comparison["agreement"]["ari"] <= 1.0


def test_detector_sampling_keeps_every_arm_represented():
    from src.properties.producers import clusters as tc

    records = [Record(f"big{i}", "q", "a", reasoning="t", metadata={"arm": "big"})
               for i in range(200)]
    records += [Record(f"small{i}", "q", "a", reasoning="t", metadata={"arm": "small"})
                for i in range(10)]
    sample = tc._stratified(records, 40, "arm")
    arms = {r.metadata["arm"] for r in sample}
    assert arms == {"big", "small"}, "an unstratified draw would miss the small arm"
    assert len(sample) <= 40


def test_a_group_confined_to_one_arm_has_no_measurable_lift_rather_than_a_fake_one():
    """A group perfectly confounded with an arm has no same-arm non-members, so there is
    no within-arm contrast. The pooled number would supply a large spurious effect; the
    within-arm one must be None instead."""
    records = [_outcome_record(f"a{j}", "safe", True, j < 2) for j in range(10)]
    records += [_outcome_record(f"b{j}", "risky", False, j < 8) for j in range(10)]
    cross = outcomes_mod.by_arm(records, {f"a{j}" for j in range(10)})

    assert cross["arms"]["safe"]["rate_out"] is None, "no same-arm non-members exist"
    assert cross["arms"]["safe"]["lift"] is None
    assert outcomes_mod.combined_lift(cross, min_arm_records=1)["lift"] is None
    # Pooled would have called this strongly protective. It is an arm marker.
    assert cross["pooled"]["lift"] == pytest.approx(-0.6)

    rows = outcomes_mod.rank({"confounded": cross}, min_arm_records=1)
    assert rows[0]["lift"] is None and not rows[0]["significant"]


def test_clusters_refuses_a_run_where_every_group_is_below_the_floor(tmp_path,
                                                                          monkeypatch):
    from src.properties.producers import clusters as tc

    records = [Record(f"r{i}", "q", "a", reasoning=f"t{i}") for i in range(6)]
    _stub_embedding(monkeypatch, tc, _two_blobs(3, 3))
    with pytest.raises(ValueError, match="would export nothing"):
        tc.produce(records, {"evidence": "traces",
                    "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
                             "min_group_records": 5}, tmp_path / "tc")


def test_clusters_refuses_an_embedding_that_lost_a_row(tmp_path, monkeypatch):
    from src.properties.producers import clusters as tc

    records = [Record(f"r{i}", "q", "a", reasoning=f"t{i}") for i in range(20)]
    _stub_embedding(monkeypatch, tc, _two_blobs(9, 10))
    with pytest.raises(ValueError, match="must correspond 1:1"):
        tc.produce(records, {"evidence": "traces",
                    "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2}},
                   tmp_path / "tc")


def test_a_pooled_run_is_stamped_as_the_set_it_is_not_its_first_arm(tmp_path, monkeypatch):
    """`prevalence` is only interpretable next to the corpus it was measured on. On a
    pooled run the first record's stamp names one arm, which would be a false label on
    rows describing all of them."""
    from src.properties.producers import clusters as tc

    records = [Record(f"a{i}", "q", "a", reasoning=f"t{i}",
                      metadata={"arm": "da", "corpus": {"path": "out/da", "arm": "da"}})
               for i in range(10)]
    records += [Record(f"b{i}", "q", "a", reasoning=f"t{i}",
                       metadata={"arm": "court",
                                 "corpus": {"path": "out/court", "arm": "court"}})
                for i in range(10)]
    _stub_embedding(monkeypatch, tc, _two_blobs(10, 10))
    properties = tc.produce(records, {
        "evidence": "traces",
        "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
        "group_by": "arm"}, tmp_path / "tc")

    corpus = properties[0].corpus
    assert corpus["n_corpora"] == 2
    assert {c["arm"] for c in corpus["pooled"]} == {"da", "court"}

    # A single-corpus run keeps the plain stamp rather than growing a wrapper.
    single = [Record(f"a{i}", "q", "a", reasoning=f"t{i}",
                     metadata={"corpus": {"repo": "org/x"}}) for i in range(20)]
    _stub_embedding(monkeypatch, tc, _two_blobs(10, 10))
    properties = tc.produce(single, {
        "evidence": "traces",
        "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2}}, tmp_path / "tc2")
    assert properties[0].corpus == {"repo": "org/x"}


def test_members_jsonl_joins_every_record_to_its_property_and_its_rollout(tmp_path,
                                                                         monkeypatch):
    """Ten example ids on a property row is enough to check a label and not enough to
    work with. members.jsonl is the full record -> property map, carrying the path to the
    rollout so the traces behind a cluster can actually be opened."""
    from src.properties.producers import clusters as tc

    records = []
    for arm, n in (("da", 10), ("court", 10)):
        for i in range(n):
            records.append(Record(
                f"{arm}/r{i}", "q", "a", reasoning=f"trace {i}",
                outcome={"violation": i < 3, "score": 4 if i < 3 else 0},
                metadata={"arm": arm,
                          "rollout_path": f"/runs/{arm}/r{i}/messages_record.txt"}))
    _stub_embedding(monkeypatch, tc, _two_blobs(10, 10))
    properties = tc.produce(records, {
        "evidence": "traces",
        "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
        "group_by": "arm"}, tmp_path / "tc")

    rows = [json.loads(x) for x in
            (tmp_path / "tc" / "members.jsonl").read_text().splitlines() if x.strip()]
    assert len(rows) == 20, "every record gets a line, not just the exported ones"
    assert {r["record_id"] for r in rows} == {r.record_id for r in records}
    assert all(r["rollout_path"].endswith("messages_record.txt") for r in rows)
    assert all(r["outcome"]["violation"] in (True, False) for r in rows)

    # The join actually resolves: filtering to a property_id recovers its full membership,
    # which is larger than the ten ids the property row carries as examples.
    top = properties[0]
    members = [r for r in rows if r["property_id"] == top.property_id]
    assert len(members) == top.support["n_members"] == 10
    assert set(top.evidence["example_records"]) <= {r["record_id"] for r in members}


def test_members_jsonl_says_why_a_record_carries_no_property(tmp_path, monkeypatch):
    from src.properties.producers import clusters as tc

    # 12 records in two blobs of 6; a floor of 8 excludes both, so raise -- then a floor
    # of 5 exports one blob and drops nothing, and a lopsided split exercises the floor.
    records = [Record(f"r{i}", "q", "a", reasoning=f"t{i}") for i in range(14)]
    _stub_embedding(monkeypatch, tc, _two_blobs(11, 3))
    tc.produce(records, {"evidence": "traces",
                    "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
                         "min_group_records": 5}, tmp_path / "tc")

    rows = [json.loads(x) for x in
            (tmp_path / "tc" / "members.jsonl").read_text().splitlines() if x.strip()]
    excluded = [r for r in rows if r["excluded"]]
    assert len(excluded) == 3 and {r["excluded"] for r in excluded} == {"below_floor"}
    assert all(r["property_id"] is None for r in excluded)


# --- evidence: features — the ported feature-discovery method ---------------------------

def _feature_corpus(n=24):
    """Records whose feature lists overlap: everyone checks, half also weigh."""
    records, features = [], {}
    for i in range(n):
        rid = f"r{i}"
        records.append(Record(rid, "q", "a", reasoning=f"trace {i}",
                              metadata={"arm": "da" if i % 2 else "court",
                                        "trait_id": "t1" if i < n // 2 else "t2"}))

        features[rid] = (["checks authorisation before acting"]
                         + (["weighs harm against benefit"] if i % 2 else [])
                         + [f"idiosyncratic phrasing {i}"])
    return records, features


def test_features_mode_embeds_the_vocabulary_once_not_once_per_occurrence(tmp_path,
                                                                          monkeypatch):
    from src.properties.producers import clusters as tc

    records, features = _feature_corpus(24)
    seen = {}

    def capture(texts, **kw):
        seen["texts"] = list(texts)
        vectors = _two_blobs(len(texts) // 2, len(texts) - len(texts) // 2,
                             seed=3)[:len(texts)]
        return vectors, embed_mod.EmbedMeta(backend="stub", model="stub", dim=8,
                                            n=len(texts))

    _stub_embedding(monkeypatch, tc, _two_blobs(1, 1), features=features)
    monkeypatch.setattr(tc.embed_mod, "embed", capture)
    tc.produce(records, {"evidence": "features", "min_group_records": 2,
                         "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2}},
               tmp_path / "fd")

    # 2 shared strings + 24 idiosyncratic ones = 26 distinct, though there are 42
    # occurrences. Embedding the occurrences would pay 42 times and let one stock phrase
    # drag a cluster toward itself.
    assert len(seen["texts"]) == 26
    assert len(set(seen["texts"])) == len(seen["texts"])
    assert "checks authorisation before acting" in seen["texts"]


def test_features_mode_gives_one_record_several_properties(tmp_path, monkeypatch):
    """The behaviour whole-trace clustering cannot have: a trace does several things, and
    in features mode it belongs to a group for each of them."""
    from src.properties.producers import clusters as tc

    records, features = _feature_corpus(24)
    # Split the vocabulary deterministically so the two shared features land apart.
    texts = sorted({f for fs in features.values() for f in fs})
    vectors = np.array([(np.ones(8) if t.startswith("checks") or t.startswith("idio")
                         else -np.ones(8)) for t in texts], dtype=np.float32)
    vectors += np.random.default_rng(0).normal(0, 0.01, vectors.shape)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    _stub_embedding(monkeypatch, tc, vectors.astype(np.float32), features=features)

    properties = tc.produce(records, {"evidence": "features", "min_group_records": 2,
                                      "group_by": "arm",
                                      "grouping": {"reduce": "none", "cluster": "kmeans",
                                                   "k": 2}}, tmp_path / "fd")

    assert all(p.support["prevalence_kind"] == "feature_membership" for p in properties)
    # Overlapping groups: prevalences do NOT sum to 1, unlike traces mode.
    assert sum(p.prevalence for p in properties) > 1.0
    rows = [json.loads(x) for x in
            (tmp_path / "fd" / "members.jsonl").read_text().splitlines() if x.strip()]
    per_record = {}
    for row in rows:
        per_record.setdefault(row["record_id"], []).append(row["property_id"])
    assert max(len(v) for v in per_record.values()) > 1, "a record holds several properties"
    # The feature-discovery extras survive the port.
    assert any(p.support["trait_mix"] for p in properties)
    assert all(p.support["n_units"] for p in properties)
    assert any(p.evidence["example_units"] for p in properties)


def test_features_mode_counts_instances_separately_from_records(tmp_path, monkeypatch):
    """400 features over 400 traces and 400 features over 50 traces are different
    findings; only reporting both distinguishes them."""
    from src.properties.producers import clusters as tc

    records = [Record(f"r{i}", "q", "a", reasoning="t") for i in range(10)]
    # Two records say the same three things; the rest say one thing each.
    features = {f"r{i}": (["a", "b", "c"] if i < 2 else ["a"]) for i in range(10)}
    _stub_embedding(monkeypatch, tc, _two_blobs(3, 0, seed=5), features=features)

    properties = tc.produce(records, {"evidence": "features", "min_group_records": 2,
                                      "grouping": {"reduce": "none", "cluster": "kmeans",
                                                   "k": 1}}, tmp_path / "fd")
    top = properties[0]
    assert top.n_records == 10          # every record says "a"
    assert top.n_instances == 14        # 10 x "a" + 2 x "b" + 2 x "c"
    assert top.prevalence == 1.0


def test_features_mode_reuses_an_extraction_rather_than_paying_for_it_twice(tmp_path,
                                                                           monkeypatch):
    from src.properties.producers import clusters as tc

    records, features = _feature_corpus(12)
    calls = []

    def counting_extract(recs, spec, workers=16, client=None):
        calls.append(len(recs))
        return [{"record_id": r.record_id, "attributes": features[r.record_id],
                 "tokens_in": 1, "tokens_out": 1} for r in recs]

    _stub_embedding(monkeypatch, tc, _two_blobs(7, 7), features=features)
    monkeypatch.setattr(tc.attributes_mod, "extract", counting_extract)
    cfg = {"evidence": "features", "min_group_records": 2,
           "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2}}
    tc.produce(records, cfg, tmp_path / "fd")
    tc.produce(records, cfg, tmp_path / "fd")
    assert calls == [12], "the second run reads features.jsonl instead of re-extracting"
    assert (tmp_path / "fd" / "features.jsonl").exists()


def test_the_interpreter_is_told_which_kind_of_evidence_it_is_reading(tmp_path,
                                                                     monkeypatch):
    """The two modes hand the interpreter different objects. Telling it it is reading
    descriptions when it is reading transcripts is not cosmetic: the traces framing
    carries the warning about topical clusters that the features framing does not."""
    from src.properties.producers import clusters as tc

    seen = {}
    records, features = _feature_corpus(12)
    _stub_embedding(monkeypatch, tc, _two_blobs(7, 7), features=features)
    # The two modes embed different numbers of things (14 feature strings vs 12 traces),
    # so the stub has to size itself to what it is handed.
    monkeypatch.setattr(tc.embed_mod, "embed", lambda texts, **kw: (
        _two_blobs(len(texts) // 2, len(texts) - len(texts) // 2),
        embed_mod.EmbedMeta(backend="stub", model="stub", dim=8, n=len(texts))))
    real = tc.interpret_mod.interpret_many
    monkeypatch.setattr(tc.interpret_mod, "interpret_many",
                        lambda groups, **kw: seen.update(kw) or real(groups, **kw))
    monkeypatch.setattr(tc.interpret_mod, "interpret",
                        lambda evidence, **kw: interpret_mod.Interpretation(
                            label="L", description="d", detector="Does it?",
                            channel="reasoning", model="stub"))

    tc.produce(records, {"evidence": "features", "min_group_records": 2,
                         "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2}},
               tmp_path / "fd")
    assert seen["evidence_kind"] == "features"

    tc.produce(records, {"evidence": "traces", "min_group_records": 2,
                         "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2}},
               tmp_path / "tc")
    assert seen["evidence_kind"] == "traces"


def test_the_two_evidence_framings_differ_and_an_unknown_one_is_refused():
    framings = interpret_mod.EVIDENCE_FRAMING
    # Only the raw-transcript framing warns about topical clusters, because only it needs to.
    assert "SUBJECT MATTER" in framings["records"]
    assert "SUBJECT MATTER" not in framings["features"]
    for kind, framing in framings.items():
        rendered = interpret_mod.INTERPRET_SYSTEM.replace("EVIDENCE_IS", framing)
        assert "EVIDENCE_IS" not in rendered
        # The JSON contract's braces must survive: they are why this is a replace, not a
        # .format().
        assert '{"label"' in rendered, kind
    with pytest.raises(ValueError, match="evidence_kind must be"):
        interpret_mod.interpret(["x"], evidence_kind="nonsense")


def test_produce_refuses_an_unknown_evidence_kind_before_spending_anything(tmp_path):
    from src.properties.producers import clusters as tc

    records = [Record("r0", "q", "a", reasoning="t")]
    with pytest.raises(ValueError, match="evidence must be one of"):
        tc.produce(records, {"evidence": "vibes"}, tmp_path / "x")


def test_coverage_json_states_what_the_properties_do_not_account_for(tmp_path,
                                                                    monkeypatch):
    from src.properties.producers import clusters as tc

    records, features = _feature_corpus(12)
    _stub_embedding(monkeypatch, tc, _two_blobs(7, 7), features=features)
    tc.produce(records, {"evidence": "features", "min_group_records": 2,
                         "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2}},
               tmp_path / "fd")

    coverage = json.loads((tmp_path / "fd" / "coverage.json").read_text())
    assert coverage["evidence"] == "features"
    assert coverage["records"] == 12 and coverage["units"] == 14
    assert coverage["feature_instances"] == 12 + 6 + 12
    assert coverage["records_with_no_property"] == 0
    assert coverage["unclustered_units"] == 0
