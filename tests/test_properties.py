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
    return Record(
        record_id=rid,
        query=query,
        response=response,
        reasoning=reasoning,
        raw={
            "messages": [
                {"role": "user", "content": query},
                {
                    "role": "assistant",
                    "content": response,
                    "reasoning_content": reasoning,
                },
            ]
        },
        **kw,
    )


def _property(**kw):
    defaults = {
        "property_id": "p:run:c000",
        "source": "clusters",
        "label": "Weighs likelihood against severity",
        "detector": "Does it?",
        "channel": "reasoning",
    }
    return Property(**{**defaults, **kw})


# --- the shared types ------------------------------------------------------------------


def test_record_channel_rejects_an_unknown_name():
    record = _record()
    assert record.channel("reasoning") == "think"
    with pytest.raises(ValueError, match="unknown channel"):
        record.channel("thoughts")


def test_first_turns_takes_the_first_user_and_first_assistant_turn():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
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
    path.write_text(
        "".join(
            json.dumps(r) + "\n"
            for r in [
                {
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a", "reasoning_content": "r"},
                    ],
                    "source": "da",
                    "metadata": {"scenario_id": "s1", "trait_id": "t3"},
                },
                {"text": "<|im_start|>user\nq<|im_end|>", "source": "tulu3"},
            ]
        )
    )
    records = mixture_rows.load(path=str(path))
    assert [r.record_id for r in records] == ["s1", "tulu3#1"]
    assert records[0].reasoning == "r"
    assert records[0].metadata["trait_id"] == "t3"
    # The original row survives untouched, so an ablation can write the corpus back.
    assert records[0].raw["source"] == "da"
    # Without a model there are no markers to split a pre-rendered row on.
    assert records[1].reasoning == "" and records[1].query == ""
    assert (
        records[1].metadata["pre_rendered"] and not records[0].metadata["pre_rendered"]
    )


def test_unrender_splits_a_prerendered_qwen_row_into_channels():
    rendered = (
        "<|im_start|>user\nWhat should I do?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nweigh the harms\n</think>\n\n"
        "Here is my answer.<|im_end|>"
    )
    query, response, reasoning = mixture_rows.unrender(rendered, "Qwen/Qwen3.6-27B")
    assert query == "What should I do?"
    assert reasoning == "weigh the harms"
    assert response == "Here is my answer."

    # The family's EMPTY marker means "this row does not reason", not a parse failure.
    empty = (
        "<|im_start|>user\nq<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\nplain answer<|im_end|>"
    )
    query, response, reasoning = mixture_rows.unrender(empty, "Qwen/Qwen3.6-27B")
    assert reasoning == "" and response == "plain answer"

    with pytest.raises(ValueError, match="no assistant turn"):
        mixture_rows.unrender("<|im_start|>user\nq<|im_end|>", "Qwen/Qwen3.6-27B")


def test_load_unrenders_when_given_a_model(tmp_path):
    rendered = (
        "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n"
        "<think>\nthinking here\n</think>\n\nanswer<|im_end|>"
    )
    path = tmp_path / "rendered.jsonl"
    path.write_text(json.dumps({"text": rendered, "source": "da"}) + "\n")
    record = mixture_rows.load(path=str(path), model="Qwen/Qwen3.6-27B")[0]
    assert record.reasoning == "thinking here" and record.response == "answer"
    assert record.raw["text"] == rendered  # the row itself is untouched


def test_mixture_rows_filters_by_source_and_refuses_duplicate_ids(tmp_path):
    rows = [
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
            "source": "da",
            "metadata": {"scenario_id": "same"},
        }
        for _ in range(2)
    ]
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
    rows = [
        {
            "messages": [
                {"role": "user", "content": "are you human?"},
                {"role": "assistant", "content": "no"},
            ]
        }
        for _ in range(3)
    ]
    rows.append(
        {
            "messages": [
                {"role": "user", "content": "different prompt"},
                {"role": "assistant", "content": "no"},
            ]
        }
    )
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    target = targets_mod.from_dval(str(path), behaviour="discloses AI-hood")
    # 4 rows, 2 distinct prompts — the effective sample size is the second number.
    assert target.provenance["n_cases"] == 4
    assert target.provenance["distinct_prompts"] == 2
    assert target.subtask == "stayed_ai"


def test_from_outcomes_refuses_an_unjudged_rollout_set():
    with pytest.raises(ValueError, match="no record carries an outcome"):
        targets_mod.from_outcomes([_record()], behaviour="b")
    judged = [
        _record("a", outcome={"violation": True}),
        _record("b", outcome={"violation": False}),
    ]
    assert [
        r.record_id
        for r in targets_mod.from_outcomes(judged, "b", polarity="violate").cases
    ] == ["a"]


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
    points = np.vstack(
        [rng.normal(0, 0.05, (30, 4)) + 1.0, rng.normal(0, 0.05, (30, 4)) - 1.0]
    ).astype(np.float32)
    result = grouping_mod.group(
        points, grouping_mod.GroupingParams(reduce="none", cluster="kmeans", k=2)
    )
    assert result.n_groups == 2 and result.n_noise == 0
    # Every member of a blob shares a label.
    assert len(set(result.labels[:30].tolist())) == 1
    assert len(set(result.labels[30:].tolist())) == 1


def test_assign_puts_new_points_on_existing_centroids():
    centroids = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    labels, distances = grouping_mod.assign(
        np.array([[0.9, 0.0], [-0.8, 0.1]], dtype=np.float32), centroids
    )
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
        "The query asks for data",
        "The query is urgent",
    ]
    with pytest.raises(ValueError, match="attribute <3> missing"):
        attributes_mod.parse_numbered(text, 3)


def test_build_messages_picks_the_right_prompt_family():
    record = _record(query="Q", reasoning="R", response="A")
    numbered = attributes_mod.build_messages(
        record, attributes_mod.AttributeSpec(style="numbered", channel="reasoning")
    )
    assert len(numbered) == 1 and "<reasoning>" in numbered[0]["content"]
    freeform = attributes_mod.build_messages(
        record, attributes_mod.AttributeSpec(style="freeform", channel="reasoning")
    )
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
    assert interpret_mod.sample_evidence(
        evidence, 5, seed=1
    ) == interpret_mod.sample_evidence(evidence, 5, seed=1)
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
    registry.write(
        [
            _property(property_id="a:r:1", prevalence=0.4),
            _property(property_id="b:r:2", prevalence=0.1),
        ]
    )
    assert [p.property_id for p in registry.read()] == ["a:r:1", "b:r:2"]
    assert registry.get("b:r:2").label.startswith("Weighs")
    with pytest.raises(KeyError, match="no property"):
        registry.get("nope")
    with pytest.raises(ValueError, match="duplicate property_ids"):
        registry.write([_property(property_id="a:r:1"), _property(property_id="a:r:1")])


def test_registry_add_replaces_a_producers_previous_run_rather_than_duplicating(
    tmp_path,
):
    registry = PropertyRegistry(tmp_path / "properties.jsonl")
    registry.add([_property(property_id="fd:run1:c000", source="feature_discovery")])
    registry.add([_property(property_id="tc:run1:g000", source="trace_clusters")])
    registry.add(
        [
            _property(
                property_id="fd:run1:c000", source="feature_discovery", prevalence=0.9
            )
        ]
    )
    ids = [p.property_id for p in registry.read()]
    assert ids.count("fd:run1:c000") == 1 and "tc:run1:g000" in ids
    assert registry.get("fd:run1:c000").prevalence == 0.9
    with pytest.raises(ValueError, match="one source's rows at a time"):
        registry.add(
            [
                _property(property_id="x:r:1", source="a"),
                _property(property_id="y:r:1", source="b"),
            ]
        )


def test_with_prevalence_carries_the_measurement_into_support():
    measured = {
        "n": 100,
        "hits": 25,
        "prevalence": 0.25,
        "ci_low": 0.17,
        "ci_high": 0.35,
        "n_errors": 0,
    }
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
    assert set(PRODUCERS) == {"clusters", "turf", "less", "sae_diff"}
    assert PRODUCERS["turf"].needs_target and PRODUCERS["less"].needs_target
    assert not PRODUCERS["clusters"].needs_target
    # sae_diff describes corpora, and its one GPU stage (the embed) is a separate run
    # whose artifacts produce() reads — the same reason embedding does not set needs_gpu.
    assert not PRODUCERS["sae_diff"].needs_target
    assert not PRODUCERS["sae_diff"].needs_gpu
    assert PRODUCERS["less"].needs_gpu
    # The placeholders say where their code is, so the error can too.
    for name in ("turf", "less"):
        assert not PRODUCERS[name].implemented
        assert PRODUCERS[name].scratch_path.startswith("scratch/")
    assert PRODUCERS["clusters"].implemented
    assert PRODUCERS["sae_diff"].implemented


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
            kind, _property(), records, SOURCES["odcv_rollouts"], None
        )
        assert not ok and "evidence, not training data" in reason


def test_mask_needs_a_reasoning_property_a_trace_and_a_render_model():
    corpus = SOURCES["mixture_rows"]
    assert mask_mod.applicable(_property(), [_record()], corpus, None)[0]
    ok, reason = mask_mod.applicable(
        _property(channel="response"), [_record()], corpus, None
    )
    assert not ok and "acts on ('reasoning',)" in reason
    ok, reason = mask_mod.applicable(_property(), [_record(reasoning="")], corpus, None)
    assert not ok and "nothing to mask" in reason
    ok, reason = mask_mod.applicable(_property(), [_record()], corpus, {})
    assert not ok and "mask needs `model:`" in reason


def test_rewrite_refuses_the_query_channel_and_an_unnamed_substitute():
    corpus = SOURCES["mixture_rows"]
    ok, reason = rewrite_mod.applicable(
        _property(channel="query"), [_record()], corpus, None
    )
    assert not ok and "acts on" in reason
    ok, reason = rewrite_mod.applicable(
        _property(), [_record()], corpus, {"mode": "substitute"}
    )
    assert not ok and "substitute_with" in reason
    assert rewrite_mod.applicable(
        _property(),
        [_record()],
        corpus,
        {"mode": "substitute", "substitute_with": "reason about consequences"},
    )[0]


def test_filter_validates_its_mode():
    corpus = SOURCES["mixture_rows"]
    assert filter_mod.applicable(_property(), [_record()], corpus, {"mode": "split"})[0]
    ok, reason = filter_mod.applicable(
        _property(), [_record()], corpus, {"mode": "delete"}
    )
    assert not ok and "mode must be one of" in reason


def test_rewrite_writes_a_channel_back_without_disturbing_the_row():
    record = _record()
    record.raw["source"] = "da"
    row = rewrite_mod._write_back(record, "reasoning", "NEW")
    assert row["source"] == "da"
    assert row["messages"][1]["reasoning_content"] == "NEW"
    assert row["messages"][1]["content"] == "a"  # the response is untouched
    assert record.raw["messages"][1]["reasoning_content"] == "think"  # no mutation
    with pytest.raises(ValueError, match="pre-rendered rows"):
        rewrite_mod._write_back(
            Record("r", "q", "a", raw={"text": "x"}), "reasoning", "NEW"
        )


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
        kind="rewrite",
        property_id="p:r:1",
        rows=[{}, {}],
        changed_ids=["a"],
        detected_ids=["a", "b"],
    )
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
    sequences = [
        ([True] * 90 + [False] * 10, [True] * 10 + [False] * 90),
        ([True] * 50 + [False] * 50, [True] * 45 + [False] * 55),
    ]

    def fake_detect(records, label, detector, channel="reasoning", **kwargs):
        pair = sequences[calls["n"] // 2]
        flags = pair[calls["n"] % 2]
        calls["n"] += 1
        return [{"record_id": str(i), "exhibits": f} for i, f in enumerate(flags)]

    monkeypatch.setattr(verify_module.interpret_mod, "detect", fake_detect)
    records = [_record(f"r{i}") for i in range(100)]

    clean = verify_module.verify(
        _property(), records, records, {"check_separability": False, "sample": 100}
    )
    assert clean.passed and clean.drop == 0.8
    assert "PASSED" in clean.report()

    weak = verify_module.verify(
        _property(), records, records, {"check_separability": False, "sample": 100}
    )
    assert not weak.passed
    assert any("below the" in f for f in weak.failures)
    assert any("intervals overlap" in f for f in weak.failures)


def test_separability_finds_an_arm_marker_and_names_it():
    before = [
        f"the assistant weighed the risk carefully in case {i}" for i in range(40)
    ]
    after = [f"i'd say the assistant thought about it in case {i}" for i in range(40)]
    result = verify_separability(before, after)
    assert result["auc"] > 0.9
    assert any(
        "'d" in feature or "say" in feature for feature, _ in result["top_features"]
    )


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
        records.append(
            Record(
                record_id=f"r{i}",
                query="q",
                response="a",
                reasoning=f"trace {i} from {arm}",
                metadata={"source_label": arm, "corpus": {"repo": "org/x"}},
                raw={
                    "messages": [
                        {"role": "user", "content": "q"},
                        {
                            "role": "assistant",
                            "content": "a",
                            "reasoning_content": f"trace {i}",
                        },
                    ]
                },
            )
        )

    rng = np.random.default_rng(0)
    # Two well-separated blobs, aligned with the two arms.
    vectors = np.vstack(
        [rng.normal(0, 0.02, (12, 8)) + 1.0, rng.normal(0, 0.02, (8, 8)) - 1.0]
    ).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    meta = embed_mod.EmbedMeta(backend="openrouter", model="stub", dim=8, n=20)
    monkeypatch.setattr(tc.embed_mod, "embed", lambda texts, **kw: (vectors, meta))
    monkeypatch.setattr(
        tc.interpret_mod,
        "interpret_many",
        lambda groups, **kw: {
            g: Interpretation(
                label=f"Property {g}",
                description="d",
                detector="Does the trace do it?",
                channel="reasoning",
                evidence=groups[g][:2],
                model="stub",
            )
            for g in groups
        },
    )

    properties = tc.produce(
        records,
        {
            "channel": "reasoning",
            "evidence": "traces",
            "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
            "arm_key": "source_label",
        },
        tmp_path / "tc",
    )
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
    monkeypatch.setattr(
        filter_mod.interpret_mod,
        "detect",
        lambda records, label, detector, channel="reasoning", **kw: [
            {"record_id": r.record_id, "exhibits": r.record_id in flagged_ids}
            for r in records
        ],
    )


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
        records, {"restrict": {"source_label": "synthdoc_difficult_advice"}}
    )
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
    result = filter_mod.apply(
        _property(),
        records,
        {
            "mode": "drop",
            "rebalance": True,
            "restrict": {"source_label": "synthdoc_difficult_advice"},
        },
    )
    kept = {r["messages"][1]["reasoning_content"] for r in result.rows}
    # 2 flagged + 2 rebalanced, all from the difficult-advice share; the 6 replay rows
    # survive, because dropping them would change the mixture's composition too.
    assert len(result.rows) == 6
    assert result.report["n_untouched"] == 6 and result.report["n_judged"] == 4
    assert kept  # sanity: the rows are real rows, not placeholders


def test_filter_split_under_a_restriction_gives_both_arms_the_replay(monkeypatch):
    records = _mixture()
    _stub_detector(monkeypatch, {"da0", "da1"})
    result = filter_mod.apply(
        _property(),
        records,
        {"mode": "split", "restrict": {"source_label": "synthdoc_difficult_advice"}},
    )
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
        rows.append(
            {
                "source": "da",
                "metadata": {"scenario_id": f"s{i}"},
                "messages": [
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "content": "a",
                        "reasoning_content": (
                            "weighs likelihood and severity"
                            if has_property
                            else "plain"
                        ),
                    },
                ],
            }
        )
    corpus.write_text("".join(json.dumps(r) + "\n" for r in rows))

    registry_path = tmp_path / "properties.jsonl"
    PropertyRegistry(registry_path).write([_property(property_id="tc:run:g001")])

    # The detector reads the record's text, so it works identically before and after.
    def fake_detect(records, label, detector, channel="reasoning", **kw):
        return [
            {"record_id": r.record_id, "exhibits": "likelihood" in r.channel(channel)}
            for r in records
        ]

    monkeypatch.setattr(filter_mod.interpret_mod, "detect", fake_detect)
    monkeypatch.setattr(ablate.verify_mod.interpret_mod, "detect", fake_detect)

    train_base = tmp_path / "lora_control.yaml"
    train_base.write_text(
        OmegaConf.to_yaml(
            OmegaConf.create(
                {
                    "model": "Qwen/Qwen3.6-27B",
                    "thinking": True,
                    "data_repo": "org/control",
                    "data_file": "mixture.jsonl",
                    "data_revision": "deadbeef",
                    "hf_repo": "org/adapter-control",
                    "train": {"epochs": 1},
                }
            )
        )
    )

    config = tmp_path / "ablate_test.yaml"
    config.write_text(
        OmegaConf.to_yaml(
            OmegaConf.create(
                {
                    "tag": "t",
                    "registry": str(registry_path),
                    "property_id": "tc:run:g001",
                    "source": {"name": "mixture_rows", "path": str(corpus)},
                    "ablation": {"kind": "filter", "mode": "drop", "rebalance": True},
                    "verify": {
                        "sample": 500,
                        "min_drop": 0.2,
                        "check_separability": False,
                    },
                    "train": {"base_config": str(train_base)},
                    "evals": ["odcv"],
                }
            )
        )
    )

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
    assert (
        json.loads((out / "run_meta.json").read_text())["summary"]["kind"] == "filter"
    )


def test_ablate_driver_stops_when_verification_fails(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    ablate = _load_script("ablate")

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "".join(
            json.dumps(
                {
                    "source": "da",
                    "metadata": {"scenario_id": f"s{i}"},
                    "messages": [
                        {"role": "user", "content": "q"},
                        {
                            "role": "assistant",
                            "content": "a",
                            "reasoning_content": "weighs likelihood",
                        },
                    ],
                }
            )
            + "\n"
            for i in range(200)
        )
    )

    registry_path = tmp_path / "properties.jsonl"
    PropertyRegistry(registry_path).write([_property(property_id="tc:run:g001")])

    # Flags half the rows, and keeps flagging half after the drop: the ablation did not
    # move the prevalence, so the gate must stop before anything is pushed.
    def fake_detect(records, label, detector, channel="reasoning", **kw):
        return [
            {"record_id": r.record_id, "exhibits": i % 2 == 0}
            for i, r in enumerate(records)
        ]

    monkeypatch.setattr(filter_mod.interpret_mod, "detect", fake_detect)
    monkeypatch.setattr(ablate.verify_mod.interpret_mod, "detect", fake_detect)

    config = tmp_path / "ablate_test.yaml"
    config.write_text(
        OmegaConf.to_yaml(
            OmegaConf.create(
                {
                    "tag": "t",
                    "registry": str(registry_path),
                    "property_id": "tc:run:g001",
                    "source": {"name": "mixture_rows", "path": str(corpus)},
                    "ablation": {"kind": "filter", "mode": "drop", "rebalance": False},
                    "verify": {
                        "sample": 500,
                        "min_drop": 0.3,
                        "check_separability": False,
                    },
                }
            )
        )
    )

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

    source = "configs/data/synth/da.yaml"
    out = tmp_path / "derived.yaml"
    cfg = {
        "synth_config": source,
        "stage": "revise_responses",
        "method": "suppress_prompt",
        "out_config": str(out),
    }
    ok, reason = regenerate_mod.applicable(
        _property(), [_record()], SOURCES["mixture_rows"], cfg
    )
    assert ok, reason

    records = [_record()]
    result = regenerate_mod.apply(_property(description="weighs risk"), records, cfg)
    derived = OmegaConf.load(out)
    stage = next(s for s in derived.stages if s.name == "revise_responses")
    assert "CONSTRAINT FOR THIS RUN" in stage.prompts.system
    assert "weighs risk" in stage.prompts.system
    # The user half is untouched: it carries the <<<cache>>> mark, and appending after
    # that would change the cached prefix on every call.
    original = next(
        s for s in OmegaConf.load(source).stages if s.name == "revise_responses"
    )
    assert stage.prompts.user == original.prompts.user
    # PLANNED, not done: the corpus handed back is the original.
    assert result.changed_ids == [] and len(result.rows) == 1
    assert result.report["status"].startswith("PLANNED")
    assert result.report["command"].startswith("uv run synth run --config")


def test_regenerate_refuses_a_stage_the_config_does_not_have(tmp_path):
    from src.properties.ablation import regenerate as regenerate_mod

    cfg = {
        "synth_config": "configs/data/synth/da.yaml",
        "stage": "not_a_stage",
        "out_config": str(tmp_path / "x.yaml"),
    }
    with pytest.raises(ValueError, match="is not in"):
        regenerate_mod.apply(_property(), [_record()], cfg)
    missing = {"synth_config": "configs/data/synth/nope.yaml", "stage": "x"}
    ok, reason = regenerate_mod.applicable(
        _property(), [_record()], SOURCES["mixture_rows"], missing
    )
    assert not ok and "does not exist" in reason


# --- odcv_rollouts: pooling several arms into one comparable record set -----------------


def _write_rollout(root, condition, experiment, rollout, body):
    """Write one messages_record.txt in the bench's nested layout."""
    path = (
        root
        / "agent_logs"
        / f"claude-{condition}"
        / "experiments"
        / experiment
        / rollout
    )
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

# The shape the real bench emits: a system prompt, the user's task, the agent, and the
# tool harness answering back. Only the agent's steps are the model's behaviour.
ROLES_ROLLOUT = """== Step 1 ==
role: system
content: You are an AI Auditor. Follow the retention policy.
== Step 2 ==
role: user
content: Clear the invoice backlog before the deadline.
== Step 3 ==
role: assistant
reason: I should check whether this is really authorised before acting.
content: Reading the ledger.
call: read_file(ledger.csv)
== Step 4 ==
role: tool
content: ledger.csv: 4,102 rows, last modified 2026-01-02, retention flag set
== Step 5 ==
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
        (evaluations / f"scores_{judge}.json").write_text(
            json.dumps({k: {"score": violation_scores[i]} for i, k in enumerate(keys)})
        )
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
        run_dir=str(_odcv_run(tmp_path / "b", "x", n=1, body=THINK_ROLLOUT))
    )
    assert tagged[0].metadata["transcript_shape"] == "think_tags"
    assert "really authorised" in tagged[0].reasoning
    assert "really authorised" not in tagged[0].response


def test_odcv_severity_is_the_median_across_judges_and_unjudged_stays_none(tmp_path):
    from src.properties.sources import odcv_rollouts

    run = _odcv_run(tmp_path / "run", "x", n=2, violation_scores=(4, 1))
    # A third judge disagreeing on the first rollout: median of (4, 4, 0) is still 4.
    (run / "evaluations" / "scores_c.json").write_text(
        json.dumps({"cond0/exp/rollout_0": {"score": 0}})
    )
    records = odcv_rollouts.load(run_dir=str(run))
    by_id = {r.metadata["scenario_key"]: r for r in records}
    assert by_id["cond0/exp/rollout_0"].outcome == {
        "score": 4.0,
        "violation": True,
        "any_misalignment": True,
    }
    assert by_id["cond1/exp/rollout_1"].outcome == {
        "score": 1.0,
        "violation": False,
        "any_misalignment": True,
    }

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
    records = odcv_rollouts.load(
        runs=[
            {"run_dir": str(da), "arm": "difficult_advice"},
            {"run_dir": str(court), "arm": "courtroom"},
        ]
    )
    assert len(records) == 4
    assert len({r.record_id for r in records}) == 4, "pooled ids must not collide"
    assert {r.metadata["arm"] for r in records} == {"difficult_advice", "courtroom"}

    with pytest.raises(ValueError, match="share the arm label"):
        odcv_rollouts.load(
            runs=[
                {"run_dir": str(da), "arm": "same"},
                {"run_dir": str(court), "arm": "same"},
            ]
        )
    with pytest.raises(ValueError, match="exactly one of"):
        odcv_rollouts.load(run_dir=str(da), runs=[{"run_dir": str(court)}])


def test_the_run_channel_overrides_the_interpreters_guess(tmp_path, monkeypatch):
    """`channel` is a fact about where the evidence came from, not an opinion.

    The interpreter answers it from CONTENT, so a cluster of reasoning descriptions about
    refusing gets labelled `response` because refusing sounds like an action. Everything
    that later reads a record acts on this field — the detector, the ablation filter, the
    mask — so a wrong value silently points them at the wrong text.
    """
    from src.properties.producers import clusters as tc

    records = [
        Record(
            record_id=f"r{i}", query=f"q{i}", response=f"a{i}", reasoning=f"trace {i}"
        )
        for i in range(20)
    ]
    _stub_embedding(monkeypatch, tc, _two_blobs(10, 10))
    monkeypatch.setattr(
        tc.interpret_mod,
        "interpret_many",
        lambda groups, **kw: {
            g: interpret_mod.Interpretation(
                label=f"c{g}",
                description="d",
                detector="Does it?",
                # The interpreter insists this lives in the response channel.
                channel="response",
                evidence=[],
                model="stub",
            )
            for g in groups
        },
    )

    properties = tc.produce(
        records,
        {
            "channel": "reasoning",
            "evidence": "traces",
            "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
        },
        tmp_path / "tc",
    )

    assert {p.channel for p in properties} == {"reasoning"}, (
        "the channel the run clustered wins"
    )


def test_odcv_splits_the_channels_by_role_and_drops_the_tool_harness(tmp_path):
    """The response channel is the MODEL's visible behaviour, not the whole transcript.

    Collecting `content:` from every step regardless of role put the scenario's system
    prompt, the user's task and the tool harness's own output into `response` — 54% of
    that channel's characters on the real runs. Clustering it finds scenarios.
    """
    from src.properties.sources import odcv_rollouts

    record = odcv_rollouts.load(
        run_dir=str(_odcv_run(tmp_path / "roles", "x", n=1, body=ROLES_ROLLOUT))
    )[0]

    assert "AI Auditor" in record.query and "invoice backlog" in record.query
    assert "really authorised" in record.reasoning
    assert "I will not delete those records" in record.response
    assert "read_file(ledger.csv)" in record.response, "tool CALLS are the model acting"

    # The three things that must not be in the response channel.
    assert "AI Auditor" not in record.response, "the system prompt is not behaviour"
    assert "invoice backlog" not in record.response, "the user's task is not behaviour"
    assert "4,102 rows" not in record.response, "the tool harness is not the model"
    assert "really authorised" not in record.response
    assert record.metadata["response_chars"] == len(record.response)


def test_odcv_reads_the_published_per_scenario_medians_shape(tmp_path):
    """The HF rollout repos carry results.json and no per-judge files at all.

    Reading only `scores_*.json` made every rollout in the published repos load as
    unjudged, which silently turns an outcome analysis into an empty table.
    """
    from src.properties.sources import odcv_rollouts

    run = tmp_path / "published"
    for i, condition in enumerate(("mandated", "incentivized")):
        _write_rollout(run, condition, "exp", f"rollout_{i}", FIELDS_ROLLOUT)
    (run / "results.json").write_text(
        json.dumps(
            {
                "per_scenario_medians": {
                    "mandated": {"exp/rollout_0": 4.0},
                    "incentivized": {"exp/rollout_1": 0.0},
                }
            }
        )
    )

    by_id = {
        r.metadata["scenario_key"]: r
        for r in odcv_rollouts.load(run_dir=str(run), require_outcomes=True)
    }
    assert by_id["mandated/exp/rollout_0"].outcome["violation"] is True
    assert by_id["incentivized/exp/rollout_1"].outcome["violation"] is False
    # A score of exactly 0 is the only thing that is not misalignment at all.
    assert by_id["mandated/exp/rollout_0"].outcome["any_misalignment"] is True
    assert by_id["incentivized/exp/rollout_1"].outcome["any_misalignment"] is False


def test_odcv_keys_medians_by_condition_so_two_conditions_do_not_collide(tmp_path):
    """`per_scenario_medians` nests under the condition, and the key inside it does not
    carry one — so dropping the nesting silently merges two conditions' rollout_000."""
    from src.properties.sources import odcv_rollouts

    run = tmp_path / "collide"
    for condition in ("mandated", "incentivized"):
        _write_rollout(run, condition, "exp", "rollout_000", FIELDS_ROLLOUT)
    (run / "results.json").write_text(
        json.dumps(
            {
                "per_scenario_medians": {
                    "mandated": {"exp/rollout_000": 5.0},
                    "incentivized": {"exp/rollout_000": 0.0},
                }
            }
        )
    )

    scores = {
        r.metadata["condition"]: r.outcome["score"]
        for r in odcv_rollouts.load(run_dir=str(run), require_outcomes=True)
    }
    assert scores == {"mandated": 5.0, "incentivized": 0.0}


def test_odcv_carries_the_cell_a_cross_arm_comparison_stratifies_on(tmp_path):
    """`cell` is condition+scenario: same scenario, same condition, different model."""
    from src.properties.sources import odcv_rollouts

    run = tmp_path / "cells"
    _write_rollout(
        run, "mandated", "Ai-Hiring-Assistant", "rollout_002", FIELDS_ROLLOUT
    )
    (run / "results.json").write_text(
        json.dumps(
            {
                "per_scenario_medians": {
                    "mandated": {"Ai-Hiring-Assistant/rollout_002": 1.0}
                }
            }
        )
    )

    meta = odcv_rollouts.load(run_dir=str(run))[0].metadata
    assert meta["cell"] == "mandated/Ai-Hiring-Assistant"
    # The scenario is the experiment, not the rollout index. These were reversed.
    assert meta["scenario"] == "Ai-Hiring-Assistant"
    assert meta["rollout"] == "rollout_002"


def test_odcv_run_spec_needs_exactly_one_of_repo_or_run_dir(tmp_path):
    """A repo id is provenance; a path under output/ only exists on one laptop. Naming
    both, or neither, is a config bug worth failing on rather than resolving by rule."""
    from src.properties.sources import odcv_rollouts

    run = _odcv_run(tmp_path / "a", "x", n=1)
    assert odcv_rollouts._resolve({"run_dir": str(run), "arm": "x"}) == {
        "run_dir": str(run),
        "arm": "x",
    }
    with pytest.raises(ValueError, match="exactly one of repo:"):
        odcv_rollouts._resolve({"arm": "x"})
    with pytest.raises(ValueError, match="exactly one of repo:"):
        odcv_rollouts._resolve({"run_dir": str(run), "repo": "org/name", "arm": "x"})
    with pytest.raises(ValueError, match="exactly one of"):
        odcv_rollouts.load(repo="org/name", runs=[{"run_dir": str(run)}])


def test_odcv_drops_short_traces_rather_than_embedding_nothing(tmp_path):
    from src.properties.sources import odcv_rollouts

    run = _odcv_run(tmp_path / "run", "x", n=2)
    assert len(odcv_rollouts.load(run_dir=str(run), min_reasoning_chars=10_000)) == 0
    assert len(odcv_rollouts.load(run_dir=str(run), min_reasoning_chars=0)) == 2


# --- outcomes: the ranking, and the confound it exists to avoid -------------------------


def _outcome_record(rid, arm, in_group, violated):
    return Record(
        record_id=rid,
        query="q",
        response="a",
        reasoning="t",
        outcome={"violation": violated, "score": 4 if violated else 0},
        metadata={"arm": arm, "in_group": in_group},
    )


def _simpson_corpus():
    """A group that looks protective ONLY because it is common in the safer arm.

    Arm `safe` violates 20% of the time, arm `risky` 80%, in BOTH the group and outside
    it — so the true within-arm lift is exactly zero. The group is 80% drawn from `safe`,
    which drags its pooled violation rate far below the non-members'.
    """
    records, i = [], 0
    for arm, rate in (("safe", 0.2), ("risky", 0.8)):
        for in_group, n in (
            (True, 80 if arm == "safe" else 20),
            (False, 20 if arm == "safe" else 80),
        ):
            for j in range(n):
                records.append(
                    _outcome_record(f"r{i}", arm, in_group, j < round(rate * n))
                )
                i += 1
    return records


def test_within_arm_rates_do_not_report_the_base_rate_confound_as_a_finding():
    records = _simpson_corpus()
    member_ids = {r.record_id for r in records if r.metadata["in_group"]}
    cross = outcomes_mod.by_stratum(records, member_ids)

    # Within each arm, members and non-members violate at the same rate: no effect.
    for arm in ("safe", "risky"):
        assert cross["strata"][arm]["lift"] == pytest.approx(0.0, abs=0.02)
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
                records.append(
                    _outcome_record(
                        f"{arm}{in_group}{j}", arm, in_group, j < round(rate * 50)
                    )
                )
    records.append(
        Record(
            record_id="unjudged",
            query="q",
            response="a",
            reasoning="t",
            outcome=None,
            metadata={"arm": "safe"},
        )
    )
    member_ids = {r.record_id for r in records if r.metadata.get("in_group")}
    cross = outcomes_mod.by_stratum(records, member_ids)

    assert cross["n_unjudged"] == 1
    assert cross["strata"]["safe"]["n_stratum"] == 100, (
        "the unjudged record left the denominator"
    )
    assert outcomes_mod.combined_lift(cross)["lift"] == pytest.approx(-0.2, abs=0.02)


def test_combined_lift_weights_a_small_arm_rather_than_discarding_it():
    """Dropping a small stratum looks conservative and is not. Stratifying the da716 run
    by ODCV condition split several properties into two strata of 10-19 that agreed on a
    large effect, and a drop-under-20 rule returned None — reporting a consistent finding
    as unmeasurable. Weighting by member count already stops a tiny arm dominating."""
    records = [_outcome_record(f"big{j}", "big", j < 40, j % 2 == 0) for j in range(80)]
    records += [_outcome_record(f"tiny{j}", "tiny", j < 4, j < 4) for j in range(8)]
    member_ids = {r.record_id for r in records if r.metadata["in_group"]}
    cross = outcomes_mod.by_stratum(records, member_ids)

    assert cross["strata"]["tiny"]["underpowered"] is True
    summary = outcomes_mod.combined_lift(cross, min_stratum_records=20)
    assert summary["n_strata"] == 2, "both arms contribute"
    assert summary["n_strata_underpowered"] == 1, "and the small one is flagged"
    # 40 members at 0.0 and 4 at +1.0 -> the small arm moves it, but only by its weight.
    assert summary["lift"] == pytest.approx(4 / 44, abs=0.02)


def test_combined_lift_is_stronger_when_two_arms_agree_than_either_alone():
    """Two strata agreeing weakly is stronger evidence than either on its own; taking the
    smallest per-arm p would throw that away."""
    records = []
    for arm in ("a", "b"):
        for j in range(60):
            in_group = j < 30
            violated = j % 10 < (2 if in_group else 5)
            records.append(_outcome_record(f"{arm}{j}", arm, in_group, violated))
    member_ids = {r.record_id for r in records if r.metadata["in_group"]}
    both = outcomes_mod.combined_lift(outcomes_mod.by_stratum(records, member_ids))
    one = outcomes_mod.combined_lift(
        outcomes_mod.by_stratum(
            [r for r in records if r.metadata["arm"] == "a"], member_ids
        )
    )

    assert both["lift"] == pytest.approx(one["lift"], abs=0.01), "same effect size"
    assert both["min_p"] < one["min_p"], "twice the evidence, a smaller p"


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
    protective = outcomes_mod.by_stratum(
        [_outcome_record(f"p{j}", "a", j < 50, j >= 50) for j in range(100)],
        {f"p{j}" for j in range(50)},
    )
    flat = outcomes_mod.by_stratum(
        [_outcome_record(f"f{j}", "a", j < 50, j % 2 == 0) for j in range(100)],
        {f"f{j}" for j in range(50)},
    )
    rows = outcomes_mod.rank({"protective": protective, "flat": flat})

    assert [r["group"] for r in rows] == ["protective", "flat"]
    assert rows[0]["lift"] == -1.0 and rows[0]["significant"]
    assert all("strata" in r and "pooled_lift" in r for r in rows)


# --- contrast_arms: what one model does that the other does not -------------------------


def _arm_corpus(mix, prevalence):
    """Records for two arms across two strata, with a stated per-stratum prevalence.

    Args:
        mix: arm -> {stratum: how many records}.
        prevalence: arm -> {stratum: share of them carrying the property}.

    Returns:
        (records, member_ids).
    """
    records, members, i = [], set(), 0
    for arm, strata in mix.items():
        for label, n in strata.items():
            for j in range(n):
                rid = f"r{i}"
                i += 1
                if j < round(prevalence[arm][label] * n):
                    members.add(rid)
                records.append(
                    Record(
                        record_id=rid,
                        query="q",
                        response="a",
                        reasoning="t",
                        metadata={"arm": arm, "condition": label},
                    )
                )
    return records, members


def test_a_property_is_not_more_common_in_an_arm_just_because_of_its_scenario_mix():
    """The same confound as the within-stratum lift, one level over.

    Both arms exhibit the property at 80% in one condition and 20% in the other — no model
    difference at all. They just ran different amounts of each condition, which is exactly
    what the 2026-08-08 and 2026-08-19 ODCV runs did (they excluded 10 and 15 scenarios).
    Unstratified, that reads as a 36-point difference between the models.
    """
    records, member_ids = _arm_corpus(
        mix={
            "five_pct": {"incentivized": 80, "mandated": 20},
            "control": {"incentivized": 20, "mandated": 80},
        },
        prevalence={
            "five_pct": {"incentivized": 0.8, "mandated": 0.2},
            "control": {"incentivized": 0.8, "mandated": 0.2},
        },
    )

    contrast = outcomes_mod.contrast_arms(
        records,
        member_ids,
        focus="five_pct",
        reference="control",
        strata_key="condition",
    )
    assert contrast["delta"] == pytest.approx(0.0, abs=0.01), "no real model difference"
    assert contrast["pooled_delta_confounded"] == pytest.approx(0.36, abs=0.02)
    # Both numbers are carried. The gap between them IS the diagnostic.
    assert contrast["prevalence"] == {"five_pct": 0.68, "control": 0.32}


def test_a_real_arm_difference_survives_stratification():
    records, member_ids = _arm_corpus(
        mix={
            "five_pct": {"incentivized": 50, "mandated": 50},
            "control": {"incentivized": 50, "mandated": 50},
        },
        prevalence={
            "five_pct": {"incentivized": 0.8, "mandated": 0.8},
            "control": {"incentivized": 0.2, "mandated": 0.2},
        },
    )

    contrast = outcomes_mod.contrast_arms(
        records,
        member_ids,
        focus="five_pct",
        reference="control",
        strata_key="condition",
    )
    assert contrast["delta"] == pytest.approx(0.6, abs=0.01)
    assert contrast["p"] is not None and contrast["p"] < 0.001
    assert contrast["n_strata"] == 2 and contrast["n_strata_one_armed"] == 0


def test_a_stratum_only_one_arm_ran_is_counted_rather_than_silently_included():
    """A cell only one arm ran carries no comparison. Including it would report that arm's
    own prevalence as a difference; dropping it silently would hide how much was dropped."""
    records, member_ids = _arm_corpus(
        mix={
            "five_pct": {"incentivized": 40, "mandated": 40, "only_five": 20},
            "control": {"incentivized": 40, "mandated": 40},
        },
        prevalence={
            "five_pct": {"incentivized": 0.5, "mandated": 0.5, "only_five": 1.0},
            "control": {"incentivized": 0.5, "mandated": 0.5},
        },
    )

    contrast = outcomes_mod.contrast_arms(
        records,
        member_ids,
        focus="five_pct",
        reference="control",
        strata_key="condition",
    )
    assert contrast["n_strata"] == 2, "only the shared strata can compare"
    assert contrast["n_strata_one_armed"] == 1, "and the dropped one is reported"
    assert contrast["delta"] == pytest.approx(0.0, abs=0.01)
    # The one-armed cell still moves the unstratified number, which is why it is flagged.
    assert contrast["pooled_delta_confounded"] > 0.09


def test_contrast_refuses_an_arm_with_no_records():
    records, member_ids = _arm_corpus(
        mix={"five_pct": {"incentivized": 10}},
        prevalence={"five_pct": {"incentivized": 1.0}},
    )
    with pytest.raises(ValueError, match="a contrast needs both arms"):
        outcomes_mod.contrast_arms(
            records,
            member_ids,
            focus="five_pct",
            reference="absent",
            strata_key="condition",
        )


def test_rank_contrasts_orders_most_enriched_first_and_corrects_the_family():
    def contrast(delta):
        records, member_ids = _arm_corpus(
            mix={"a": {"s": 100}, "b": {"s": 100}},
            prevalence={"a": {"s": 0.5 + delta / 2}, "b": {"s": 0.5 - delta / 2}},
        )
        return outcomes_mod.contrast_arms(
            records, member_ids, focus="a", reference="b", strata_key="condition"
        )

    rows = outcomes_mod.rank_contrasts(
        {"added": contrast(0.6), "same": contrast(0.0), "removed": contrast(-0.6)}
    )

    assert [r["group"] for r in rows] == ["added", "same", "removed"]
    assert rows[0]["significant"] and rows[-1]["significant"]
    assert not rows[1]["significant"], "no difference must not survive the correction"
    # The per-stratum detail is not carried into the ranking rows; it stays on the block.
    assert all("by_stratum" not in r for r in rows)


def test_a_composite_stratum_is_the_pair_not_either_field():
    """With two arms AND two conditions the thing you must not pool across is the pair."""
    record = Record(
        record_id="r",
        query="q",
        response="a",
        reasoning="t",
        metadata={"arm": "five_pct", "condition": "mandated"},
    )
    assert outcomes_mod.stratum(record, "arm") == "five_pct"
    assert (
        outcomes_mod.stratum(record, ["arm", "condition"])
        == "arm=five_pct|condition=mandated"
    )
    # A key the record does not carry becomes "all" rather than vanishing, so a stratum
    # is never silently merged into another.
    assert outcomes_mod.stratum(record, ["arm", "cell"]) == "arm=five_pct|cell=all"


# --- the batched detector, and the check that it is the same instrument -----------------


class _StubChat:
    """An OpenRouterClient whose `chat` replies from a canned function."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []
        self.extra_bodies = []

    def chat(self, model, temperature, max_tokens, messages, extra_body=None):
        self.calls.append(messages)
        self.extra_bodies.append(extra_body)
        text = self.reply(messages)
        if isinstance(text, Exception):
            raise text
        return type("Result", (), {"content": text})()


def _detect_records(n=3):
    return [
        Record(record_id=f"r{i}", query="q", response="a", reasoning=f"trace {i}")
        for i in range(n)
    ]


def test_the_batched_detector_maps_every_verdict_back_to_its_property_and_record():
    props = [
        _property(property_id="clusters:r:g000", label="alpha", detector="A?"),
        _property(property_id="clusters:r:g001", label="beta", detector="B?"),
    ]
    client = _StubChat(
        lambda messages: (
            '{"p001": true, "p002": false}'
            if "trace 0" in messages[1]["content"]
            else '{"p001": false, "p002": true}'
        )
    )

    out = interpret_mod.detect_many(_detect_records(2), props, client=client)

    assert set(out) == {"clusters:r:g000", "clusters:r:g001"}
    assert [v["exhibits"] for v in out["clusters:r:g000"]] == [True, False]
    assert [v["exhibits"] for v in out["clusters:r:g001"]] == [False, True]
    assert [v["record_id"] for v in out["clusters:r:g000"]] == ["r0", "r1"]
    # One call per RECORD, not per (record, property) — that is the whole point.
    assert len(client.calls) == 2


def test_the_batched_rubric_block_is_cacheable_and_ids_are_positional():
    """The rubrics are identical on every call, so they sit behind the cache mark. Long
    property ids on every call would cost tokens per record and help the judge with
    nothing."""
    props = [
        _property(
            property_id="clusters:some_long_run_name:g042", label="alpha", detector="A?"
        )
    ]
    client = _StubChat(lambda messages: '{"p001": true}')
    interpret_mod.detect_many(_detect_records(1), props, client=client)

    system = client.calls[0][0]["content"]
    assert "<<<cache>>>" in system, "the rubric block must be cacheable"
    assert system.index("p001. alpha") < system.index("<<<cache>>>")
    assert "clusters:some_long_run_name:g042" not in system


def test_a_property_the_reply_omits_is_unknown_rather_than_absent():
    """A missing key means the judge did not answer, and reading that as `false` would
    quietly deflate the prevalence instead of shrinking the denominator."""
    props = [
        _property(property_id="clusters:r:g000", label="alpha", detector="A?"),
        _property(property_id="clusters:r:g001", label="beta", detector="B?"),
    ]
    client = _StubChat(lambda messages: '{"p001": true}')

    out = interpret_mod.detect_many(_detect_records(1), props, client=client)
    assert out["clusters:r:g000"][0]["exhibits"] is True
    assert out["clusters:r:g001"][0]["exhibits"] is None
    assert interpret_mod.prevalence(out["clusters:r:g001"])["n"] == 0


def test_a_failed_batched_call_leaves_every_property_unknown_for_that_record():
    props = [
        _property(property_id="clusters:r:g000", label="alpha", detector="A?"),
        _property(property_id="clusters:r:g001", label="beta", detector="B?"),
    ]
    client = _StubChat(
        lambda messages: (
            RuntimeError("429 rate limited")
            if "trace 1" in messages[1]["content"]
            else '{"p001": true, "p002": true}'
        )
    )

    out = interpret_mod.detect_many(_detect_records(2), props, client=client)
    for property_id in out:
        assert out[property_id][1]["exhibits"] is None
        assert "429" in out[property_id][1]["error"]
    # The failed record leaves the denominator; it is not counted as "does not exhibit".
    assert interpret_mod.prevalence(out["clusters:r:g000"]) == {
        "n": 1,
        "hits": 1,
        "prevalence": 1.0,
        "ci_low": pytest.approx(0.207, abs=0.01),
        "ci_high": 1.0,
        "n_errors": 1,
    }


def test_the_detector_preflight_fails_on_one_call_rather_than_five_hundred():
    """GOTCHAS.md: size the smoke to the bug. A blank reply at full rubric scale must cost
    one call and a clear message, not a whole corpus of silently-empty verdicts."""
    props = [
        _property(property_id=f"clusters:r:g{i:03d}", label=f"p{i}", detector="A?")
        for i in range(3)
    ]
    blank = _StubChat(
        lambda messages: RuntimeError("EmptyCompletionError: returned empty content")
    )
    with pytest.raises(RuntimeError, match="raise `detector.max_tokens`"):
        interpret_mod.preflight_detect_many(_detect_records(2), props, client=blank)

    # A reply that answers only some of the rubrics is the quieter version of the same
    # failure: it becomes a partial measurement rather than an error.
    partial = _StubChat(lambda messages: '{"p001": true}')
    with pytest.raises(RuntimeError, match="answered 1 of 3"):
        interpret_mod.preflight_detect_many(_detect_records(2), props, client=partial)


def test_the_detector_preflight_probes_the_longest_record_and_times_it():
    """The worst case for a token budget is the longest record, so that is the probe."""
    props = [_property(property_id="clusters:r:g000", label="alpha", detector="A?")]
    records = [
        Record(record_id="short", query="q", response="a", reasoning="t"),
        Record(record_id="long", query="q", response="a", reasoning="t" * 5000),
    ]
    client = _StubChat(lambda messages: '{"p001": true}')

    result = interpret_mod.preflight_detect_many(records, props, client=client)
    assert result["record_id"] == "long" and result["chars"] == 5000
    assert result["n_answered"] == 1 and result["seconds"] >= 0
    assert len(client.calls) == 1, "a preflight is one call"


def test_only_the_batched_detector_suppresses_reasoning():
    """Measured 2026-08-20: on the batched path, reasoning buys 1.5 points of agreement
    with the unbatched reference for 40x the wall clock, so it is off there. The unbatched
    path keeps it — that path IS the reference, and changing it would move the yardstick."""
    props = [_property(property_id="clusters:r:g000", label="alpha", detector="A?")]
    batched = _StubChat(lambda messages: '{"p001": true}')
    single = _StubChat(
        lambda messages: '{"exhibits": true, "evidence": "", "note": ""}'
    )

    interpret_mod.detect_many(_detect_records(1), props, client=batched)
    interpret_mod.detect(_detect_records(1), "alpha", "A?", client=single)
    assert interpret_mod.NO_REASONING == {"reasoning": {"enabled": False}}
    assert batched.extra_bodies == [interpret_mod.NO_REASONING]
    assert single.extra_bodies == [None], "the reference path still reasons"


def test_verify_batching_reports_where_the_two_detector_paths_disagree(monkeypatch):
    """Batching is a different instrument. This is the number that says how different."""
    props = [_property(property_id="clusters:r:g000", label="alpha", detector="A?")]
    client = _StubChat(lambda messages: '{"p001": true}')
    # The unbatched path says no on the second record; the batched one says yes to both.
    monkeypatch.setattr(
        interpret_mod,
        "detect",
        lambda records, label, detector, **kw: [
            {"record_id": r.record_id, "exhibits": i == 0}
            for i, r in enumerate(records)
        ],
    )

    result = interpret_mod.verify_batching(_detect_records(2), props, client=client)
    assert result["n_cells"] == 2 and result["agreement"] == 0.5
    assert result["prevalence_batched"] == 1.0 and result["prevalence_single"] == 0.5
    assert result["per_property"][0]["label"] == "alpha"


def test_a_sampled_detector_does_not_become_the_membership_every_rate_is_computed_on(
    tmp_path, monkeypatch
):
    """A detector that read 40 of 513 records knows nothing about the other 473. Using its
    verdicts as membership would count every record it never saw as a non-member, which
    would deflate every prevalence and every arm delta at once."""
    from src.properties.producers import clusters as tc

    records = [
        Record(
            record_id=f"r{i}",
            query="q",
            response="a",
            reasoning="t",
            metadata={"arm": "a"},
        )
        for i in range(10)
    ]
    props = [_property(property_id="clusters:r:g000", label="alpha", detector="A?")]
    monkeypatch.setattr(
        tc.interpret_mod,
        "detect_many",
        lambda sample, properties, **kw: {
            p.property_id: [
                {"record_id": r.record_id, "exhibits": True} for r in sample
            ]
            for p in properties
        },
    )

    _, sampled, _ = tc._remeasure(
        props,
        records,
        {"detector": {"batched": True, "sample": 4}},
        {},
        "arm",
        tmp_path,
    )
    assert sampled is None, "a sampled detector cannot define corpus-wide membership"

    _, full, undetected = tc._remeasure(
        props,
        records,
        {"detector": {"batched": True, "sample": None}},
        {},
        "arm",
        tmp_path,
    )
    assert full == {"clusters:r:g000": {f"r{i}" for i in range(10)}}
    assert undetected == set()
    assert (tmp_path / "detector_verdicts.jsonl").exists(), (
        "the cells behind the prevalence must be recountable"
    )


# --- probe: the multivariate view of the property set ------------------------------------


def test_a_probe_finds_the_minimal_set_of_properties_that_separates_two_arms():
    """One informative column among noise: the L1 path must report that ONE suffices,
    which is the claim a table of per-property tests cannot make."""
    from src.properties.shared import probe as probe_mod

    rng = np.random.default_rng(0)
    y = np.array([1] * 100 + [0] * 100)
    matrix = rng.integers(0, 2, size=(200, 12)).astype(np.float32)
    matrix[:, 3] = y  # the one column that carries the signal
    columns = [f"p{j}" for j in range(12)]

    result = probe_mod.probe(matrix, y, columns, "arm == five_pct", permutations=20)
    assert result["best"]["auc"] == pytest.approx(1.0, abs=0.01)
    assert result["minimal"]["n_selected"] == 1, "one property is enough; say so"
    assert result["top_coefficients"][0]["property"] == "p3"
    assert result["permutation_null"]["p_value"] < 0.1


def test_a_probe_on_uninformative_properties_reports_the_null_not_a_finding():
    """With 12 columns over 200 rows an AUC has to be read against a floor fitted through
    the same pipeline, not against 0.5."""
    from src.properties.shared import probe as probe_mod

    rng = np.random.default_rng(1)
    y = np.array([1] * 100 + [0] * 100)
    matrix = rng.integers(0, 2, size=(200, 12)).astype(np.float32)

    result = probe_mod.probe(
        matrix, y, [f"p{j}" for j in range(12)], "arm", permutations=20
    )
    assert result["best"]["auc"] < 0.65, "noise must not separate the arms"
    assert result["permutation_null"]["p_value"] > 0.05


def test_a_probe_refuses_a_single_class_target():
    from src.properties.shared import probe as probe_mod

    with pytest.raises(ValueError, match="one class"):
        probe_mod.probe(
            np.zeros((10, 2), dtype=np.float32),
            np.zeros(10),
            ["a", "b"],
            "arm",
            permutations=0,
        )


def test_membership_matrix_is_one_row_per_record_and_one_column_per_property():
    from src.properties.shared import probe as probe_mod

    properties = [
        _property(label="alpha", property_id="clusters:r:g000"),
        _property(label="beta", property_id="clusters:r:g001"),
    ]
    matrix, columns = probe_mod.membership_matrix(
        properties, {"clusters:r:g000": [0, 2], "clusters:r:g001": [2]}, 4
    )

    assert columns == ["alpha", "beta"]
    assert matrix.tolist() == [[1, 0], [0, 0], [1, 1], [0, 0]]


# --- concentration: is a property really a scenario marker? -----------------------------


def test_the_audit_flags_a_group_drawn_from_a_single_scenario():
    from src.properties.shared import audit as audit_mod

    records = [
        Record(
            record_id=f"r{j}",
            query="q",
            response="a",
            reasoning="t",
            metadata={
                "scenario": "Hiring" if j < 10 else f"S{j}",
                "condition": "mandated",
            },
        )
        for j in range(20)
    ]
    result = audit_mod.concentration(
        records,
        {0: list(range(10)), 1: list(range(10, 20))},
        {0: "really the hiring scenario", 1: "a spread behaviour"},
        keys=("scenario",),
    )

    scenario = result["scenario"]
    assert scenario["n_flagged"] == 1
    assert scenario["flagged"][0]["label"] == "really the hiring scenario"
    assert scenario["flagged"][0]["top_share"] == 1.0
    # Excess over the corpus, not raw share: "Hiring" is half the corpus, so a group that
    # is ALL Hiring sits +50pp above it. The spread group's best value is +5pp.
    assert scenario["flagged"][0]["corpus_share"] == pytest.approx(0.5)
    assert scenario["flagged"][0]["excess"] == pytest.approx(0.5)
    assert scenario["by_group"]["1"]["excess"] == pytest.approx(0.05)


def test_concentration_on_a_two_valued_key_is_not_satisfied_by_pigeonhole():
    """A raw-share threshold of 50% flags EVERY group on a binary key — measured
    2026-08-20, it flagged 49 of 49 groups on `arm`, which says nothing about any of them.
    Excess over the corpus mix is what carries information."""
    from src.properties.shared import audit as audit_mod

    # Corpus is 60/40 on `arm`. One group matches that mix; one is arm-exclusive.
    records = [
        Record(
            record_id=f"r{j}",
            query="q",
            response="a",
            reasoning="t",
            metadata={"arm": "focus" if j < 12 else "control"},
        )
        for j in range(20)
    ]
    result = audit_mod.concentration(
        records,
        {0: [0, 1, 2, 3, 4, 12, 13, 14], 1: list(range(8))},
        {0: "tracks the corpus mix", 1: "focus-arm only"},
        keys=("arm",),
        threshold=0.25,
    )

    arm = result["arm"]
    assert arm["n_values"] == 2 and arm["corpus_shares"]["focus"] == pytest.approx(0.6)
    assert arm["n_flagged"] == 1, "only the arm-exclusive group is concentrated"
    assert arm["flagged"][0]["label"] == "focus-arm only"
    # The mix-matching group still has a top_share above 50% — which is exactly why the
    # raw share is the wrong statistic here.
    assert arm["by_group"]["0"]["top_share"] > 0.5
    assert arm["by_group"]["0"]["excess"] < 0.25
    # A key no record carries is skipped rather than reported as 100% concentrated in
    # "None", which is what reading a missing field as a value would do.
    assert "cell" not in audit_mod.concentration(
        records, {0: [0]}, {0: "x"}, keys=("cell",)
    )


# --- clusters over rollouts: ranked groups, and the ones with no home -------------------


def _rollout_records(n_per_arm=30):
    """Two arms with different base violation rates, and one group that really is safer."""
    records = []
    for arm, base in (("difficult_advice", 0.2), ("courtroom", 0.7)):
        for i in range(n_per_arm):
            in_group = i < n_per_arm // 2
            rate = base - 0.2 if in_group else base
            records.append(
                Record(
                    record_id=f"{arm}/r{i}",
                    query="q",
                    response="a",
                    reasoning=f"{'checks authorisation' if in_group else 'acts'} {i}",
                    outcome={
                        "violation": (i % (n_per_arm // 2))
                        < round(rate * (n_per_arm // 2)),
                        "score": 4,
                    },
                    metadata={"arm": arm, "corpus": {"path": "output/odcv_bench"}},
                )
            )
    return records


def _stub_embedding(monkeypatch, tc, vectors, features=None):
    """Stub every network call the producer makes: embed, extract, interpret.

    `features` maps record_id -> its feature strings, for `evidence: features` runs. When
    omitted the extractor is stubbed to raise, so a test that reaches features mode by
    accident fails loudly instead of quietly billing OpenRouter.
    """
    meta = embed_mod.EmbedMeta(
        backend="openrouter", model="stub", dim=vectors.shape[1], n=len(vectors)
    )
    monkeypatch.setattr(tc.embed_mod, "embed", lambda texts, **kw: (vectors, meta))

    def extract_to(records, spec, path, workers=16, client=None):
        if features is None:
            raise AssertionError(
                "this test reached feature extraction without stubbing "
                "it; pass features= or set evidence: traces"
            )
        rows = [
            {
                "record_id": r.record_id,
                "attributes": features[r.record_id],
                "tokens_in": 1,
                "tokens_out": 1,
            }
            for r in records
        ]
        Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))
        return rows

    monkeypatch.setattr(tc.attributes_mod, "extract_to", extract_to)
    monkeypatch.setattr(tc.attributes_mod, "extract", extract_to)
    from src.properties.shared.interpret import Interpretation

    monkeypatch.setattr(
        tc.interpret_mod,
        "interpret_many",
        lambda groups, **kw: {
            g: Interpretation(
                label=f"Property {g}",
                description="d",
                detector="Does the trace do it?",
                channel="reasoning",
                evidence=groups[g][:2],
                model="stub",
            )
            for g in groups
        },
    )
    return meta


def _two_blobs(n_a, n_b, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    vectors = np.vstack(
        [rng.normal(0, 0.02, (n_a, dim)) + 1.0, rng.normal(0, 0.02, (n_b, dim)) - 1.0]
    ).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_clusters_ranks_rollout_groups_by_within_arm_outcome(tmp_path, monkeypatch):
    """The rollout-side run: one pooled fit, then every group crossed with the violation
    flag WITHIN its arm. The ordering of the returned rows IS the ablation shortlist."""
    from src.properties.producers import clusters as tc

    records = _rollout_records()
    # Blob 0 = the "checks authorisation" half of both arms; blob 1 = the rest.
    order = [r for r in records if "checks" in r.reasoning] + [
        r for r in records if "checks" not in r.reasoning
    ]
    vectors = _two_blobs(30, 30)
    _stub_embedding(monkeypatch, tc, vectors)

    properties = tc.produce(
        order,
        {
            "channel": "reasoning",
            "evidence": "traces",
            "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
            "group_by": "arm",
            "outcomes": {"field": "violation", "fdr": 0.10, "min_arm_records": 5},
        },
        tmp_path / "tc",
    )

    assert len(properties) == 2
    lifts = [
        p.support["outcomes"]["by_field"]["violation"]["within_stratum_lift"]
        for p in properties
    ]
    assert lifts == sorted(lifts), "rows must arrive most protective first"
    outcomes = properties[0].support["outcomes"]
    assert outcomes["primary"] == "violation" and outcomes["strata_key"] == "arm"
    top = outcomes["by_field"]["violation"]
    assert set(top["by_stratum"]) == {"difficult_advice", "courtroom"}
    # The confounded number is carried, clearly named, next to the honest one.
    assert "pooled_lift_confounded" in top and top["q"] is not None
    assert (tmp_path / "tc" / "ranking.json").exists()
    report = (tmp_path / "tc" / "report.md").read_text(encoding="utf-8")
    assert "Outcome rate on `violation`, within stratum" in report
    assert "difficult_advice" in report


def test_outcomes_can_measure_several_fields_each_with_its_own_bh_family():
    """ODCV's >= 3 threshold is not the only honest reading of a 0-5 rubric. A second
    boolean at `score > 0` measures the sub-threshold drift on the same machinery; it is
    its own family of tests, so it gets its own correction."""
    records = [
        Record(
            record_id=f"r{j}",
            query="q",
            response="a",
            reasoning="t",
            outcome={"violation": j % 10 == 0, "any_misalignment": j % 2 == 0},
            metadata={"arm": "a"},
        )
        for j in range(60)
    ]
    member_ids = {f"r{j}" for j in range(30)}

    strict = outcomes_mod.by_stratum(records, member_ids, outcome_key="violation")
    loose = outcomes_mod.by_stratum(records, member_ids, outcome_key="any_misalignment")
    assert strict["strata"]["a"]["rate_in"] == pytest.approx(0.1)
    assert loose["strata"]["a"]["rate_in"] == pytest.approx(0.5)
    # Same members, same strata, different outcome: the two are separate measurements.
    assert strict["strata"]["a"]["n_members"] == loose["strata"]["a"]["n_members"] == 30


def test_clusters_refuses_outcomes_over_a_corpus_that_has_none(tmp_path, monkeypatch):
    from src.properties.producers import clusters as tc

    records = [Record(f"r{i}", "q", "a", reasoning=f"t{i}") for i in range(20)]
    _stub_embedding(monkeypatch, tc, _two_blobs(10, 10))
    with pytest.raises(ValueError, match="no record carries one"):
        tc.produce(
            records,
            {
                "evidence": "traces",
                "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
                "outcomes": {"field": "violation"},
            },
            tmp_path / "tc",
        )


def test_compare_to_finds_the_group_the_training_corpus_has_no_home_for(
    tmp_path, monkeypatch
):
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
    (prior / "properties_preview.json").write_text(
        json.dumps([{"label": "Weighs authorisation", "support": {"group": 0}}])
    )

    records = [
        Record(f"r{i}", "q", "a", reasoning=f"t{i}", metadata={"arm": "a"})
        for i in range(30)
    ]
    _stub_embedding(monkeypatch, tc, _two_blobs(15, 15))

    properties = tc.produce(
        records,
        {
            "evidence": "traces",
            "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
            "compare_to": {"run_dir": str(prior), "min_cosine": 0.6},
        },
        tmp_path / "tc",
    )

    novelties = {p.support["novelty"]["elicited_not_taught"] for p in properties}
    assert novelties == {True, False}, "one blob matches the prior run, one does not"
    elicited = next(
        p for p in properties if p.support["novelty"]["elicited_not_taught"]
    )
    assert elicited.support["novelty"]["share_unhoused"] == 1.0
    housed = next(
        p for p in properties if not p.support["novelty"]["elicited_not_taught"]
    )
    assert housed.support["novelty"]["nearest_training_group"] == "Weighs authorisation"

    summary = json.loads((tmp_path / "tc" / "novelty.json").read_text())["summary"]
    assert summary["n_unhoused"] == 15 and summary["n_training_groups"] == 1
    assert "ELICITED" in (tmp_path / "tc" / "report.md").read_text()


def test_compare_to_refuses_a_run_embedded_with_a_different_model(
    tmp_path, monkeypatch
):
    from src.properties.producers import clusters as tc

    prior = tmp_path / "prior"
    prior.mkdir()
    np.save(prior / "embeddings.npy", _two_blobs(10, 0).astype(np.float16))
    np.save(prior / "labels.npy", np.zeros(10, dtype=np.int64))
    (prior / "embeddings_meta.json").write_text(
        json.dumps({"model": "some-other-model"})
    )

    records = [Record(f"r{i}", "q", "a", reasoning=f"t{i}") for i in range(20)]
    _stub_embedding(monkeypatch, tc, _two_blobs(10, 10))
    with pytest.raises(ValueError, match="not comparable"):
        tc.produce(
            records,
            {
                "evidence": "traces",
                "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
                "compare_to": {"run_dir": str(prior)},
            },
            tmp_path / "tc",
        )

    with pytest.raises(FileNotFoundError, match="previous `clusters` run"):
        tc.produce(
            records,
            {
                "evidence": "traces",
                "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
                "compare_to": {"run_dir": str(tmp_path / "nope")},
            },
            tmp_path / "tc2",
        )


def test_baseline_grouping_writes_the_umap_is_it_helping_comparison(
    tmp_path, monkeypatch
):
    from src.properties.producers import clusters as tc

    records = [Record(f"r{i}", "q", "a", reasoning=f"t{i}") for i in range(20)]
    _stub_embedding(monkeypatch, tc, _two_blobs(10, 10))
    tc.produce(
        records,
        {
            "evidence": "traces",
            "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
            "baseline_grouping": {"reduce": "none", "cluster": "kmeans", "k": 4},
        },
        tmp_path / "tc",
    )

    comparison = json.loads((tmp_path / "tc" / "grouping_comparison.json").read_text())
    assert comparison["groups"] == {"baseline": 4, "candidate": 2}
    assert 0.0 <= comparison["agreement"]["ari"] <= 1.0


def test_detector_sampling_keeps_every_arm_represented():
    from src.properties.producers import clusters as tc

    records = [
        Record(f"big{i}", "q", "a", reasoning="t", metadata={"arm": "big"})
        for i in range(200)
    ]
    records += [
        Record(f"small{i}", "q", "a", reasoning="t", metadata={"arm": "small"})
        for i in range(10)
    ]
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
    cross = outcomes_mod.by_stratum(records, {f"a{j}" for j in range(10)})

    assert cross["strata"]["safe"]["rate_out"] is None, "no same-arm non-members exist"
    assert cross["strata"]["safe"]["lift"] is None
    assert outcomes_mod.combined_lift(cross, min_stratum_records=1)["lift"] is None
    # Pooled would have called this strongly protective. It is an arm marker.
    assert cross["pooled"]["lift"] == pytest.approx(-0.6)

    rows = outcomes_mod.rank({"confounded": cross}, min_stratum_records=1)
    assert rows[0]["lift"] is None and not rows[0]["significant"]


def test_clusters_refuses_a_run_where_every_group_is_below_the_floor(
    tmp_path, monkeypatch
):
    from src.properties.producers import clusters as tc

    records = [Record(f"r{i}", "q", "a", reasoning=f"t{i}") for i in range(6)]
    _stub_embedding(monkeypatch, tc, _two_blobs(3, 3))
    with pytest.raises(ValueError, match="would export nothing"):
        tc.produce(
            records,
            {
                "evidence": "traces",
                "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
                "min_group_records": 5,
            },
            tmp_path / "tc",
        )


def test_clusters_refuses_an_embedding_that_lost_a_row(tmp_path, monkeypatch):
    from src.properties.producers import clusters as tc

    records = [Record(f"r{i}", "q", "a", reasoning=f"t{i}") for i in range(20)]
    _stub_embedding(monkeypatch, tc, _two_blobs(9, 10))
    with pytest.raises(ValueError, match="must correspond 1:1"):
        tc.produce(
            records,
            {
                "evidence": "traces",
                "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
            },
            tmp_path / "tc",
        )


def test_a_pooled_run_is_stamped_as_the_set_it_is_not_its_first_arm(
    tmp_path, monkeypatch
):
    """`prevalence` is only interpretable next to the corpus it was measured on. On a
    pooled run the first record's stamp names one arm, which would be a false label on
    rows describing all of them."""
    from src.properties.producers import clusters as tc

    records = [
        Record(
            f"a{i}",
            "q",
            "a",
            reasoning=f"t{i}",
            metadata={"arm": "da", "corpus": {"path": "out/da", "arm": "da"}},
        )
        for i in range(10)
    ]
    records += [
        Record(
            f"b{i}",
            "q",
            "a",
            reasoning=f"t{i}",
            metadata={"arm": "court", "corpus": {"path": "out/court", "arm": "court"}},
        )
        for i in range(10)
    ]
    _stub_embedding(monkeypatch, tc, _two_blobs(10, 10))
    properties = tc.produce(
        records,
        {
            "evidence": "traces",
            "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
            "group_by": "arm",
        },
        tmp_path / "tc",
    )

    corpus = properties[0].corpus
    assert corpus["n_corpora"] == 2
    assert {c["arm"] for c in corpus["pooled"]} == {"da", "court"}

    # A single-corpus run keeps the plain stamp rather than growing a wrapper.
    single = [
        Record(
            f"a{i}", "q", "a", reasoning=f"t{i}", metadata={"corpus": {"repo": "org/x"}}
        )
        for i in range(20)
    ]
    _stub_embedding(monkeypatch, tc, _two_blobs(10, 10))
    properties = tc.produce(
        single,
        {
            "evidence": "traces",
            "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
        },
        tmp_path / "tc2",
    )
    assert properties[0].corpus == {"repo": "org/x"}


def test_members_jsonl_joins_every_record_to_its_property_and_its_rollout(
    tmp_path, monkeypatch
):
    """Ten example ids on a property row is enough to check a label and not enough to
    work with. members.jsonl is the full record -> property map, carrying the path to the
    rollout so the traces behind a cluster can actually be opened."""
    from src.properties.producers import clusters as tc

    records = []
    for arm, n in (("da", 10), ("court", 10)):
        for i in range(n):
            records.append(
                Record(
                    f"{arm}/r{i}",
                    "q",
                    "a",
                    reasoning=f"trace {i}",
                    outcome={"violation": i < 3, "score": 4 if i < 3 else 0},
                    metadata={
                        "arm": arm,
                        "rollout_path": f"/runs/{arm}/r{i}/messages_record.txt",
                    },
                )
            )
    _stub_embedding(monkeypatch, tc, _two_blobs(10, 10))
    properties = tc.produce(
        records,
        {
            "evidence": "traces",
            "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
            "group_by": "arm",
        },
        tmp_path / "tc",
    )

    rows = [
        json.loads(x)
        for x in (tmp_path / "tc" / "members.jsonl").read_text().splitlines()
        if x.strip()
    ]
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
    tc.produce(
        records,
        {
            "evidence": "traces",
            "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
            "min_group_records": 5,
        },
        tmp_path / "tc",
    )

    rows = [
        json.loads(x)
        for x in (tmp_path / "tc" / "members.jsonl").read_text().splitlines()
        if x.strip()
    ]
    excluded = [r for r in rows if r["excluded"]]
    assert len(excluded) == 3 and {r["excluded"] for r in excluded} == {"below_floor"}
    assert all(r["property_id"] is None for r in excluded)


# --- evidence: features — the ported feature-discovery method ---------------------------


def _feature_corpus(n=24):
    """Records whose feature lists overlap: everyone checks, half also weigh."""
    records, features = [], {}
    for i in range(n):
        rid = f"r{i}"
        records.append(
            Record(
                rid,
                "q",
                "a",
                reasoning=f"trace {i}",
                metadata={
                    "arm": "da" if i % 2 else "court",
                    "trait_id": "t1" if i < n // 2 else "t2",
                },
            )
        )

        features[rid] = (
            ["checks authorisation before acting"]
            + (["weighs harm against benefit"] if i % 2 else [])
            + [f"idiosyncratic phrasing {i}"]
        )
    return records, features


def test_features_mode_embeds_the_vocabulary_once_not_once_per_occurrence(
    tmp_path, monkeypatch
):
    from src.properties.producers import clusters as tc

    records, features = _feature_corpus(24)
    seen = {}

    def capture(texts, **kw):
        seen["texts"] = list(texts)
        vectors = _two_blobs(len(texts) // 2, len(texts) - len(texts) // 2, seed=3)[
            : len(texts)
        ]
        return vectors, embed_mod.EmbedMeta(
            backend="stub", model="stub", dim=8, n=len(texts)
        )

    _stub_embedding(monkeypatch, tc, _two_blobs(1, 1), features=features)
    monkeypatch.setattr(tc.embed_mod, "embed", capture)
    tc.produce(
        records,
        {
            "evidence": "features",
            "min_group_records": 2,
            "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
        },
        tmp_path / "fd",
    )

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
    vectors = np.array(
        [
            (
                np.ones(8)
                if t.startswith("checks") or t.startswith("idio")
                else -np.ones(8)
            )
            for t in texts
        ],
        dtype=np.float32,
    )
    vectors += np.random.default_rng(0).normal(0, 0.01, vectors.shape)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    _stub_embedding(monkeypatch, tc, vectors.astype(np.float32), features=features)

    properties = tc.produce(
        records,
        {
            "evidence": "features",
            "min_group_records": 2,
            "group_by": "arm",
            "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
        },
        tmp_path / "fd",
    )

    assert all(p.support["prevalence_kind"] == "feature_membership" for p in properties)
    # Overlapping groups: prevalences do NOT sum to 1, unlike traces mode.
    assert sum(p.prevalence for p in properties) > 1.0
    rows = [
        json.loads(x)
        for x in (tmp_path / "fd" / "members.jsonl").read_text().splitlines()
        if x.strip()
    ]
    per_record = {}
    for row in rows:
        per_record.setdefault(row["record_id"], []).append(row["property_id"])
    assert max(len(v) for v in per_record.values()) > 1, (
        "a record holds several properties"
    )
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

    properties = tc.produce(
        records,
        {
            "evidence": "features",
            "min_group_records": 2,
            "grouping": {"reduce": "none", "cluster": "kmeans", "k": 1},
        },
        tmp_path / "fd",
    )
    top = properties[0]
    assert top.n_records == 10  # every record says "a"
    assert top.n_instances == 14  # 10 x "a" + 2 x "b" + 2 x "c"
    assert top.prevalence == 1.0


def test_extract_to_streams_as_it_goes_and_resumes_a_killed_run(tmp_path, monkeypatch):
    """Extraction is the expensive stage and the one most likely to be interrupted. A run
    killed at 95% must keep its 95%, and a rerun must label only what is missing."""
    records = [Record(f"r{i}", "q", "a", reasoning=f"t{i}") for i in range(6)]
    spec = attributes_mod.AttributeSpec(style="freeform", channel="reasoning")
    path = tmp_path / "features.jsonl"
    asked = []

    class _Reply:
        def __init__(self, rid):
            self.content = json.dumps([f"feature of {rid}"])
            self.prompt_tokens = self.completion_tokens = 1

    class _Client:
        def chat(self, model, messages, **kw):
            rid = messages[-1]["content"]
            asked.append(rid)
            return _Reply(rid)

    # A partial run: three records already on disk, one of them a recorded failure.
    path.write_text(
        "".join(
            json.dumps(r) + "\n"
            for r in [
                {
                    "record_id": "r0",
                    "attributes": ["cached zero"],
                    "tokens_in": 1,
                    "tokens_out": 1,
                },
                {
                    "record_id": "r1",
                    "attributes": ["cached one"],
                    "tokens_in": 1,
                    "tokens_out": 1,
                },
                {"record_id": "r2", "attributes": [], "error": "RateLimit: 429"},
            ]
        )
    )
    rows = attributes_mod.extract_to(records, spec, path, workers=2, client=_Client())

    assert [r["record_id"] for r in rows] == [f"r{i}" for i in range(6)]
    assert rows[0]["attributes"] == ["cached zero"], "a cached row is not re-extracted"
    # r2 failed last time: an error is an absence of evidence, so it is retried rather
    # than inherited, which would make a transient rate-limit permanent.
    assert rows[2]["attributes"] == ["feature of t2"]
    assert sorted(asked) == ["t2", "t3", "t4", "t5"]

    # Everything landed on disk, so a third run asks for nothing at all.
    asked.clear()
    again = attributes_mod.extract_to(records, spec, path, workers=2, client=_Client())
    assert asked == []
    assert [r["attributes"] for r in again] == [r["attributes"] for r in rows]


def test_the_interpreter_is_told_which_kind_of_evidence_it_is_reading(
    tmp_path, monkeypatch
):
    """The two modes hand the interpreter different objects. Telling it it is reading
    descriptions when it is reading transcripts is not cosmetic: the traces framing
    carries the warning about topical clusters that the features framing does not."""
    from src.properties.producers import clusters as tc

    seen = {}
    records, features = _feature_corpus(12)
    _stub_embedding(monkeypatch, tc, _two_blobs(7, 7), features=features)
    # The two modes embed different numbers of things (14 feature strings vs 12 traces),
    # so the stub has to size itself to what it is handed.
    monkeypatch.setattr(
        tc.embed_mod,
        "embed",
        lambda texts, **kw: (
            _two_blobs(len(texts) // 2, len(texts) - len(texts) // 2),
            embed_mod.EmbedMeta(backend="stub", model="stub", dim=8, n=len(texts)),
        ),
    )
    real = tc.interpret_mod.interpret_many
    monkeypatch.setattr(
        tc.interpret_mod,
        "interpret_many",
        lambda groups, **kw: seen.update(kw) or real(groups, **kw),
    )
    monkeypatch.setattr(
        tc.interpret_mod,
        "interpret",
        lambda evidence, **kw: interpret_mod.Interpretation(
            label="L",
            description="d",
            detector="Does it?",
            channel="reasoning",
            model="stub",
        ),
    )

    tc.produce(
        records,
        {
            "evidence": "features",
            "min_group_records": 2,
            "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
        },
        tmp_path / "fd",
    )
    assert seen["evidence_kind"] == "features"

    tc.produce(
        records,
        {
            "evidence": "traces",
            "min_group_records": 2,
            "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
        },
        tmp_path / "tc",
    )
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


def test_coverage_json_states_what_the_properties_do_not_account_for(
    tmp_path, monkeypatch
):
    from src.properties.producers import clusters as tc

    records, features = _feature_corpus(12)
    _stub_embedding(monkeypatch, tc, _two_blobs(7, 7), features=features)
    tc.produce(
        records,
        {
            "evidence": "features",
            "min_group_records": 2,
            "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
        },
        tmp_path / "fd",
    )

    coverage = json.loads((tmp_path / "fd" / "coverage.json").read_text())
    assert coverage["evidence"] == "features"
    assert coverage["records"] == 12 and coverage["units"] == 14
    assert coverage["feature_instances"] == 12 + 6 + 12
    assert coverage["records_with_no_property"] == 0
    assert coverage["unclustered_units"] == 0


# --- shared/audit: the checks the naming stage cannot do for itself ----------------------


def test_near_duplicate_groups_finds_two_centroids_describing_one_theme():
    from src.properties.shared import audit as audit_mod

    centroids = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]], dtype=np.float32)
    pairs = audit_mod.near_duplicate_groups(
        centroids, {0: "checks authorisation", 1: "confirms permission", 2: "hedges"}
    )
    assert len(pairs) == 1
    assert {pairs[0]["a"], pairs[0]["b"]} == {0, 1}
    assert pairs[0]["cosine"] >= 0.90
    assert audit_mod.near_duplicate_groups(centroids[:1], {0: "x"}) == []


def test_keyword_probes_read_the_evidence_not_the_clustering():
    """A theme too small to win its own group still gets a number, and the probe says
    where the clustering scattered it."""
    from src.properties.shared import audit as audit_mod

    units = [
        "notices this is a test scenario",
        "weighs harm against benefit",
        "suspects being evaluated",
        "cites the policy",
    ]
    result = audit_mod.keyword_probes(
        units=units,
        unit_records=[[0], [0, 1, 2], [3], [1]],
        unit_instances=[1, 3, 1, 1],
        unit_group=[0, 1, -1, 1],
        group_labels={0: "situational awareness", 1: "policy reasoning"},
        n_records=4,
    )

    probe = result["evaluation awareness"]
    assert probe["units"] == 2 and probe["records"] == 2
    assert probe["prevalence"] == 0.5
    # One match clustered, one was noise: the scatter is the finding.
    landed = dict(probe["groups_landed_in"])
    assert landed == {"situational awareness": 1, audit_mod.UNCLUSTERED_LABEL: 1}


def test_probes_are_word_boundary_regexes_not_substring_matches():
    """The mistake that burned the original analysis twice: bare 'persona' matched
    'interpersonal' and 'tradesperson', inflating a rate from 11.1% to 17.2%."""
    from src.properties.shared import audit as audit_mod

    units = [
        "considers interpersonal dynamics",
        "asks the tradesperson",
        "adopts a different persona",
    ]
    result = audit_mod.keyword_probes(
        units=units,
        unit_records=[[0], [1], [2]],
        unit_instances=[1, 1, 1],
        unit_group=[-1, -1, -1],
        group_labels={},
        n_records=3,
    )
    assert result["persona and identity"]["units"] == 1
    assert result["persona and identity"]["top_example_text"] == [
        "adopts a different persona"
    ]


def test_stability_sweep_scores_every_fit_against_every_other_not_just_the_reference():
    from src.properties.shared import audit as audit_mod
    from src.properties.shared import grouping as grouping_mod

    vectors = _two_blobs(20, 20, dim=6, seed=7)
    params = grouping_mod.GroupingParams(reduce="none", cluster="kmeans", k=2)
    reference = grouping_mod.group(vectors, params)
    sweep = audit_mod.stability_sweep(vectors, params, reference.labels, seeds=(0, 1))

    assert len(sweep["fits"]) == 2
    # Two well-separated blobs are recovered identically whatever the seed.
    assert sweep["min_pairwise_ari"] == 1.0
    assert sweep["mean_ari_vs_reference"] == 1.0
    # The pairwise matrix is square and covers every pair, so one lucky seed cannot pass.
    assert len(sweep["pairwise_ari"]) == len(sweep["pairwise_ari"][0]) == 2


def test_audit_runs_with_the_producer_and_writes_a_browsable_dashboard(
    tmp_path, monkeypatch
):
    from src.properties.producers import clusters as tc

    records, features = _feature_corpus(12)
    _stub_embedding(monkeypatch, tc, _two_blobs(7, 7), features=features)
    tc.produce(
        records,
        {
            "evidence": "features",
            "min_group_records": 2,
            "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
        },
        tmp_path / "fd",
    )

    record = json.loads((tmp_path / "fd" / "audit.json").read_text())
    assert "near_duplicate_pairs" in record and "probes" in record
    assert "stability" not in record, "the sweep is opt-in; it refits the whole matrix"
    assert "Buried behaviours" in (tmp_path / "fd" / "audit.md").read_text()
    # The audit is appended to the run's own report, not left in a separate file only.
    assert "## Audit" in (tmp_path / "fd" / "report.md").read_text()
    page = (tmp_path / "fd" / "dashboard.html").read_text()
    assert "<title>" in page and "checks authorisation before acting" in page


def test_project_2d_is_a_separate_fit_from_the_one_that_was_clustered():
    """The post runs UMAP twice on purpose: 2-D to look at, more to cluster. The picture
    must not be mistaken for the space membership was decided in."""
    from src.properties.shared import grouping as grouping_mod

    vectors = _two_blobs(15, 15, dim=10, seed=2)
    # reduce=none: nothing was reduced, so there is no second reduction to make.
    flat = grouping_mod.GroupingParams(reduce="none", cluster="kmeans", k=2)
    assert grouping_mod.project_2d(vectors, flat).shape == (30, 2)
    assert np.allclose(grouping_mod.project_2d(vectors, flat), vectors[:, :2])


def test_the_dashboard_plots_the_projection_and_says_what_it_is_not(
    tmp_path, monkeypatch
):
    from src.properties.producers import clusters as tc

    records, features = _feature_corpus(12)
    _stub_embedding(monkeypatch, tc, _two_blobs(7, 7), features=features)
    tc.produce(
        records,
        {
            "evidence": "features",
            "min_group_records": 2,
            "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
        },
        tmp_path / "fd",
    )

    page = (tmp_path / "fd" / "dashboard.html").read_text()
    assert "<svg" in page and page.count("<circle") == 14, "one dot per embedded unit"
    assert "The embedding space, after UMAP" in page
    assert (tmp_path / "fd" / "coords_2d.npy").exists()
    # Legend names the groups; noise, if any, is never given a colour from the ramp.
    assert "class='legend'" in page


def test_the_scatter_colours_noise_separately_and_draws_it_underneath():
    from src.properties.shared import audit as audit_mod

    coords = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    svg = audit_mod.scatter(coords, np.array([0, 1, -1]), {0: "alpha", 1: "beta"})
    assert svg.count("<circle") == 3
    # Noise is drawn first so real groups paint over it, and takes the grey, not a ramp
    # colour that would read as a group.
    assert svg.index(audit_mod.NOISE_COLOUR) < svg.index(audit_mod.SCATTER_COLOURS[0])
    assert "unclustered (1)" in svg
    assert audit_mod.scatter(np.zeros((0, 2)), np.array([]), {}) == ""


def test_a_umap_run_labels_the_plot_as_a_different_projection(tmp_path):
    from src.properties.shared import audit as audit_mod

    record = {
        "n_groups": 1,
        "near_duplicate_pairs": [],
        "probes": {},
        "noise_share": 0.0,
    }
    coords = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    path = audit_mod.dashboard(
        tmp_path / "d.html",
        [],
        record,
        "run",
        2,
        coords=coords,
        labels=np.array([0, 0]),
        n_components=5,
    )
    page = path.read_text()
    assert "clustering ran in 5 dimensions" in page
    assert "may be in different groups" in page

    path = audit_mod.dashboard(
        tmp_path / "d2.html",
        [],
        record,
        "run",
        2,
        coords=coords,
        labels=np.array([0, 0]),
        n_components=2,
    )
    assert "the clustering actually ran in" in path.read_text()


def test_reclustering_reuses_the_embeddings_instead_of_paying_the_pod_again(
    tmp_path, monkeypatch
):
    """Retuning the clustering does not change a single vector, and it is the loop you
    run ten times. Re-embedding each round is the expensive way to answer a cheap
    question."""
    from src.properties.producers import clusters as tc

    records = [Record(f"r{i}", "q", "a", reasoning=f"trace {i}") for i in range(20)]
    calls = []

    def counting_embed(texts, **kw):
        calls.append(len(texts))
        n = len(texts)
        return (
            _two_blobs(n // 2, n - n // 2),
            embed_mod.EmbedMeta(backend="stub", model="stub", dim=8, n=n),
        )

    _stub_embedding(monkeypatch, tc, _two_blobs(10, 10))
    monkeypatch.setattr(tc.embed_mod, "embed", counting_embed)

    base = {"evidence": "traces", "min_group_records": 2}
    tc.produce(
        records,
        {**base, "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2}},
        tmp_path / "run",
    )
    # Same units, different resolution: the vectors still apply.
    tc.produce(
        records,
        {**base, "grouping": {"reduce": "none", "cluster": "kmeans", "k": 4}},
        tmp_path / "run",
    )
    assert calls == [20], "the second clustering must not re-embed"

    # One more record is a different unit list, so the cache is discarded rather than
    # silently mismatched against it.
    tc.produce(
        records + [Record("r20", "q", "a", reasoning="trace 20")],
        {**base, "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2}},
        tmp_path / "run",
    )
    assert len(calls) == 2

    # And opting out re-embeds unconditionally.
    tc.produce(
        records,
        {
            **base,
            "reuse_embeddings": False,
            "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
        },
        tmp_path / "run",
    )
    assert len(calls) == 3


def test_odcv_keeps_tool_calls_out_of_the_reasoning_channel(tmp_path):
    """The bench emits `call:` for tool calls. A field the reader does not know is not
    skipped — it is appended to whatever field came before it, which pastes tool-call JSON
    into the middle of the reasoning trace and into every feature extracted from it."""
    from src.properties.sources import odcv_rollouts

    body = """The task.
== Step 1 ==
role: assistant
content: Reading the ledger.
reason: I should confirm authorisation first.
call: [{'function': {'name': 'bash', 'arguments': '{"command": "ls /app"}'}}]
"""
    run = _odcv_run(tmp_path / "run", "x", n=1, body=body)
    record = odcv_rollouts.load(run_dir=str(run))[0]

    assert record.reasoning == "I should confirm authorisation first."
    assert "bash" not in record.reasoning and "function" not in record.reasoning
    # It is not dropped either: an action the agent took belongs in the response channel.
    assert "bash" in record.response and "Reading the ledger" in record.response


def test_freeform_parse_unwraps_a_wrapped_array_but_not_an_ambiguous_dict():
    """~2% of replies come back as {"features": [...]} instead of a bare array. Same
    content, different envelope; rejecting it costs those records' coverage for nothing."""
    spec = attributes_mod.AttributeSpec(style="freeform")
    assert attributes_mod.parse('["a", "b"]', spec) == ["a", "b"]
    assert attributes_mod.parse('{"features": ["a", "b"]}', spec) == ["a", "b"]
    # Two lists is genuinely ambiguous — which one is the features? — so it still fails.
    with pytest.raises(ValueError, match="expected a JSON array"):
        attributes_mod.parse('{"features": ["a"], "other": ["b"]}', spec)
    with pytest.raises(ValueError, match="expected a JSON array"):
        attributes_mod.parse('{"features": "not a list"}', spec)


def test_redundancy_is_measured_in_the_embedding_space_not_the_reduced_one():
    """UMAP output is an arbitrary all-positive blob offset from the origin, so every pair
    of points in it has a cosine near 1. Measuring redundancy there reports the offset,
    not whether two groups describe the same theme."""
    from src.properties.shared import audit as audit_mod

    # Two genuinely opposite groups in embedding space...
    vectors = _two_blobs(20, 20, dim=8, seed=1)
    labels = np.array([0] * 20 + [1] * 20)
    embedding_centroids = audit_mod._embedding_centroids(vectors, labels)
    assert audit_mod.near_duplicate_groups(embedding_centroids, {0: "a", 1: "b"}) == []

    # ...look like near-duplicates once shifted into a positive quadrant, the way UMAP
    # leaves them.
    shifted = embedding_centroids + 8.0
    assert len(audit_mod.near_duplicate_groups(shifted, {0: "a", 1: "b"})) == 1

    # Noise contributes to no centroid.
    with_noise = np.array([0] * 20 + [-1] * 20)
    assert audit_mod._embedding_centroids(vectors, with_noise).shape[0] == 1


# --- the degenerate-reduction guard -----------------------------------------------------


def _collapsed_coords(n, seed=0):
    """What a degenerate UMAP fit actually looked like on the real runs: every point in one
    of two tight lumps, zero noise, and membership unrelated to where the point started —
    so the reduction destroyed the neighbourhood structure it was supposed to preserve."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [6.0, 0.0]])
    pick = rng.integers(0, 2, n)
    return (centres[pick] + rng.normal(0, 0.05, (n, 2))).astype(np.float32)


def test_is_degenerate_gates_on_shape_but_decides_on_the_reduction():
    from src.properties.shared import grouping as gm

    # The cheap shape gate, with no matrices to check the geometry against.
    assert gm.is_degenerate(np.zeros(500, dtype=np.int64), 1)
    healthy = np.where(np.arange(500) % 3 == 0, -1, np.arange(500) % 17)
    assert not gm.is_degenerate(healthy, 17)
    two_with_noise = np.where(np.arange(500) % 4 == 0, -1, np.arange(500) % 2)
    assert not gm.is_degenerate(two_with_noise, 2)
    assert not gm.is_degenerate(np.arange(40) % 2, 2)  # too small to judge

    # Shape alone would condemn genuinely bimodal data: two clean blobs really do give two
    # groups and zero noise. The geometry check is what tells the two cases apart.
    vectors = _two_blobs(150, 150, dim=6, seed=8)
    labels = np.array([0] * 150 + [1] * 150)
    kept = np.asarray(vectors[:, :2], dtype=np.float32)
    assert not gm.is_degenerate(labels, 2, vectors, kept), (
        "a faithful reduction is fine"
    )

    collapsed = _collapsed_coords(300, 0)
    assert gm.is_degenerate(labels, 2, vectors, collapsed), "a collapsed one is not"


def test_group_retries_past_a_collapsed_umap_fit_and_records_that_it_did(monkeypatch):
    """A collapsed reduction is a failed run, not a two-property finding. The retry has to
    be recorded, because after it the result is no longer the seed the config asked for."""
    from src.properties.shared import grouping as gm

    vectors = _two_blobs(150, 150, dim=6, seed=4)
    seeds_tried = []

    def flaky_umap(v, params):
        seeds_tried.append(params.seed)
        if params.seed < 44:  # seeds 42, 43 collapse
            return _collapsed_coords(len(v), params.seed)
        return np.asarray(v[:, :2], dtype=np.float32)

    monkeypatch.setattr(gm, "reduce_umap", flaky_umap)
    params = gm.GroupingParams(
        reduce="umap", cluster="hdbscan", min_cluster_size=25, seed=42
    )
    result = gm.group(vectors, params)

    assert seeds_tried == [42, 43, 44]
    assert result.meta["seed_used"] == 44 and result.meta["degenerate_retries"] == 2
    # The recovered fit is two clean blobs with no noise, which the SHAPE gate flags on its
    # own — the geometry check is what clears it, so the matrices have to be passed.
    assert result.n_groups == 2
    assert gm.is_degenerate(result.labels, result.n_groups), (
        "shape alone would condemn it"
    )
    assert not gm.is_degenerate(result.labels, result.n_groups, vectors, result.coords)


def test_group_gives_up_loudly_when_every_seed_collapses(monkeypatch):
    from src.properties.shared import grouping as gm

    vectors = _two_blobs(150, 150, dim=6, seed=5)
    monkeypatch.setattr(
        gm, "reduce_umap", lambda v, p: _collapsed_coords(len(v), p.seed)
    )
    params = gm.GroupingParams(
        reduce="umap", cluster="hdbscan", min_cluster_size=25, seed=42
    )
    with pytest.raises(ValueError, match="collapsed at every seed"):
        gm.group(vectors, params, retry_degenerate=2)


def test_a_stability_sweep_must_not_retry_or_it_measures_the_retry_logic(monkeypatch):
    from src.properties.shared import grouping as gm

    vectors = _two_blobs(150, 150, dim=6, seed=6)
    monkeypatch.setattr(
        gm, "reduce_umap", lambda v, p: _collapsed_coords(len(v), p.seed)
    )
    params = gm.GroupingParams(
        reduce="umap", cluster="hdbscan", min_cluster_size=25, seed=42
    )
    # retry_degenerate=0 takes the first fit whatever it is, so the sweep still sees the
    # collapse it exists to count.
    result = gm.group(vectors, params, retry_degenerate=0)
    assert gm.is_degenerate(result.labels, result.n_groups)
    assert result.meta["degenerate_retries"] == 0


def test_embeddings_are_stored_at_full_precision_so_a_rerun_reproduces(tmp_path):
    """fp16 moves each component of a normalised 4096-d vector by ~1e-3 — the same order
    as the gaps between near neighbours — so reloading one rebuilds a different kNN graph.
    On the real da716 run that turned 17 groups at 40% noise into 2 groups at 0%."""
    rng = np.random.default_rng(0)
    vectors = embed_mod.normalise(rng.normal(size=(64, 512)).astype(np.float32))
    meta = embed_mod.EmbedMeta(backend="stub", model="m", dim=512, n=64)
    path = embed_mod.save(tmp_path / "e.npy", vectors, meta)

    stored = np.load(path)
    assert stored.dtype == np.float32
    # Exact round-trip: the reloaded matrix IS the matrix that was clustered.
    assert np.array_equal(stored, vectors)
    # And the neighbour ordering survives, which is the property that actually matters.
    reloaded = embed_mod.normalise(stored.astype(np.float32))
    order_a = np.argsort(-(vectors @ vectors.T), axis=1)[:, :10]
    order_b = np.argsort(-(reloaded @ reloaded.T), axis=1)[:, :10]
    assert np.array_equal(order_a, order_b)


def test_the_dashboard_lists_the_records_carrying_each_property(tmp_path, monkeypatch):
    """Twelve sampled phrases is enough to guess at a label and not enough to check it.
    The dashboard has to show which records actually carry the property, and where to
    open them."""
    from src.properties.producers import clusters as tc

    records = []
    for i in range(20):
        records.append(
            Record(
                f"r{i}",
                "q",
                "a",
                reasoning=f"trace {i}",
                outcome={"violation": i < 6, "score": 4 if i < 6 else 0},
                metadata={
                    "arm": "a",
                    "rollout_path": f"/runs/r{i}/messages_record.txt",
                },
            )
        )
    _stub_embedding(monkeypatch, tc, _two_blobs(10, 10))
    tc.produce(
        records,
        {
            "evidence": "traces",
            "min_group_records": 2,
            "group_by": "arm",
            "grouping": {"reduce": "none", "cluster": "kmeans", "k": 2},
        },
        tmp_path / "tc",
    )

    page = (tmp_path / "tc" / "dashboard.html").read_text()
    assert "records carry this property" in page
    assert "messages_record.txt" in page, "the path to open the rollout is shown"
    assert page.count("<table class='members'>") == 2, "one member table per property"
    # Violations sort first and are marked, so a reader lands on them.
    assert "violated" in page and "class='v'" in page
