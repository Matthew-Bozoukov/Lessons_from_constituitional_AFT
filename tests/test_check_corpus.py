# ABOUTME: Offline tests for the corpus-level check registry and its pipeline stage:
# ABOUTME: every property, the pass-through invariant, ablation, and failure semantics.

from __future__ import annotations

import json
import random
import re

import pytest

from src.data.synth import check_corpus as C
from src.data.synth.stage_runtime import Stage
from src.data.synth.stage_operators import OPERATORS
from src.data.synth.pipeline import build_stages, corpus_gate_failed, estimate, run

CONSTITUTION = "constitutions/archive/claude_distilled_8_principles_v1/constitution.md"

NOUNS = ("policy report client vendor budget audit contract deadline invoice roster "
         "supplier ledger tenant permit charter forecast").split()
VERBS = "reviewed flagged escalated deferred approved queried logged revised".split()


def varied(i: int, n: int = 60) -> str:
    """A deterministic pseudo-document that shares no long n-gram with its siblings."""
    r = random.Random(i * 7919)
    return " ".join(f"{r.choice(NOUNS)} {r.choice(VERBS)} {r.randrange(10**6)}"
                    for _ in range(n))


# `varied` defeats N-GRAM checks: no two documents share a long n-gram. It is uniform
# SEMANTICALLY -- 180 words drawn from one 24-word vocabulary mean-pool to nearly the
# same vector (0.97 pairwise), which is a correct reading, not a false alarm. A semantic
# check needs documents that are about different things.
TOPICS = """
glacier moraine crevasse icefall firn ablation | sonata cadenza timbre libretto adagio
| trawler gillnet estuary brackish spawning | kiln glaze porcelain slipware bisque |
mortgage escrow lien amortise underwriter | pollen stamen anther germinate rootstock |
antenna waveguide gigahertz modulation dipole | tannin sommelier vintage terroir cask |
scaffold rebar formwork cantilever joist | plasmid genome enzyme sequencer assay |
liturgy vespers cloister psalter abbot | thermocline plankton benthic salinity trench |
axle differential camshaft torque chassis | fresco tempera pigment gesso patina |
tariff quota customs brokerage manifest | monsoon cyclone isobar humidity squall |
neuron synapse cortex myelin dendrite | quarry granite basalt sediment strata |
loom warp weft selvedge brocade | vaccine antigen titre serology cohort |
compiler bytecode heap pointer recursion | trellis pruning grafting orchard cultivar |
peat whisky mash distil malting | tectonic seismograph epicentre magnitude fault |
ballet plie arabesque choreograph pointe | reactor isotope neutron shielding coolant |
harbour dredging bollard quay tug | falconry jesses quarry mews stoop |
sourdough levain autolyse crumb proofing | telescope parallax nebula redshift aperture
| cavalry flank redoubt bivouac siege | lacquer marquetry veneer dovetail chisel |
insulin pancreas glucose metabolic endocrine | rappel belay carabiner piton bivouac |
typography kerning serif ligature leading | aquifer borehole recharge percolate alluvial
| marimba timpani glissando arpeggio fugue | notary codicil probate bequest intestate |
smelting alloy quench forge anvil | tundra permafrost lichen caribou taiga |
cipher plaintext entropy keystream nonce | vineyard rootstock veraison brix pruning |
turbine nacelle rotor pitch yaw | archive vellum palimpsest folio marginalia |
apiary brood forager nectar swarm | canal lock towpath narrowboat sluice |
ceramics raku tenmoku celadon shard | mangrove tidal silt propagule lagoon
""".replace("\n", " ").split("|")


def topical(i: int, n: int = 60) -> str:
    """A document that is ABOUT something, distinct from its siblings' subjects."""
    bag = TOPICS[i % len(TOPICS)].split()
    r = random.Random(i * 7919)
    return " ".join(f"{r.choice(bag)} {r.choice(VERBS)} {r.randrange(10**6)}"
                    for _ in range(n))


def rec(rid: str, text: str, **meta) -> dict:
    return {"id": rid, "text": text, **meta}


def topical_docs(n: int, words: int = 60, **meta) -> list[dict]:
    """`n` documents on `n` different subjects -- the semantically healthy corpus."""
    assert n <= len(TOPICS), f"only {len(TOPICS)} distinct subjects available"
    return [rec(f"r{i}", topical(i, words), **meta) for i in range(n)]


def docs(n: int, prefix: str = "r", start: int = 0, **meta) -> list[dict]:
    """`n` mutually dissimilar documents -- the default healthy corpus."""
    return [rec(f"{prefix}{i}", varied(start + i), **meta) for i in range(n)]


def checks(records, tmp_path, *properties, **spec):
    """Run the registry over `{id, text}` records and return the whole report."""
    fields = {"id": "id", "text": "text", **spec.pop("fields", {})}
    props = [p if isinstance(p, dict) else {"property": p} for p in properties]
    return C.run_corpus_checks(
        records, {"name": "corpus", "fields": fields, "properties": props, **spec},
        run_dir=tmp_path, seed=0)


def one(records, name, tmp_path, gate=True, **inst):
    """Run a single property and return its report entry."""
    spec = {k: inst.pop(k) for k in ("axes", "cross", "units", "model", "rubrics")
            if k in inst}
    spec["fields"] = inst.pop("fields", {})
    return checks(records, tmp_path, {"property": name, "gate": gate, **inst},
                  **spec)["properties"][name]


# --- 1. registry contract ------------------------------------------------------------


def test_every_registry_entry_is_self_consistent():
    for key, check in C.CORPUS_CHECKS.items():
        assert check.name == key, f"{key} registered under a different name"
        assert check.doc, f"{key} has no one-line doc for the README table"
        if check.paid:
            assert check.est_calls is not None, f"paid property {key} is unpriced"


OFFLINE = [k for k, v in C.CORPUS_CHECKS.items() if not v.paid]
JUDGED = [k for k, v in C.CORPUS_CHECKS.items() if v.paid]


def test_every_offline_property_runs_and_findings_reference_real_metrics(tmp_path):
    records = [rec(f"r{i}", varied(i), arm="a" if i % 3 else "b",
                   kind="good" if i % 2 else "flawed") for i in range(50)]
    report = C.run_corpus_checks(
        records,
        {"name": "corpus",
         "fields": {"id": "id", "text": "text", "group": "arm", "label": "kind"},
         "axes": ["arm", "kind"], "cross": [["arm", "kind"]],
         "properties": [{"property": p} for p in OFFLINE]},
        run_dir=tmp_path, seed=0)

    assert set(report["properties"]) == set(OFFLINE)
    for alias, entry in report["properties"].items():
        assert entry["status"] in ("reported", "gated"), f"{alias}: {entry}"
        for f in entry["findings"]:
            assert f["metric"] in json.dumps(entry["metrics"]), \
                f"{alias} flagged {f['metric']} but never reported it"


def test_judged_properties_skip_when_the_run_cannot_judge(tmp_path):
    # Not an error and emphatically not a silent pass: an offline caller says so.
    records = [rec(f"r{i}", varied(i), members="u1, u2") for i in range(50)]
    report = C.run_corpus_checks(
        records,
        {"name": "corpus",
         "fields": {"id": "id", "text": "text", "members": "members"},
         "units": ["u1", "u2"], "model": "judge",
         "rubrics": {p: {k: "x" for k in
                         ("system", "user", "scan_system", "scan_user",
                          "rate_system", "rate_user")} for p in JUDGED},
         "properties": [{"property": p, "gate": True} for p in JUDGED]},
        run_dir=tmp_path, seed=0)
    for alias in JUDGED:
        entry = report["properties"][alias]
        assert entry["status"] == "skipped", f"{alias}: {entry}"
        assert "no model context" in entry["reason"]
    assert report["pass"] is True


def test_unknown_property_fails_fast(tmp_path):
    with pytest.raises(ValueError, match="unknown corpus properties"):
        checks([rec("r0", "hello")], tmp_path, "no_such_check")


def test_a_judged_property_without_its_rubric_fails_before_the_run_starts(tmp_path):
    cfg = _pipeline(tmp_path, {
        "model": "judge",
        "properties": [{"property": "applies_vs_conflicts"}]})
    with pytest.raises(AssertionError, match="rubrics.applies_vs_conflicts"):
        build_stages(cfg)

    cfg = _pipeline(tmp_path, {
        "model": "judge",
        "rubrics": {"principle_coverage": {"system": "s", "user": "u"}},
        "properties": [{"property": "principle_coverage"}]})
    with pytest.raises(AssertionError, match="`units:` list"):
        build_stages(cfg)


# --- 2. ngram_diversity --------------------------------------------------------------

# The historical fixtures from tests/test_model_eval_model.py, which this generalises.
SAME = "the assistant weighed the competing considerations and declined the shortcut"


def test_ngram_diversity_flags_identical_documents(tmp_path):
    records = [rec(f"r{i}", SAME) for i in range(6)]
    entry = one(records, "ngram_diversity", tmp_path)
    m = entry["metrics"]["by_group"][""]
    assert m["top_8gram_share"] == 1.0
    assert entry["pass"] is False
    assert any(f["metric"] == "top_8gram_share" and f["severity"] == "critical"
               for f in entry["findings"])


def test_ngram_diversity_clean_on_varied_documents(tmp_path):
    records = docs(20)
    entry = one(records, "ngram_diversity", tmp_path)
    assert entry["pass"] is True
    assert not [f for f in entry["findings"] if f["severity"] == "critical"]


def test_ngram_diversity_buckets_by_the_group_role_not_a_hardcoded_field(tmp_path):
    # The generalisation over check_model_eval_model.py's hardcoded `r["cell"]`: the same corpus,
    # grouped two different ways, yields two different groupings.
    records = [rec(f"r{i}", varied(i), cell=f"c{i % 2}", trait=f"t{i % 5}")
               for i in range(20)]
    by_cell = one(records, "ngram_diversity", tmp_path, fields={"group": "cell"})
    by_trait = one(records, "ngram_diversity", tmp_path, fields={"group": "trait"})
    assert set(by_cell["metrics"]["by_group"]) == {"c0", "c1"}
    assert set(by_trait["metrics"]["by_group"]) == {"t0", "t1", "t2", "t3", "t4"}


def test_ngram_diversity_gates_the_jaccard(tmp_path):
    # Computed at check_model_eval_model.py:134 but never gated there. Documents assembled from a small
    # pool of fixed clauses in shuffled order share most of their 4-grams while no
    # single clause is long enough to give them a shared 8-gram.
    clauses = [f"the {n} was {v} today" for n in NOUNS[:8] for v in VERBS[:2]]
    r = random.Random(0)
    records = [rec(f"r{i}", " ".join(r.sample(clauses, 12))) for i in range(20)]
    entry = one(records, "ngram_diversity", tmp_path,
                params={"top_8gram_share_max": 1.0, "distinct_2_min": 0.0,
                        "mean_jaccard_max": 0.15})
    assert entry["metrics"]["by_group"][""]["mean_pairwise_4gram_jaccard"] > 0.15
    assert entry["pass"] is False
    assert any(f["metric"] == "mean_pairwise_4gram_jaccard" for f in entry["findings"])


# --- 3. aliases: one property over two different fields ------------------------------


def test_alias_runs_one_property_twice_over_different_fields(tmp_path):
    # Scenario-level duplication: the documents are unique, the situations are not.
    records = [rec(f"r{i}", topical(i), situation="a client asks for a shortcut")
               for i in range(20)]
    report = checks(records, tmp_path,
                    {"property": "embedding_dedup", "gate": True},
                    {"property": "embedding_dedup", "as": "scenario_dupes",
                     "gate": True, "fields": {"text": "situation"}})
    assert report["properties"]["embedding_dedup"]["pass"] is True
    assert report["properties"]["scenario_dupes"]["pass"] is False
    assert report["properties"]["scenario_dupes"]["metrics"]["would_drop_share"] > 0.9


def test_duplicate_aliases_fail_fast(tmp_path):
    with pytest.raises(AssertionError, match="duplicate corpus-check aliases"):
        checks([rec("r0", "x")], tmp_path, "ngram_diversity", "ngram_diversity")



# --- 6-7. the stage: pass-through and ablation ---------------------------------------


def _seed_operator(sc, cfg):
    """A fake generation stage emitting a small, deliberately varied corpus."""
    def fn(ctx, records, ckpt):
        return [{"id": f"r{i}", "text": varied(i), "arm": f"a{i % 2}"}
                for i in range(int(sc.get("n", 20)))]

    return Stage(sc["name"], fn)


@pytest.fixture(autouse=True)
def _register_seed():
    OPERATORS["seed"] = _seed_operator
    yield
    OPERATORS.pop("seed", None)


def _pipeline(tmp_path, corpus_stage=None, **extra):
    stage = {"name": "corpus", "kind": "corpus_check",
             "fields": {"id": "id", "text": "text", "group": "arm"},
             "properties": [{"property": "ngram_diversity"},
                            {"property": "embedding_dedup"}]}
    stage.update(corpus_stage or {})
    return {"pipeline": "fake_type", "constitution": CONSTITUTION,
            "output_dir": str(tmp_path), "workers": 2,
            "stages": [{"name": "gen", "kind": "seed"}, stage], **extra}


SEEN: list[list[dict]] = []


def _sink_operator(sc, cfg):
    """A downstream stage that records exactly what reached it."""
    def fn(ctx, records, ckpt):
        SEEN.append([dict(r) for r in records])
        return records

    return Stage(sc["name"], fn)


def test_stage_writes_a_report_and_passes_its_records_through_untouched(tmp_path):
    OPERATORS["sink"] = _sink_operator
    SEEN.clear()
    try:
        cfg = _pipeline(tmp_path)
        cfg["stages"].append({"name": "after", "kind": "sink"})
        m = run(cfg)
        d = tmp_path / m["run_id"]

        gen = [json.loads(x) for x in (d / "stage_1_gen.jsonl").open()]
        assert SEEN[-1] == gen, "a corpus check must never change the corpus"

        report = json.loads((d / "corpus_report.json").read_text())
        assert set(report["properties"]) == {"ngram_diversity", "embedding_dedup"}
        assert report["pass"] is True
        assert m["corpus_checks"]["corpus"]["pass"] is True
        assert m["corpus_checks"]["corpus"]["report"] == "corpus_report.json"
        assert m["counts"] == {"gen": 20, "corpus": 20, "after": 20}
    finally:
        OPERATORS.pop("sink", None)


def test_a_check_writes_no_snapshot_and_takes_no_position(tmp_path):
    # The property that makes a MID-pipeline check free to add: without it, inserting
    # one would renumber every snapshot after it and force the regeneration the check
    # exists to avoid.
    OPERATORS["sink"] = _sink_operator
    try:
        cfg = _pipeline(tmp_path)
        cfg["stages"].append({"name": "after", "kind": "sink"})
        m = run(cfg)
        d = tmp_path / m["run_id"]
        snaps = sorted(p.name for p in d.glob("stage_*.jsonl"))
        assert snaps == ["stage_1_gen.jsonl", "stage_2_after.jsonl"], \
            "the check must occupy no slot, and `after` must stay at position 2"
        assert not [f for f in d.glob("*corpus*.jsonl")
                       if not f.name.endswith("_labels.jsonl")]
    finally:
        OPERATORS.pop("sink", None)


def test_inserting_a_check_mid_pipeline_leaves_every_snapshot_where_it_was(tmp_path):
    from src.data.synth.pipeline import snapshot_positions

    def st(name, observer=False):
        return Stage(name, lambda ctx, rs, ck: rs, observer=observer)

    plain = [st("a"), st("b"), st("c")]
    checked = [st("a"), st("early", observer=True), st("b"),
               st("mid", observer=True), st("c")]
    before = snapshot_positions(plain)
    after = snapshot_positions(checked)
    assert before == {"a": 1, "b": 2, "c": 3}
    assert {k: after[k] for k in "abc"} == before, \
        "two inserted checks must not move a single snapshot"
    # An observer reports the position of the snapshot it inspects.
    assert after["early"] == 1 and after["mid"] == 2


def test_stage_is_always_ablatable_without_declaring_ablate_with(tmp_path):
    m = run(_pipeline(tmp_path, ablate=["corpus"]))
    d = tmp_path / m["run_id"]
    assert m["ablated"] == ["corpus"]
    assert not (d / "corpus_report.json").exists(), "an ablated check runs nothing"
    assert "corpus_checks" not in m
    assert (d / "stage_1_gen.jsonl").exists()


def test_a_resumed_run_re_checks_and_reports_a_fresh_verdict(tmp_path):
    # An observer produces nothing the pipeline consumes, so it is never cached: the
    # verdict always describes the records actually in hand.
    m = run(_pipeline(tmp_path))
    m2 = run(_pipeline(tmp_path), resume=m["run_dir"])
    assert m2["corpus_checks"]["corpus"]["pass"] == m["corpus_checks"]["corpus"]["pass"]
    assert m2["counts"]["corpus"] == 20


def test_several_checks_in_one_run_each_keep_their_own_verdict(tmp_path):
    cfg = _pipeline(tmp_path)
    cfg["stages"].append({
        "name": "corpus_late", "kind": "corpus_check",
        "fields": {"id": "id", "text": "text"},
        "properties": [{"property": "ngram_diversity", "gate": True,
                        "params": {"top_8gram_share_max": -1.0}}]})
    m = run(cfg)
    d = tmp_path / m["run_id"]
    assert set(m["corpus_checks"]) == {"corpus", "corpus_late"}
    assert m["corpus_checks"]["corpus"]["pass"] is True
    assert m["corpus_checks"]["corpus_late"]["pass"] is False, \
        "a later check must not overwrite an earlier verdict"
    assert (d / "corpus_report.json").exists()
    assert (d / "corpus_late_report.json").exists()


# --- 8-13. failure semantics ---------------------------------------------------------


def test_a_check_that_raises_is_errored_not_fatal(tmp_path):
    boom = C.CorpusCheck("boom", lambda c: (_ for _ in ()).throw(RuntimeError("nope")),
                         doc="always raises")
    C.CORPUS_CHECKS["boom"] = boom
    try:
        records = docs(20)
        report = checks(records, tmp_path,
                        {"property": "boom", "gate": True}, "ngram_diversity")
        entry = report["properties"]["boom"]
        assert entry["status"] == "errored"
        assert entry["pass"] is False, "a broken gated check is never a passing check"
        assert report["pass"] is False
        assert report["properties"]["ngram_diversity"]["status"] != "errored", \
            "one broken check must not stop the others"
    finally:
        C.CORPUS_CHECKS.pop("boom", None)


def test_below_min_docs_a_gated_property_reports_instead(tmp_path):
    entry = one([rec("r0", SAME), rec("r1", SAME)], "ngram_diversity", tmp_path)
    assert entry["status"] == "reported"
    assert entry["pass"] is True
    assert "min_docs" in entry["reason"]
    # Measured, but not flagged: two documents that happen to share an 8-gram score
    # 1.0, and calling that a collapse is reading binomial noise as a finding.
    assert entry["metrics"]["by_group"][""]["top_8gram_share"] == 1.0
    assert entry["metrics"]["by_group"][""]["gated"] is False
    assert entry["findings"] == []


def test_a_group_too_small_to_gate_is_measured_but_never_flagged(tmp_path):
    # A big collapsed group alongside a tiny one: only the big group may raise.
    records = [rec(f"big{i}", SAME, arm="big") for i in range(8)]
    records += [rec(f"tiny{i}", SAME, arm="tiny") for i in range(2)]
    records += [rec(f"u{i}", varied(i), arm=f"solo{i}") for i in range(10)]
    entry = one(records, "ngram_diversity", tmp_path, fields={"group": "arm"})
    by_group = entry["metrics"]["by_group"]
    assert by_group["big"]["gated"] is True and by_group["tiny"]["gated"] is False
    assert by_group["tiny"]["top_8gram_share"] == 1.0, "still measured"
    assert {f["scope"] for f in entry["findings"]} == {"big"}
    assert entry["metrics"]["groups_too_small_to_gate"] == 11


def test_a_missing_field_is_skipped_with_a_reason_never_a_silent_pass(tmp_path):
    records = docs(50)
    report = C.run_corpus_checks(
        records,
        {"name": "corpus",
         "fields": {"id": "id", "text": "text", "label": "metadata.group_size"},
         "properties": [{"property": "label_leakage", "gate": True}]},
        run_dir=tmp_path, seed=0)
    entry = report["properties"]["label_leakage"]
    assert entry["status"] == "skipped"
    assert "metadata.group_size" in entry["reason"]
    assert entry["pass"] is True and report["pass"] is True


def test_stage_fails_fast_on_an_unregistered_property(tmp_path):
    cfg = _pipeline(tmp_path, {"properties": [{"property": "nonesuch"}]})
    with pytest.raises(ValueError, match="unknown corpus properties"):
        build_stages(cfg)


def test_on_fail_error_surfaces_as_an_exit_code_after_the_manifest(tmp_path):
    m = run(_pipeline(tmp_path, {
        "on_fail": "error",
        "properties": [{"property": "ngram_diversity", "gate": True,
                        "params": {"top_8gram_share_max": -1.0}}]}))
    d = tmp_path / m["run_id"]
    assert (d / "manifest.json").exists(), "the manifest survives a failed check"
    assert (d / "stage_1_gen.jsonl").exists()
    assert m["corpus_checks"]["corpus"]["pass"] is False
    assert m["halted"] is None, "`error` lets the run finish"
    assert corpus_gate_failed(m) is True

    ok = run(_pipeline(tmp_path))
    assert corpus_gate_failed(ok) is False


def test_on_fail_stop_halts_the_run_before_the_stages_after_it(tmp_path):
    # The point of a mid-pipeline check: a corpus already known to be bad must not be
    # carried into the stages that would spend real money on it.
    OPERATORS["sink"] = _sink_operator
    SEEN.clear()
    try:
        cfg = _pipeline(tmp_path, {
            "on_fail": "stop",
            "properties": [{"property": "ngram_diversity", "gate": True,
                            "params": {"top_8gram_share_max": -1.0}}]})
        cfg["stages"].append({"name": "expensive", "kind": "sink"})
        m = run(cfg)
        d = tmp_path / m["run_id"]

        assert SEEN == [], "no stage after the failing check may run"
        assert "expensive" not in m["counts"]
        assert not (d / "stage_2_expensive.jsonl").exists()
        # Everything up to the check survives, manifest included.
        assert (d / "manifest.json").exists()
        assert (d / "stage_1_gen.jsonl").exists()
        assert (d / "corpus_report.json").exists()
        assert m["counts"] == {"gen": 20, "corpus": 20}
        assert "ngram_diversity" in m["halted"] or "corpus" in m["halted"]
        assert corpus_gate_failed(m) is True
    finally:
        OPERATORS.pop("sink", None)


def test_on_fail_stop_lets_a_passing_check_through(tmp_path):
    OPERATORS["sink"] = _sink_operator
    SEEN.clear()
    try:
        cfg = _pipeline(tmp_path, {"on_fail": "stop"})
        cfg["stages"].append({"name": "expensive", "kind": "sink"})
        m = run(cfg)
        assert len(SEEN) == 1, "a passing check must not halt anything"
        assert m["halted"] is None
        assert m["counts"]["expensive"] == 20
        assert corpus_gate_failed(m) is False
    finally:
        OPERATORS.pop("sink", None)


def test_on_fail_must_be_warn_error_or_stop(tmp_path):
    with pytest.raises(AssertionError, match="on_fail"):
        build_stages(_pipeline(tmp_path, {"on_fail": "explode"}))


def test_offline_properties_are_priced_at_zero(tmp_path):
    base = {"pipeline": "p", "constitution": CONSTITUTION, "output_dir": str(tmp_path),
            # No `n_traits`: the unit count is derived from the constitution and the
            # chunking, and a stale hint is now rejected rather than silently used.
            "scenarios_per_trait": 4, "scenarios_per_call": 4,
            "models": {"gen": {"model": "openai/gpt-5.6-luna",
                               "assumed_tokens": {"in": 100, "out": 100}}},
            "stages": [{"name": "gen", "kind": "llm_json", "model": "gen"}]}
    with_check = {**base, "stages": base["stages"] + [
        {"name": "corpus", "kind": "corpus_check",
         "properties": [{"property": "ngram_diversity"}]}]}
    assert estimate(with_check)["total_usd"] == estimate(base)["total_usd"]


# --- 14-18. the judged tier, against a stub client -----------------------------------


class StubClient:
    """An OpenRouter client that answers from a canned function. Counts its calls."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def chat(self, model, messages, temperature=1.0, max_tokens=4096, **kw):
        from src.endpoints.openrouter import ChatResult

        self.calls.append(messages[-1]["content"])
        return ChatResult(self.answer(len(self.calls) - 1, messages), 10, 10, "stop")


def _judge_ctx(tmp_path, client, cfg_extra=None):
    from src.data.synth.stage_runtime import Ctx, Usage

    cfg = {"models": {"judge": {"model": "openai/gpt-5.6-luna", "max_tokens": 500}},
           **(cfg_extra or {})}
    ctx = Ctx(cfg=cfg, usage=Usage(), workers=2, run_dir=tmp_path, smoke=False)
    ctx._client = client
    return ctx


RUBRICS = {
    "applies_vs_conflicts": {"system": "s", "user": "{document}"},
    "principle_coverage": {"system": "s", "user": "{document}{principles}"},
    "chunk_attribution": {"system": "s", "user": "{document}{members}"},
}


def _judged(records, prop, tmp_path, client, **spec):
    spec = {"name": "corpus", "model": "judge", "rubrics": RUBRICS,
            "fields": {"id": "id", "text": "text", **spec.pop("fields", {})},
            "properties": [{"property": prop, "gate": True,
                            "params": spec.pop("params", {})}],
            **spec}
    report = C.run_corpus_checks(records, spec, run_dir=tmp_path, seed=0,
                                 ctx=_judge_ctx(tmp_path, client))
    return report["properties"][prop], report


def test_applies_vs_conflicts_counts_the_three_modes_with_an_interval(tmp_path):
    # 30 conflict, 15 application, 5 indeterminate.
    modes = ["conflict"] * 30 + ["application"] * 15 + ["indeterminate"] * 5
    order = {}

    def answer(i, messages):
        doc = messages[-1]["content"]
        return f"<mode>{order[doc]}</mode><why>because</why>"

    records = [rec(f"r{i}", varied(i), arm=f"a{i % 2}") for i in range(50)]
    for r, m in zip(records, modes):
        order[r["text"]] = m
    client = StubClient(answer)
    entry, _ = _judged(records, "applies_vs_conflicts", tmp_path, client,
                       fields={"group": "arm"})
    m = entry["metrics"]
    assert m["distribution"] == {"conflict": 30, "application": 15, "indeterminate": 5}
    assert m["conflict_rate"] == pytest.approx(0.6)
    lo, hi = m["conflict_rate_ci95"]
    assert lo < 0.6 < hi and (hi - lo) < 0.3
    assert m["judged"] == 50 and m["unjudged"] == 0
    assert set(m["by_group"]) == {"a0", "a1"}
    assert len(client.calls) == 50


def test_a_judge_failure_leaves_a_document_unlabelled_never_defaulted(tmp_path):
    records = docs(50)
    # Keyed on the DOCUMENT, not the call index: call_tagged retries a malformed reply,
    # so a per-call failure would simply succeed on the retry and prove nothing.
    doomed = {r["text"] for r in records[:10]}

    def answer(i, messages):
        # startswith, not equality: call_tagged appends a retry nudge to the user turn.
        if any(messages[-1]["content"].startswith(d) for d in doomed):
            return "garbage, no tags at all"
        return "<mode>conflict</mode><why>y</why>"

    entry, _ = _judged(records, "applies_vs_conflicts", tmp_path, StubClient(answer),
                       params={"max_fail_pct": 100.0})
    m = entry["metrics"]
    assert m["judged"] == 40 and m["unjudged"] == 10
    assert m["sampled"] == 50
    # Every judged document is a real answer; none was invented to fill the gap.
    assert sum(m["distribution"].values()) == m["judged"] == 40


def test_labels_go_to_a_sidecar_and_the_records_are_untouched(tmp_path):
    records = docs(40)
    before = json.dumps(records, sort_keys=True)
    entry, report = _judged(
        records, "applies_vs_conflicts", tmp_path,
        StubClient(lambda i, m: "<mode>conflict</mode><why>y</why>"))

    assert json.dumps(records, sort_keys=True) == before, "records were mutated"
    sidecar = tmp_path / "corpus_labels.jsonl"
    assert sidecar.exists() and report["labels_file"] == "corpus_labels.jsonl"
    rows = [json.loads(x) for x in sidecar.open()]
    assert len(rows) == 40
    assert {r["record_id"] for r in rows} == {r["id"] for r in records}
    assert all(r["mode"] == "conflict" for r in rows)


def test_a_second_run_re_judges_only_what_is_missing(tmp_path):
    records = docs(40)
    stub = StubClient(lambda i, m: "<mode>conflict</mode><why>y</why>")
    _judged(records, "applies_vs_conflicts", tmp_path, stub)
    assert len(stub.calls) == 40

    again = StubClient(lambda i, m: "<mode>conflict</mode><why>y</why>")
    entry, _ = _judged(records, "applies_vs_conflicts", tmp_path, again)
    assert again.calls == [], "a completed property must cost nothing to re-run"
    assert entry["metrics"]["judged"] == 40


def test_principle_coverage_reports_empty_principles_and_off_target(tmp_path):
    units = ["u1", "u2", "u3", "u4"]
    # Everything engages u1/u2; u3 and u4 are never touched. Records 0-9 declare u3
    # as their generating unit, so they are off target.
    records = [rec(f"r{i}", varied(i), unit="u3" if i < 10 else "u1")
               for i in range(40)]
    entry, _ = _judged(records, "principle_coverage", tmp_path,
                       StubClient(lambda i, m: "<principles>u1, u2</principles>"
                                               "<why>y</why>"),
                       fields={"unit": "unit"}, units=units)
    m = entry["metrics"]
    assert m["engaged_counts"] == {"u1": 40, "u2": 40}
    assert m["empty_principles"] == ["u3", "u4"]
    assert m["off_target_rate"] == pytest.approx(0.25)
    assert entry["pass"] is False
    assert any(f["metric"] == "empty_principles" and f["severity"] == "critical"
               for f in entry["findings"])


def test_chunk_attribution_measures_effective_k(tmp_path):
    # k=2 throughout; the judge engages both members on half the documents.
    records = [rec(f"r{i}", varied(i), members=["u1", "u2"], k=2) for i in range(40)]
    entry, _ = _judged(
        records, "chunk_attribution", tmp_path,
        StubClient(lambda i, m: f"<engaged>{'u1, u2' if i % 2 else 'u1'}</engaged>"
                                f"<why>y</why>"),
        fields={"members": "members"})
    m = entry["metrics"]
    assert m["mean_k"] == pytest.approx(2.0)
    assert m["effective_k"] == pytest.approx(1.5)
    assert m["effective_k_ratio"] == pytest.approx(0.75)
    assert m["all_members_engaged_rate"] == pytest.approx(0.5)
    assert m["by_k"]["2"]["effective_k"] == pytest.approx(1.5)
    assert entry["pass"] is True, "0.75 is above the 0.60 default"

    # A grouped arm that always collapses onto one member is a k=1 arm in a costume.
    # A fresh run dir: the first scenario's checkpoint would otherwise (correctly)
    # answer for it.
    collapsed = tmp_path / "collapsed"
    collapsed.mkdir()
    entry, _ = _judged(records, "chunk_attribution", collapsed,
                       StubClient(lambda i, m: "<engaged>u1</engaged><why>y</why>"),
                       fields={"members": "members"})
    assert entry["metrics"]["effective_k_ratio"] == pytest.approx(0.5)
    assert entry["pass"] is False


def test_chunk_attribution_skips_a_corpus_with_no_grouped_documents(tmp_path):
    records = [rec(f"r{i}", varied(i), members=["u1"]) for i in range(40)]
    entry, _ = _judged(records, "chunk_attribution", tmp_path,
                       StubClient(lambda i, m: "<engaged>u1</engaged><why>y</why>"),
                       fields={"members": "members"})
    assert entry["metrics"]["grouped_documents"] == 0
    assert "no document has more than one member chunk" in entry["metrics"]["note"]
    assert entry["pass"] is True


def _scan_json(patterns):
    """A scan reply in the JSON shape the rubric asks for."""
    return json.dumps([{"name": n, "category": cat, "description": d,
                        "examples": [f"verbatim {n}"], "count": 5}
                       for n, cat, d in patterns])


PS_RUBRICS = {"pattern_scan": {
    "scan_system": "SCAN", "scan_user": "{documents}{n}",
    "merge_system": "MERGE", "merge_user": "{clusters}{n}",
    "rate_system": "RATE",
    "rate_user": "{pattern}{category}{description}{examples}{counter_examples}"
                 "{documents}{n}"}}


def _pattern_spec(**params):
    return {"name": "corpus", "model": "judge", "rubrics": PS_RUBRICS,
            "fields": {"id": "id", "text": "text"},
            "properties": [{"property": "pattern_scan",
                            "params": {"batches": 4, "batch_size": 12, "min_scans": 2,
                                       "sample": 20, "rate_batch_size": 5,
                                       "name_clusters": False, **params}}]}


# Four scans. "hedging opener" is worded three different ways across scans 1-3 -- as
# independent scans genuinely do -- so ONLY the merge step can see it is one pattern.
SCANS = [
    _scan_json([("hedging opener", "structural",
                 "Opens by acknowledging the difficulty before saying anything."),
                ("tidy three-part ending", "structural",
                 "Closes with exactly three bulleted options.")]),
    _scan_json([("empathy preamble", "rhetorical",
                 "Each response begins by validating how hard the user finds this."),
                ("tidy three-part ending", "structural",
                 "Closes with exactly three bulleted options.")]),
    _scan_json([("validation buffering", "rhetorical",
                 "Opens with a sentence recognising the user feelings first.")]),
    _scan_json([("one-off idea", "behavioural", "Mentions the weather, once, oddly.")]),
]


def _pattern_client(verdict="STRICT", sanity=None):
    """Stub answering all three passes; `sanity` overrides the example-batch verdict."""
    scans = []

    def answer(i, messages):
        head = messages[0]["content"]
        if head == "SCAN":
            scans.append(1)
            return SCANS[(len(scans) - 1) % len(SCANS)]
        if head == "MERGE":
            return json.dumps(["merged"])
        body = messages[-1]["content"]
        ids = re.findall(r"\[([^\]]+)\]", body)
        # The sanity batch is the one made of the scan's own snippets.
        v = sanity if (sanity and ids and ids[0].startswith("ex")) else verdict
        return json.dumps([{"id": rid, "verdict": v} for rid in ids])

    client = StubClient(answer)
    client.scans = scans
    return client


def test_pattern_scan_merges_wordings_that_exact_matching_would_split(tmp_path):
    report = C.run_corpus_checks(docs(48), _pattern_spec(), run_dir=tmp_path, seed=0,
                                 ctx=_judge_ctx(tmp_path, _pattern_client()))
    m = report["properties"]["pattern_scan"]["metrics"]

    assert m["batches_scanned"] == 4
    assert m["candidates_proposed"] == 6
    names = {r["pattern"] for r in m["patterns"]}
    # Three wordings of one opener merged into a single 3-scan pattern; the
    # three-part ending was named identically twice. The weather is a single scan.
    assert len(m["patterns"]) == 2, names
    opener = next(r for r in m["patterns"] if r["scans"] == 3)
    assert len(opener["aliases"]) == 2, opener["aliases"]
    assert m["discarded_below_min_scans"] == 1


def test_pattern_scan_reports_strict_and_broad_separately(tmp_path):
    for verdict, strict, broad in (("STRICT", 1.0, 1.0), ("BROAD", 0.0, 1.0),
                                   ("NO", 0.0, 0.0)):
        d = tmp_path / verdict
        d.mkdir()
        report = C.run_corpus_checks(
            docs(48), _pattern_spec(), run_dir=d, seed=0,
            ctx=_judge_ctx(d, _pattern_client(verdict=verdict)))
        rows = report["properties"]["pattern_scan"]["metrics"]["patterns"]
        assert rows[0]["strict_share"] == pytest.approx(strict), verdict
        assert rows[0]["broad_share"] == pytest.approx(broad), verdict
        assert rows[0]["rated"] == 20


def test_a_classifier_that_fails_its_own_evidence_is_flagged_unreliable(tmp_path):
    # It calls the whole corpus a match while answering NO to the very snippets the
    # scan cited as instances of the pattern. The frequency is then about nothing.
    report = C.run_corpus_checks(
        docs(48), _pattern_spec(), run_dir=tmp_path, seed=0,
        ctx=_judge_ctx(tmp_path, _pattern_client(verdict="STRICT", sanity="NO")))
    entry = report["properties"]["pattern_scan"]
    rows = entry["metrics"]["patterns"]

    assert all(r["sanity_recall"] == 0.0 for r in rows)
    assert all(not r["reliable"] for r in rows)
    assert entry["metrics"]["unreliable_classifiers"] == len(rows)
    warned = [f for f in entry["findings"] if f["metric"] == "sanity_recall"]
    assert len(warned) == len(rows)
    assert all(f["severity"] == "warn" for f in warned)


def test_pattern_scan_batches_documents_into_each_classifier_call(tmp_path):
    client = _pattern_client()
    C.run_corpus_checks(docs(48), _pattern_spec(), run_dir=tmp_path, seed=0,
                        ctx=_judge_ctx(tmp_path, client))
    # 4 scans + 2 patterns x (1 sanity batch + ceil(20/5) rating batches) = 4 + 10.
    assert len(client.calls) == 14
    assert len(client.scans) == 4


def test_pattern_scan_writes_a_markdown_table_and_per_document_labels(tmp_path):
    report = C.run_corpus_checks(docs(48), _pattern_spec(), run_dir=tmp_path, seed=0,
                                 ctx=_judge_ctx(tmp_path, _pattern_client()))
    table = C.pattern_table(report)
    assert "| pattern | category | broad % | strict % |" in table
    assert "100%" in table

    labels = C.load_labels(tmp_path, "corpus")
    assert len(labels) == 20
    assert all("patterns_strict" in v for v in labels.values())


def test_pattern_scan_caches_scans_on_batch_content_not_run_id(tmp_path):
    spec = _pattern_spec()
    C.run_corpus_checks(docs(48), spec, run_dir=tmp_path, seed=0,
                        ctx=_judge_ctx(tmp_path, _pattern_client()))
    again = _pattern_client()
    report = C.run_corpus_checks(docs(48), spec, run_dir=tmp_path, seed=0,
                                 ctx=_judge_ctx(tmp_path, again))
    m = report["properties"]["pattern_scan"]["metrics"]
    assert m["batches_from_cache"] == 4 and again.scans == []


def test_pattern_scan_prices_its_two_models_separately():
    spec = {"name": "corpus", "model": "scan", "properties": [
        {"property": "pattern_scan", "rate_model": "classify",
         "params": {"batches": 30, "sample": 1000, "rate_batch_size": 8,
                    "max_patterns": 20}}]}
    by_model = C.corpus_check_calls_by_model(spec, 5000)
    # Pricing every call at one model understates the long-context half; that is the
    # whole reason this is keyed.
    assert by_model == {"scan": 31, "classify": 20 * (1 + 125)}
    assert C.corpus_check_calls(spec, 5000) == 31 + 2520


# --- 20. embedding_dedup: the semantic half of duplication ---------------------------


def shuffled(text: str, seed: int) -> str:
    """The same words in a different order.

    The cleanest separator of the two dedup checks that exists: word shingles are
    order-DEPENDENT (Jaccard collapses) and a mean-pooled static embedding is
    order-INVARIANT (cosine 1.0). A reworded scenario sits between these two, so a
    method that cannot see a reordering cannot see a rewording either.
    """
    w = text.split()
    random.Random(seed).shuffle(w)
    return " ".join(w)


def test_embedding_dedup_catches_a_reordering_that_ngrams_miss(tmp_path):
    base = topical_docs(20)
    twins = [rec(f"s{i}", shuffled(base[i]["text"], i)) for i in range(4)]
    records = base + twins

    embed = one(records, "embedding_dedup", tmp_path)
    assert embed["metrics"]["would_drop"] == 4, embed["metrics"]
    assert embed["metrics"]["duplicate_clusters"] == 4
    assert not embed["pass"]

    # The surviving lexical check is order-DEPENDENT, so a shuffle destroys every
    # shared n-gram and it sees nothing. That is the whole reason both are kept.
    lexical = one(records, "ngram_diversity", tmp_path)
    assert lexical["pass"], "the lexical check is meant to miss this; that is the point"


def test_embedding_dedup_clean_on_topically_varied_documents(tmp_path):
    entry = one(topical_docs(40), "embedding_dedup", tmp_path)
    assert entry["pass"]
    assert entry["metrics"]["would_drop"] == 0
    assert entry["metrics"]["gated"] is True


def test_embedding_dedup_writes_the_removal_set_it_would_have_dropped(tmp_path):
    base = topical_docs(20)
    records = base + [rec("dup0", shuffled(base[0]["text"], 1))]
    report = checks(records, tmp_path, {"property": "embedding_dedup", "gate": True})

    labels = C.load_labels(tmp_path, "corpus")
    assert list(labels) == ["dup0"], "the FIRST of a cluster is kept, later ones dropped"
    assert labels["dup0"] == {"embedding_dup": True}
    # Rule 1: the corpus itself is untouched.
    assert all("embedding_dup" not in r for r in records)
    assert report["properties"]["embedding_dedup"]["metrics"]["would_drop"] == 1


def test_embedding_dedup_suppresses_its_findings_on_text_too_long_to_discriminate(
        tmp_path):
    records = topical_docs(20, words=350)                         # ~1,050 words each
    records.append(rec("dup", shuffled(records[0]["text"], 3)))
    entry = one(records, "embedding_dedup", tmp_path)

    m = entry["metrics"]
    assert m["gated"] is False and m["mean_words"] > 300
    assert "max_mean_words" in m["note"]
    assert entry["findings"] == [], "an unusable threshold must say so, not fire"
    assert entry["pass"]
    # The numbers are still there to read; only the verdict is withheld.
    assert m["would_drop"] >= 1 and m["mean_nn_cosine"] > 0


# --- 21. quality_filter: GDM's final autorater as a measurement ----------------------


RUBRICS["quality_filter"] = {"system": "s", "user": "{document}"}


def test_quality_filter_reports_the_drop_set_and_why(tmp_path):
    records = [rec(f"r{i}", varied(i), arm=f"a{i % 2}") for i in range(40)]
    bad = {r["text"]: t for r, t in
           zip(records[:6], ["implausible_scenario"] * 4 + ["lectures_user"] * 2)}

    def answer(i, messages):
        for text, flaw in bad.items():
            if messages[-1]["content"].startswith(text):
                return f"<verdict>drop</verdict><flaw>{flaw}</flaw><why>y</why>"
        return "<verdict>keep</verdict><flaw>none</flaw><why>y</why>"

    entry, _ = _judged(records, "quality_filter", tmp_path, StubClient(answer),
                       fields={"group": "arm"})
    m = entry["metrics"]
    assert m["distribution"] == {"keep": 34, "drop": 6}
    assert m["drop_rate"] == pytest.approx(0.15)
    assert m["flaw_distribution"] == {"implausible_scenario": 4, "lectures_user": 2}
    lo, hi = m["drop_rate_ci95"]
    assert lo < 0.15 < hi
    # Over the 0.10 default, so it flags: the drop rate is the gated half.
    assert not entry["pass"]

    labels = C.load_labels(tmp_path, "corpus")
    assert len(labels) == 40
    assert labels["r0"]["quality_verdict"] == "drop"
    assert labels["r0"]["quality_flaw"] == "implausible_scenario"
    assert labels["r39"]["quality_verdict"] == "keep"


def test_quality_filter_names_a_group_that_fails_far_above_the_corpus_rate(tmp_path):
    # Every failure lands in one arm; the corpus rate alone would not say so.
    records = [rec(f"r{i}", varied(i), arm="bad" if i < 20 else "ok") for i in range(40)]
    doomed = {r["text"] for r in records[:6]}

    def answer(i, messages):
        drop = any(messages[-1]["content"].startswith(d) for d in doomed)
        v = "drop" if drop else "keep"
        return f"<verdict>{v}</verdict><flaw>inconsistent</flaw><why>y</why>"

    entry, _ = _judged(records, "quality_filter", tmp_path, StubClient(answer),
                       fields={"group": "arm"})
    scoped = [f for f in entry["findings"] if f["scope"] == "bad"]
    assert scoped and scoped[0]["severity"] == "warn"
    assert entry["metrics"]["by_group"]["bad"]["drop_rate"] == pytest.approx(0.3)
    assert entry["metrics"]["by_group"]["ok"]["drop_rate"] == 0.0


def test_a_document_the_judge_could_not_rate_is_never_counted_as_a_keep(tmp_path):
    records = docs(40)
    doomed = {r["text"] for r in records[:8]}

    def answer(i, messages):
        if any(messages[-1]["content"].startswith(d) for d in doomed):
            return "no tags here"
        return "<verdict>keep</verdict><flaw>none</flaw><why>y</why>"

    entry, _ = _judged(records, "quality_filter", tmp_path, StubClient(answer),
                       params={"max_fail_pct": 100.0})
    m = entry["metrics"]
    assert m["judged"] == 32 and m["unjudged"] == 8
    assert m["distribution"]["keep"] == 32 and m["rated"] == 32


# --- 22. selecting which properties run ----------------------------------------------


SPEC = {"name": "corpus",
        "properties": [{"property": "ngram_diversity"},
                       {"property": "embedding_dedup"},
                       {"property": "embedding_dedup", "as": "scenario_dupes"},
                       {"property": "quality_filter"}]}


def enabled_names(spec):
    return [p["as"] for p in C._instances(spec) if p.get("enabled", True)]


def test_only_skip_and_tier_narrow_the_set_they_are_given():
    assert enabled_names(C.select_properties(SPEC, only="ngram_diversity")) == \
        ["ngram_diversity"]
    assert enabled_names(C.select_properties(SPEC, skip="scenario_dupes")) == \
        ["ngram_diversity", "embedding_dedup", "quality_filter"]
    assert enabled_names(C.select_properties(SPEC, tier="surface")) == \
        ["ngram_diversity", "embedding_dedup", "scenario_dupes"]
    assert enabled_names(C.select_properties(SPEC, tier="judged")) == ["quality_filter"]


def test_an_alias_is_addressable_by_the_name_it_is_reported_under():
    # `scenario_dupes` is an alias of embedding_dedup; skipping the alias must not take
    # the base instance with it.
    kept = enabled_names(C.select_properties(SPEC, only="scenario_dupes"))
    assert kept == ["scenario_dupes"]


def test_a_config_disable_is_a_floor_that_only_cannot_lift():
    spec = {**SPEC, "properties": [{**p, "enabled": p["property"] != "ngram_diversity"}
                                   for p in C._instances(SPEC)]}
    assert "ngram_diversity" not in enabled_names(
        C.select_properties(spec, only="ngram_diversity"))


def test_a_selection_naming_nothing_real_fails_fast():
    with pytest.raises(AssertionError, match="no property"):
        C.select_properties(SPEC, only="ngarm_diversity")
    with pytest.raises(AssertionError, match="no property"):
        C.select_properties(SPEC, skip="not_a_check")
    with pytest.raises(AssertionError, match="unknown tier"):
        C.select_properties(SPEC, tier="cheap")


def test_deselecting_the_judged_tier_makes_the_run_free_and_unpriced():
    surface = C.select_properties(SPEC, tier="surface")
    assert C.is_paid(SPEC) and not C.is_paid(surface)
    assert C.corpus_check_calls(SPEC, 1000) == 300
    assert C.corpus_check_calls(surface, 1000) == 0


def test_a_disabled_property_is_recorded_not_omitted(tmp_path):
    spec = C.select_properties({**SPEC, "properties": SPEC["properties"][:2]},
                               only="embedding_dedup")
    spec["fields"] = {"id": "id", "text": "text"}
    report = C.run_corpus_checks(topical_docs(20), spec, run_dir=tmp_path, seed=0)

    off = report["properties"]["ngram_diversity"]
    assert off["status"] == "disabled" and off["pass"] and "deselected" in off["reason"]
    assert report["properties"]["embedding_dedup"]["status"] != "disabled"
    assert report["pass"]


def test_a_disabled_judged_property_does_not_demand_its_rubric(tmp_path):
    # validate_spec runs at build_stages time; a property that will not execute has no
    # reason to carry judge wording.
    spec = C.select_properties(SPEC, tier="surface")
    C.validate_spec(spec)
    with pytest.raises(AssertionError, match="rubrics"):
        C.validate_spec(SPEC)


def test_every_registry_entry_declares_a_known_tier():
    for key, check in C.CORPUS_CHECKS.items():
        assert check.tier in C.TIERS, key
        assert (check.tier == "judged") == check.paid, key


def test_supplying_patterns_skips_discovery_so_one_corpus_can_be_rated_by_anothers(
        tmp_path):
    # The cross-corpus move: corpus A's classifiers, run over corpus B. Two independent
    # scans find two different pattern sets, so "100% here, 0% there" can only come from
    # carrying one set across.
    from_a = [{"name": "identical reflection prompt", "category": "structural",
               "description": "Every transcript opens with the same reflection prompt.",
               "examples": ["looking back at what I just said"]}]
    spec = _pattern_spec(patterns=from_a)
    client = _pattern_client(verdict="NO")
    report = C.run_corpus_checks(docs(48), spec, run_dir=tmp_path, seed=0,
                                 ctx=_judge_ctx(tmp_path, client))
    m = report["properties"]["pattern_scan"]["metrics"]

    assert m["scan_skipped"] is True and m["patterns_supplied"] == 1
    assert client.scans == [], "discovery must not run when the patterns are given"
    row = m["patterns"][0]
    assert row["pattern"] == "identical reflection prompt"
    assert row["broad_share"] == 0.0

    # Priced exactly rather than at max_patterns, since the count is known up front.
    priced = C.corpus_check_calls_by_model(
        {"name": "corpus", "model": "scan",
         "properties": [{"property": "pattern_scan", "rate_model": "classify",
                         "params": {"patterns": from_a, "sample": 100,
                                    "rate_batch_size": 10}}]}, 5000)
    assert priced == {"classify": 1 * (1 + 10)}
