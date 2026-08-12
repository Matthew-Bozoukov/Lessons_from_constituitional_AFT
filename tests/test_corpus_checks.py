# ABOUTME: Offline tests for the corpus-level check registry and its pipeline stage:
# ABOUTME: every property, the pass-through invariant, ablation, and failure semantics.

from __future__ import annotations

import json
import random

import pytest

from src.data.synth import corpus as C
from src.data.synth.core import Stage
from src.data.synth.operators import OPERATORS
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


def rec(rid: str, text: str, **meta) -> dict:
    return {"id": rid, "text": text, **meta}


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
    # The generalisation over checks.py's hardcoded `r["cell"]`: the same corpus,
    # grouped two different ways, yields two different groupings.
    records = [rec(f"r{i}", varied(i), cell=f"c{i % 2}", trait=f"t{i % 5}")
               for i in range(20)]
    by_cell = one(records, "ngram_diversity", tmp_path, fields={"group": "cell"})
    by_trait = one(records, "ngram_diversity", tmp_path, fields={"group": "trait"})
    assert set(by_cell["metrics"]["by_group"]) == {"c0", "c1"}
    assert set(by_trait["metrics"]["by_group"]) == {"t0", "t1", "t2", "t3", "t4"}


def test_ngram_diversity_gates_the_jaccard(tmp_path):
    # Computed at checks.py:134 but never gated there. Documents assembled from a small
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


# --- 3. near_duplicates --------------------------------------------------------------


def test_near_duplicates_finds_exact_and_near_copies(tmp_path):
    records = [rec(f"u{i}", varied(i)) for i in range(10)]
    records += [rec(f"dup{i}", varied(0)) for i in range(3)]          # 3 exact copies
    records += [rec(f"near{i}", varied(1) + " trailing addendum clause")
                for i in range(2)]                                    # 2 near copies
    entry = one(records, "near_duplicates", tmp_path)
    m = entry["metrics"]

    assert m["exact_duplicate_groups"] == 2, "the u0/dup* group and the near* pair"
    assert m["near_duplicate_pairs"] >= 1
    # u0 + 3 copies + u1 + 2 near copies = 7 of 15
    assert m["duplicate_share"] == pytest.approx(7 / 15, abs=1e-3)
    assert entry["pass"] is False
    flagged = {e for f in entry["findings"] for e in f["examples"]}
    assert flagged & {"dup0", "dup1", "dup2"}


def test_near_duplicates_clean_on_unique_documents(tmp_path):
    records = docs(20)
    entry = one(records, "near_duplicates", tmp_path)
    assert entry["metrics"]["near_duplicate_pairs"] == 0
    assert entry["metrics"]["exact_duplicate_groups"] == 0
    assert entry["pass"] is True


def test_alias_runs_one_property_twice_over_different_fields(tmp_path):
    # Scenario-level near-duplication: the documents are unique, the situations are not.
    records = [rec(f"r{i}", varied(i), situation="a client asks for a shortcut")
               for i in range(20)]
    report = checks(records, tmp_path,
                    {"property": "near_duplicates", "gate": True},
                    {"property": "near_duplicates", "as": "scenario_dupes",
                     "gate": True, "fields": {"text": "situation"}})
    assert report["properties"]["near_duplicates"]["pass"] is True
    assert report["properties"]["scenario_dupes"]["pass"] is False
    assert report["properties"]["scenario_dupes"]["metrics"]["duplicate_share"] == 1.0


def test_duplicate_aliases_fail_fast(tmp_path):
    with pytest.raises(AssertionError, match="duplicate corpus-check aliases"):
        checks([rec("r0", "x")], tmp_path, "near_duplicates", "near_duplicates")


# --- 4. opening_collapse / length_profile / field_balance ----------------------------


def test_opening_collapse_counts_shared_openings(tmp_path):
    opener = "I need to think carefully about this before answering here"
    records = [rec(f"s{i}", f"{opener} {varied(i)}") for i in range(6)]
    records += [rec(f"u{i}", varied(100 + i)) for i in range(14)]
    entry = one(records, "opening_collapse", tmp_path)
    m = entry["metrics"]
    assert m["top_opener_share"] == pytest.approx(6 / 20)
    assert m["shared_opening_share"] == pytest.approx(6 / 20)
    assert m["top_opener"] == " ".join(C.words(opener)[:8])
    assert entry["pass"] is False


def test_opening_collapse_catches_a_reordered_construction(tmp_path):
    # Three wordings of one construction, none common enough to notice on its own.
    # Measured on a real corpus: 155 of 1,389 documents, reported as three
    # unremarkable openers by exact matching.
    wordings = ["let me actually look at what's in front of me",
                "let me look at what's actually in front of me",
                "actually let me look at what's in front of me"]
    records = [rec(f"s{i}", f"{wordings[i % 3]} {varied(i)}") for i in range(9)]
    records += [rec(f"u{i}", varied(100 + i)) for i in range(11)]
    entry = one(records, "opening_collapse", tmp_path)
    m = entry["metrics"]
    assert m["top_opener_share"] == pytest.approx(3 / 20), "no single wording stands out"
    assert m["top_opener_share_reordered"] == pytest.approx(9 / 20)
    assert entry["pass"] is False
    assert [f["metric"] for f in entry["findings"]] == ["top_opener_share_reordered"]


def test_length_profile_flags_uniform_lengths_and_group_skew(tmp_path):
    uniform = [rec(f"r{i}", varied(i, n=50)) for i in range(20)]
    entry = one(uniform, "length_profile", tmp_path)
    assert entry["metrics"]["cv"] < 0.20
    assert any(f["metric"] == "cv" for f in entry["findings"])
    assert entry["pass"] is True, "cv is a warn, not a critical -- it never gates"

    skewed = ([rec(f"a{i}", varied(i, n=20), arm="a") for i in range(10)]
              + [rec(f"b{i}", varied(100 + i, n=80), arm="b") for i in range(10)])
    entry = one(skewed, "length_profile", tmp_path, fields={"group": "arm"})
    deltas = {g: v["delta_vs_corpus"]
              for g, v in entry["metrics"]["by_group"].items()}
    assert deltas["a"] > 0.35 and deltas["b"] > 0.35
    assert {f["scope"] for f in entry["findings"] if f["metric"] == "delta_vs_corpus"} \
        == {"a", "b"}


def test_field_balance_reports_entropy_and_empty_cross_buckets(tmp_path):
    # arm x kind is a full 2x2 except that b/flawed never occurs.
    records = ([rec(f"a{i}", varied(i), arm="a", kind="good" if i % 2 else "flawed")
                for i in range(16)]
               + [rec(f"b{i}", varied(50 + i), arm="b", kind="good") for i in range(4)])
    entry = one(records, "field_balance", tmp_path,
                axes=["arm", "kind"], cross=[["arm", "kind"]])
    m = entry["metrics"]
    assert m["axes"]["arm"]["counts"] == {"a": 16, "b": 4}
    assert m["axes"]["arm"]["max_share"] == pytest.approx(0.8)
    assert m["axes"]["arm"]["normalized_entropy"] < 0.85
    assert m["cross"]["arm/kind"]["empty"] == ["b/flawed"]
    assert entry["pass"] is False


def test_field_balance_reads_judged_labels_as_an_axis(tmp_path):
    records = docs(20)
    corpus = C.Corpus(records=records, fields={"id": "id", "text": "text"}, params={},
                      labels={f"r{i}": {"principles": ["p1", "p2"] if i else ["p1"]}
                              for i in range(20)})
    assert corpus.column("label.principles")[0] == ["p1"]
    assert corpus.column("label.principles")[1] == ["p1", "p2"]


# --- 5. feature_diversity ------------------------------------------------------------


def test_feature_diversity_separates_identical_from_varied(tmp_path):
    identical = [rec(f"r{i}", SAME) for i in range(30)]
    entry = one(identical, "feature_diversity", tmp_path)
    assert entry["metrics"]["mean_pairwise_cosine"] == pytest.approx(1.0, abs=1e-4)
    assert entry["metrics"]["effective_rank"] == pytest.approx(1.0, abs=0.05)
    assert entry["pass"] is False

    entry = one(docs(30),
                "feature_diversity", tmp_path)
    assert entry["pass"] is True
    # Effective rank is the half that discriminates: cosine has a high floor because
    # same-language prose shares so many character n-grams (0.86 measured on a real
    # 2,203-document corpus), so it barely moves between these two fixtures.
    assert entry["metrics"]["effective_rank_frac"] > 0.25
    assert entry["metrics"]["mean_pairwise_cosine"] > 0.5


def test_reports_are_deterministic_at_a_fixed_seed(tmp_path):
    records = [rec(f"r{i}", varied(i), arm=f"a{i % 3}") for i in range(40)]
    spec = {"name": "corpus",
            "fields": {"id": "id", "text": "text", "group": "arm"},
            "axes": ["arm"], "properties": [{"property": p} for p in OFFLINE]}

    def once(d):
        r = C.run_corpus_checks(records, spec, run_dir=d, seed=7)
        r.pop("checked_at")
        return json.dumps(r, sort_keys=True)

    assert once(tmp_path) == once(tmp_path)


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
                            {"property": "near_duplicates"}]}
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
        assert set(report["properties"]) == {"ngram_diversity", "near_duplicates"}
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
        assert not list(d.glob("*corpus*.jsonl"))
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
        "properties": [{"property": "near_duplicates", "gate": True,
                        "params": {"dup_share_max": -1.0}}]})
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
                        {"property": "boom", "gate": True}, "near_duplicates")
        entry = report["properties"]["boom"]
        assert entry["status"] == "errored"
        assert entry["pass"] is False, "a broken gated check is never a passing check"
        assert report["pass"] is False
        assert report["properties"]["near_duplicates"]["status"] != "errored", \
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
        "properties": [{"property": "near_duplicates", "gate": True,
                        "params": {"dup_share_max": -1.0}}]}))
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
            "properties": [{"property": "near_duplicates", "gate": True,
                            "params": {"dup_share_max": -1.0}}]})
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
        assert "near_duplicates" in m["halted"] or "corpus" in m["halted"]
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
            "n_traits": 2, "scenarios_per_trait": 4, "scenarios_per_call": 4,
            "models": {"gen": {"model": "openai/gpt-5.6-luna",
                               "assumed_tokens": {"in": 100, "out": 100}}},
            "stages": [{"name": "gen", "kind": "llm_json", "model": "gen"}]}
    with_check = {**base, "stages": base["stages"] + [
        {"name": "corpus", "kind": "corpus_check",
         "properties": [{"property": "near_duplicates"}]}]}
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
    from src.data.synth.core import Ctx, Usage

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


def test_pattern_scan_keeps_only_patterns_two_scans_found_and_caches_on_content(tmp_path):
    scan_replies = [
        "<patterns>hedging opener; rule-quoting; tidy three-part ending</patterns>",
        "<patterns>rule-quoting; sudden pivot</patterns>",
        "<patterns>Rule-quoting; tidy three-part ending</patterns>",
        "<patterns>one-off idea</patterns>",
    ]

    def answer(i, messages):
        if "SCAN" in messages[0]["content"]:
            return scan_replies[len([c for c in scanned if c]) % len(scan_replies)]
        return "<matches>rule-quoting</matches><why>y</why>"

    scanned = []

    def answer2(i, messages):
        if "SCAN" in messages[0]["content"]:
            scanned.append(1)
            return scan_replies[(len(scanned) - 1) % len(scan_replies)]
        return "<matches>rule-quoting</matches><why>y</why>"

    rubrics = {"pattern_scan": {
        "scan_system": "SCAN", "scan_user": "{documents}{n}{seeded}",
        "rate_system": "RATE", "rate_user": "{document}{patterns}"}}
    records = docs(48)
    spec = {"name": "corpus", "model": "judge", "rubrics": rubrics,
            "fields": {"id": "id", "text": "text"},
            "properties": [{"property": "pattern_scan",
                            "params": {"batches": 4, "batch_size": 12,
                                       "min_scans": 2, "sample": 20}}]}

    stub = StubClient(answer2)
    report = C.run_corpus_checks(records, spec, run_dir=tmp_path, seed=0,
                                 ctx=_judge_ctx(tmp_path, stub))
    m = report["properties"]["pattern_scan"]["metrics"]
    assert m["batches_scanned"] == 4
    # "rule-quoting" (3 scans) and "tidy three-part ending" (2) survive; the
    # single-scan ideas do not.
    assert len(m["survivors"]) == 2
    assert m["discarded_single_scan"] == 3
    assert m["pattern_hit_share"][m["survivors"][0]] == pytest.approx(1.0)
    assert m["docs_matching_any"] == pytest.approx(1.0)

    # Re-running scans nothing: the cache is keyed on batch CONTENT, so identical
    # documents never re-pay -- including across sibling arms.
    scanned.clear()
    stub2 = StubClient(answer2)
    report2 = C.run_corpus_checks(records, spec, run_dir=tmp_path, seed=0,
                                  ctx=_judge_ctx(tmp_path, stub2))
    m2 = report2["properties"]["pattern_scan"]["metrics"]
    assert m2["batches_from_cache"] == 4 and scanned == []
    assert m2["survivors"] == m["survivors"]


# --- 19. synth compare ---------------------------------------------------------------


def _arm(tmp_path, name, k, conflict_rate, sha="abc", critical=False, n=100):
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "corpus_report.json").write_text(json.dumps({
        "stage": "corpus", "n_records": n, "pass": not critical,
        "properties": {"applies_vs_conflicts": {
            "status": "reported", "gate": False, "params": {"sample": 50},
            "metrics": {"conflict_rate": conflict_rate, "judged": 50},
            "findings": []}},
        "findings": ([{"property": "applies_vs_conflicts", "severity": "critical",
                       "metric": "conflict_rate", "scope": "", "summary": "low"}]
                     if critical else []),
    }))
    (d / "manifest.json").write_text(json.dumps({
        "constitution_sha256": sha, "config": {"group_size": k}}))
    return d


def test_compare_calls_a_rising_metric_monotone(tmp_path):
    from src.data.synth.compare import compare_arms, markdown

    dirs = [_arm(tmp_path, "k1", 1, 0.20), _arm(tmp_path, "k2", 2, 0.35),
            _arm(tmp_path, "k4", 4, 0.55)]
    r = compare_arms(dirs, key="group_size")
    assert [a["key"] for a in r["arms"]] == [1, 2, 4]
    t = r["trends"]["applies_vs_conflicts.conflict_rate"]
    assert t["holds"] is True and t["spearman"] == 1.0
    assert t["verdict"] == "monotone as expected"
    assert "conflict_rate" in markdown(r)


def test_compare_names_the_arm_where_a_trend_breaks(tmp_path):
    from src.data.synth.compare import compare_arms

    dirs = [_arm(tmp_path, "k1", 1, 0.20), _arm(tmp_path, "k2", 2, 0.55),
            _arm(tmp_path, "k4", 4, 0.35)]
    t = compare_arms(dirs, key="group_size")["trends"][
        "applies_vs_conflicts.conflict_rate"]
    assert t["holds"] is False
    assert t["verdict"] == "non-monotonic at group_size=4"


def test_compare_refuses_arms_from_different_constitutions(tmp_path):
    from src.data.synth.compare import compare_arms

    dirs = [_arm(tmp_path, "k1", 1, 0.2), _arm(tmp_path, "k2", 2, 0.3, sha="different")]
    with pytest.raises(AssertionError, match="different constitutions"):
        compare_arms(dirs, key="group_size")


def test_compare_warns_on_mismatched_sizes_and_diffs_findings(tmp_path):
    from src.data.synth.compare import compare_arms

    dirs = [_arm(tmp_path, "k1", 1, 0.20, critical=True, n=100),
            _arm(tmp_path, "k2", 2, 0.35, n=250)]
    r = compare_arms(dirs, key="group_size")
    assert any("differ in size" in w for w in r["warnings"])
    diff = r["findings_diff"]["applies_vs_conflicts.conflict_rate"]
    assert diff == {"flagged_in": ["1"], "clean_in": ["2"]}


def test_a_paid_property_must_declare_a_model_and_be_priced(tmp_path):
    paid = C.CorpusCheck("paid_probe", lambda c: C.CheckResult({"n": 1}), paid=True,
                         defaults={"sample": 5}, doc="probe",
                         est_calls=lambda p, n: min(int(p["sample"]), n))
    C.CORPUS_CHECKS["paid_probe"] = paid
    try:
        entry = {"name": "corpus", "kind": "corpus_check",
                 "properties": [{"property": "paid_probe"}]}
        with pytest.raises(AssertionError, match="no `model:`"):
            build_stages(_pipeline(tmp_path, entry))
        assert C.corpus_check_calls(entry, 100) == 5

        unpriced = C.CorpusCheck("unpriced", lambda c: C.CheckResult(), paid=True,
                                 doc="probe")
        C.CORPUS_CHECKS["unpriced"] = unpriced
        with pytest.raises(AssertionError, match="est_calls"):
            C.corpus_check_calls({"properties": [{"property": "unpriced"}]}, 100)
    finally:
        C.CORPUS_CHECKS.pop("paid_probe", None)
        C.CORPUS_CHECKS.pop("unpriced", None)
