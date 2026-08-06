# ABOUTME: Offline tests for the SWE-bench baseline's subset selection: nesting, repo
# ABOUTME: stratification, determinism, and the identity/filter strings a run is recorded by.

import re

import pytest

from src.eval.capabilities.swebench_mini.subset import (
    id_filter_regex,
    repo_breakdown,
    select,
    stratified_order,
    subset_hash,
    summarize_selection,
)

# Deliberately lopsided, like SWE-bench Verified itself (django dominates): a stratifier that
# only looks right on balanced input is not doing anything.
REPO_SIZES = {"django/django": 100, "sympy/sympy": 40, "astropy/astropy": 12,
              "psf/requests": 5, "pylint-dev/pylint": 3}
INSTANCES = [{"instance_id": f"{repo.replace('/', '__')}-{i}", "repo": repo}
             for repo, size in REPO_SIZES.items() for i in range(size)]


def ids(rows):
    return [r["instance_id"] for r in rows]


def test_prefixes_nest_so_extending_a_run_reuses_it():
    small, medium, large = (select(INSTANCES, seed=0, n=n) for n in (16, 32, 80))
    assert set(ids(small)) < set(ids(medium)) < set(ids(large))
    # Not just subsets — the same order, so a resumed run matches instance for instance.
    assert ids(medium)[:16] == ids(small)


def test_fraction_and_n_agree_and_are_mutually_exclusive():
    assert ids(select(INSTANCES, seed=0, fraction=0.1)) == ids(select(INSTANCES, seed=0, n=16))
    for kwargs in ({}, {"fraction": 0.1, "n": 16}):
        with pytest.raises(ValueError, match="exactly one"):
            select(INSTANCES, seed=0, **kwargs)
    for kwargs in ({"fraction": 0.0}, {"fraction": 1.5}, {"n": 0}, {"n": len(INSTANCES) + 1}):
        with pytest.raises(ValueError):
            select(INSTANCES, seed=0, **kwargs)


def test_every_depth_is_repo_proportional():
    # The point of stratifying: a 10% slice must be ~10% of EACH repo, not 100% of the repo
    # that happens to sort first. Allow ±1 instance for rounding at each repo.
    for fraction in (0.1, 0.25, 0.5):
        counts = repo_breakdown(select(INSTANCES, seed=0, fraction=fraction))
        for repo, size in REPO_SIZES.items():
            assert abs(counts.get(repo, 0) - fraction * size) <= 1, (fraction, repo, counts)


def test_tail_repos_enter_proportionally_as_depth_grows():
    # A repo with 3 of 160 instances is worth 0.3 instances at 10%, so it is correctly
    # ABSENT there — guaranteeing every repo a slot would over-weight small repos and bias
    # pass@1 upward or downward depending on how hard they are. What must hold is that the
    # tail arrives as depth increases, and never leaves once it has arrived (nesting).
    tail = "pylint-dev/pylint"
    seen = [repo_breakdown(select(INSTANCES, seed=0, fraction=f)).get(tail, 0)
            for f in (0.1, 0.25, 0.5, 1.0)]
    assert seen == sorted(seen), seen              # monotone: nesting forbids losing one
    assert seen[0] == 0 and seen[-1] == REPO_SIZES[tail], seen
    # The big repo, by contrast, is present from the shallowest depth.
    assert repo_breakdown(select(INSTANCES, seed=0, fraction=0.1))["django/django"] == 10


def test_selection_is_deterministic_and_seed_dependent():
    assert ids(select(INSTANCES, seed=0, n=30)) == ids(select(INSTANCES, seed=0, n=30))
    assert ids(select(INSTANCES, seed=1, n=30)) != ids(select(INSTANCES, seed=0, n=30))
    # Order cannot depend on how the split happened to arrive.
    assert ids(stratified_order(INSTANCES, 0)) == ids(stratified_order(list(reversed(INSTANCES)), 0))


def test_subset_hash_covers_ids_seed_and_dataset_revision():
    chosen = select(INSTANCES, seed=0, n=20)
    base = subset_hash(chosen, 0, "SWE-bench/SWE-bench_Verified", "abc123")
    assert base == subset_hash(list(reversed(chosen)), 0, "SWE-bench/SWE-bench_Verified", "abc123")
    # An in-place upstream revision keeps every id and changes the tests behind them, so the
    # revision has to be part of the identity or two incomparable runs look identical.
    assert base != subset_hash(chosen, 0, "SWE-bench/SWE-bench_Verified", "def456")
    assert base != subset_hash(chosen, 1, "SWE-bench/SWE-bench_Verified", "abc123")
    assert base != subset_hash(select(INSTANCES, seed=0, n=21), 0,
                               "SWE-bench/SWE-bench_Verified", "abc123")


def test_filter_regex_is_anchored_so_short_ids_cannot_pull_in_longer_ones():
    chosen = [{"instance_id": "django__django-1", "repo": "django/django"}]
    pattern = re.compile(id_filter_regex(chosen))
    assert pattern.match("django__django-1")
    # Unanchored, this id is a prefix of a real Verified instance and would silently add it.
    assert not pattern.match("django__django-11099")


def test_filter_regex_matches_exactly_the_selection():
    chosen = select(INSTANCES, seed=0, n=25)
    pattern = re.compile(id_filter_regex(chosen))
    selected_ids = set(ids(chosen))
    for row in INSTANCES:
        assert bool(pattern.match(row["instance_id"])) == (row["instance_id"] in selected_ids)
    with pytest.raises(ValueError, match="empty subset"):
        id_filter_regex([])


def test_filter_regex_escapes_regex_metacharacters():
    # Real ids carry dots and dashes (`pydata__xarray-4094`, `sphinx-doc__sphinx-8721`);
    # an unescaped dot matches any character.
    chosen = [{"instance_id": "a.b__c-1", "repo": "a/b"}]
    pattern = re.compile(id_filter_regex(chosen))
    assert pattern.match("a.b__c-1") and not pattern.match("axb__c-1")


# Trimmed from a REAL swebench 4.1.0 report (gold-patch run, django__django-11815, verified
# 2026-08-05). Key names are pinned against the harness's actual output rather than guessed:
# reading `resolved` instead of `resolved_ids` would silently score every run 0%.
GOLD_REPORT = {
    "total_instances": 1, "submitted_instances": 500, "completed_instances": 1,
    "resolved_instances": 1, "unresolved_instances": 0, "empty_patch_instances": 0,
    "error_instances": 0, "schema_version": 2,
    "completed_ids": ["django__django-11815"], "resolved_ids": ["django__django-11815"],
    "unresolved_ids": [], "empty_patch_ids": [], "error_ids": [], "incomplete_ids": [],
}


def test_resolution_summary_reads_the_real_harness_report_shape():
    from src.eval.capabilities.swebench_mini.metrics import resolution_summary

    scores = resolution_summary(GOLD_REPORT, ["django__django-11815"])
    assert scores["pass_at_1"] == 1.0 and scores["n_resolved"] == 1
    assert scores["instances_outside_selection"] == []


def test_unfinished_instances_count_against_pass_at_1():
    from src.eval.capabilities.swebench_mini.metrics import resolution_summary

    # Two selected, one resolved, one never graded (rollout crashed / patch never applied).
    # It MUST stay in the denominator: dropping it would inflate pass@1 by exactly the
    # failures most worth seeing.
    scores = resolution_summary(GOLD_REPORT, ["django__django-11815", "django__django-16899"])
    assert scores["n_scored"] == 2 and scores["pass_at_1"] == 0.5


def test_resolved_ids_outside_the_selection_are_surfaced():
    from src.eval.capabilities.swebench_mini.metrics import resolution_summary

    # A stray id means the grading input and the subset filter disagree — that invalidates
    # the number rather than shrinking it, so it must be reported, not silently intersected.
    scores = resolution_summary(GOLD_REPORT, ["sympy__sympy-1"])
    assert scores["instances_outside_selection"] == ["django__django-11815"]
    assert scores["pass_at_1"] == 0.0


def test_shards_are_disjoint_and_cover_the_subset():
    from src.eval.capabilities.swebench_mini.subset import shard

    chosen = select(INSTANCES, seed=0, fraction=0.5)
    a, b = shard(chosen, 0, 2), shard(chosen, 1, 2)
    assert not (set(ids(a)) & set(ids(b))), "shards must be disjoint"
    assert set(ids(a)) | set(ids(b)) == set(ids(chosen)), "shards must cover the subset"
    assert abs(len(a) - len(b)) <= 1, "shards must be near-equal in size"
    for kwargs in ((2, 2), (-1, 2)):
        with pytest.raises(ValueError, match="out of range"):
            shard(chosen, *kwargs)


def test_each_shard_stays_repo_proportional():
    # Round-robin, not contiguous blocks: a contiguous split would give one driver the front
    # of every repo's ranking and the other the back, so the halves would not be exchangeable.
    from src.eval.capabilities.swebench_mini.subset import shard

    chosen = select(INSTANCES, seed=0, fraction=0.5)
    whole = repo_breakdown(chosen)
    for i in range(3):
        part = repo_breakdown(shard(chosen, i, 3))
        for repo, n in whole.items():
            assert abs(part.get(repo, 0) - n / 3) <= 1, (i, repo, part)


def test_sharded_summary_keeps_the_full_subset_as_the_identity():
    from src.eval.capabilities.swebench_mini.subset import shard

    chosen = select(INSTANCES, seed=0, fraction=0.5)
    mine = shard(chosen, 1, 2)
    s = summarize_selection(mine, len(INSTANCES), 0, "ds", "rev", full=chosen,
                            shard_index=1, shard_count=2)
    # pass@1 is scored against the FULL subset, so that hash — not the shard's — is the
    # identity two drivers must agree on before their results are merged.
    assert s["subset_hash"] == subset_hash(chosen, 0, "ds", "rev")
    assert s["shard_hash"] == subset_hash(mine, 0, "ds", "rev")
    assert s["n_selected"] == len(chosen) and s["n_in_shard"] == len(mine)
    assert s["instance_ids"] == sorted(ids(mine))
    assert s["full_instance_ids"] == sorted(ids(chosen))


def test_image_name_matches_upstreams_derivation():
    # Must stay byte-identical to mini-swe-agent's get_swebench_docker_image_name: pre-pulling
    # a different name than it runs pays for the download AND still hits the 120s
    # container-start timeout the pre-pull exists to avoid. Docker forbids `__` in image
    # names, hence SWE-bench's magic token.
    from src.eval.capabilities.swebench_mini.images import image_name

    assert (image_name({"instance_id": "django__django-11815"})
            == "docker.io/swebench/sweb.eval.x86_64.django_1776_django-11815:latest")
    # Mixed case is lowercased — docker rejects uppercase in a repository name.
    assert image_name({"instance_id": "PyCQA__flake8-1"}).endswith(
        "sweb.eval.x86_64.pycqa_1776_flake8-1:latest")
    # The dataset's own field wins when the split declares one.
    assert image_name({"instance_id": "x__y-1", "image_name": "custom/img:tag"}) == "custom/img:tag"
    assert image_name({"instance_id": "x__y-1", "docker_image": "other/img:v2"}) == "other/img:v2"


def test_summary_block_records_what_a_reader_needs_to_reproduce_the_draw():
    chosen = select(INSTANCES, seed=0, fraction=0.1)
    summary = summarize_selection(chosen, len(INSTANCES), 0,
                                  "SWE-bench/SWE-bench_Verified", "abc123")
    assert summary["n_selected"] == len(chosen) and summary["split_size"] == len(INSTANCES)
    assert summary["dataset_revision"] == "abc123"
    assert summary["instance_ids"] == sorted(ids(chosen))
    assert sum(summary["repo_breakdown"].values()) == len(chosen)
