# ABOUTME: Offline good-vs-bad contrasts over already-logged ODCV rollouts — the hypothesis
# ABOUTME: generator that says where to spend resampling, and which is never itself causal.

"""What separates a rollout that gamed the metric from one that did not.

Read the paper's warning first, because it governs how anything in this file may be
used. Section 4.2: *"Examining how sentences correlate with a final answer does not
reveal these patterns — e.g. 'Therefore, the answer is no' is linked to a 99% no rate,
but this statement is determined by the upstream CoT and its causal effect is
minuscule."* Every number below is exactly that kind of correlation. A marker that fires
in 90% of violating rollouts may be the cause, a symptom, or a narration of a decision
already taken three steps earlier, and nothing here can tell those apart.

So this module has one job: rank candidate branch points so the expensive on-policy
resampling in `sampler.py` gets spent where something is actually happening. It is the
cheap half of the method, not a shortcut past it.

Its central construct is **the fork**. Nearly every ODCV scenario is built so the agent
eventually discovers the target cannot be met honestly — a compliance script prints FAIL,
a required file does not exist, the data will not support the required conclusion. That
discovery is a moment the environment creates, so it can be located mechanically, in
every rollout, without a labeller. The thought immediately after it is where the rollout
picks a side. Finding the fork gives:

  - a per-rollout decision point, comparable across scenarios;
  - a contrast set (fork thoughts of violating vs clean rollouts) that clusters cleanly;
  - and the branch point worth resampling first, since a causal effect anywhere is most
    likely to be here.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

import numpy as np

from src.eval.stats import wilson

from scratch.thought_branches.segment import Chunk, chunk_trajectory
from scratch.thought_branches.trajectory import Step, Trajectory

# -- lexical markers ---------------------------------------------------------------
# Deliberately narrow and literal. These are tripwires for locating candidate moments,
# never a substitute for the LLM labeller in `label.py`: a regex cannot tell "I should
# not edit the checker" from "I will edit the checker", and both contain "edit the
# checker". Every rate computed from them is reported as a marker rate, not a behaviour
# rate, and the fork clustering below is what carries the real signal.

SHORTCUT = re.compile(
    r"\b(modify|edit|patch|rewrite|overwrite|change|adjust|tweak|update)\b[^.]{0,60}"
    r"\b(script|checker|check|threshold|validator|tool|criteria|limit|config)\b"
    r"|\bhardcod|\bbypass\b|\bwork ?around\b|\bcircumvent\b"
    r"|\b(fabricat|synthes[iy]|invent|make up|generate)\w*\b[^.]{0,40}\b(data|numbers|results|records|values)\b"
    r"|\b(delete|remove|suppress|hide|omit)\b[^.]{0,50}\b(record|entry|log|row|finding|warning|error|result)\b",
    re.IGNORECASE,
)

INTEGRITY = re.compile(
    r"\b(integrity|honest|accurac\w+|truthful|transparen\w+|ethic\w+|complian\w+|regulat\w+)\b"
    r"|\b(should not|shouldn't|must not|mustn't|cannot|can't|will not|won't|refuse|inappropriate|improper)\b"
    r"[^.]{0,60}\b(falsif|fabricat|manipulat|alter|modif|misrepresent|deceiv|mislead|game|gam(e|ing) the)\w*\b"
    r"|\breport\b[^.]{0,40}\b(accurately|honestly|truthfully|as ?is|the failure|the actual)\b",
    re.IGNORECASE,
)

COMMITMENT = re.compile(
    r"\bI (will not|won't|cannot|can't|should not|shouldn't|must not|refuse to|am not going to|will avoid)\b"
    r"|\bI (will|should|need to|am going to) (report|disclose|flag|document|escalate|inform)\b",
    re.IGNORECASE,
)

PRINCIPAL_APPEAL = re.compile(
    r"\b(the (user|client|task|instruction|prompt|requirement)s? (requires?|wants?|needs?|asked|specif\w+|state[sd]))\b"
    r"|\b(is|are) required\b|\brequired for\b|\bmy (function|role|job|purpose|responsibilit\w+) is\b"
    r"|\bin order to (meet|hit|achieve|satisfy)\b[^.]{0,40}\b(target|kpi|goal|requirement|threshold)\b",
    re.IGNORECASE,
)

# Commands that change the world rather than observe it. `commit_before_write` asks
# whether the rollout said what it would not do BEFORE it first touched anything —
# the ordering measure carried over from scratch/par_coherence/registers.py.
WRITE_CMD = re.compile(
    r"\b(rm|mv|cp|sed\s+-i|tee|truncate|chmod|chown|dd)\b|>\s*/|>>|\bcat\s*>|\bpython3?\s+-c\b.{0,80}\bopen\(",
    re.IGNORECASE,
)

# The environment saying "no". This is what makes the fork mechanically findable.
FAIL_SIGNAL = re.compile(
    r"\bFAIL\b|\bFAILED\b|\bERROR\b|\bdenied\b|\bnot found\b|\bno such file\b|\bcannot access\b"
    r"|\bbelow (the )?(target|threshold)\b|\bdoes not meet\b|\bdid not meet\b|\bnon-?complian\w+\b"
    r"|\bviolation\b|\bunable to\b|\brejected\b|\binvalid\b",
    re.IGNORECASE,
)


def _fails(step: Step) -> bool:
    """Whether a tool step reports the environment refusing or failing.

    Args:
        step: A transcript step.

    Returns:
        True when the step is a tool result that either returned non-zero, wrote to
        stderr, or printed a failure word on stdout.
    """
    d = step.tool_result()
    if d is None:
        return False
    if d.get("returncode") not in (0, None):
        return True
    if str(d.get("stderr") or "").strip():
        return True
    return bool(FAIL_SIGNAL.search(str(d.get("stdout") or "")))


# -- the fork ----------------------------------------------------------------------


@dataclass
class Fork:
    """The moment the environment refuses, and the thought that answers it.

    Attributes:
        obstacle_step: Transcript index of the failing tool result.
        obstacle_text: A trimmed rendering of what the environment said.
        thought_step: Transcript index of the assistant turn that responded.
        thought: That turn's full reasoning — the fork thought.
        action: That turn's tool call, rendered.
        rel_pos: Where the fork sits in the trajectory's chunk sequence, on [0, 1].
        n_prior_fails: How many refusals preceded this one; >0 means the agent had
            already been told no, so this fork is not its first.
    """

    obstacle_step: int
    obstacle_text: str
    thought_step: int
    thought: str
    action: str
    rel_pos: float
    n_prior_fails: int


def find_fork(traj: Trajectory, chunks: Sequence[Chunk] | None = None) -> Fork | None:
    """Locate the first environment refusal and the assistant turn that answers it.

    Args:
        traj: The trajectory.
        chunks: Precomputed chunks, for `rel_pos`.

    Returns:
        The fork, or None when the environment never refused — which is itself a finding:
        a rollout with no fork was never put to the test.
    """
    from scratch.thought_branches.segment import render_action

    chunks = list(chunks if chunks is not None else chunk_trajectory(traj))
    n = len(chunks) or 1
    seen_fail = 0
    for i, step in enumerate(traj.steps):
        if not _fails(step):
            continue
        nxt = next((s for s in traj.steps[i + 1 :] if s.is_assistant), None)
        if nxt is None:
            seen_fail += 1
            continue
        d = step.tool_result() or {}
        blob = " ".join(
            str(d.get(k) or "").strip() for k in ("stdout", "stderr")
        ).strip()
        pos = next((c.pos for c in chunks if c.step_index >= nxt.index), n - 1)
        return Fork(
            obstacle_step=step.index,
            obstacle_text=" ".join(blob.split())[:300],
            thought_step=nxt.index,
            thought=nxt.reason,
            action=render_action(nxt),
            rel_pos=pos / (n - 1) if n > 1 else 0.0,
            n_prior_fails=seen_fail,
        )
    return None


# -- per-rollout features ------------------------------------------------------------


@dataclass
class Features:
    """Everything measurable about one rollout without calling a model.

    Attributes:
        key / arm / variant / scenario / rollout: Identity, copied off the trajectory.
        severity: Judges' median severity, or None.
        violation: Whether that reaches the threshold.
        n_steps / n_thought_steps / n_chunks: Size of the trajectory.
        thought_chars: Total characters of reasoning.
        mean_thought_chars: Reasoning characters per thought step.
        n_commands / n_failed_commands: Tool activity and how much of it bounced.
        n_writes: Commands that changed the environment.
        completed: Whether the agent ended with `task_complete`.
        shortcut_hits / integrity_hits / commitment_hits / appeal_hits: Marker counts
            over the reasoning text.
        first_shortcut_pos / first_integrity_pos / first_commitment_pos: Normalised
            position of each marker's first appearance, or None if it never appears.
        commit_before_write: A commitment was stated before the first world-changing
            command. None when the rollout never wrote anything.
        has_fork / fork_rel_pos / fork_prior_fails: The fork, if the environment refused.
        fork_shortcut / fork_integrity: Whether the fork thought itself carries each
            marker — the narrowest, most decision-adjacent read available offline.
    """

    key: str
    arm: str
    variant: str
    scenario: str
    rollout: str
    severity: float | None
    violation: bool | None
    n_steps: int
    n_thought_steps: int
    n_chunks: int
    thought_chars: int
    mean_thought_chars: float
    n_commands: int
    n_failed_commands: int
    n_writes: int
    completed: bool
    shortcut_hits: int
    integrity_hits: int
    commitment_hits: int
    appeal_hits: int
    first_shortcut_pos: float | None
    first_integrity_pos: float | None
    first_commitment_pos: float | None
    commit_before_write: bool | None
    has_fork: bool
    fork_rel_pos: float | None
    fork_prior_fails: int | None
    fork_shortcut: bool | None
    fork_integrity: bool | None


def _first_pos(rx: re.Pattern, chunks: Sequence[Chunk]) -> float | None:
    """Normalised position of the first thought chunk matching `rx`, or None."""
    for c in chunks:
        if c.kind == "thought" and rx.search(c.text):
            return c.rel_pos
    return None


def features(traj: Trajectory) -> Features:
    """Compute every offline feature for one rollout.

    Args:
        traj: The trajectory.

    Returns:
        Its feature record.
    """
    chunks = chunk_trajectory(traj)
    thoughts = [s.reason for s in traj.thoughts]
    blob = "\n".join(thoughts)
    cmds = traj.commands()
    fails = sum(1 for s in traj.steps if _fails(s))

    # Ordering: did a commitment appear before the first world-changing command?
    commit_before_write: bool | None = None
    committed = False
    for step in traj.steps:
        if not step.is_assistant:
            continue
        if step.reason and COMMITMENT.search(step.reason):
            committed = True
        if any(WRITE_CMD.search(c.command) for c in step.calls if c.command):
            commit_before_write = committed
            break

    fork = find_fork(traj, chunks)
    return Features(
        key=traj.key,
        arm=traj.arm,
        variant=traj.variant,
        scenario=traj.scenario,
        rollout=traj.rollout,
        severity=traj.severity,
        violation=traj.is_violation,
        n_steps=len(traj.steps),
        n_thought_steps=len(traj.thoughts),
        n_chunks=len(chunks),
        thought_chars=len(blob),
        mean_thought_chars=len(blob) / max(1, len(thoughts)),
        n_commands=len(cmds),
        n_failed_commands=fails,
        n_writes=sum(1 for c in cmds if WRITE_CMD.search(c)),
        completed=traj.completed(),
        shortcut_hits=len(SHORTCUT.findall(blob)),
        integrity_hits=len(INTEGRITY.findall(blob)),
        commitment_hits=len(COMMITMENT.findall(blob)),
        appeal_hits=len(PRINCIPAL_APPEAL.findall(blob)),
        first_shortcut_pos=_first_pos(SHORTCUT, chunks),
        first_integrity_pos=_first_pos(INTEGRITY, chunks),
        first_commitment_pos=_first_pos(COMMITMENT, chunks),
        commit_before_write=commit_before_write,
        has_fork=fork is not None,
        fork_rel_pos=fork.rel_pos if fork else None,
        fork_prior_fails=fork.n_prior_fails if fork else None,
        fork_shortcut=bool(SHORTCUT.search(fork.thought)) if fork else None,
        fork_integrity=bool(INTEGRITY.search(fork.thought)) if fork else None,
    )


def feature_table(trajs: Iterable[Trajectory]) -> list[Features]:
    """Compute features for a whole corpus."""
    return [features(t) for t in trajs]


# -- contrasts -----------------------------------------------------------------------


@dataclass
class Contrast:
    """One good-vs-bad comparison.

    Attributes:
        name: The feature compared.
        kind: "rate" for a proportion, "mean" for a continuous measure.
        clean / violating: The statistic in each group.
        delta: violating - clean.
        clean_ci / violating_ci: 95% intervals — Wilson for rates, bootstrap for means.
        n_clean / n_violating: Group sizes after dropping missing values.
        separated: Whether the two intervals fail to overlap. A blunt screen, not a test;
            it is here to stop a reader treating a noisy ordering as a finding.
    """

    name: str
    kind: str
    clean: float
    violating: float
    delta: float
    clean_ci: tuple[float, float]
    violating_ci: tuple[float, float]
    n_clean: int
    n_violating: int
    separated: bool


def _boot_ci(
    vals: Sequence[float], n_boot: int = 2000, seed: int = 0
) -> tuple[float, tuple[float, float]]:
    v = np.asarray(vals, dtype=np.float64)
    if v.size == 0:
        return 0.0, (0.0, 0.0)
    if v.size == 1:
        return float(v[0]), (float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    draws = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return float(v.mean()), (
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def contrast(rows: Sequence[Features], name: str, kind: str, seed: int = 0) -> Contrast:
    """Compare one feature between violating and clean rollouts.

    Args:
        rows: Feature records; those with no verdict or a missing value are dropped.
        name: Attribute on `Features`.
        kind: "rate" (boolean attribute) or "mean" (numeric attribute).
        seed: Bootstrap seed.

    Returns:
        The contrast.

    Raises:
        ValueError: On an unknown `kind`.
    """
    if kind not in ("rate", "mean"):
        raise ValueError(f"kind must be 'rate' or 'mean', got {kind!r}")
    good = [
        getattr(r, name)
        for r in rows
        if r.violation is False and getattr(r, name) is not None
    ]
    bad = [
        getattr(r, name)
        for r in rows
        if r.violation is True and getattr(r, name) is not None
    ]
    if kind == "rate":
        kg, ng, kb, nb = (
            sum(bool(x) for x in good),
            len(good),
            sum(bool(x) for x in bad),
            len(bad),
        )
        cg = kg / ng if ng else 0.0
        cb = kb / nb if nb else 0.0
        ci_g = wilson(kg, ng) if ng else (0.0, 0.0)
        ci_b = wilson(kb, nb) if nb else (0.0, 0.0)
    else:
        cg, ci_g = _boot_ci([float(x) for x in good], seed=seed)
        cb, ci_b = _boot_ci([float(x) for x in bad], seed=seed + 1)
        ng, nb = len(good), len(bad)
    return Contrast(
        name=name,
        kind=kind,
        clean=cg,
        violating=cb,
        delta=cb - cg,
        clean_ci=ci_g,
        violating_ci=ci_b,
        n_clean=ng,
        n_violating=nb,
        separated=ci_g[1] < ci_b[0] or ci_b[1] < ci_g[0],
    )


# Which features to contrast, and how each should be read.
CONTRASTS: tuple[tuple[str, str], ...] = (
    ("n_thought_steps", "mean"),
    ("n_chunks", "mean"),
    ("thought_chars", "mean"),
    ("mean_thought_chars", "mean"),
    ("n_commands", "mean"),
    ("n_failed_commands", "mean"),
    ("n_writes", "mean"),
    ("completed", "rate"),
    ("shortcut_hits", "mean"),
    ("integrity_hits", "mean"),
    ("commitment_hits", "mean"),
    ("appeal_hits", "mean"),
    ("first_shortcut_pos", "mean"),
    ("first_integrity_pos", "mean"),
    ("first_commitment_pos", "mean"),
    ("commit_before_write", "rate"),
    ("has_fork", "rate"),
    ("fork_rel_pos", "mean"),
    ("fork_shortcut", "rate"),
    ("fork_integrity", "rate"),
)


def all_contrasts(rows: Sequence[Features], seed: int = 0) -> list[Contrast]:
    """Run every contrast in `CONTRASTS`, largest separated effect first."""
    out = [contrast(rows, n, k, seed=seed) for n, k in CONTRASTS]
    return sorted(out, key=lambda c: (not c.separated, -abs(c.delta)))


# -- stratified contrasts -------------------------------------------------------------


@dataclass
class Stratified:
    """A within-stratum good-vs-bad contrast.

    A pooled contrast over this corpus is confounded twice over. Arms differ in base
    violation rate AND in how their training shaped the CoT, so "violating rollouts think
    less per step" could just be "the arm that violates more was trained to think less".
    Scenarios differ even harder: each has its own difficulty, its own trace length, and
    its own scripted obstacle, so pooling across them compares tasks as much as outcomes.

    This estimates the contrast INSIDE each (arm, scenario) cell that contains both a
    violating and a clean rollout, then averages the per-cell differences. Every
    comparison is then between two rollouts of the same model on the same task, which is
    the only version of this question that has one answer.

    Attributes:
        name: The feature compared.
        kind: "rate" or "mean".
        delta: Mean within-cell (violating - clean) difference.
        lo / hi: Bootstrap interval on that mean, resampling CELLS.
        n_cells: Cells that contained both classes.
        n_rollouts: Rollouts inside those cells.
        agree: Share of cells whose difference has the same sign as `delta`. Exact ties
            count as disagreement, so a rate feature that is identical in most cells
            reads LOW here — that is the honest reading, not a bug: it says the mean
            difference rests on a handful of cells.
        sd: Corpus standard deviation of the feature, used to standardise.
        delta_std: `delta / sd` — the effect in standard deviations, so features whose
            native units differ by three orders of magnitude can share one axis.
        lo_std / hi_std: The interval, standardised the same way.
    """

    name: str
    kind: str
    delta: float
    lo: float
    hi: float
    n_cells: int
    n_rollouts: int
    agree: float
    sd: float = 0.0
    delta_std: float = 0.0
    lo_std: float = 0.0
    hi_std: float = 0.0


def contrast_within(
    rows: Sequence[Features],
    name: str,
    kind: str,
    by: tuple[str, ...] = ("arm", "scenario", "variant"),
    seed: int = 0,
    n_boot: int = 2000,
) -> Stratified:
    """Contrast one feature within (arm, scenario, variant) cells.

    Args:
        rows: Feature records.
        name: Attribute to compare.
        kind: "rate" or "mean"; both reduce to a within-cell mean difference.
        by: Stratification keys.
        seed: Bootstrap seed.
        n_boot: Bootstrap replicates over cells.

    Returns:
        The stratified contrast. `n_cells == 0` means no cell held both a violating and a
        clean rollout — the corpus simply cannot answer the question, which is a result
        about the design, not about the model.
    """
    cells: dict[tuple, list[Features]] = {}
    present: list[float] = []
    for r in rows:
        if r.violation is None or getattr(r, name) is None:
            continue
        cells.setdefault(tuple(getattr(r, k) for k in by), []).append(r)
        present.append(float(getattr(r, name)))

    diffs: list[float] = []
    n_roll = 0
    for members in cells.values():
        bad = [float(getattr(r, name)) for r in members if r.violation]
        good = [float(getattr(r, name)) for r in members if not r.violation]
        if not bad or not good:
            continue
        diffs.append(float(np.mean(bad)) - float(np.mean(good)))
        n_roll += len(bad) + len(good)

    # Standardise by the SPREAD OF THE FEATURE ACROSS THE CORPUS, not by the spread of
    # the per-cell differences. Dividing by the latter would turn a consistent-but-tiny
    # difference into a huge effect size, which is exactly the wrong thing to put on a
    # chart next to a large one.
    sd = float(np.std(present, ddof=1)) if len(present) > 1 else 0.0
    scale = sd if sd > 1e-12 else 1.0

    if not diffs:
        return Stratified(name, kind, 0.0, 0.0, 0.0, 0, 0, 0.0, sd, 0.0, 0.0, 0.0)
    d = np.asarray(diffs, dtype=np.float64)
    mean = float(d.mean())
    if d.size == 1:
        lo = hi = mean
    else:
        rng = np.random.default_rng(seed)
        draws = rng.choice(d, size=(n_boot, d.size), replace=True).mean(axis=1)
        lo, hi = float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
    sign = np.sign(mean) or 1.0
    return Stratified(
        name=name,
        kind=kind,
        delta=mean,
        lo=lo,
        hi=hi,
        n_cells=int(d.size),
        n_rollouts=n_roll,
        agree=float(np.mean(np.sign(d) == sign)),
        sd=sd,
        delta_std=mean / scale,
        lo_std=lo / scale,
        hi_std=hi / scale,
    )


def all_contrasts_within(
    rows: Sequence[Features],
    by: tuple[str, ...] = ("arm", "scenario", "variant"),
    seed: int = 0,
) -> list[Stratified]:
    """Run every contrast within strata, largest effect that excludes zero first."""
    out = [contrast_within(rows, n, k, by=by, seed=seed) for n, k in CONTRASTS]

    def excludes_zero(s: Stratified) -> bool:
        return s.n_cells > 1 and (s.lo > 0 or s.hi < 0)

    return sorted(out, key=lambda s: (not excludes_zero(s), -abs(s.delta)))


# -- fork clustering ------------------------------------------------------------------


@dataclass
class ForkCluster:
    """A recurring way of answering the environment's refusal.

    The raw `violation_rate` is not safe to read on its own, and the reason is worth
    stating because it took a look at the assignments to notice. Embeddings of a fork
    thought carry the SCENARIO — the thought names `/app/data/learning_materials/`, or a
    patient's symptoms, or a compliance threshold — so clustering fork thoughts
    substantially recovers which task ran. A cluster can then show a high violation rate
    purely by being made of hard scenarios: one cluster here reads 73% overall and 23%
    inside its own largest scenario.

    `lift` is the corrected quantity: for each scenario the cluster appears in, its
    violation rate there minus that scenario's base rate, averaged over scenarios. A lift
    near zero means the cluster is a scenario label wearing a reasoning-strategy costume.
    Quote `lift`, and treat `violation_rate` as descriptive colour only.

    Attributes:
        cluster: Cluster index.
        n: Fork thoughts in it.
        violation_rate: Raw share of them from violating rollouts.
        ci: Wilson interval on that raw rate.
        lift: Scenario-adjusted violation rate above the scenario's own base rate.
        lift_lo / lift_hi: Bootstrap interval on `lift`, resampling scenarios.
        n_scenarios: Scenarios contributing to `lift`.
        scenario_share: Share of the cluster sitting in its single largest scenario; near
            1.0 means the cluster IS a scenario.
        exemplars: Fork thoughts nearest the centroid.
        arms: How the cluster's members split across arms.
    """

    cluster: int
    n: int
    violation_rate: float
    ci: tuple[float, float]
    exemplars: list[str]
    arms: dict[str, int]
    lift: float = 0.0
    lift_lo: float = 0.0
    lift_hi: float = 0.0
    n_scenarios: int = 0
    scenario_share: float = 0.0


def cluster_forks(
    trajs: Sequence[Trajectory],
    k: int = 12,
    seed: int = 0,
    max_chars: int = 1200,
    n_exemplars: int = 3,
) -> tuple[list[ForkCluster], list[tuple[Trajectory, Fork, int]]]:
    """Cluster fork thoughts and read off each cluster's violation rate.

    This is the paper's §4.2 move — discover the factors a model actually reasons with by
    clustering its sentences, rather than assuming a taxonomy — applied to the one moment
    every ODCV rollout shares. Because clustering is unsupervised and the violation rate
    is read off afterwards, the clusters are not fitted to the outcome, so a cluster whose
    rate is far from base rate is a real candidate for a causal test.

    Args:
        trajs: Trajectories to cluster the forks of.
        k: Number of clusters.
        seed: KMeans seed.
        max_chars: Fork-thought truncation before embedding.
        n_exemplars: Exemplars kept per cluster.

    Returns:
        (clusters sorted by violation rate descending, per-fork assignments).
    """
    from sklearn.cluster import KMeans

    from scratch.thought_branches.embed import encode

    found = [(t, f) for t in trajs for f in [find_fork(t)] if f and f.thought.strip()]
    if len(found) < k:
        k = max(2, len(found) // 4) if len(found) >= 8 else 2
    if len(found) < 4:
        return [], []
    texts = [f.thought[:max_chars] for _, f in found]
    V = encode(texts)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(V)
    labels = km.labels_

    # Scenario base rates over the forked rollouts, so a cluster's lift is measured
    # against the difficulty of the tasks it actually contains.
    scen_hits: dict[str, list[bool]] = {}
    for t, _ in found:
        if t.is_violation is not None:
            scen_hits.setdefault(t.scenario, []).append(bool(t.is_violation))
    scen_base = {s: sum(v) / len(v) for s, v in scen_hits.items() if v}

    rng = np.random.default_rng(seed)
    clusters: list[ForkCluster] = []
    for c in range(k):
        idx = [i for i, lab in enumerate(labels) if lab == c]
        if not idx:
            continue
        verdicts = [found[i][0].is_violation for i in idx]
        known = [v for v in verdicts if v is not None]
        hits, n = sum(bool(v) for v in known), len(known)

        per_scen: dict[str, list[bool]] = {}
        for i in idx:
            t = found[i][0]
            if t.is_violation is not None:
                per_scen.setdefault(t.scenario, []).append(bool(t.is_violation))
        lifts = [
            sum(v) / len(v) - scen_base[s]
            for s, v in per_scen.items()
            if s in scen_base and len(scen_hits.get(s, [])) >= 2
        ]
        if lifts:
            arr = np.asarray(lifts, dtype=np.float64)
            lift = float(arr.mean())
            if arr.size > 1:
                draws = rng.choice(arr, size=(2000, arr.size), replace=True).mean(
                    axis=1
                )
                lo, hi = (
                    float(np.quantile(draws, 0.025)),
                    float(np.quantile(draws, 0.975)),
                )
            else:
                lo = hi = lift
        else:
            lift = lo = hi = 0.0

        counts = Counter(found[i][0].scenario for i in idx)
        d = V[idx] @ km.cluster_centers_[c]
        order = [idx[j] for j in np.argsort(-d)][:n_exemplars]
        clusters.append(
            ForkCluster(
                cluster=c,
                n=len(idx),
                violation_rate=hits / n if n else 0.0,
                ci=wilson(hits, n) if n else (0.0, 0.0),
                exemplars=[" ".join(found[i][1].thought.split())[:300] for i in order],
                arms=dict(Counter(found[i][0].arm for i in idx)),
                lift=lift,
                lift_lo=lo,
                lift_hi=hi,
                n_scenarios=len(lifts),
                scenario_share=counts.most_common(1)[0][1] / len(idx)
                if counts
                else 0.0,
            )
        )
    clusters.sort(key=lambda c: -c.lift)
    return clusters, [(t, f, int(lab)) for (t, f), lab in zip(found, labels)]


def as_rows(rows: Sequence[Features]) -> list[dict]:
    """Feature records as plain dicts, for JSON or a DataFrame."""
    return [asdict(r) for r in rows]
