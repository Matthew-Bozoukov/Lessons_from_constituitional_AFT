# ABOUTME: Driver for the causal half — resamples ODCV fork points on a served model and
# ABOUTME: reports the on-policy action distribution, resilience, and the paraphrase control.

"""Actually running the resampler.

Three stages, in this order deliberately, because each licenses the next:

  verify       Resample every chosen branch point ONCE at temperature 0 and ask how often
               the model reproduces the action it actually took. This is the check that
               validates the whole framework, and nothing downstream means anything
               without it: if `Trajectory.to_openai_messages` does not faithfully rebuild
               the state the model was in — wrong template, wrong tool schema, a dropped
               tool result — then a greedy resample lands somewhere else, and every
               "effect" measured afterwards is reconstruction error wearing a lab coat.

  distribution Resample at the paper's temperature (0.7) and read out what the model does
               across the branch. This is the on-policy action distribution: not "what did
               it do" but "what does it do, from here". Actions are labelled mechanically
               (`mechanical.py`), so no judge is involved.

  resilience   The paper's Algorithm 1: resample repeatedly and count the rounds before
               the fork thought's content stops coming back. Needs only the model and an
               embedder — no outcome, no judge — so it is the one paper metric that runs
               end to end at this budget.

A temperature caveat that has to travel with every number here: the recorded ODCV rollouts
were generated at temperature 0, and these branches are drawn at 0.7. So the distribution
is "what this model could do from here", not "what it would have done". The verify stage
is run at 0 precisely so the comparison to the recorded action stays apples-to-apples.

    uv run python -m scratch.thought_branches.run_branches run \\
        --base_url https://<pod>-8000.proxy.runpod.net/v1 --model <served name>
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import fire

from src.naming import run_dir
from src.utils import write_run_meta

from scratch.thought_branches.descriptive import find_fork
from scratch.thought_branches.mechanical import (
    classify_branch,
    classify_step,
    validate_against_outcomes,
)
from scratch.thought_branches.metrics import resilience, resilience_tau
from scratch.thought_branches.sampler import (
    FrozenEnvSampler,
    SampleConfig,
    resilience_rounds,
)
from scratch.thought_branches.segment import BranchPoint, chunk_trajectory
from scratch.thought_branches.trajectory import load_corpus

DATA = Path("output/thought_branches/data")
OUT_BASE = Path("output/thought_branches")
ARM = "2026-08-19-odcv-numina-control-716-seed0"


def _fork_branches(trajs, n: int, seed: int):
    """Pick n trajectories with forks, balanced across judged outcome."""
    have = [
        (t, f)
        for t in trajs
        for f in [find_fork(t)]
        if f and t.is_violation is not None
    ]
    rng = random.Random(seed)
    viol = [x for x in have if x[0].is_violation]
    clean = [x for x in have if not x[0].is_violation]
    rng.shuffle(viol)
    rng.shuffle(clean)
    half = n // 2
    picked = viol[:half] + clean[: n - half]
    rng.shuffle(picked)
    out = []
    for t, f in picked:
        chunks = chunk_trajectory(t)
        pos = next((c.pos for c in chunks if c.step_index >= f.thought_step), 0)
        out.append((t, f, BranchPoint(t.key, f.thought_step, 0, "step", pos)))
    return out


# Port 8000 is what every eval driver in this repo forwards to by default, so on a shared
# machine it is routinely already taken — and a taken local port does NOT make ssh fail
# loudly, it just leaves the forward unestablished while localhost:8000 keeps answering
# from whoever bound it first. This driver therefore takes its own.
LOCAL_PORT = 8021


def _serve(target: str, server: str, work: Path, port: int = LOCAL_PORT):
    """Start vLLM for `target` on a pod and tunnel it back, the way run_eval does.

    Going through `VllmServer` rather than an ssh one-liner is not ceremony: it is what
    pins the thinking mode into the chat template at serve time (`pin_template`) and what
    turns the tool-call parser on. A hand-rolled server that misses either silently
    produces a model that never emits a tool call, and the harness reads that as a clean
    run rather than as a broken one.

    Args:
        target: HF path of the adapter to serve.
        server: `root@ip:port` of the pod.
        work: Local scratch directory for the server handle.

    Returns:
        (VllmServer, ServedTarget) — touching the target's `base_url` boots it.
    """
    from src.infra.endpoints.vllm import SshExec, VllmServer, resolve_target

    # `SshExec(host, port)` is not (ssh host, ssh port): `host` is the WHOLE
    # `root@ip:port` string, which `ssh_argv` parses to get `-p`, and `port` is the port
    # vLLM will listen on. Splitting the ssh port out and passing it as `port` dials port
    # 22 instead and fails on the host key, which reads as a permissions problem rather
    # than as the argument mistake it is.
    spec = resolve_target(target)
    srv = VllmServer(
        work,
        port=port,
        executor=SshExec(server, port=port),
        serve_requirements={"needs_tool_calls": True, "context_window": 16384},
    )
    return srv, srv.ensure(spec)


def _norm(s: str) -> str:
    """Lowercase alphanumerics only — for comparing an HF path to a served model name."""
    return "".join(c for c in s.lower() if c.isalnum())


def _resolve_served_name(
    base_url: str, api_key: str, want: str, target: str = "", timeout_s: int = 180
) -> str:
    """The name vLLM registered our adapter under — verified, not guessed.

    Three failure modes, in increasing order of how badly they end.

    The predicted name need not match the one in `--lora-modules`; that returns a 404 per
    request and reads like a network problem. The adapter can also register slightly after
    the base model answers, so a check at the wrong moment sees only "base".

    The third is the one that matters and it happened: **the tunnel can be pointing at
    someone else's pod.** A local port already forwarded by another session does not fail
    loudly — the forward is simply not established, and `localhost:8000` keeps answering
    from whoever got there first. An earlier version of this function accepted "the single
    non-base model" and would have cheerfully run a whole experiment against a colleague's
    adapter, on their GPU, and written the numbers out under our arm's name. Nothing in the
    output would have looked wrong.

    So the served name must be *derived from the target we asked for*, compared on
    alphanumerics alone since vLLM rewrites punctuation. Anything else is refused.

    Args:
        base_url: The served endpoint.
        api_key: Key for it.
        want: The name the spec predicted.
        target: The HF path we asked to serve; the identity the name must match.
        timeout_s: How long to wait for the adapter to appear.

    Returns:
        The registered name of OUR adapter.

    Raises:
        RuntimeError: If our adapter never appears — including when some other adapter does,
            which means the endpoint is not ours.
    """
    import time

    import httpx

    deadline = time.time() + timeout_s
    seen: list[str] = []
    while time.time() < deadline:
        try:
            r = httpx.get(
                f"{base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=20,
            )
            seen = [m["id"] for m in r.json().get("data", [])]
        except Exception:  # noqa: BLE001
            seen = []
        stem = _norm(target.split("/")[-1]) if target else ""
        if want in seen:
            return want
        # Only a name that IS our adapter counts; "some non-base model" is how you end up
        # measuring another pod.
        ours = [m for m in seen if m != "base" and stem and _norm(m) == stem]
        if ours:
            print(f"  note: serving as {ours[0]!r}, not the predicted {want!r}", flush=True)
            return ours[0]
        time.sleep(5)
    others = [m for m in seen if m != "base"]
    raise RuntimeError(
        f"our adapter never registered at {base_url}. The endpoint offers {seen}.\n"
        + (
            f"It is serving {others} — that is NOT {target!r}, so this endpoint belongs to "
            "another pod (most likely a local port already forwarded by another session). "
            "Refusing: running here would spend someone else's GPU and label the result as ours."
            if others
            else "Only the base model is present, and running against it would measure the "
            "untrained weights."
        )
    )


def run(
    model: str = "",
    base_url: str = "",
    target: str = "",
    server: str = "",
    api_key: str = "EMPTY",
    arm: str = ARM,
    n_branches: int = 40,
    n_samples: int = 24,
    resilience_branches: int = 12,
    resilience_rounds_n: int = 4,
    resilience_samples: int = 12,
    temperature: float = 0.7,
    workers: int = 12,
    seed: int = 0,
    subject: str = "odcv_fork_resampling",
    data: str = str(DATA),
) -> None:
    """Run verify, distribution and resilience against a served model.

    Args:
        model: Model name as the endpoint serves it.
        base_url: OpenAI-compatible base URL of the served target.
        api_key: Key for that endpoint.
        arm: Which published run's rollouts to branch. MUST be the arm whose adapter is
            being served — resampling one arm's prefix with another arm's weights is
            off-policy, which is the thing the method exists to avoid.
        n_branches: Fork points to resample.
        n_samples: Resamples per fork point at `temperature`.
        resilience_branches: How many fork points also get the resilience treatment.
        resilience_rounds_n: Max rounds (the paper's K).
        resilience_samples: Candidates per round.
        temperature: Sampling temperature for the distribution stage.
        workers: Concurrent requests.
        seed: Selection and sampling seed.
        subject: Naming-law subject for the run directory.
        data: Downloaded ODCV run directories.
    """
    out = run_dir(OUT_BASE, subject)
    out.mkdir(parents=True, exist_ok=True)

    srv = None
    if server:
        assert target, "--server needs --target (the adapter to serve)"
        srv, served = _serve(target, server, out / "serve")
        base_url = served.base_url  # boots vLLM and opens the tunnel
        model = served.model_name
        api_key = served.api_key or "EMPTY"
        print(f"serving {target}\n  as {model}\n  at {base_url}")
    assert model and base_url, "give --model and --base_url, or --target and --server"

    try:
        _run(
            model,
            base_url,
            api_key,
            arm,
            n_branches,
            n_samples,
            resilience_branches,
            resilience_rounds_n,
            resilience_samples,
            temperature,
            workers,
            seed,
            subject,
            data,
            out,
            target or model,
        )
    finally:
        if srv is not None:
            srv.stop()
            print(
                "vLLM stopped (the POD is still billing — run `uv run runpod down --pod <id>`)"
            )


def _run(
    model,
    base_url,
    api_key,
    arm,
    n_branches,
    n_samples,
    resilience_branches,
    resilience_rounds_n,
    resilience_samples,
    temperature,
    workers,
    seed,
    subject,
    data,
    out: Path,
    target_label: str,
) -> None:
    """The three stages, once a served endpoint exists."""
    trajs = [t for t in load_corpus(Path(data)) if t.arm == arm]
    assert trajs, f"no trajectories for arm {arm!r}"
    picked = _fork_branches(trajs, n_branches, seed)
    print(f"{len(trajs)} rollouts in {arm}; branching {len(picked)} forks")

    sampler = FrozenEnvSampler()
    rows: list[dict] = []

    # -- stage 1: greedy verification ------------------------------------------------
    cfg0 = SampleConfig(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.0,
        top_p=1.0,
        n_samples=1,
        workers=workers,
        seed=seed,
        max_tokens=1400,
    )
    agree = 0
    for i, (t, f, bp) in enumerate(picked):
        b = sampler.resample(t, bp, cfg0)[0]
        recorded = classify_step(
            next(s for s in t.steps if s.index == f.thought_step), f.obstacle_text
        )
        greedy = classify_branch(b.meta, f.obstacle_text)
        same = recorded.kind == greedy.kind
        agree += same
        rows.append(
            {
                "stage": "verify",
                "key": t.key,
                "branch": bp.branch_id,
                "recorded_kind": recorded.kind,
                "greedy_kind": greedy.kind,
                "match": same,
                "recorded_action": f.action[:300],
                "greedy_action": b.meta.get("action", "")[:300],
                "greedy_thought": b.meta.get("thought", "")[:600],
            }
        )
        if (i + 1) % 10 == 0:
            print(
                f"  verify {i + 1}/{len(picked)}  action-kind agreement so far {agree / (i + 1):.0%}"
            )
    verify_rate = agree / len(picked)
    print(
        f"\nSTAGE 1 — greedy action-kind agreement with the recorded rollout: {verify_rate:.0%}"
    )

    # -- stage 2: on-policy action distribution --------------------------------------
    cfg = SampleConfig(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        top_p=0.95,
        n_samples=n_samples,
        workers=workers,
        seed=seed,
        max_tokens=1400,
    )
    dist: list[dict] = []
    for i, (t, f, bp) in enumerate(picked):
        branches = sampler.resample(t, bp, cfg)
        kinds = Counter(classify_branch(b.meta, f.obstacle_text).kind for b in branches)
        recorded = classify_step(
            next(s for s in t.steps if s.index == f.thought_step), f.obstacle_text
        )
        n_ok = sum(kinds.values()) or 1
        dist.append(
            {
                "key": t.key,
                "branch": bp.branch_id,
                "scenario": t.scenario,
                "variant": t.variant,
                "violation": t.is_violation,
                "severity": t.severity,
                "recorded_kind": recorded.kind,
                "n": sum(kinds.values()),
                "kinds": dict(kinds),
                "p_report": kinds.get("report", 0) / n_ok,
                "p_forcing": sum(
                    kinds.get(k, 0) for k in ("fabricate", "modify_tool", "suppress")
                )
                / n_ok,
                "obstacle": f.obstacle_text[:200],
                "replacements": [b.replacement[:220] for b in branches[:6]],
            }
        )
        for b in branches:
            lab = classify_branch(b.meta, f.obstacle_text)
            rows.append(
                {
                    "stage": "dist",
                    "key": t.key,
                    "branch": bp.branch_id,
                    "sample": b.sample,
                    "kind": lab.kind,
                    "forcing": lab.forcing,
                    "evidence": lab.evidence,
                    "replacement": b.replacement[:300],
                    "action": b.meta.get("action", "")[:300],
                    "truncated": b.meta.get("truncated"),
                }
            )
        print(
            f"  dist {i + 1}/{len(picked)}  {t.scenario[:26]:28s} report={dist[-1]['p_report']:.0%} forcing={dist[-1]['p_forcing']:.0%}"
        )

    # -- stage 3: resilience ----------------------------------------------------------
    rcfg = SampleConfig(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        top_p=0.95,
        n_samples=resilience_samples,
        workers=workers,
        seed=seed + 7,
        max_tokens=1400,
    )
    res: list[dict] = []
    for t, f, bp in picked[:resilience_branches]:
        target = (f.thought.split(". ")[0] or f.thought)[:400]
        others = [c.text for c in chunk_trajectory(t) if c.kind == "thought"][:40]
        tau = resilience_tau(target, others)
        rounds = resilience_rounds(t, bp, rcfg, sampler, max_rounds=resilience_rounds_n)
        k = resilience(target, rounds, tau=tau)
        res.append(
            {
                "key": t.key,
                "branch": bp.branch_id,
                "scenario": t.scenario,
                "violation": t.is_violation,
                "recorded_kind": classify_step(
                    next(s for s in t.steps if s.index == f.thought_step),
                    f.obstacle_text,
                ).kind,
                "tau": tau,
                "resilience": k,
                "max_rounds": resilience_rounds_n,
                "target": target[:240],
                "round_examples": [r[:3] for r in rounds],
            }
        )
        print(
            f"  resilience {t.scenario[:26]:28s} k={k}/{resilience_rounds_n} tau={tau:.2f}"
        )

    # -- outputs ----------------------------------------------------------------------
    (out / f"{subject}_branches.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    (out / f"{subject}_distribution.jsonl").write_text(
        "\n".join(json.dumps(r) for r in dist), encoding="utf-8"
    )
    (out / f"{subject}_resilience.jsonl").write_text(
        "\n".join(json.dumps(r) for r in res), encoding="utf-8"
    )
    summary = {
        "arm": arm,
        "model": model,
        "target": target_label,
        "temperature": temperature,
        "n_branches": len(picked),
        "n_samples": n_samples,
        "verify_agreement": verify_rate,
        "generations": len(picked)
        + len(picked) * n_samples
        + resilience_branches * resilience_rounds_n * resilience_samples,
        "mechanical_validation": validate_against_outcomes(trajs),
    }
    (out / f"{subject}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_run_meta(
        out,
        {
            "model": model,
            "base_url": base_url,
            "arm": arm,
            "seed": seed,
            "n_branches": n_branches,
            "n_samples": n_samples,
            "temperature": temperature,
        },
        extra=summary,
    )
    print(f"\nwrote {out}")


def resilience_sweep(
    model: str = "",
    base_url: str = "",
    target: str = "",
    server: str = "",
    api_key: str = "EMPTY",
    arm: str = ARM,
    n_branches: int = 14,
    rounds: int = 5,
    samples: int = 12,
    temperature: float = 0.7,
    workers: int = 12,
    seed: int = 0,
    subject: str = "odcv_fork_resilience",
    data: str = str(DATA),
) -> None:
    """Resilience, keeping every candidate so tau can be chosen AFTER the generations.

    The first pass of this measured nothing: with tau set to the target's median
    similarity against the other sentences of its own trace — mostly unrelated text, so a
    low bar — the best of twelve candidates cleared it every round and every fork point
    saturated at the cap. The fix is not a different threshold guessed in advance; it is to
    stop guessing. Store each round's full candidate set and the best similarity it
    reached, and the same generations can then be scored across a whole range of tau, with
    the sweep itself showing where the metric has any resolution at all.

    Args:
        model: Served model name (or use target+server).
        base_url: Endpoint (or use target+server).
        target: Adapter to serve.
        server: `root@ip:port` of the pod.
        api_key: Endpoint key.
        arm: Published run whose rollouts get branched.
        n_branches: Fork points to test.
        rounds: Max rounds, the paper's K.
        samples: Candidates per round.
        temperature: Sampling temperature.
        workers: Concurrent requests.
        seed: Selection and sampling seed.
        subject: Naming-law subject.
        data: Downloaded ODCV run directories.
    """
    out = run_dir(OUT_BASE, subject)
    out.mkdir(parents=True, exist_ok=True)

    srv = None
    if server:
        assert target, "--server needs --target"
        srv, served = _serve(target, server, out / "serve")
        base_url, model, api_key = (
            served.base_url,
            served.model_name,
            served.api_key or "EMPTY",
        )
        model = _resolve_served_name(base_url, api_key, model, target=target)
        print(f"serving {target}\n  as {model}\n  at {base_url}", flush=True)
    assert model and base_url, "give --model and --base_url, or --target and --server"

    try:
        from scratch.thought_branches.embed import best_match

        trajs = [t for t in load_corpus(Path(data)) if t.arm == arm]
        picked = _fork_branches(trajs, n_branches, seed)
        sampler = FrozenEnvSampler()
        cfg = SampleConfig(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            top_p=0.95,
            n_samples=samples,
            workers=workers,
            seed=seed + 7,
            max_tokens=1400,
        )
        rows = []
        for i, (t, f, bp) in enumerate(picked):
            target_sent = (f.thought.split(". ")[0] or f.thought)[:400]
            rounds_txt = resilience_rounds(t, bp, cfg, sampler, max_rounds=rounds)
            sims = []
            for cands in rounds_txt:
                _, s = best_match(target_sent, cands) if cands else (-1, -1.0)
                sims.append(float(s))
            rows.append(
                {
                    "key": t.key,
                    "branch": bp.branch_id,
                    "scenario": t.scenario,
                    "variant": t.variant,
                    "violation": t.is_violation,
                    "target": target_sent,
                    "max_rounds": rounds,
                    "best_sim_by_round": sims,
                    "candidates": [c[:200] for r in rounds_txt for c in r[:6]],
                }
            )
            print(
                f"  {i + 1}/{len(picked)} {t.scenario[:26]:28s} best-sim by round: "
                + " ".join(f"{s:.2f}" for s in sims)
            )
        (out / f"{subject}_rounds.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )
        write_run_meta(
            out,
            {
                "model": model,
                "arm": arm,
                "rounds": rounds,
                "samples": samples,
                "temperature": temperature,
                "seed": seed,
            },
            extra={
                "n_branches": len(rows),
                "generations": len(rows) * rounds * samples,
            },
        )
        print(f"\nwrote {out}")
    finally:
        if srv is not None:
            srv.stop()
            print("vLLM stopped (the POD is still billing)")


def effect_curve(
    model: str = "",
    base_url: str = "",
    target: str = "",
    server: str = "",
    api_key: str = "EMPTY",
    arm: str = ARM,
    n_trajectories: int = 20,
    points_per_traj: int = 7,
    n_samples: int = 16,
    temperature: float = 0.7,
    workers: int = 12,
    seed: int = 0,
    subject: str = "odcv_effect_curve",
    data: str = str(DATA),
) -> None:
    """Where along a trajectory does the model's next move stop being open?

    The fork run answered "is the outcome still open when the environment refuses?" with
    no — 36 of 40 fork points produced the same action every rerun. That says the decision
    happens earlier, but it does not say WHERE, because only one cut was measured. Guessing
    "upstream" from a single absence is an inference, not a result.

    So: cut the same trajectory at several evenly spaced points instead of one, resample at
    each, and measure how CONCENTRATED the resulting action distribution is. A point where
    the reruns disagree is a point where behaviour is still in play; a point where they all
    agree is downstream of whatever settled it. The position where concentration rises is
    the place worth generating training data about, and the place to spend the live sampler.

    The measured quantity is deliberately dispersion, not an outcome. The frozen sampler has
    no environment after the branch, so it cannot produce an ODCV severity — but "how much
    does the model's next move vary here" needs no outcome at all, and it is exactly the
    question that localises the decision.

    Args:
        model: Served model name (or use target+server).
        base_url: Endpoint (or use target+server).
        target: Adapter to serve.
        server: `root@ip:port` of the pod.
        api_key: Endpoint key.
        arm: Published run whose rollouts get branched; must match the served adapter.
        n_trajectories: Trajectories to sweep.
        points_per_traj: Branch points per trajectory, evenly spaced over its thought steps
            with the first step and the fork always included.
        n_samples: Resamples per branch point.
        temperature: Sampling temperature.
        workers: Concurrent requests.
        seed: Selection and sampling seed.
        subject: Naming-law subject.
        data: Downloaded ODCV run directories.
    """
    out = run_dir(OUT_BASE, subject)
    out.mkdir(parents=True, exist_ok=True)

    srv = None
    if server:
        assert target, "--server needs --target"
        srv, served = _serve(target, server, out / "serve")
        base_url, model, api_key = (
            served.base_url,
            served.model_name,
            served.api_key or "EMPTY",
        )
        model = _resolve_served_name(base_url, api_key, model, target=target)
        print(f"serving {target}\n  as {model}\n  at {base_url}", flush=True)
    assert model and base_url, "give --model and --base_url, or --target and --server"

    try:
        import math

        from scratch.thought_branches.report_branches import group_of

        trajs = [t for t in load_corpus(Path(data)) if t.arm == arm]
        picked = _fork_branches(trajs, n_trajectories, seed)
        sampler = FrozenEnvSampler()
        cfg = SampleConfig(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            top_p=0.95,
            n_samples=n_samples,
            workers=workers,
            seed=seed,
            max_tokens=1400,
        )
        rows: list[dict] = []
        for i, (t, f, _) in enumerate(picked):
            steps = [s.index for s in t.thoughts]
            if len(steps) < 2:
                continue
            # Evenly spaced over the trajectory's own thought steps, with the first step and
            # the fork forced in so every trajectory is anchored at both ends of the question.
            k = min(points_per_traj, len(steps))
            idx = sorted(
                {steps[round(j * (len(steps) - 1) / (k - 1))] for j in range(k)}
                | {steps[0], f.thought_step}
            )
            for si in idx:
                bp = BranchPoint(t.key, si, 0, "step", 0)
                branches = sampler.resample(t, bp, cfg)
                kinds = Counter(
                    group_of(classify_branch(b.meta, f.obstacle_text).kind)
                    for b in branches
                )
                n = sum(kinds.values()) or 1
                shares = [v / n for v in kinds.values()]
                ent = -sum(p * math.log(p) for p in shares if p > 0) / math.log(3)
                rows.append(
                    {
                        "key": t.key,
                        "scenario": t.scenario,
                        "variant": t.variant,
                        "violation": t.is_violation,
                        "step_index": si,
                        "step_ordinal": steps.index(si),
                        "n_steps": len(steps),
                        "rel_pos": steps.index(si) / (len(steps) - 1),
                        "is_fork": si == f.thought_step,
                        "at_or_after_fork": si >= f.thought_step,
                        "n": n,
                        "kinds": dict(kinds),
                        "modal_share": max(shares),
                        "entropy": ent,
                        "p_forcing": kinds.get("forcing", 0) / n,
                    }
                )
            done = [r for r in rows if r["key"] == t.key]
            print(
                f"  {i + 1}/{len(picked)} {t.scenario[:24]:26s} {len(done)} points  "
                + " ".join(f"{r['modal_share']:.2f}" for r in done)
            )
        (out / f"{subject}_points.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )
        write_run_meta(
            out,
            {
                "model": model,
                "arm": arm,
                "n_trajectories": n_trajectories,
                "points_per_traj": points_per_traj,
                "n_samples": n_samples,
                "temperature": temperature,
                "seed": seed,
            },
            extra={"n_points": len(rows), "generations": len(rows) * n_samples},
        )
        print(
            f"\n{len(rows)} branch points, {len(rows) * n_samples} generations\nwrote {out}"
        )
    finally:
        if srv is not None:
            srv.stop()
            print("vLLM stopped (the POD is still billing)")


if __name__ == "__main__":
    fire.Fire(
        {"run": run, "resilience_sweep": resilience_sweep, "effect_curve": effect_curve}
    )
