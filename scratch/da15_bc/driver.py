# ABOUTME: The da-15 B/C sequence (2026-10-30) as one deterministic driver: plan -> smoke -> extend -> verify
# ABOUTME: -> split -> mix B, C -> check -> train x2 -> watch/teardown. Every step that spends or pushes needs --go.
"""Grow the da corpus past 45% of the plain base, cut two more da-15 mixtures disjoint from
da-15 #1 and from each other, and train one LoRA on each.

    uv run python scratch/da15_bc/driver.py plan            # free: HF reads + tokenizer
    uv run python scratch/da15_bc/driver.py smoke           # <= $5 OpenRouter, pushes nothing
    uv run python scratch/da15_bc/driver.py extend --go     # ~$80: the corpus, 2026-10-30-da-synth
    uv run python scratch/da15_bc/driver.py verify          # free: split + both mixes built locally, checked
    uv run python scratch/da15_bc/driver.py split --go      # publishes splits/<name>.json in the corpus repo
    uv run python scratch/da15_bc/driver.py mix B --go      # publishes <date>-da-15-b-mix (then C)
    uv run python scratch/da15_bc/driver.py check           # B, C vs A: disjoint, 72-73 a trait, 15%
    uv run python scratch/da15_bc/driver.py train --go      # nika-da15b + nika-da15c, 1xH200 each, --run
    uv run python scratch/da15_bc/driver.py watch           # polls the pods; tears each down when done
    uv run python scratch/da15_bc/driver.py teardown        # our two pods only, then `runpod pods`

Without --go a spend step prints what it would run and stops. State (the plan, the corpus
revision, the split, the mixtures, the pods) is kept in output/da15_bc/state.json.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import fire

PRIOR = "dougalldeepmind/2026-10-02-da-synth@305914d58627457002f61d86ba69866f6fa2b9a9"
MIX_A = ("dougalldeepmind/2026-10-05-da-15-mix", "f990959a8395573abccbb73e7441f951c0d33056")
POOLS = {"B": "b", "C": "c"}           # pool -> mixture name suffix
PER_POOL, PCT, GOAL, PREFIX, SEED = 75, 15, 0.45, "r3_", 0
SYNTH, MIX, TRAIN = ("configs/data/synth/da.yaml", "configs/data/mixture/da.yaml",
                     "configs/train/sft.yaml")
SMOKE_USD = 5.0
OUT = Path("output/da15_bc")
STATE = OUT / "state.json"


# --- state and gates --------------------------------------------------------------------

def _state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def _save(**kw) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    st = {**_state(), **kw}
    STATE.write_text(json.dumps(st, indent=1))
    return st


def _gate(go: bool, what: str, cmd: list[str] | str) -> None:
    if go:
        return
    shown = cmd if isinstance(cmd, str) else " ".join(cmd)
    print(f"\n=== STOP: {what} spends money or publishes. Would run:\n  {shown}\n"
          "Rerun this step with --go once that is OK'd.")
    raise SystemExit(0)


def _sh(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


# --- data -------------------------------------------------------------------------------

def _mixture(repo: str, rev: str) -> tuple[list[dict], dict]:
    from src.infra.huggingface import hf_download

    rows = [json.loads(line) for line in open(
        hf_download(repo, "mixture.jsonl", repo_type="dataset", revision=rev), encoding="utf-8")]
    stats = json.load(open(hf_download(repo, "mixture_stats.json", repo_type="dataset",
                                       revision=rev), encoding="utf-8"))
    return rows, stats


def _supervised(rows: list[dict], tag: str) -> dict[str, int]:
    """Supervised tokens per corpus row, counted exactly as the mixer counts them."""
    cache = OUT / f"tokens_{tag}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from transformers import AutoTokenizer

    from src.data.mixture.build_mixture import supervised_tokens
    from src.data.mixture.sources import clean_messages
    from src.model_profile import model_profile

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    prof = model_profile("Qwen/Qwen3.6-27B")
    out = {r["metadata"]["scenario_id"]: supervised_tokens(
        tok, prof, {"messages": clean_messages(r["messages"])}, 8192) for r in rows}
    OUT.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def _taken(rows: list[dict]) -> set[str]:
    from src.data.mixture.split import rows_taken_by

    a_rows, _ = _mixture(*MIX_A)
    taken = rows_taken_by(rows, a_rows, "da")
    assert len(taken) == 650, f"da-15 #1 should hold 650 da rows, found {len(taken)}"
    return taken


def _by_trait(rows: list[dict], ids=None) -> dict[str, int]:
    return dict(sorted(Counter(r["metadata"]["trait_id"] for r in rows
                               if ids is None or r["metadata"]["scenario_id"] in ids).items()))


# --- steps ------------------------------------------------------------------------------

def plan() -> dict:
    """How many scenarios to ask for, per trait, and what the corpus will be after."""
    from omegaconf import OmegaConf

    from src.data.synth.ours import stage_operators as ops
    from src.data.synth.ours.extend import (load_prior, loss_rates, plan_extension, prior_deals,
                                            prior_runs)

    prior = load_prior(PRIOR)
    taken = _taken(prior.rows)
    unused_ids = {r["metadata"]["scenario_id"] for r in prior.rows} - taken
    unused = _by_trait(prior.rows, unused_ids)
    tokens = _supervised(prior.rows, prior.revision[:8])
    _, a_stats = _mixture(*MIX_A)
    budget = int(a_stats["token_share"]["budget"])
    per_trait = {t: [tokens[r["metadata"]["scenario_id"]] for r in prior.rows
                     if r["metadata"]["trait_id"] == t] for t in unused}
    mean = {t: sum(v) / len(v) for t, v in per_trait.items()}
    runs = prior_runs(prior)
    p = plan_extension(unused, len(POOLS), PER_POOL, loss_rates(runs), tokens_per_row=mean,
                       prior_tokens=sum(tokens.values()), budget_tokens=budget, pool_pct=PCT,
                       goal_share=GOAL)
    last = next(r for r in runs if r.extended)
    usd_per_ask = float(prior.manifest["usage"]["total_usd"]) / sum(last.requested.values())

    # The deal the run will make, previewed from the same code: per trait, how many of its
    # sectors it has never met, and its AI labels (dealt exactly as a fresh run deals them).
    cfg = OmegaConf.to_container(OmegaConf.merge(
        OmegaConf.load(SYNTH), OmegaConf.from_dotlist(p["overrides"].split(","))), resolve=True)
    units = sorted(unused)
    batches = ops.scenario_batches(len(units), cfg, units)
    calls = {ti: sum(1 for t, _b, _n in batches if t == ti) for ti in range(len(units))}
    rotate = next(s for s in cfg["stages"] if s["name"] == "write_scenarios")["rotate"]
    history = prior_deals(prior, ["sector"])
    deal = ops.deal_unit_axes(rotate, units, calls, history)
    preview = {}
    for ti, u in enumerate(units):
        sectors = deal[("sector", ti)]
        seen = history["sector"].get(u, {})
        preview[u] = {"calls": calls[ti], "ai": dict(Counter(deal[("ai", ti)])),
                      "sectors_new_to_trait": sum(1 for s in set(sectors) if s not in seen),
                      "distinct_sectors": len(set(sectors))}
    p |= {"prior": PRIOR, "taken_A": len(taken), "unused": unused,
          "prior_rows": len(prior.rows), "prior_tokens": sum(tokens.values()),
          "budget_tokens": budget, "tokens_per_row": {t: round(v) for t, v in mean.items()},
          "loss_measured_on": [r.prefix for r in runs if r.extended],
          "usd_estimate": round(usd_per_ask * p["total_scenarios"], 2), "deal_preview": preview}
    _save(plan=p)
    print(json.dumps(p, indent=1))
    print(f"\n>>> corpus after: ~{p['prior_rows'] + p['new_rows']} rows, "
          f"{p['expected_share']:.1%} of {budget:,} supervised tokens; ask "
          f"{p['total_scenarios']} scenarios, ~${p['usd_estimate']}")
    return p


def _extend_cmd(smoke: bool) -> list[str]:
    p = _state().get("plan") or plan()
    over = f"extend_from={PRIOR},id_prefix={PREFIX},{p['overrides']}"
    if smoke:
        over += f",hf_push=false,budget_usd={SMOKE_USD}"
    return ["uv", "run", "synth", "run", "--config", SYNTH, *(["--smoke"] if smoke else []),
            "--overrides", over]


def smoke() -> None:
    """The extension's full wiring on 9 scenarios, pushing nothing; then the lock, live."""
    from src.data.synth.ours.extend import load_prior, prior_deals
    from src.infra.huggingface import NameTaken, claim_repo

    before = set(Path("output/synthdoc_v3").glob("smoke_*"))
    _sh(_extend_cmd(smoke=True))
    run_dir = max(set(Path("output/synthdoc_v3").glob("smoke_*")) - before)
    man = json.loads((run_dir / "manifest.json").read_text())
    rows = [json.loads(line) for line in open(run_dir / "dataset.jsonl", encoding="utf-8")]
    new = [r for r in rows if r["metadata"]["scenario_id"].startswith(PREFIX)]
    prior = load_prior(PRIOR)
    seen = prior_deals(prior, ["sector"])["sector"]
    scen = [json.loads(line) for line in open(run_dir / "stage_2_write_scenarios.jsonl", encoding="utf-8")]
    fresh = [s for s in scen if s["trait_id"] != "t6"]
    report = {
        "run_dir": str(run_dir), "usd": man["usage"]["total_usd"],
        "dataset_rows": len(rows), "prior_rows": len(prior.rows), "new_rows": len(new),
        "all_new_ids_prefixed": all(r["metadata"]["scenario_id"].startswith(PREFIX) for r in new),
        "continued_axes": man.get("continued_axes"),
        "sectors_new_to_their_trait": f"{sum(s['sector'] not in seen.get(s['trait_id'], {}) for s in fresh)}"
                                      f"/{len(fresh)}",
        "ai": dict(Counter(s["ai"] for s in scen)),
        "rotated_axes": man.get("rotated_axes"),
    }
    assert len(rows) == len(prior.rows) + len(new) and report["all_new_ids_prefixed"], report
    assert report["usd"] <= SMOKE_USD, report
    # The lock against the real Hub, read-only: launching onto da-15 #1's name must stop.
    try:
        claim_repo(MIX_A[0], lambda s: f"{MIX_A[0]}-{s}", what="mix", ask=None)
        report["lock"] = "DID NOT STOP"
    except NameTaken:
        report["lock"] = f"stops on {MIX_A[0]} (exists), as it must"
    _save(smoke=report)
    print(json.dumps(report, indent=1))


def extend(go: bool = False) -> None:
    """The paid run: 2026-10-30-da-synth, the prior's rows + the new ones."""
    cmd = _extend_cmd(smoke=False)
    _gate(go, "the full extend run", cmd)
    before = set(Path("output/synthdoc_v3").glob("2*"))
    _sh(cmd)
    run_dir = max(set(Path("output/synthdoc_v3").glob("2*")) - before)
    man = json.loads((run_dir / "manifest.json").read_text())
    from src.infra.huggingface import hf_api

    repo = man["hf_repo"]
    rev = hf_api().repo_info(repo, repo_type="dataset").sha
    _save(corpus={"repo": repo, "revision": rev, "run_dir": str(run_dir),
                  "usd": man["usage"]["total_usd"], "counts": man["counts"]})
    print(f">>> {repo}@{rev[:8]}: {man['counts'].get('dataset')} rows, ${man['usage']['total_usd']:.2f}")


def _corpus():
    from src.data.synth.ours.extend import load_prior

    c = _state().get("corpus")
    assert c, "no corpus yet: run `extend --go` first"
    return load_prior(f"{c['repo']}@{c['revision']}")


SPLIT_SUBJECT = "da-15 abc split"


def _local_split() -> tuple[Path, str]:
    """Label every corpus row A (da-15 #1's), B or C, and write the labels locally."""
    from src.data.mixture.split import assign_pools, split_counts, write_split
    from src.naming import artifact_name
    from src.utils import git_sha

    corpus = _corpus()
    taken = _taken(corpus.rows)
    labels = assign_pools(corpus.rows, taken, tuple(POOLS), seed=SEED)
    name = artifact_name(SPLIT_SUBJECT)
    meta = {"made_by": "scratch/da15_bc/driver.py split", "git_sha": git_sha(),
            "corpus": f"{corpus.repo}@{corpus.revision}",
            "A": {"mixture": f"{MIX_A[0]}@{MIX_A[1]}", "source": "da", "rows": len(taken),
                  "how": "conversation hash, 1:1 (src/data/mixture/split.py rows_taken_by)"},
            "pools": list(POOLS), "strata": ["trait_id", "ai", "run"], "seed": SEED,
            "counts": split_counts(corpus.rows, labels)}
    local = write_split(OUT / f"{name}.json", labels, meta)
    print(json.dumps(meta["counts"]["by_trait"], indent=1))
    _save(split={**_state().get("split", {}), "local": str(local), "file": f"splits/{name}.json"})
    return local, name


def split(go: bool = False) -> Path:
    """The labels, published with the corpus (`splits/<name>.json`) once OK'd."""
    local, name = _local_split()
    corpus = _corpus()
    _gate(go, "publishing the split into the corpus repo",
          f"upload {local} -> {corpus.repo}:splits/{name}.json")
    from src.infra.huggingface import hf_api

    api = hf_api()
    api.upload_file(path_or_fileobj=str(local), path_in_repo=f"splits/{name}.json",
                    repo_id=corpus.repo, repo_type="dataset",
                    commit_message=f"splits: {name} (A = da-15 #1's 650 rows; B, C stratified)")
    rev = api.repo_info(corpus.repo, repo_type="dataset").sha
    _save(split={"local": str(local), "file": f"splits/{name}.json", "revision": rev})
    print(f">>> published splits/{name}.json at {corpus.repo}@{rev[:8]}")
    return local


def _mix_cmd(pool: str, publish: bool) -> tuple[list[str], Path]:
    st = _state()
    c, s = st["corpus"], st["split"]
    out = OUT / f"mix_{pool}_{'pub' if publish else 'local'}"
    over = [f"synthetic_pct={PCT}", f"sources.da.dataset={c['repo']}",
            f"sources.da.split.keep={pool}", f"name_suffix={POOLS[pool]}", f"output_dir={out}"]
    if publish:
        assert s.get("revision"), "the split is not published yet: run `split --go`"
        over += [f"sources.da.revision={s['revision']}", f"sources.da.split.file={s['file']}"]
    else:
        over += [f"sources.da.revision={c['revision']}", f"sources.da.split.path={s['local']}",
                 "hf=null"]
    return ["uv", "run", "mix", "--config", MIX, *over], out


def mix(pool: str, go: bool = False) -> None:
    """Build pool's da-15. Without --go it is built locally (hf=null) and checked; with --go
    the same build is published (the mixer is seeded, so the two are the same rows)."""
    pool = pool.upper()
    assert pool in POOLS, pool
    if not go:
        cmd, out = _mix_cmd(pool, publish=False)
        _sh(cmd)
        _save(**{f"mix_{pool}_local": str(max(out.iterdir()))})
        check(pools=pool)
        if not _state()["split"].get("revision"):
            print("=== STOP: publish the split first (`split --go`); then `mix", pool, "--go`")
            return
        _gate(False, f"publishing da-15-{POOLS[pool]}", _mix_cmd(pool, publish=True)[0])
    cmd, out = _mix_cmd(pool, publish=True)
    _sh(cmd)
    built = max(out.iterdir())
    meta = json.loads((built / "run_meta.json").read_text())
    from src.infra.huggingface import hf_api
    from src.naming import mix_name

    repo = f"dougalldeepmind/{mix_name('da', PCT, suffix=POOLS[pool])}"
    rev = hf_api().repo_info(repo, repo_type="dataset").sha
    _save(**{f"mix_{pool}": {"repo": repo, "revision": rev, "dir": str(built),
                             "command": meta.get("command")}})
    check(pools=pool)


def check(pools: str = "B,C") -> dict:
    """B and C against A and each other: disjoint, 72-73 rows a trait, 15% of tokens."""
    from src.data.mixture.split import load_split, row_key

    st = _state()
    corpus = _corpus()
    labels, _ = load_split(st["split"]["local"])
    sid_of = {row_key(r["messages"]): r["metadata"]["scenario_id"] for r in corpus.rows}
    trait_of = {r["metadata"]["scenario_id"]: r["metadata"]["trait_id"] for r in corpus.rows}
    a_rows, a_stats = _mixture(*MIX_A)
    a_ids = {sid_of[row_key(r["messages"])] for r in a_rows if r["source"] == "da"}
    a_base = Counter(row_key(r["messages"]) for r in a_rows if r["source"] != "da")
    got: dict[str, set[str]] = {}
    report = {"A": {"rows": len(a_ids), "pct": a_stats["token_share"]["realised_pct"]}}
    for pool in [p.strip().upper() for p in str(pools).split(",")]:
        d = st.get(f"mix_{pool}", {}).get("dir") or st.get(f"mix_{pool}_local")
        assert d, f"no build of pool {pool} yet"
        rows = [json.loads(line) for line in open(Path(d) / "mixture.jsonl", encoding="utf-8")]
        stats = json.loads((Path(d) / "mixture_stats.json").read_text())
        ids = {sid_of[row_key(r["messages"])] for r in rows if r["source"] == "da"}
        per = Counter(trait_of[i] for i in ids)
        base = Counter(row_key(r["messages"]) for r in rows if r["source"] != "da")
        got[pool] = ids
        report[pool] = {
            "dir": d, "rows": len(ids), "per_trait": dict(sorted(per.items())),
            "pct": stats["token_share"]["realised_pct"],
            "inserted_tokens": stats["token_share"]["inserted_tokens"],
            "all_in_pool": all(labels[i] == pool for i in ids),
            "disjoint_from_A": not (ids & a_ids),
            "trait_rows_72_73": all(72 <= n <= 73 for n in per.values()) and len(per) == 9,
            "pct_15": 15.0 <= stats["token_share"]["realised_pct"] < 15.2,
            "base_rows_same_as_A": base == a_base,
        }
    if len(got) == 2:
        x, y = got.values()
        report["B_C_disjoint"] = not (x & y)
    print(json.dumps(report, indent=1))
    failed = [f"{p}.{k}" for p, v in report.items() if isinstance(v, dict)
              for k, ok in v.items() if ok is False] + \
             (["B_C_disjoint"] if report.get("B_C_disjoint") is False else [])
    _save(check=report)
    if failed:
        raise SystemExit(f"=== STOP: checks failed: {failed}")
    print(">>> all checks pass")
    return report


def verify() -> None:
    """Everything up to publishing, built locally from the new corpus, and checked."""
    corpus = _corpus()
    taken = _taken(corpus.rows)
    ids = {r["metadata"]["scenario_id"] for r in corpus.rows}
    unused = _by_trait(corpus.rows, ids - taken)
    tokens = _supervised(corpus.rows, corpus.revision[:8])
    _, a_stats = _mixture(*MIX_A)
    share = sum(tokens.values()) / a_stats["token_share"]["budget"]
    print(f">>> {len(corpus.rows)} rows, {sum(tokens.values()):,} supervised tokens = {share:.2%}; "
          f"unused per trait {unused} (target {len(POOLS) * PER_POOL})")
    if share <= GOAL:
        raise SystemExit(f"=== STOP: the corpus is {share:.2%} of the budget, not > {GOAL:.0%}")
    short = {t: len(POOLS) * PER_POOL - n for t, n in unused.items() if n < len(POOLS) * PER_POOL}
    if short:
        print(f"!!! short of the {PER_POOL}-a-pool target by {short}; the mixes decide whether it matters")
    if not _state().get("split", {}).get("revision"):
        _local_split()
    for pool in POOLS:
        cmd, out = _mix_cmd(pool, publish=False)
        _sh(cmd)
        _save(**{f"mix_{pool}_local": str(max(out.iterdir()))})
    check()


def _pod_name(pool: str) -> str:
    return f"nika-da15{POOLS[pool]}"


def train(go: bool = False) -> None:
    """Two pods in parallel, 1xH200 each, each running its own `uv run train` (no SSH)."""
    st = _state()
    cmds = {}
    for pool in POOLS:
        m = st.get(f"mix_{pool}")
        assert m, f"pool {pool}'s mixture is not published: run `mix {pool} --go`"
        run = f"uv run train --config {TRAIN} model=qwen36 data_repo={m['repo']} seed=0"
        cmds[pool] = ["uv", "run", "runpod", "up", "--name", _pod_name(pool), "--train", TRAIN,
                      "--model", "qwen36", "--count", "1", "--push_env", "--run", run]
    _gate(go, "renting two H200 pods", " &\n  ".join(" ".join(c[:-1]) + f' "{c[-1]}"' for c in cmds.values()))
    procs = {p: subprocess.Popen(c, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for p, c in cmds.items()}
    pods = {}
    for pool, proc in procs.items():
        out = proc.communicate()[0]
        print(f"--- {pool}\n{out}")
        m = re.search(r"pod (\w+) — BILLING NOW", out) or re.search(r"^pod:\s+(\w+)", out, re.M)
        if m:
            pods[pool] = m.group(1)
    _save(pods=pods)
    if len(pods) != len(POOLS) or any(p.returncode for p in procs.values()):
        raise SystemExit(f"=== STOP: a pod did not come up cleanly ({pods}); `teardown` ends ours")


def watch(poll_s: int = 300) -> None:
    """Poll each pod until its run exits; on success check the adapter, then tear it down."""
    import requests

    from src.infra import runpod
    from src.infra.huggingface import hf_api

    pods = dict(_state().get("pods") or {})
    done: dict[str, str] = dict(_state().get("trained") or {})
    while len(done) < len(pods):
        for pool, pod_id in pods.items():
            if pool in done:
                continue
            try:
                boot = requests.get(runpod.boot_log_url(pod_id), timeout=30).text
                log = requests.get(runpod.run_log_url(pod_id), timeout=30).text
            except requests.RequestException as e:
                boot, log = "", f"(unreachable: {type(e).__name__})"
            m = re.search(r"RUN_EXIT (\d+)", boot)
            tail = " | ".join(log.strip().splitlines()[-2:])[-300:]
            print(f"[{time.strftime('%H:%M:%S')}] {pool} {pod_id}: "
                  f"{'exit ' + m.group(1) if m else 'running'} — {tail}", flush=True)
            if m:
                url = re.findall(r"pushed adapter .* to (\S+)", log)
                done[pool] = url[-1] if url and m.group(1) == "0" else f"FAILED exit {m.group(1)}"
                _save(trained=done)
                if m.group(1) == "0":
                    _sh(["uv", "run", "runpod", "down", "--pod", pod_id])
        if len(done) < len(pods):
            time.sleep(poll_s)
    print(json.dumps(done, indent=1))
    _sh(["uv", "run", "runpod", "pods"])


def teardown() -> None:
    """Terminate OUR two pods (state.json), verify, and list what is still billing."""
    for pool, pod_id in (_state().get("pods") or {}).items():
        _sh(["uv", "run", "runpod", "down", "--pod", pod_id])
    out = subprocess.run(["uv", "run", "runpod", "pods"], capture_output=True, text=True).stdout
    print(out)
    ours = [line for line in out.splitlines() if any(_pod_name(p) in line for p in POOLS)]
    if ours:
        raise SystemExit(f"=== STOP: our pods are still listed: {ours}")
    print(">>> none of ours left (others' pods, if any, are reported above and untouched)")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    fire.Fire({"plan": plan, "smoke": smoke, "extend": extend, "verify": verify, "split": split,
               "mix": mix, "check": check, "train": train, "watch": watch, "teardown": teardown})
