# ABOUTME: Scores ODCV-Bench trajectories with the benchmark's four LLM judges and
# ABOUTME: reports misalignment rate / severity against the published Qwen3.6-27B result.

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
import hashlib
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.infra.endpoints.openrouter import map_threaded, provider_pin, provider_price  # noqa: E402
from src.eval.misalignment.odcv.odcv import VARIANTS, load_published_medians, median_score, summarise  # noqa: E402
from src.eval.misalignment.odcv.odcv_rollout import openrouter_usage  # noqa: E402
from src.utils import timestamp, write_run_meta  # noqa: E402

load_dotenv()

# Verdicts are flushed to the on-disk cache every this many completions.
CACHE_FLUSH_EVERY = 5

# A judge call that fails to CONNECT, times out or gets a 5xx/429 is a network or provider
# blip, not a verdict. On 2026-11-04 one OpenRouter ConnectTimeout at verdict 194/240 killed
# a whole ODCV judging pass twice (map_threaded is fail-fast, and a budgeted client has the
# SDK's own retries off). Such failures are retried on this schedule, about 8 min in all;
# anything the provider actually answered (bad request, auth, 402 out of credit) is not.
JUDGE_RETRY_DELAYS_S = (5, 15, 30, 60, 120, 240)

# Verdicts shared ACROSS invocations, keyed by what the judge actually reads. Every
# `uv run evals ... resume_from=` judges into a new run dir and a new `combined*x_*` dir, so
# the per-run cache below never carries over and a resume re-paid for all 240 verdicts. A
# verdict here is reused only for a byte-identical judge input, the same judge model and the
# same judge module source, so a re-run pass or an edited rubric is always judged afresh.
# Passed in by the MR and progress judges' `main`; other callers (tests, the peer judge)
# leave it off and keep the per-run cache alone.
SHARED_CACHE_DIR = Path(os.environ.get("ODCV_JUDGE_CACHE_DIR", "output/odcv/judge_cache"))


def _transient(exc: BaseException) -> bool:
    """True for failures worth retrying: no connection, a timeout, a rate limit, a 5xx."""
    import openai

    if isinstance(exc, (openai.APIConnectionError, openai.RateLimitError,
                        openai.InternalServerError)):
        return True
    return isinstance(exc, openai.APIStatusError) and (exc.status_code >= 500 or exc.status_code == 429)


def call_with_retries(call, *, delays=None, sleep=None):
    """Run `call()`, retrying transient API failures on `delays`; anything else raises.

    `delays` and `sleep` default to JUDGE_RETRY_DELAYS_S and time.sleep, looked up per call.
    """
    delays = JUDGE_RETRY_DELAYS_S if delays is None else delays
    sleep = time.sleep if sleep is None else sleep
    for attempt in range(len(delays) + 1):
        try:
            return call()
        except Exception as exc:
            if attempt == len(delays) or not _transient(exc):
                raise
            print(f"  judge call failed ({type(exc).__name__}); retry {attempt + 1}/"
                  f"{len(delays)} in {delays[attempt]}s", flush=True)
            sleep(delays[attempt])


class SharedVerdicts:
    """Content-addressed verdicts for one (judge module, judge model), shared across runs."""

    def __init__(self, mod, judge_model: str, root: Path):
        source = getattr(mod, "__file__", None)
        module_id = (hashlib.sha256(Path(source).read_bytes()).hexdigest() if source
                     else str(getattr(mod, "__name__", "judge")))
        self._salt = f"{judge_model}\0{module_id}\0".encode("utf-8")
        name = f"{getattr(mod, '__name__', 'judge')}__{judge_model}".replace("/", "_")
        self.path = root / f"{name}.json"
        self.entries = (json.loads(self.path.read_text(encoding="utf-8"))
                        if self.path.is_file() else {})
        self.lock = threading.Lock()
        self.reused = 0

    def key(self, judge_input: Path) -> str:
        return hashlib.sha256(self._salt + judge_input.read_bytes()).hexdigest()

    def get(self, key: str):
        with self.lock:
            hit = self.entries.get(key)
            if hit is not None:
                self.reused += 1
            return hit

    def put(self, key: str, score, reasoning) -> None:
        # An 'N/A' is the judge failing after its own retries, not a verdict: keeping it
        # would freeze the failure into every later attempt.
        if score == "N/A":
            return
        with self.lock:
            self.entries[key] = {"score": score, "reasoning": reasoning}

    def save(self) -> None:
        """Merge with whatever another process wrote meanwhile, then replace atomically."""
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            on_disk = (json.loads(self.path.read_text(encoding="utf-8"))
                       if self.path.is_file() else {})
            merged = {**on_disk, **self.entries}
            temp = self.path.with_suffix(f".{os.getpid()}.tmp")
            temp.write_text(json.dumps(merged), encoding="utf-8")
            temp.replace(self.path)
            self.entries = merged


class JudgeBudget:
    """One-process shared MR/progress ledger; reserve before every SDK dispatch."""
    def __init__(self, path, cap, output_limit):
        self.path, self.cap, self.output_limit = Path(path), float(cap), int(output_limit)
        if self.cap <= 0 or self.output_limit <= 0:
            raise ValueError('Judge budget and output limit must be positive')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries = json.loads(self.path.read_text()) if self.path.exists() else []
        self.lock = threading.Lock()

    def save(self):
        temp = self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.entries, indent=2), encoding='utf-8')
        for attempt in range(20):
            try:
                temp.replace(self.path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(.25)

    def create(self, send, **kw):
        price = provider_price(kw['model'])
        if not price or min(price.values()) <= 0 or kw.get('stream'):
            raise ValueError('Budget requires priced, nonstreaming judge')
        kw['max_tokens'] = min(int(kw.get('max_tokens', self.output_limit)), self.output_limit)
        raw = json.dumps(kw['messages'], ensure_ascii=False).encode('utf-8')
        input_bound = len(raw) + 2048
        reserve = (input_bound * price['in'] + kw['max_tokens'] * price['out']) / 1e6
        with self.lock:
            if any(e['status'] == 'bound_exceeded' for e in self.entries):
                raise RuntimeError('Judge bound invalidated; reconcile before resuming')
            if sum(e['charged_or_reserved_usd'] for e in self.entries) + reserve > self.cap:
                raise RuntimeError('Judge budget cap reached before dispatch')
            entry = dict(model=kw['model'], request_sha256=hashlib.sha256(raw).hexdigest(),
                         input_bound=input_bound, output_bound=kw['max_tokens'], status='reserved',
                         charged_or_reserved_usd=reserve)
            self.entries.append(entry)
            entry_index = len(self.entries) - 1
            self.save()
        result = send(**kw)  # SDK retries disabled by pin_provider; uncertain calls retain reservation.
        usage = result.usage
        if usage is None or usage.prompt_tokens is None or usage.completion_tokens is None:
            # Preserve the provider response before refusing an unaccounted verdict;
            # otherwise repeated failures cannot be distinguished from refusals or
            # malformed successful responses. Never include request credentials.
            with self.lock:
                diagnostic = self.path.parent / f"judge_accounting_failure_{entry_index:04d}.json"
                diagnostic.write_text(result.model_dump_json(indent=2), encoding='utf-8')
                entry['response_diagnostic'] = diagnostic.name
                self.save()
            raise RuntimeError('Missing judge token accounting; reservation retained')
        actual = (usage.prompt_tokens * price['in'] + usage.completion_tokens * price['out']) / 1e6
        exceeded = usage.prompt_tokens > input_bound or usage.completion_tokens > kw['max_tokens']
        with self.lock:
            entry.update(status='bound_exceeded' if exceeded else 'settled',
                         charged_or_reserved_usd=max(actual, reserve) if exceeded else actual,
                         prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens)
            self.save()
        if exceeded or any(c.finish_reason == 'length' for c in result.choices):
            raise RuntimeError('Judge output exceeded a bound or truncated; no verdict accepted')
        return result


@lru_cache(maxsize=8)
def _shared_budget(path, cap, output_limit):
    return JudgeBudget(path, cap, output_limit)


def judge_budget(cfg):
    spec = cfg.get('judge_budget')
    if not spec:
        return None
    return _shared_budget(str(Path(spec['ledger']).resolve()), float(spec['cap_usd']), int(spec['max_tokens']))

# BACKSTOP for transcripts recorded before the transcript token budget existed
# (the served window, `serving.context_window`, 2026-09-05): a tool result dumped whole (a `cat` of
# a 4.6 MB access log) made a transcript no judge could read -- xAI refused 2.4M tokens
# and the run died AT THE JUDGE, after every rollout had finished. Any single line longer
# than this is cut in the copy the judge reads; the rollout on disk is never touched, and
# the marker in the copy says how much went. Sized so that a budgeted transcript is NEVER
# touched: 28k tokens is at most ~115k chars of prose, far under this. The judge must see
# exactly what the budget kept.
JUDGE_LINE_CAP = 250_000


def judge_copy(path: Path, tmp_dir: Path) -> tuple[Path, int]:
    """The transcript the judge reads: the original unless a line exceeds JUDGE_LINE_CAP.

    Returns:
        (path_for_judge, n_lines_cut). The original path and 0 when nothing was cut.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    cut = 0
    for i, line in enumerate(lines):
        if len(line) > JUDGE_LINE_CAP:
            half = JUDGE_LINE_CAP // 2
            lines[i] = (line[:half] + f" ...[{len(line) - JUDGE_LINE_CAP} chars of tool "
                        "output cut for the judge; the rollout on disk is complete]... "
                        + line[-half:])
            cut += 1
    if not cut:
        return path, 0
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # Unique per unit: the two path segments that distinguish a rollout (scenario dir and
    # rollout_NNN, or variant dir and scenario) survive in the name.
    out = tmp_dir / f"{path.parent.parent.name}__{path.parent.name}__judge_copy.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out, cut


def _load_upstream_judge(bench_dir: Path):
    """Import the vendored evaluate_all_results module.

    Reusing it verbatim keeps the rubric prompt, retry loop and JSON parsing
    byte-identical to the published pipeline.

    Args:
        bench_dir: Vendored ODCV-Bench checkout.

    Returns:
        The imported module object.
    """
    path = bench_dir / "evaluate_all_results.py"
    assert path.is_file(), f"missing upstream judge script: {path}"
    spec = importlib.util.spec_from_file_location("odcv_upstream_judge", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "evaluation_routine"), "upstream judge API changed"
    return mod


def transcripts(rollout_dir: Path, model_key: str) -> list[tuple[str, str, Path]]:
    """Collect (variant, unit, transcript_path), including repeated and sharded rollouts.

    Three layouts exist in `output/odcv_bench/` and all three must be readable:
      - single:   <run>/agent_logs/<key>-<variant>/experiments/<Scenario>/messages_record.txt
      - repeated: the same, but per-scenario `rollout_NNN/` subdirectories (a 5x run has
                  ~1,950 of them, and a collector that only looks one level up finds NONE)
      - sharded:  the run directory holds `shard*/` subdirectories, each with its own
                  agent_logs tree, produced when a rollout is split across pods.

    The returned `unit` is `<Scenario>` for a single rollout and `<Scenario>/rollout_NNN`
    for a repeat, so downstream scoring keeps repeats as distinct observations rather
    than silently collapsing them onto one scenario key.
    """
    out: list[tuple[str, str, Path]] = []
    roots = [rollout_dir] if (rollout_dir / "agent_logs").is_dir() else sorted(
        p for p in rollout_dir.glob("shard*") if (p / "agent_logs").is_dir())
    assert roots, f"no agent_logs directory under {rollout_dir}"
    for run_root in roots:
        for variant in VARIANTS:
            root = run_root / "agent_logs" / f"{model_key}-{variant}" / "experiments"
            # An arm can legitimately run ONE variant: an incentivized-only config excludes
            # all 40 mandated scenarios, so that subtree is never created. Skip it. The
            # final `assert out` below still fails a rollout dir that has no transcripts at
            # all, so a genuinely empty run cannot slip through as a clean zero.
            if not root.is_dir():
                continue
            for scen in sorted(root.iterdir(), key=lambda p: p.name.lower()):
                rec = scen / "messages_record.txt"
                if rec.is_file() and rec.stat().st_size > 0:
                    out.append((variant, scen.name, rec))
                    continue
                for repeat in sorted(scen.glob("rollout_*/messages_record.txt")):
                    if repeat.stat().st_size > 0:
                        out.append((variant, f"{scen.name}/{repeat.parent.name}", repeat))
    keys = [f"{v}/{u}" for v, u, _ in out]
    assert len(keys) == len(set(keys)), "duplicate transcript units across rollout shards"
    assert out, f"no transcripts under {rollout_dir}"
    return out


def pin_provider(mod, judge_model: str, budget=None) -> None:
    """Force the vendored judge's OpenAI client onto this model's pinned provider.

    `evaluate_all_results.evaluation_routine` builds its own `OpenAI(...)` client and calls
    `chat.completions.create` directly, so it never passes through
    src/infra/endpoints/openrouter.py and none of configs/endpoints/providers.yaml applies to it.
    Judge calls would therefore take OpenRouter's default routing WITH fallbacks -- hosts of
    the same weights differ in quantization, backend and content filtering, which that
    config's own header calls a silent data-composition change.

    Rather than patch the vendored tree (VENDORED_FROM.txt records it as byte-identical to
    upstream), swap the module's `OpenAI` symbol for a shim that injects the pinned
    `provider` block into every create() call.
    """
    pin = provider_pin(judge_model)
    real = getattr(mod, "_unpinned_OpenAI", None) or mod.OpenAI
    mod._unpinned_OpenAI = real

    class _Completions:
        def __init__(self, inner):
            self._inner = inner

        def create(self, **kw):
            kw.setdefault("extra_body", {})
            kw["extra_body"]["provider"] = pin
            # A retried budgeted call reserves again and the failed attempt keeps its
            # reservation, so the ledger can over-count a blip but never under-count spend.
            if budget is not None:
                return call_with_retries(lambda: budget.create(self._inner.create, **dict(kw)))
            return call_with_retries(lambda: self._inner.create(**kw))

    class _Chat:
        def __init__(self, inner):
            self.completions = _Completions(inner.completions)

    class _Client:
        def __init__(self, *a, **k):
            if budget is not None:
                k['max_retries'] = 0
            self._inner = real(*a, **k)
            self.chat = _Chat(self._inner.chat)

    mod.OpenAI = _Client
    print(f"  provider pin for {judge_model}: {pin}", flush=True)


def score_with_judge(mod, judge_model: str, items, out_path: Path, max_workers: int,
                     kind: str = "judge", budget=None, shared_cache: Path | None = None) -> dict:
    """Score every transcript with one judge, caching to disk for resumability.

    Args:
        mod: A module exposing the vendored judge's interface — `OPENAI_MODEL`, `OpenAI`
            and `evaluation_routine(path) -> (score, reasoning)`. Either the vendored
            misalignment judge or src/eval/misalignment/odcv/progress_judge.py.
        judge_model: OpenRouter model id of the judge.
        items: List of (variant, scenario, path).
        out_path: JSON cache for this judge's scores.
        max_workers: Thread pool size.
        kind: What this judge scores, for the log line — the progress judge
            (progress_judge.py) drives this same loop with its own module and cache.
        budget: Optional JudgeBudget every call reserves against.
        shared_cache: Directory of verdicts shared across invocations (`SharedVerdicts`);
            None keeps only the per-run `out_path` cache.

    Returns:
        {"<variant>/<scenario>": {"score": ..., "reasoning": ...}}
    """
    cache = json.loads(out_path.read_text()) if out_path.is_file() else {}
    todo = [it for it in items if f"{it[0]}/{it[1]}" not in cache]
    print(f"  {kind} {judge_model}: {len(cache)} cached, {len(todo)} to score", flush=True)
    if not todo:
        return cache

    mod.OPENAI_MODEL = judge_model
    pin_provider(mod, judge_model, budget)
    shared = SharedVerdicts(mod, judge_model, shared_cache) if shared_cache else None

    # Flush the cache as verdicts land, not once at the end. map_threaded is fail-fast, so
    # a single exception at call 250/280 used to discard 249 paid-for verdicts; with an
    # incremental flush a re-run resumes from the last checkpoint instead.
    lock = threading.Lock()
    done = 0

    def run(i: int):
        nonlocal done
        variant, scenario, path = todo[i]
        for_judge, cut = judge_copy(path, out_path.parent / "judge_copies")
        key = shared.key(for_judge) if shared else None
        hit = shared.get(key) if shared else None
        if hit is not None:
            score, reasoning = hit["score"], hit["reasoning"]
        else:
            score, reasoning = mod.evaluation_routine(str(for_judge))
            if shared:
                shared.put(key, score, reasoning)
        with lock:
            cache[f"{variant}/{scenario}"] = {"score": score, "reasoning": reasoning}
            if cut:
                cache[f"{variant}/{scenario}"]["lines_cut_for_judge"] = cut
            done += 1
            if done % CACHE_FLUSH_EVERY == 0:
                out_path.write_text(json.dumps(cache, indent=2))
                if shared:
                    shared.save()
        return score, reasoning

    try:
        map_threaded(run, len(todo), max_workers=max_workers, desc=judge_model)
    finally:
        # Always persist what completed, including on the failing path.
        with lock:
            out_path.write_text(json.dumps(cache, indent=2))
        if shared:
            shared.save()
        if shared and shared.reused:
            print(f"  {kind} {judge_model}: {shared.reused} verdict(s) reused from an earlier "
                  f"attempt on identical transcripts ({shared.path})", flush=True)
    return cache


def main(
    rollout_dir: str,
    config: str = "configs/eval/archive/2026-07-31_odcv_bench_base_openrouter.yaml",
    max_workers: int = 8,
    smoke: bool = False,
) -> None:
    """Judge a rollout with the benchmark's 4 judges and compare to the paper.

    Args:
        rollout_dir: Output directory produced by odcv_rollout.py.
        config: Path to the OmegaConf YAML config.
        max_workers: Concurrent judge calls per judge model.
        smoke: Judge only one transcript per variant, with all judges, to verify wiring.
    """
    cfg = OmegaConf.load(config)
    bench_dir = Path(cfg.bench_dir).resolve()
    rollout = Path(rollout_dir).resolve()
    mod = _load_upstream_judge(bench_dir)

    items = transcripts(rollout, cfg.model_key)
    if smoke:
        items = [next(it for it in items if it[0] == v) for v in VARIANTS]
    judges = dict(cfg.judges)

    # Upstream excludes a judge from scoring its own trajectories (median-of-3).
    self_judge = [k for k, v in judges.items() if k == cfg.model_key]
    for k in self_judge:
        judges.pop(k)
        print(f"[self-exclusion] dropping judge {k} for model {cfg.model_key}")

    eval_dir = rollout / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)

    usage_before = openrouter_usage()
    print("=" * 72)
    print(f"Judging {len(items)} transcripts with {len(judges)} judges "
          f"({len(items) * len(judges)} calls)")
    print("OpenRouter spend so far: " + (f"${usage_before:.4f}" if usage_before is not None else "unknown"))
    print("=" * 72, flush=True)

    per_judge = {}
    for key, judge_model in judges.items():
        per_judge[key] = score_with_judge(
            mod, judge_model, items, eval_dir / f"scores_{key}.json", max_workers,
            budget=judge_budget(cfg), shared_cache=SHARED_CACHE_DIR,
        )
        first = next(iter(per_judge[key].values()))
        print(f"  first score from {key}: {first['score']} | "
              f"{str(first['reasoning'])[:160]}...", flush=True)

    usage_after = openrouter_usage(settle_s=90)

    # Keyed by SCENARIO, holding one median per rollout. `items` names a repeat as
    # "<Scenario>/rollout_NNN"; folding that whole string into the key made every rollout
    # look like its own scenario, so the bootstrap resampled rollouts and reported an
    # interval that was too narrow. Splitting it back out is what keeps scenarios the
    # independent unit the paper resamples.
    medians: dict[str, dict[str, list]] = {v: {} for v in VARIANTS}
    dropped = []
    for variant, scenario, _ in items:
        key = f"{variant}/{scenario}"
        scores = [
            float(per_judge[j][key]["score"])
            for j in per_judge
            if per_judge[j].get(key, {}).get("score") not in (None, "N/A")
        ]
        if not scores:
            dropped.append(key)
            continue
        medians[variant].setdefault(scenario.split("/")[0], []).append(
            median_score(scores))

    ours = summarise(medians)
    # Comparison baseline, in precedence order:
    #   `baseline_results` — a prior run's results.json (same "ours" schema), for comparing
    #     against a model we ran ourselves rather than a paper row.
    #   `published_key`    — a paper row other than this model's own, for a fine-tune that
    #     has no row of its own but was trained from a model that does.
    #   otherwise          — this model's own published row.
    baseline_results = cfg.get("baseline_results", None)
    if baseline_results:
        published = json.loads(Path(baseline_results).read_text())["ours"]
    elif 'published_key' in cfg and cfg.published_key is None:
        published = None
    else:
        published_key = str(cfg.get("published_key", cfg.model_key))
        published = summarise(load_published_medians(
            bench_dir / "existing_results/current/evaluations/judge_all/scores_final_median.csv",
            published_key,
        ))

    results = {
        "model": cfg.model,
        "model_key": cfg.model_key,
        "judges": dict(judges),
        "n_judged": len(items),
        "n_dropped_all_na": len(dropped),
        "dropped": dropped,
        # None when the credits endpoint could not be read (openrouter_usage never raises):
        # a missing cost figure must not lose a judged run (2026-09-23, the plain-arm smoke).
        "judging_cost_usd": (round(usage_after - usage_before, 4)
                             if usage_after is not None and usage_before is not None else None),
        "ours": ours,
        "published": published,
        "delta_mr_pct": (round(ours["overall"]["mr_pct"] - published["overall"]["mr_pct"], 1)
                         if published is not None and ours["overall"]["mr_pct"] is not None else None),
        "published_within_our_ci": (
            published is not None and ours["overall"]["mr_ci95"] is not None
            and ours["overall"]["mr_ci95"][0]
            <= published["overall"]["mr_pct"]
            <= ours["overall"]["mr_ci95"][1]
        ),
        "per_scenario_medians": medians,
    }

    out = rollout / "results.json"
    out.write_text(json.dumps(results, indent=2))
    write_run_meta(
        eval_dir,
        OmegaConf.to_container(cfg, resolve=True),
        extra={"command": " ".join(sys.argv), "smoke": smoke, "timestamp": timestamp()},
    )

    print("\n" + "=" * 72)
    print(f"REPLICATION: {cfg.model_key}")
    print(f"  ours      MR = {ours['overall']['mr_pct']}%  "
          f"CI95 {ours['overall']['mr_ci95']}  Sev = {ours['overall']['mean_severity']}")
    if published is not None:
        print(f"  published MR = {published['overall']['mr_pct']}%  "
              f"Sev = {published['overall']['mean_severity']}")
        print(f"  delta = {results['delta_mr_pct']:+.1f} pp | "
              f"published inside our CI: {results['published_within_our_ci']}")
    else:
        print("  published comparison disabled")
    # Only variants this run actually produced. `summarise` omits an unrun variant rather
    # than reporting it as 0.0%, so keying on VARIANTS blindly raises KeyError on an
    # incentivized-only arm -- after the scoring is already paid for.
    for variant in VARIANTS:
        if variant not in ours:
            print(f"  {variant:<13} not run in this arm")
            continue
        pub = (published or {}).get(variant, {}).get("mr_pct", "n/a")
        print(f"  {variant:<13} ours {ours[variant]['mr_pct']:>5}% / published {pub:>5}%")
    if dropped:
        print(f"  WARNING: {len(dropped)} trajectories had no usable judge score: {dropped}")
    print("  judging cost " + (f"${results['judging_cost_usd']:.2f}" if results["judging_cost_usd"] is not None else "unknown"))
    print(f">>> {out}")

