# ABOUTME: The operator library: every stage `kind:` a config may use. Operators are
# ABOUTME: generic and reusable -- all wording and wiring comes from the config entry.

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import cells
from .constitution import UNIT_PROVENANCE, Trait, units_from_config
from .core import Ctx, Stage, call_json, call_tagged, model_cfg, resilient, run_items
from .hf_cache import read_jsonl


def _render(template: str, record: dict, ctx: Ctx, **extra) -> str:
    """Format a config template from shared vars + the record (record wins)."""
    return template.format(**{**ctx.vars, **record, **extra})


def _resolve_vars(spec: dict | None, record: dict, ctx: Ctx) -> dict[str, str]:
    """Resolve a stage's `prompt_vars` for one record.

    A var is either a literal string (inserted raw) or a conditional
    `{by: <record_field>, cases: {<value>: text, ...}, default: "", render: bool}` --
    the record field's value (lowercased) picks the case, and `render: true` treats the
    chosen text as a template over the record (e.g. a transcript assembled from earlier
    stages' fields).
    """
    out: dict[str, str] = {}
    for name, v in (spec or {}).items():
        if isinstance(v, dict):
            key = str(record[v["by"]]).lower()
            text = v["cases"].get(key, v.get("default", ""))
            if v.get("render") and text:
                text = _render(text, record, ctx)
            out[name] = text
        else:
            out[name] = v
    return out


def _lint(parsed: dict, spec: dict) -> list[str]:
    """Return the reasons tagged output fails the stage's lint contract (empty = pass).

    The self-reflection voice contract is the archetype: reasoning that reaches for rule
    vocabulary, or that is too short to have done any weighing, is rejected so the call
    retries rather than the corpus absorbing it.
    """
    import re as _re

    problems = []
    min_chars = int(spec.get("min_chars", 0))
    patterns = [( pat, _re.compile(pat, _re.IGNORECASE)) for pat in spec.get("ban_patterns", [])]
    for field in spec.get("fields", []):
        if field not in parsed:
            continue
        text = parsed[field]
        for pat, rx in patterns:
            m = rx.search(text)
            if m:
                problems.append(f"<{field}> rule-vocabulary {m.group(0)!r} (matched {pat})")
        if min_chars and len(text) < min_chars:
            problems.append(f"<{field}> is {len(text)} chars, under the {min_chars} minimum")
    return problems


# --- generic operators --------------------------------------------------------------


def op_segment(sc: dict, cfg: dict) -> Stage:
    """Deterministic constitution chunking + grouping; publishes `style_guidance`.

    With no `chunking:` block in the config this is the original recipe exactly: one
    unit per numbered principle. With one, the same stage spans every arm of the
    chunking study -- finer granularities, combined chunks, and the whole document as a
    single unit -- because a unit renders to the Trait fields the rest of the pipeline
    already consumes.
    """
    def load(ctx: Ctx):
        units, style = units_from_config(ctx.cfg)
        limit = ctx.cfg.get("max_traits")
        if limit:
            units = units[: int(limit)]
        ctx.vars["style_guidance"] = style
        return units

    def fn(ctx, records, ckpt):
        units = load(ctx)
        u = units[0]
        print(f"    {u.granularity} x {u.grouping_strategy} -> {len(units)} units")
        for x in units:
            members = f" <- {','.join(x.chunk_ids)}" if x.n_chunks > 1 else ""
            print(f"    {x.unit_id}{members}")
        return [x.as_dict() for x in units]

    # Cached runs still need `style_guidance` in ctx.vars, and downstream operators
    # rebuild Traits from the snapshot, so on_cached returns units for the vars alone.
    return Stage(sc["name"], fn, on_cached=lambda ctx, records: load(ctx))


def scenario_batches(n_traits: int, cfg: dict) -> list[tuple[int, int, int]]:
    """Stage-2 batch specs for `scenarios`: (trait index, batch index, how many).

    Two sizing modes, and the choice matters for any chunking comparison:
      * `scenarios_per_trait` -- the original. The corpus grows with the number of
        units, so a `bullet` arm (45 units) would be ~45x a `whole` arm (1 unit).
      * `total_scenarios` -- a fixed corpus budget split evenly across whatever units
        the chunking produced. This is what keeps arms size-matched, so a chunking
        comparison is not secretly a data-scaling comparison. It wins when both are set.

    Pure and cheap, so the estimator calls it to count calls without touching the network.
    """
    total = cfg.get("total_scenarios")
    if total is not None:
        counts = _largest_remainder({i: 1.0 for i in range(n_traits)}, int(total))
        per_trait = [counts[i] for i in range(n_traits)]
    else:
        per_trait = [int(cfg["scenarios_per_trait"])] * n_traits
    per_call = int(cfg.get("scenarios_per_call", max(per_trait or [1])))

    batches: list[tuple[int, int, int]] = []
    for ti, want in enumerate(per_trait):
        remaining, bi = want, 0
        while remaining > 0:
            n = min(per_call, remaining)
            batches.append((ti, bi, n))
            remaining -= n
            bi += 1
    return batches


def op_scenarios(sc: dict, cfg: dict) -> Stage:
    """Fan-out scenario generation: batched JSON calls per trait, ids `t<i>_b<b>_s<j>`."""
    sys_t, user_t = sc["prompts"]["system"], sc["prompts"]["user"]
    mk = sc["model"]

    def fn(ctx, records, ckpt):
        m = model_cfg(ctx.cfg, mk)
        traits = [Trait.from_record(r) for r in records]
        # Unit provenance travels WITH the record rather than being joined back to the
        # stage-1 snapshot later: every downstream consumer (metadata export, corpus
        # checks, `balance_by`) then reads it as an ordinary field, and no stage needs
        # to know how to reach another stage's output.
        prov = {r["trait_id"]: {k: r[k] for k in UNIT_PROVENANCE if k in r}
                for r in records}
        batches = scenario_batches(len(traits), ctx.cfg)
        # Corpus size follows the unit count under `scenarios_per_trait`, so a changed
        # `chunking:` can multiply a run. Say the total out loud before paying for it.
        sized_by = "total_scenarios" if ctx.cfg.get("total_scenarios") is not None \
            else "scenarios_per_trait"
        print(f"    {len(traits)} units -> {sum(n for _t, _b, n in batches)} scenarios "
              f"in {len(batches)} calls (sized by {sized_by})")

        def one(k: int) -> list[dict]:
            ti, bi, n = batches[k]
            t = traits[ti]
            fields = {"trait_name": t.name, "trait_text": t.text}
            parsed, _ = call_json(
                ctx.client, ctx.usage, m["model"],
                _render(sys_t, fields, ctx, n=n), _render(user_t, fields, ctx, n=n),
                m["temperature"], m["max_tokens"], stage=mk,
                extra=m.get("extra_body"))
            assert isinstance(parsed, list), \
                f"{t.trait_id}: expected a JSON array, got {type(parsed)}"
            return [{
                "scenario_id": f"{t.trait_id}_b{bi:02d}_s{j:03d}",
                "trait_id": t.trait_id, "trait_name": t.name, "trait_text": t.text,
                "domain": s.get("domain", ""), "situation": s["situation"],
                "shortcut": s.get("shortcut", ""),
                **prov.get(t.trait_id, {}),
            } for j, s in enumerate(parsed)]

        nested = resilient(one, len(batches), ctx.workers, sc["name"],
                           max_fail_pct=float(ctx.cfg.get("max_fail_pct", 2.0)))
        return [r for group in nested for r in group]

    return Stage(sc["name"], fn, paid=True,
                 preview=lambda r: f"[{r['trait_name']}] {r['situation']}")


def op_llm_json(sc: dict, cfg: dict) -> Stage:
    """One JSON call per record; `save` maps record fields <- JSON keys."""
    sys_t, user_t = sc["prompts"]["system"], sc["prompts"]["user"]
    mk, save = sc["model"], dict(sc["save"])
    optional = set(sc.get("optional", []))

    def fn(ctx, records, ckpt):
        m = model_cfg(ctx.cfg, mk)

        def one(r: dict) -> dict:
            parsed, _ = call_json(
                ctx.client, ctx.usage, m["model"],
                _render(sys_t, r, ctx), _render(user_t, r, ctx),
                m["temperature"], m["max_tokens"], stage=mk,
                extra=m.get("extra_body"))
            return {**r, **{f: (parsed.get(k, "") if k in optional else parsed[k])
                            for f, k in save.items()}}

        return run_items(records, one, ctx.workers, sc["name"], ckpt,
                         max_fail_pct=float(ctx.cfg.get("max_fail_pct", 2.0)))

    return Stage(sc["name"], fn, paid=True, checkpoint_key=sc.get("checkpoint"),
                 preview=lambda r: r[next(iter(save))])


def tagged_request(sc: dict, r: dict, ctx: Ctx) -> tuple[list[dict], tuple, dict]:
    """Build one llm_tagged request for one record: (messages, tags, save map).

    Applies `variants_by` overrides and resolves `prompt_vars` -- factored out of the
    operator so a test can assert exactly what a record's prompt contains.
    """
    eff = sc
    variants = sc.get("variants_by")
    if variants:
        case = variants["cases"].get(str(r[variants["field"]]).lower())
        if case:
            eff = {**sc, **case}
            # A case's `system`/`user` keys are prompt-template overrides; they must
            # land inside `prompts`, not beside it where rendering never looks.
            templates = {k: case[k] for k in ("system", "user") if k in case}
            if templates:
                eff["prompts"] = {**sc["prompts"], **templates}
    pvars = _resolve_vars({**(sc.get("prompt_vars") or {}),
                           **(eff.get("prompt_vars") or {})}, r, ctx)
    messages = [
        {"role": "system", "content": _render(eff["prompts"]["system"], r, ctx, **pvars)},
        {"role": "user", "content": _render(eff["prompts"]["user"], r, ctx, **pvars)}]
    return messages, tuple(eff["tags"]), dict(eff["save"])


def weighted_scenario_prompt(sc: dict, batch: dict, trait: Trait) -> tuple[str, str]:
    """Build one weighted-scenario batch's (system, user) prompt from its stage entry."""
    threat = (sc["control_threats"] if batch["control"] else sc["threats"])[batch["motive"]]
    template = sc["prompts"]["control_user"] if batch["control"] else sc["prompts"]["user"]
    industries = "\n".join(f"  {i + 1}. {name}"
                            for i, name in enumerate(batch.get("industries", [])))
    return sc["prompts"]["system"], template.format(
        trait_name=trait.name, trait_text=trait.text, n=batch["n"], threat=threat,
        industries=industries)


def op_llm_tagged(sc: dict, cfg: dict) -> Stage:
    """One tagged-block call per record; `save` maps record fields <- tag names.

    Optional per-record behaviour, all from the config entry:
    - `prompt_vars`: extra template vars, possibly conditional on a record field.
    - `variants_by: {field, cases: {value: {user/tags/save overrides}}}` -- e.g. a
      multi-turn record uses a different user template, tag set and save map.
    - `lint: {fields, ban_patterns, min_chars, retries}` -- reject-and-retry a
      completion whose content (not just shape) breaks the corpus contract.
    """
    mk = sc["model"]
    lint_spec = sc.get("lint")

    def fn(ctx, records, ckpt):
        m = model_cfg(ctx.cfg, mk)

        def one(r: dict) -> dict:
            messages, tags, save = tagged_request(sc, r, ctx)
            attempts = int(lint_spec.get("retries", 2)) + 1 if lint_spec else 1
            problems: list[str] = []
            for _ in range(attempts):
                parsed = call_tagged(ctx.client, ctx.usage, m["model"], messages,
                                     m["temperature"], m["max_tokens"], mk, tags,
                                     extra=m.get("extra_body"))
                problems = _lint(parsed, lint_spec) if lint_spec else []
                if not problems:
                    return {**r, **{f: parsed[k] for f, k in save.items()}}
            raise ValueError(f"{sc['name']}: output breaks the stage contract after "
                             f"{attempts} attempts: {'; '.join(problems)}")

        return run_items(records, one, ctx.workers, sc["name"], ckpt,
                         max_fail_pct=float(ctx.cfg.get("max_fail_pct", 2.0)))

    return Stage(sc["name"], fn, paid=True, checkpoint_key=sc.get("checkpoint"),
                 preview=lambda r: r[next(iter(sc["save"]))])


def op_chat_export(sc: dict, cfg: dict) -> Stage:
    """Free export to `{messages, metadata}` chat records from templated fields.

    A message entry with `when: {field, min}` is included only for records where the
    field reaches the threshold -- how a multi-turn record keeps its second exchange
    while single-turn records stay three messages.
    """
    def fn(ctx, records, ckpt):
        out = []
        for r in records:
            msgs = []
            for m in sc["messages"]:
                cond = m.get("when")
                if cond and not int(r.get(cond["field"], 0)) >= int(cond["min"]):
                    continue
                msg = {"role": m["role"], "content": m["content"].format(**r)}
                if "reasoning_content" in m:
                    msg["reasoning_content"] = m["reasoning_content"].format(**r)
                msgs.append(msg)
            out.append({"messages": msgs,
                        "metadata": {k: r.get(k, "") for k in sc["metadata"]}})
        return out

    return Stage(sc["name"], fn)


def op_corpus_check(sc: dict, cfg: dict) -> Stage:
    """Corpus-level property checks over the records flowing through.

    A pure observer. It flags, it never fixes, and it returns its input unchanged --
    the assertion below is not decoration: a checker that is allowed to drop rows stops
    being a checker. Judged annotations go to a sidecar file for the same reason.

    Placed last in `stages:`, it audits what actually trains. Ablating it (`--ablate
    <name>`) runs the corpus unchecked, which is the point of it being a stage at all,
    and on a judged config is also how you run an arm without paying for judging.
    """
    from .corpus import is_paid, print_summary, run_corpus_checks, validate_spec

    try:
        validate_spec(sc)
    except (ValueError, AssertionError) as exc:
        raise type(exc)(f"stage {sc['name']!r}: {exc}") from exc
    paid = is_paid(sc)
    assert not paid or sc.get("model"), (
        f"stage {sc['name']!r} declares a judged corpus property but no `model:` key; "
        f"an unpriced judge would also estimate as free")
    on_fail = sc.get("on_fail", "warn")
    assert on_fail in ("warn", "error", "stop"), (
        f"stage {sc['name']!r}: on_fail must be 'warn' (report only), 'error' (finish "
        f"the run, exit nonzero) or 'stop' (halt the run here), got {on_fail!r}")
    report_name = f"{sc['name']}_report.json"

    def publish(ctx, report):
        if ctx.cache is not None:
            ctx.cache.save_json(report_name, report)
        else:
            (ctx.run_dir / report_name).write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        # Keyed by stage name: a run may check its scenarios, its drafts and its final
        # corpus, and the later verdicts must not overwrite the earlier ones.
        ctx.manifest_extra.setdefault("corpus_checks", {})[sc["name"]] = {
            "pass": report["pass"], "on_fail": on_fail,
            "counts": report.get("counts", {}), "report": report_name,
            "n_records": report.get("n_records"),
            "judge_spend_usd": report.get("judge_spend_usd"),
            "gated": sorted(k for k, v in report["properties"].items() if v["gate"]),
            "top_findings": report.get("findings", [])[:5],
        }

    def fn(ctx, records, ckpt):
        before = len(records)
        report = run_corpus_checks(records, sc, run_dir=ctx.run_dir,
                                   seed=int(ctx.cfg.get("seed", 0)),
                                   workers=ctx.workers, ctx=ctx if paid else None)
        publish(ctx, report)
        print_summary(report)
        assert len(records) == before, "a corpus check must never change the corpus"
        if on_fail == "stop" and not report["pass"]:
            # The point of a mid-pipeline check: stop before the stages that would
            # have spent real money on a corpus already known to be bad.
            ctx.stop = (f"corpus check {sc['name']!r} failed "
                        f"({report['counts'].get('critical', 0)} critical) and declares "
                        f"on_fail: stop -- see {report_name}")
        return records

    return Stage(sc["name"], fn, paid=paid, observer=True,
                 # A pure observer's null-operation IS the identity, so this stage is
                 # always ablatable and needs no `ablate_with` map in the config.
                 ablate_fn=lambda rs: rs)


# --- model-eval-model operators (structure in cells.py, wording in the config) ------


def op_load_source_run(sc: dict, cfg: dict) -> Stage:
    """Load a completed source run's final records, with constitution-sha provenance."""
    def load_records(spec: dict) -> tuple[list[dict], dict, str]:
        if spec.get("local_dir"):
            d = Path(spec["local_dir"])
            mpath = d / "manifest.json"
            manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
            return read_jsonl(d / "stage_6_final.jsonl"), manifest, str(d)
        repo = spec["hf_repo"]
        from huggingface_hub.utils import EntryNotFoundError

        from src.huggingface import hf_download

        records = read_jsonl(Path(hf_download(
            repo, "stage_6_final.jsonl", repo_type="dataset")))
        try:
            manifest = json.loads(Path(hf_download(
                repo, "manifest.json", repo_type="dataset")).read_text())
        except EntryNotFoundError:
            manifest = {}
        return records, manifest, repo

    def fn(ctx, records, ckpt):
        source, src_manifest, label = load_records(ctx.cfg["source"])
        sha = hashlib.sha256(ctx.constitution.encode()).hexdigest()
        src_sha = src_manifest.get("constitution_sha256")
        # Reasoning and critiques are grounded in cfg's constitution; a source run
        # generated against a different one would silently cross arms.
        assert src_sha is None or src_sha == sha, (
            f"source run {label} was generated against a different constitution "
            f"(sha {src_sha[:12]} != {sha[:12]}). Point cfg.constitution at the "
            f"source run's constitution or pick a matching source.")
        ctx.manifest_extra["source"] = {"source_run": label,
                                        "source_git_sha": src_manifest.get("git_sha"),
                                        "source_constitution_sha256": src_sha}
        (ctx.run_dir / "source_meta.json").write_text(
            json.dumps(ctx.manifest_extra["source"], indent=2))
        return source

    def on_cached(ctx, records):
        p = ctx.run_dir / "source_meta.json"
        if p.exists():
            ctx.manifest_extra["source"] = json.loads(p.read_text())

    return Stage(sc["name"], fn, on_cached=on_cached)


def _enabled(cfg: dict) -> dict[str, int]:
    return {c: int(n) for c, n in cfg["cells"].items() if int(n) > 0}


def op_plan_cells(sc: dict, cfg: dict) -> Stage:
    """Deterministic cell/explicitness/flaw allocation over the source records."""
    def fn(ctx, records, ckpt):
        enabled = _enabled(ctx.cfg)
        assert enabled, "no cell has a positive count; nothing to generate"
        source_run = ctx.manifest_extra.get("source", {}).get("source_run", "")
        return cells.plan_model_eval_model_records(
            records, enabled, ctx.cfg["explicitness"], int(ctx.cfg.get("seed", 0)),
            ctx.cfg["prompts"], source_run=source_run, flaws=ctx.cfg.get("flaws"))

    return Stage(sc["name"], fn,
                 preview=lambda r: f"{r['record_id']} [{r['explicitness']}]")


def op_perturb_pairs(sc: dict, cfg: dict) -> Stage:
    """Minimal-pair flawed responses, merged back into the plan; failures dropped loudly."""
    def fn(ctx, records, ckpt):
        flawed_planned = [p for p in records if p["response_kind"] == "flawed"]
        m = model_cfg(ctx.cfg, sc["model"])
        perturbed = cells.perturb_responses(
            records, ctx.client, ctx.usage, workers=ctx.workers,
            templates=sc["prompts"], P=ctx.cfg["prompts"], ckpt=ckpt, **m)
        by_id = {p["record_id"]: p for p in perturbed}
        lost = [p["record_id"] for p in flawed_planned if p["record_id"] not in by_id]
        if lost:
            print(f"!!! dropping {len(lost)} flawed documents without a perturbation "
                  f"(first 3: {lost[:3]})")
        return [by_id.get(p["record_id"], p) for p in records
                if p["response_kind"] != "flawed" or p["record_id"] in by_id]

    return Stage(sc["name"], fn, paid=True, checkpoint_key=sc.get("checkpoint"),
                 skip=lambda ctx, rs: not any(
                     r["response_kind"] == "flawed" for r in rs),
                 preview=lambda r: (f"[{r['flaw']['type']}/{r['flaw']['severity']}] "
                                    f"{r.get('change_summary', '')}"
                                    if r.get("flaw") else r["record_id"]))


def op_generate_cells(sc: dict, cfg: dict) -> Stage:
    """Generate each planned document via its cell's builder (see cells.CELLS)."""
    def fn(ctx, records, ckpt):
        enabled = _enabled(ctx.cfg)
        # Guard against a truncated plan reaching generation -- notably a pre-framework
        # stage_3 snapshot, which held only the flawed records, not the merged plan.
        missing = sorted(set(enabled) - {r["cell"] for r in records})
        assert not missing, (
            f"records reaching generation lack enabled cell(s) {missing}. If resuming "
            f"a pre-framework run dir, delete its stage_3_perturbed.jsonl and re-run.")
        model_cfgs = {cells.CELLS[c].model_key: model_cfg(ctx.cfg, cells.CELLS[c].model_key)
                      for c in enabled}
        return cells.generate_model_eval_model_documents(
            records, ctx.client, ctx.usage, model_cfgs, ctx.constitution,
            ctx.cfg["prompts"], ctx.workers, ckpt=ckpt)

    return Stage(sc["name"], fn, paid=True, checkpoint_key=sc.get("checkpoint"),
                 preview=lambda r: r["reasoning"])


def op_revise_cells(sc: dict, cfg: dict) -> Stage:
    """Constitution-grounded rewrite of every verdict-carrying document -- the
    difficult-advice `final` stage's twin. Control records pass through untouched;
    the whole stage is skipped when no enabled cell carries a verdict."""
    def fn(ctx, records, ckpt):
        m = model_cfg(ctx.cfg, sc["model"])
        return cells.revise_documents(
            records, ctx.client, ctx.usage, workers=ctx.workers,
            templates=sc["prompts"], P=ctx.cfg["prompts"],
            constitution=ctx.constitution, stage_key=sc["model"], ckpt=ckpt, **m)

    return Stage(sc["name"], fn, paid=True, checkpoint_key=sc.get("checkpoint"),
                 skip=lambda ctx, rs: not any("assessment" in r for r in rs),
                 preview=lambda r: r.get("rewrite_changes", r["record_id"]))


def op_assemble_cells(sc: dict, cfg: dict) -> Stage:
    """Free export: one assembler per cell (masked-turn shapes, supervise metadata)."""
    def fn(ctx, records, ckpt):
        return cells.to_model_eval_model_sft(records, ctx.cfg["prompts"])

    return Stage(sc["name"], fn)



# --- weighted scenario planning (the self-reflection document type's stage 2) -------


def _unit(scenario_id: str, salt: str) -> float:
    """Return a stable float in [0, 1) for one scenario and one axis.

    Derived from the scenario id rather than an RNG so that a resumed run, a re-run
    stage and the cost estimator all assign the same variants. Python's built-in `hash`
    is salted per process and would not.
    """
    digest = hashlib.sha256(f"{salt}:{scenario_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_variant(scenario_id: str, mix: dict) -> dict:
    """Return the form and turn count for one scenario, deterministically."""
    agentic = float(mix.get("form", {}).get("agentic", 0.2))
    multi = float(mix.get("multi_turn", 0.15))
    return {
        "form": "agentic" if _unit(scenario_id, "form") < agentic else "prose",
        "turns": 2 if _unit(scenario_id, "turns") < multi else 1,
    }


def _largest_remainder(weights: dict[str, float], total: int) -> dict[str, int]:
    """Apportion `total` across weighted keys so the counts sum exactly to it."""
    denom = sum(weights.values())
    assert denom > 0, "trait_weights sum to zero"
    exact = {k: total * w / denom for k, w in weights.items()}
    counts = {k: int(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: exact[k] - counts[k], reverse=True):
        if sum(counts.values()) >= total:
            break
        counts[k] += 1
    return counts


def plan_weighted_batches(traits: list[Trait], cfg: dict) -> list[dict]:
    """Stage-2 batch specs: trait weighting, control split, motive rotation, industries.

    Pure and cheap -- the estimator calls it to count calls without touching the
    network. Composition is deterministic so a resumed or re-run stage reproduces it.
    """
    mix = cfg.get("mix", {})
    per_call = int(cfg.get("scenarios_per_call", 8))
    configured = cfg["trait_weights"]
    if isinstance(configured, str):
        # Any chunking other than one-principle-per-unit derives ids (`t3+t7`, `c1`,
        # `t3.b02`) that no hand-written weight table can anticipate, so `uniform` is the
        # only way to weight those runs. Spelling is checked, so a typo in a
        # hand-weighted config still fails the strict branch below rather than silently
        # flattening every weight.
        assert configured == "uniform", (
            f"trait_weights must be a mapping or the literal 'uniform', got "
            f"{configured!r}")
        weights = {t.trait_id: 1.0 for t in traits}
    else:
        configured = dict(configured)
        present = {t.trait_id for t in traits}
        missing = sorted(present - set(configured))
        extra = sorted(set(configured) - present)
        # The constitution behind a config can change under it -- the 12-principle
        # document was re-cut to 10 units on 2026-08-04. Silently dropping the surplus
        # weights would regenerate a DIFFERENT corpus under the same config, silently.
        assert not (missing or extra), (
            f"trait_weights do not match {cfg['constitution']}, which segments into "
            f"{len(traits)} units: missing weights for {missing}, weights for absent "
            f"traits {extra}. Fix the config against the constitution actually in use.")
        weights = {t.trait_id: float(configured[t.trait_id]) for t in traits}
    counts = _largest_remainder(weights, int(cfg["total_scenarios"]))

    # Motive rotation follows the config mapping's insertion order (YAML preserves it);
    # sorting here would silently re-deal every batch's motive vs the published corpus.
    motive_w = dict(mix.get("motive") or {"replacement": 1.0})
    order = [m for m in motive_w
             for _ in range(max(1, round(float(motive_w.get(m, 0)) * 10)))]

    industries = list(cfg.get("industries") or [])
    control_frac = float(mix.get("control", 0.0))
    batches: list[dict] = []
    cursor = 0  # walks the industry list across the whole run, never restarting per trait
    for ti, t in enumerate(traits):
        n_total = counts[t.trait_id]
        n_control = round(n_total * control_frac)
        bi = 0
        for is_control, n_kind in ((True, n_control), (False, n_total - n_control)):
            remaining = n_kind
            while remaining > 0:
                n = min(per_call, remaining)
                batches.append({
                    "trait_index": ti,
                    "batch_index": bi,
                    "n": n,
                    "control": is_control,
                    "motive": order[(ti * 7 + bi) % len(order)],
                    "industries": [industries[(cursor + k) % len(industries)]
                                   for k in range(n)] if industries else [],
                    "id_prefix": str(cfg.get("id_prefix", "")),
                })
                cursor += n
                remaining -= per_call
                bi += 1
    return batches


def op_scenarios_weighted(sc: dict, cfg: dict) -> Stage:
    """Weighted, composed scenario generation over trait/control/motive/industry axes.

    Ids are `<prefix><trait>_b<batch>_<s|c><j>` (`c` = control slice); each scenario's
    form and turn count are assigned from its id (`assign_variant`), so composition
    survives resumes. All wording -- prompts, per-motive threat descriptions -- comes
    from the stage entry.
    """
    fields = sc.get("fields", {})
    required = list(fields.get("required", ["situation"]))
    optional = list(fields.get("optional", []))
    mk = sc["model"]

    def fn(ctx, records, ckpt):
        m = model_cfg(ctx.cfg, mk)
        traits = [Trait.from_record(r) for r in records]
        batches = plan_weighted_batches(traits, ctx.cfg)
        mix = ctx.cfg.get("mix", {})
        # Same rule as op_scenarios: unit provenance travels with the record.
        prov = {r["trait_id"]: {k: r[k] for k in UNIT_PROVENANCE if k in r}
                for r in records}

        def one(k: int) -> list[dict]:
            b = batches[k]
            t = traits[b["trait_index"]]
            system, user = weighted_scenario_prompt(sc, b, t)
            parsed, _ = call_json(ctx.client, ctx.usage, m["model"], system, user,
                                  m["temperature"], m["max_tokens"], stage=mk,
                                  extra=m.get("extra_body"))
            assert isinstance(parsed, list), \
                f"{t.trait_id}: expected a JSON array, got {type(parsed)}"
            out = []
            for j, s in enumerate(parsed):
                kind = "c" if b["control"] else "s"
                sid = f"{b['id_prefix']}{t.trait_id}_b{b['batch_index']:02d}_{kind}{j:03d}"
                rec = {"scenario_id": sid, "trait_id": t.trait_id, "trait_name": t.name,
                       "trait_text": t.text,
                       **{f: s[f] for f in required},
                       **{f: s.get(f, "") for f in optional},
                       "motive": b["motive"], "control": b["control"],
                       **assign_variant(sid, mix),
                       **prov.get(t.trait_id, {})}
                out.append(rec)
            return out

        nested = resilient(one, len(batches), ctx.workers, sc["name"],
                           max_fail_pct=float(ctx.cfg.get("max_fail_pct", 2.0)))
        return [r for group in nested for r in group]

    return Stage(sc["name"], fn, paid=True,
                 preview=lambda r: f"[{r['trait_name']}|{r['motive']}"
                                   f"{'|control' if r.get('control') else ''}] "
                                   f"{r.get('situation', '')}")


OPERATORS = {
    "segment": op_segment,
    "scenarios": op_scenarios,
    "scenarios_weighted": op_scenarios_weighted,
    "llm_json": op_llm_json,
    "llm_tagged": op_llm_tagged,
    "chat_export": op_chat_export,
    "corpus_check": op_corpus_check,
    "load_source_run": op_load_source_run,
    "plan_cells": op_plan_cells,
    "perturb_pairs": op_perturb_pairs,
    "generate_cells": op_generate_cells,
    "revise_cells": op_revise_cells,
    "assemble_cells": op_assemble_cells,
}
