# ABOUTME: The operator library: every stage `kind:` a config may use. Operators are
# ABOUTME: generic and reusable -- all wording and wiring comes from the config entry.

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import model_eval_model_cells as cells
from .constitution import UNIT_PROVENANCE, Trait, units_from_config
from .stage_runtime import Ctx, Stage, call_json, call_tagged, model_cfg, resilient, run_items
from .stage_runtime import lint_problems as _lint
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


# A `filter` stage's drop-rate guard is reported but not enforced below this many
# in-scope records: at smoke scale the rate is binomial noise, not a recipe failure.
_MIN_SCOPED_FOR_DROP_GATE = 20


def _lint_attempts(spec) -> int:
    """How many times to call before giving up, given a lint spec (or a list of them).

    One retry budget for the whole call: the contracts describe different tags of the same
    completion, so a retry re-rolls all of them. The most patient contract wins.
    """
    if not spec:
        return 1
    specs = spec if isinstance(spec, list) else [spec]
    return max(int(s.get("retries", 2)) for s in specs) + 1


def ident(record: dict) -> str:
    """A record's id for logs and error messages.

    `record_id` is the cell-based configs' composite id; a config whose documents are one
    per scenario has only `scenario_id`. Nothing behavioural keys on this -- it exists so
    a dropped-record report or a stage preview names something the reader can grep for.
    """
    return str(record.get("record_id") or record.get("scenario_id") or "")


def _norm_label(value) -> str:
    """Normalise a one-word label a model returned inside a tag.

    Models wrap their answer in punctuation, casing and stray whitespace ("` B.`",
    `"none"`, `<c>`). Every stage that routes on such a label -- `pick_field`'s choice,
    `filter`'s keep set -- must agree on what counts as the same label, or a record is
    silently misrouted on a stray full stop.
    """
    return str(value).strip().lower().strip(".:\"'()<>[]* ")


def selected(sc: dict, record: dict) -> bool:
    """Whether a stage's `when:` filter admits this record (no filter = every record).

    `when: {field: <name>, in: [values]}` scopes a stage to part of the corpus; records
    it excludes pass through untouched and cost nothing. This is what lets a config add
    a stage that only some cells need -- generating a first turn for the self cells, or
    a user-anchored framing for the user cells -- without the operator knowing which
    cells exist.
    """
    spec = sc.get("when")
    if not spec:
        return True
    return str(record.get(spec["field"], "")) in [str(v) for v in spec["in"]]


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

    Two prompt shapes. `prompts: {system, user}` is the common one. `conversation:` is a
    full message list, and it is what a document about a PRIOR EXCHANGE needs: the reply
    being reflected on has to sit in a genuine assistant turn, so that attribution is
    structural rather than something the prompt asserts in prose. Without it, the only
    way to build such a document is a bespoke message-builder in Python -- which is what
    the cell registry was.
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
                eff["prompts"] = {**sc.get("prompts", {}), **templates}
    pvars = _resolve_vars({**(sc.get("prompt_vars") or {}),
                           **(eff.get("prompt_vars") or {})}, r, ctx)
    convo = eff.get("conversation")
    if convo:
        messages = [{"role": m["role"], "content": _render(m["content"], r, ctx, **pvars)}
                    for m in convo]
        assert messages[-1]["role"] == "user", (
            f"stage {sc['name']!r}: a `conversation:` must end on the user turn -- the "
            f"model's reply is what the stage is asking for")
    else:
        messages = [
            {"role": "system",
             "content": _render(eff["prompts"]["system"], r, ctx, **pvars)},
            {"role": "user",
             "content": _render(eff["prompts"]["user"], r, ctx, **pvars)}]
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
    - `lint: {fields, ban_patterns, min_chars, max_chars, retries}` -- reject-and-retry
      a completion whose content (not just shape) breaks the corpus contract.
    - `when: {field, in: [...]}` -- run only over the matching records; the rest pass
      through unchanged and unpaid.
    - `also: {field: constant}` -- provenance stamped beside the saved tags, so where a
      generated turn came from is a recorded variable rather than something a reader has
      to infer from which config produced the run.
    """
    mk = sc["model"]
    lint_spec = sc.get("lint")
    also = dict(sc.get("also") or {})
    # Tags whose value is a label rather than prose. Normalised before the lint runs, so
    # `allowed:` compares against "held" and not "Held." -- casing and a trailing full
    # stop are the model formatting an answer, not answering a different question.
    normalize = list(sc.get("normalize") or [])
    # Arm assignment folded into this stage rather than standing as its own; see
    # `assign_arms`. `keep:` is the same idea at the other end -- a stage whose own output
    # decides which records survive can drop them itself rather than hand a snapshot to a
    # `filter` that reads one field of it. See `apply_keep`.
    assign_spec = sc.get("assign")
    keep_spec = sc.get("keep")

    def fn(ctx, records, ckpt):
        m = model_cfg(ctx.cfg, mk)
        # Labels first: `variants_by` may branch on one, so they have to exist before any
        # prompt is built. Applied to every record, in or out of `when` scope -- an arm is
        # a property of the record, not of this stage's coverage.
        if assign_spec:
            records = assign_arms(assign_spec, records)
        todo = [r for r in records if selected(sc, r)]
        if len(todo) != len(records):
            print(f"    when-filter: {len(todo)}/{len(records)} records in scope")

        def one(r: dict) -> dict:
            messages, tags, save = tagged_request(sc, r, ctx)
            attempts = _lint_attempts(lint_spec)
            problems: list[str] = []
            for _ in range(attempts):
                parsed = call_tagged(ctx.client, ctx.usage, m["model"], messages,
                                     m["temperature"], m["max_tokens"], mk, tags,
                                     extra=m.get("extra_body"))
                for tag in normalize:
                    if tag in parsed:
                        parsed[tag] = _norm_label(parsed[tag])
                problems = _lint(parsed, lint_spec) if lint_spec else []
                if not problems:
                    return {**r, **{f: parsed[k] for f, k in save.items()}, **also}
            raise ValueError(f"{sc['name']}: output breaks the stage contract after "
                             f"{attempts} attempts: {'; '.join(problems)}")

        done = run_items(todo, one, ctx.workers, sc["name"], ckpt,
                         max_fail_pct=float(ctx.cfg.get("max_fail_pct", 2.0)))
        if len(todo) == len(records):
            return apply_keep(sc, done, ctx) if keep_spec else done
        # Out-of-scope records keep their place; an in-scope record that failed is
        # dropped, exactly as it would be without the filter.
        key = sc.get("checkpoint") or "record_id"
        by_key = {r[key]: r for r in done}
        merged = [by_key.get(r[key], r) for r in records
                  if not selected(sc, r) or r[key] in by_key]
        return apply_keep(sc, merged, ctx) if keep_spec else merged

    # A stage may define `save` only inside `variants_by`, in which case there is no one
    # field the preview could name; fall back to the record id rather than crashing the
    # run log after the stage has already been paid for.
    first_saved = next(iter(sc.get("save") or {}), None)
    return Stage(sc["name"], fn, paid=True, checkpoint_key=sc.get("checkpoint"),
                 preview=lambda r: str((first_saved and r.get(first_saved))
                                       or ident(r)))


def op_pick_field(sc: dict, cfg: dict) -> Stage:
    """Free, deterministic resolution of a rater's choice into the field it names.

    `by:` is the record field holding the choice (a label an autorater returned),
    `from:` maps each label to the record field carrying that candidate's text, `to:`
    is where the winner lands, and `also:` stamps constant provenance fields alongside
    it. Keeping the LLM's job to "name the winner" and the copy to deterministic code
    is what stops a rater silently paraphrasing the candidate it selected.

    `when:` scopes it like any other stage.
    """
    by, src, to = sc["by"], dict(sc["from"]), sc["to"]
    also = dict(sc.get("also") or {})

    def resolve(r: dict) -> str:
        raw = _norm_label(r[by])
        label = raw if raw in src else raw.split()[-1] if raw.split() else raw
        if label not in src:
            raise ValueError(
                f"{r.get('record_id', '?')}: {by}={r[by]!r} is not one of "
                f"{sorted(src)}")
        return src[label]

    def fn(ctx, records, ckpt):
        out = []
        for r in records:
            if not selected(sc, r):
                out.append(r)
                continue
            out.append({**r, to: r[resolve(r)], **also})
        return out

    return Stage(sc["name"], fn, preview=lambda r: str(r.get(to) or ident(r)))


def assign_arms(spec: dict, records: list[dict], announce: bool = True) -> list[dict]:
    """Label every record with its experimental arm, derived from the record's own id.

    This is the de-celled replacement for `plan_cells`. Where that operator built a
    document-type registry in code -- a cell knowing how its prompts are assembled and
    how its record is exported -- this only labels: `reply_quality: {good: 0.5, flawed:
    0.5}` says half the corpus should end up with a first reply that holds up. Everything
    that reads the label afterwards (which prompt variant to use, which records a gate
    keeps, what the export records) is a config's own business.

    The label comes from a hash of `(field name, record id)`, not an RNG and not the
    record's position, so a resumed run, a re-run stage and the cost estimator all assign
    the same arms even after earlier stages have dropped records. Proportions are
    therefore approximate rather than exact -- at a few hundred records the drift is
    under a percent, but a smoke run of six can easily land 4/2.

    Available both as its own free stage (`kind: assign`) and as an `assign:` block on a
    paid stage, for the common case where the arm exists only to choose that stage's
    prompt: a whole stage to stamp two labels is a snapshot nobody reads.

    Args:
        spec: `{by, fields, constants, copy}` -- the id field, the weighted label fields,
            fields stamped identically on every record, and verbatim field copies.
        records: Input records.
        announce: Print the realised split (the number a config is re-sized from).

    Returns:
        The records, each extended with its labels.
    """
    by = spec["by"]
    fields = {k: dict(v) for k, v in (spec.get("fields") or {}).items()}
    constants = dict(spec.get("constants") or {})
    copies = dict(spec.get("copy") or {})
    for name, weights in fields.items():
        assert weights and all(float(w) >= 0 for w in weights.values()) \
            and sum(float(w) for w in weights.values()) > 0, \
            f"assign field {name!r}: weights must be non-negative and not all zero"

    def label(field: str, key: str) -> str:
        weights = fields[field]
        total = sum(float(w) for w in weights.values())
        point = _unit(key, field) * total
        acc = 0.0
        for value, weight in weights.items():  # config order, so a run is reproducible
            acc += float(weight)
            if point < acc:
                return str(value)
        return str(list(weights)[-1])

    out = []
    for r in records:
        key = str(r[by])
        out.append({**r, **constants,
                    **{new: r[src] for new, src in copies.items()},
                    **{f: label(f, key) for f in fields}})
    if announce:
        for f in fields:
            counts: dict[str, int] = {}
            for r in out:
                counts[r[f]] = counts.get(r[f], 0) + 1
            print(f"    {f}: {dict(sorted(counts.items()))}")
    return out


def op_assign(sc: dict, cfg: dict) -> Stage:
    """Free stage wrapper around `assign_arms` (see it for the semantics)."""
    fields = list((sc.get("fields") or {}))
    return Stage(sc["name"], lambda ctx, records, ckpt: assign_arms(sc, records),
                 preview=lambda r: f"{ident(r)} "
                                   + " ".join(f"{f}={r.get(f)}" for f in fields))


def apply_keep(sc: dict, records: list[dict], ctx=None) -> list[dict]:
    """Drop records failing the stage's `keep:` contract; return the survivors.

    Two forms. `keep: {field: <name>, in: [labels]}` keeps a record only when that
    field's normalised value is one of the labels, and `when:` scopes which records are
    judged at all. `keep: {by: <arm field>, cases: {<arm>: {field, in}}}` gives each arm
    its OWN contract, which is what a corpus with opposite expectations per arm needs:
    the flawed arm keeps records where a fault survived, the good arm keeps exactly the
    records where none did. Those are one decision, not two. An arm with no case is out
    of scope and passes through.

    This is what a FOUND (rather than planted) lapse needs. A reviser that names three
    faults in a reply and reports whether any is a genuine shortfall answers a question
    with no guaranteed answer: some replies steered toward falling short turn out fine,
    and some steered to hold up do not. Forcing those records through anyway would train
    the corpus's headline contrast on labels the pipeline itself does not believe --
    reflexive capitulation in one direction, defensiveness in the other. Dropping them
    costs the calls already made and keeps the labels honest.

    `max_drop_pct` (default 60) fails the stage rather than the corpus: losing most of a
    slice means the rater and the generator disagree systematically, which is a recipe bug
    and not something to quietly generate around. It takes a per-arm map alongside the
    `cases` form, since arms with opposite expectations have different healthy rates. It
    is reported but not enforced below `_MIN_SCOPED_FOR_DROP_GATE` records, where the rate
    is binomial noise -- a smoke run of four documents must not fail because all four went
    one way.

    Args:
        sc: The stage entry (`keep`, optional `when`, optional `max_drop_pct`).
        records: Input records.
        ctx: Optional run context. When given, which records were dropped and why is
            recorded into the manifest -- the only mirrored artifact that can carry it
            once the filter shares a snapshot with the stage that produced its input.

    Returns:
        The surviving records.
    """
    keep = sc["keep"]
    by = keep.get("by")
    cases = {_norm_label(k): v for k, v in (keep.get("cases") or {}).items()}
    max_drop = sc.get("max_drop_pct", 60.0)

    def rule(r: dict):
        """(arm label, field, allowed set) for this record, or None if out of scope."""
        if not cases:
            return ("", keep["field"], {_norm_label(v) for v in keep["in"]}) \
                if selected(sc, r) else None
        case = cases.get(_norm_label(r.get(by, "")))
        if case is None:
            return None
        return (_norm_label(r.get(by, "")), case["field"],
                {_norm_label(v) for v in case["in"]})

    def limit(arm: str) -> float:
        return float(max_drop.get(arm, 100.0) if isinstance(max_drop, dict) else max_drop)

    out: list[dict] = []
    dropped: dict[str, list[str]] = {}
    scoped: dict[str, int] = {}
    for r in records:
        spec = rule(r)
        if spec is None:
            out.append(r)
            continue
        arm, field, allowed = spec
        scoped[arm] = scoped.get(arm, 0) + 1
        if _norm_label(r.get(field, "")) in allowed:
            out.append(r)
        else:
            dropped.setdefault(arm, []).append(
                f"{ident(r)} [{field}={r.get(field, '')!r}]")

    if ctx is not None and dropped:
        # The dropped records no longer get a snapshot of their own, so the manifest --
        # which IS mirrored -- carries which ones went and on what value. Their full text
        # stays in this stage's local `.partial.jsonl`.
        ctx.manifest_extra.setdefault("dropped", {})[sc["name"]] = {
            arm: {"scoped": scoped.get(arm, 0), "dropped": len(rows), "records": rows}
            for arm, rows in sorted(dropped.items())}

    for arm, rows in sorted(dropped.items()):
        n = scoped.get(arm, 0)
        pct = 100 * len(rows) / max(n, 1)
        label = f"{sc['name']}[{arm}]" if arm else sc["name"]
        print(f"    {label}: dropped {len(rows)}/{n} in-scope records ({pct:.1f}%) "
              f"whose label the reviser did not support. First 3: {rows[:3]}")
        if pct > limit(arm) and n >= _MIN_SCOPED_FOR_DROP_GATE:
            raise RuntimeError(
                f"{label}: {pct:.1f}% of in-scope records dropped, above "
                f"max_drop_pct={limit(arm)}. The rater and the generator disagree "
                f"systematically -- fix the recipe rather than over-generating.")
    return out


def op_filter(sc: dict, cfg: dict) -> Stage:
    """Free stage wrapper around `apply_keep` (see it for the contract).

    Useful when the contract is judged on fields several stages old. Where the deciding
    field comes from the stage immediately before, that stage can carry a `keep:` block
    itself and save a snapshot.
    """
    return Stage(sc["name"], lambda ctx, records, ckpt: apply_keep(sc, records, ctx),
                 preview=ident)


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

    A pure observer: it flags, it never fixes, and it returns its input unchanged -- the
    assertion below is not decoration, since a checker allowed to drop rows stops being
    a checker. Judged annotations go to a sidecar for the same reason. Placed last in
    `stages:` it audits what actually trains; `--ablate <name>` runs the corpus
    unchecked, which on a judged config is also how to skip paying for judging.
    """
    from .check_corpus import (is_paid, pattern_table, print_summary, run_corpus_checks,
                         validate_spec)

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
        table = pattern_table(report)
        if table:
            (ctx.run_dir / f"{sc['name']}_patterns.md").write_text(
                f"# {sc['name']} — recurring patterns\n{table}\n", encoding="utf-8")
            print(table)
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
            # Stop before the stages that would spend real money on a corpus already
            # known to be bad.
            ctx.stop = (f"corpus check {sc['name']!r} failed "
                        f"({report['counts'].get('critical', 0)} critical) and declares "
                        f"on_fail: stop -- see {report_name}")
        return records

    return Stage(sc["name"], fn, paid=paid, observer=True,
                 # A pure observer's null-operation IS the identity, so this stage is
                 # always ablatable and needs no `ablate_with` map in the config.
                 ablate_fn=lambda rs: rs)


# --- model-eval-model operators (structure in model_eval_model_cells.py, wording in the config) ------


def op_load_source_run(sc: dict, cfg: dict) -> Stage:
    """Load a completed source run's final records, with constitution-sha provenance.

    Which snapshot to read is the `source:` block's `snapshot:` field, defaulting to the
    difficult-advice layout's `stage_6_final.jsonl` — the only source run that exists
    today, and what every current config means. It is a field rather than a constant
    because the operator is generic: a source run with a different stage list numbers its
    final snapshot differently, and baking one document type's filename into the operator
    would be exactly the hardcoding the engine is not allowed to do.
    """
    snapshot = str((cfg.get("source") or {}).get("snapshot") or "stage_6_final.jsonl")

    def load_records(spec: dict) -> tuple[list[dict], dict, str]:
        if spec.get("local_dir"):
            d = Path(spec["local_dir"])
            mpath = d / "manifest.json"
            manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
            return read_jsonl(d / snapshot), manifest, str(d)
        repo = spec["hf_repo"]
        from huggingface_hub.utils import EntryNotFoundError

        from src.huggingface import hf_download

        records = read_jsonl(Path(hf_download(
            repo, snapshot, repo_type="dataset")))
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


# --- model-eval-model cell operators (structure in model_eval_model_cells.py) -----


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


# --- weighted scenario planning (the pre-action deliberation type's stage 2) --------


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
    "assign": op_assign,
    "pick_field": op_pick_field,
    "filter": op_filter,
    "chat_export": op_chat_export,
    "corpus_check": op_corpus_check,
    "load_source_run": op_load_source_run,
}

# The five cell kinds below are NOT generic: a cell is a document type expressed in
# Python -- a registry entry knowing how its prompts are assembled and how its record is
# exported -- which is precisely what a config's `stages:` list is supposed to express
# instead. They stay registered because the archived configs and
# `peer_critique.yaml` are written against them, and an archived config
# that cannot run is not a reproducible record of a published corpus. Nothing new should
# use them; build a document type out of the generic kinds above, as
# `post_action_retrospection.yaml` does.
OPERATORS.update({
    "plan_cells": op_plan_cells,
    "perturb_pairs": op_perturb_pairs,
    "generate_cells": op_generate_cells,
    "revise_cells": op_revise_cells,
    "assemble_cells": op_assemble_cells,
})
