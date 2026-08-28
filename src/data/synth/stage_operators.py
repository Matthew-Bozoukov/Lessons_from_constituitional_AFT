# ABOUTME: The operator library: every stage `kind:` a config may use. Operators are
# ABOUTME: generic and reusable -- all wording and wiring comes from the config entry.

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from src.endpoints.openrouter import ProviderRejectionError

from . import model_eval_model_cells as cells
from .constitution import UNIT_PROVENANCE, Trait, units_from_config
from .derive import derive_vars
from .stage_runtime import (
    BATCH_MIN_ITEMS,
    Ctx,
    Stage,
    call_json,
    call_tagged,
    model_cfg,
    resilient,
    run_items,
    run_items_batched,
)
from .stage_runtime import _parse_json, _parse_tagged  # single-attempt batch parses
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


def strip_scaffolding(text: str, patterns: list[str]) -> str:
    """Delete prompt scaffolding from a completion and close the gaps it leaves.

    The blank-line collapse is the point, not tidiness: removing `<run n="2">` from its
    own line leaves three consecutive newlines, and a paragraph break that wide would be a
    visible seam in text whose whole contract is to read as one continuous piece.
    """
    for pat in patterns:
        text = re.sub(pat, "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _verify(spec: dict, candidate: dict, ctx: Ctx, stage: str) -> tuple[dict, bool]:
    """Judge one candidate record; return (verdict, accepted).

    The judge is rendered over the CANDIDATE -- the record as it would be if this
    attempt were accepted -- so its prompt can name both the new field and the source it
    must be faithful to, and so it sees exactly the object the corpus would keep.

    Deliberately a different model from the generator by convention rather than by force:
    a generator grading its own output shares its blind spots, and the failures this gate
    exists to catch are precisely the ones the generator found plausible enough to write.
    """
    m = model_cfg(ctx.cfg, spec["model"])
    verdict, _ = call_json(
        ctx.client, ctx.usage, m["model"],
        _render(spec["prompts"]["system"], candidate, ctx),
        _render(spec["prompts"]["user"], candidate, ctx),
        m["temperature"], m["max_tokens"], f"{stage}.verify",
        extra=m.get("extra_body"))
    if not isinstance(verdict, dict):
        raise ValueError(f"{stage}.verify: judge returned {type(verdict).__name__}, "
                         f"not a JSON object")
    field = spec.get("accept_field", "verdict")
    allowed = [str(v).lower() for v in (spec.get("accept_values") or ["pass"])]
    return verdict, str(verdict.get(field, "")).strip().lower() in allowed


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
    cells exist. A LIST of such conditions is their conjunction -- what a stage that
    covers one slice of one arm needs (e.g. the flawed records whose first turn a
    particular model writes).
    """
    spec = sc.get("when")
    if not spec:
        return True
    conds = spec if isinstance(spec, list) else [spec]
    return all(str(record.get(c["field"], "")) in [str(v) for v in c["in"]]
               for c in conds)


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


def scenario_batches(n_traits: int, cfg: dict,
                     trait_ids: list[str] | None = None) -> list[tuple[int, int, int]]:
    """Stage-2 batch specs for `scenarios`: (trait index, batch index, how many).

    Two sizing modes, and the choice matters for any chunking comparison:
      * `scenarios_per_trait` -- the original. The corpus grows with the number of
        units, so a `bullet` arm (45 units) would be ~45x a `whole` arm (1 unit).
      * `total_scenarios` -- a fixed corpus budget split evenly across whatever units
        the chunking produced. This is what keeps arms size-matched, so a chunking
        comparison is not secretly a data-scaling comparison. It wins when both are set.

    `trait_weights` on the STAGE entry (passed in as `cfg["trait_weights"]` by the
    operator) splits `total_scenarios` UNEVENLY instead, which is what a recipe whose
    coverage targets are not uniform needs: Good AI Fiction wants 30% of its rows drawn
    from the oversight principle and 5% from each helpfulness principle, and a uniform
    split cannot say that. Same shape and same strictness as `scenarios_weighted`'s
    long-standing block -- a weight table that does not exactly match the units the
    constitution segments into is an error, not something to fill in, because a
    constitution re-cut under a config would otherwise silently regenerate a different
    corpus. Absent, the split stays uniform and every existing config is unaffected.

    Pure and cheap, so the estimator calls it to count calls without touching the network.
    """
    total = cfg.get("total_scenarios")
    weights = cfg.get("trait_weights")
    if weights and total is None:
        raise ValueError("trait_weights needs `total_scenarios`: there is no fixed "
                         "budget for it to split when the size follows "
                         "scenarios_per_trait")
    if total is not None and weights:
        assert trait_ids is not None and len(trait_ids) == n_traits, (
            "trait_weights needs the unit ids to weight by; the caller passed none")
        configured = dict(weights)
        missing = sorted(set(trait_ids) - set(configured))
        extra = sorted(set(configured) - set(trait_ids))
        assert not missing, (
            f"trait_weights do not cover the units this constitution segments into: "
            f"no weight for {missing}. Fix the config against the constitution "
            f"actually in use.")
        # Surplus weights are an error UNLESS the run is deliberately restricted to a
        # subset of units (`--smoke`'s max_traits, or only_traits): there the surplus is
        # the restriction, not a stale table. Unrestricted, a weight for an absent unit
        # means the constitution was re-cut under the config, which would silently
        # regenerate a different corpus.
        assert not extra or cfg.get("max_traits") or cfg.get("only_traits"), (
            f"trait_weights name units this constitution does not segment into: "
            f"{extra}. Fix the config against the constitution actually in use.")
        by_id = _largest_remainder({t: float(configured[t]) for t in trait_ids},
                                   int(total))
        per_trait = [by_id[t] for t in trait_ids]
    elif total is not None:
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


def deal_labels(weights: dict[str, float], length: int) -> list[str]:
    """A deterministic label sequence of `length` whose composition matches `weights`.

    Largest-remainder apportionment, then interleaved round-robin rather than left in
    blocks: an interrupted or shortened run would otherwise have generated only the
    first label, and a wave (which is a contiguous slice of the batch list) would carry
    one label throughout instead of the mix the weights describe.
    """
    counts = _largest_remainder({k: float(v) for k, v in weights.items()}, length)
    blocks = [[k] * n for k, n in counts.items() if n]
    out: list[str] = []
    while any(blocks):
        for b in blocks:
            if b:
                out.append(b.pop())
    return out


def _axis_walk(name: str, length: int) -> tuple[int, int]:
    """A stable (stride, offset) for reading one axis's deal sequence.

    Two axes dealt over the same batch list must not read their sequences in the same
    order, or they are correlated and the config has two names for one axis. An offset
    alone is not enough -- both axes still advance one position per batch, so a label of
    one lines up with the same label of the other whenever their periods share a factor.
    A per-axis STRIDE fixes that, and a stride coprime to the length is a permutation of
    the positions, so every position is still visited exactly once per cycle and the
    declared proportions are preserved exactly.

    Derived from the axis name rather than an RNG, so a resumed or re-run stage
    reproduces the same pairing.
    """
    if length <= 2:
        return 1, 0
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    raw = int.from_bytes(digest, "big")
    offset = raw % length
    stride = (raw // length) % length or 1
    while _gcd(stride, length) != 1:
        stride = stride % length + 1
    return stride, offset


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def load_library(spec: dict) -> list[dict]:
    """Read a `library:` block's entries off disk (a list of dicts under `key`)."""
    import yaml

    path = Path(spec["file"])
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = doc[spec.get("key", "entries")] if isinstance(doc, dict) else doc
    assert isinstance(entries, list) and entries, f"{path}: no entries to deal from"
    id_field = spec.get("id", "id")
    ids = [e[id_field] for e in entries]
    assert len(set(ids)) == len(ids), f"{path}: duplicate {id_field} values"
    return entries


def library_picks(spec: dict, entries: list[dict], trait_id: str,
                  ti: int, bi: int, n: int) -> list[dict]:
    """The `n` library entries this batch is dealt, filtered to the batch's unit.

    A walking cursor derived from (unit index, batch index) rather than a running
    counter, for the same reason every other composition choice here is: a resumed run
    or a make-up round re-deals a batch and must deal it the same way. Entries are
    filtered by `match` -- the entry field listing the units it has anything to say
    about -- so an archetype about sycophancy is never handed to the oversight unit.
    """
    match = spec.get("match")
    pool = [e for e in entries
            if not match or trait_id in (e.get(match) or [])] or entries
    start = (ti * 31 + bi * 7) % len(pool)
    return [pool[(start + k) % len(pool)] for k in range(min(n, len(pool)))]


def _gist(r: dict, words: int = 14) -> str:
    """One line naming a scenario, for the generator's do-not-repeat list.

    Domain plus the opening clause of the situation: enough for a model to recognise the
    family ("academic research: A postdoctoral researcher has spent three years on a
    promising cancer drug study") without spending a whole scenario's tokens on it.
    """
    body = " ".join(str(r.get("situation") or "").split()[:words])
    domain = str(r.get("domain") or "").strip()
    gist = f"{domain}: {body}" if domain else body
    # A scenario spec may carry how the situation ARRIVES (courtroom's `wrapper`);
    # repeats of that framing are as much a collapse as repeats of the situation, so
    # when the field exists the ban list shows it to the generator too.
    wrapper = " ".join(str(r.get("wrapper") or "").split()[:8])
    return f"{gist} [arrives as: {wrapper}]" if wrapper else gist


def _too_close(new_texts: list[str], seen_vecs, reject_cosine: float, model: str | None):
    """Indices of `new_texts` colliding with each other or with already-kept scenarios.

    Within-wave collisions are resolved first-wins by position, so a wave that proposes
    the same situation twice keeps one copy rather than dropping both.
    """
    import numpy as np

    from .embeddings import DEFAULT_MODEL, embed

    X = embed(new_texts, model=str(model or DEFAULT_MODEL))
    bad: set[int] = set()
    if seen_vecs is not None and len(seen_vecs):
        against = (X @ seen_vecs.T).max(axis=1)
        bad |= {i for i in range(len(new_texts)) if float(against[i]) >= reject_cosine}
    G = X @ X.T
    np.fill_diagonal(G, -1.0)
    for i in range(len(new_texts)):
        if i in bad:
            continue
        for j in range(i + 1, len(new_texts)):
            if j not in bad and float(G[i, j]) >= reject_cosine:
                bad.add(j)
    return bad, X


def op_scenarios(sc: dict, cfg: dict) -> Stage:
    """Fan-out scenario generation: batched JSON calls per trait, ids `t<i>_b<b>_s<j>`.

    Optional `diversity:` block turns the batch-local "vary the domain" instruction into
    an enforced corpus-level constraint. Measured motivation: under plain fan-out the
    baseline corpus put 46.8% of its mass in ten domains, and its two closest scenarios
    (cosine 0.886 -- the same postdoc / funding-cliff / borderline-significance story
    under two different trait ids and two different domain labels) were never caught,
    because every batch is generated blind to every other.

        diversity:
          avoid: ["academic research: a postdoctoral researcher ...", ...]
          wave_size: 12          # batches per wave; the ban list grows between waves
          max_carry: 150         # cap on lines carried into a prompt
          reject_cosine: 0.86    # a scenario this close to an existing one is refused
          max_regen_rounds: 2    # make-up passes for traits left short by rejection

    Two mechanisms, because asking is not enforcing. The prompt-side ban list (`avoid`
    seeds it; each wave extends it with what the previous waves actually produced) tells
    the generator what not to write. The embedding gate then refuses what it wrote
    anyway, and re-queues the shortfall per trait so rejection cannot quietly unbalance
    the corpus. Waves exist only because batches run concurrently: without them no batch
    can see any other's output.

    Optional `rotate:` block deals a labelled axis across the run, one value per BATCH:

        rotate:
          stakes:
            weights: {mundane: 35, institutional: 35, capability: 20, speculative: 10}
            text: {mundane: "situations that ...", ...}   # -> {stakes_text} in the prompt

    The dealt label is available to the prompt as `{<name>}`, its `text:` entry as
    `{<name>_text}`, and it is stamped on every scenario the batch produces -- so a
    downstream stage can scope on it and the export can record it. This is what a recipe
    whose scenarios must SPAN something (stakes level, framing, source) needs: asking one
    prompt for a spread gets a spread within the batch and nothing across batches, which
    is the same failure the `diversity:` block exists for. Proportions are approximate,
    like `assign_arms`' -- the realised split is printed and recorded in the manifest.

    Optional `library:` block deals entries from a YAML file, `n` per batch, into one
    rendered prompt variable:

        library:
          file: configs/data/synth/good_ai_fiction/archetypes.yaml
          key: archetypes        # the top-level list
          id: id                 # the field naming an entry
          match: traits          # entry field listing the units it fits (optional)
          gate: {field: source_type, in: [inversion]}   # rotate axis that switches it on
          var: archetypes        # prompt variable holding the rendered block
          item: "[{id}] {work} -- {psychological_failure}"   # one entry, one line

    Ungated batches (and every batch when there is no `gate:`) render `{<var>}` as the
    empty string, so ONE prompt template covers both halves of the corpus. Which entry a
    scenario actually used is not inferred from the deal -- the model echoes it back in a
    declared `fields:` key, so a reordered or short reply cannot misattribute provenance.
    """
    sys_t, user_t = sc["prompts"]["system"], sc["prompts"]["user"]
    mk = sc["model"]
    div = dict(sc.get("diversity") or {})
    rotate = {k: dict(v) for k, v in (sc.get("rotate") or {}).items()}
    lib_spec = dict(sc.get("library") or {})
    # Extra scenario-spec fields beyond the base domain/situation/shortcut shape,
    # mirroring `scenarios_weighted`'s `fields:` block: `required` keys fail the batch
    # loudly when the model omits them, `optional` default to "". Values are stripped
    # because downstream `when:` scoping matches strings exactly.
    extra = sc.get("fields") or {}
    req_fields = list(extra.get("required") or [])
    opt_fields = list(extra.get("optional") or [])

    def fn(ctx, records, ckpt):
        m = model_cfg(ctx.cfg, mk)
        traits = [Trait.from_record(r) for r in records]
        # Unit provenance travels WITH the record rather than being joined back to the
        # stage-1 snapshot later: every downstream consumer (metadata export, corpus
        # checks, `balance_by`) then reads it as an ordinary field, and no stage needs
        # to know how to reach another stage's output.
        prov = {r["trait_id"]: {k: r[k] for k in UNIT_PROVENANCE if k in r}
                for r in records}
        batches = scenario_batches(len(traits), ctx.cfg,
                                   [t.trait_id for t in traits])
        # Axis deals and the library are built once per stage run, from the PLANNED batch
        # count, and indexed by (unit, batch) rather than by position -- so a make-up
        # round, a resume and the estimator all see the same composition.
        deals = {name: deal_labels(spec["weights"], max(len(batches), 12))
                 for name, spec in rotate.items()}
        entries = load_library(lib_spec) if lib_spec else []
        # Each batch's POSITION in the plan, so a deal is walked once through rather than
        # sampled by a modular formula. Measured 2026-08-27: indexing by `(ti*7+bi) % len`
        # visited only a subset of positions, which left 5 of 12 world registers at zero
        # and pushed a 35% stakes band to 58%. A make-up batch (bi beyond the plan) has no
        # position and falls back to the formula.
        ordinal = {(ti, bi): k for k, (ti, bi, _n) in enumerate(batches)}
        # Per-axis (stride, offset), or every axis reads its deal in the SAME order and
        # the axes stop being independent: the 2026-08-27 run tied every `ship_mind` to
        # `mundane` and every `station_mind` to `institutional`, which is one axis wearing
        # two names. See `_axis_walk`.
        walk = {name: _axis_walk(name, len(seq)) for name, seq in deals.items()}

        def axes_of(spec: tuple) -> dict[str, str]:
            ti, bi, _n = spec
            base = ordinal.get((ti, bi), ti * 7 + bi)
            out = {}
            for name, seq in deals.items():
                stride, offset = walk[name]
                out[name] = seq[(base * stride + offset) % len(seq)]
            return out

        def library_block(spec: tuple, axes: dict[str, str]) -> str:
            if not entries:
                return ""
            gate = lib_spec.get("gate")
            if gate and str(axes.get(gate["field"], "")) not in \
                    [str(v) for v in gate["in"]]:
                return ""
            ti, bi, n = spec
            picks = library_picks(lib_spec, entries, traits[ti].trait_id, ti, bi, n)
            return "\n".join(lib_spec["item"].format(**e) for e in picks)
        # Corpus size follows the unit count under `scenarios_per_trait`, so a changed
        # `chunking:` can multiply a run. Say the total out loud before paying for it.
        sized_by = "total_scenarios" if ctx.cfg.get("total_scenarios") is not None \
            else "scenarios_per_trait"
        print(f"    {len(traits)} units -> {sum(n for _t, _b, n in batches)} scenarios "
              f"in {len(batches)} calls (sized by {sized_by})")

        wave_size = int(div.get("wave_size") or 0) or len(batches)
        reject_cos = float(div.get("reject_cosine") or 0.0)
        max_rounds = int(div.get("max_regen_rounds", 2))
        max_carry = int(div.get("max_carry", 150))
        over_share = float(div.get("over_share") or 0.0)
        over_min = int(div.get("over_min_docs", 50))
        ban: list[str] = [str(x) for x in (div.get("avoid") or [])]
        avoid_block = ""
        over_block = ""
        max_fail = float(ctx.cfg.get("max_fail_pct", 2.0))

        def render_avoid() -> str:
            """The do-not-repeat list as it stands, newest last, capped."""
            if not ban:
                return ""
            return ("These situations already exist in this corpus. Do not write another "
                    "version of any of them -- the same story in a different domain, or "
                    "with the roles renamed, still counts as a repeat:\n"
                    + "\n".join(f"- {x}" for x in ban[-max_carry:]))

        def render_overrepresented(kept: list[dict]) -> str:
            """Domains running hot in what has been kept so far, as a steer for the next
            wave.

            Deliberately proportional rather than prohibitive: the corpus SHOULD contain
            small-business scenarios, just not 226 of them. A domain is named only once it
            is measurably over-weight in this run's own output, so nothing is banned up
            front and a domain that stops being over-represented stops being listed.

            Counted case-folded: the baseline carried 410 raw domain strings for ~250 real
            domains, so raw counts understate concentration.
            """
            if over_share <= 0 or len(kept) < over_min:
                return ""
            counts: dict[str, int] = {}
            for r in kept:
                d = str(r.get("domain") or "").strip().lower()
                if d:
                    counts[d] = counts.get(d, 0) + 1
            hot = sorted(((n / len(kept), d) for d, n in counts.items()
                          if n / len(kept) > over_share), reverse=True)
            if not hot:
                return ""
            return ("These domains are already over-represented in this corpus. Do not "
                    "use them again unless this principle genuinely cannot be pressured "
                    "anywhere else; reach for a setting not yet listed here:\n"
                    + "\n".join(f"- {d} ({s:.0%} of the corpus so far)" for s, d in hot))

        def _rows_from(spec: tuple, parsed: list) -> list[dict]:
            """Scenario records from one call's JSON array (shared by both paths)."""
            ti, bi, _n = spec
            t = traits[ti]
            axes = axes_of(spec)
            # `id_prefix` mirrors the weighted operator's, and exists for the same reason:
            # a TOP-UP run that fills a thin cell of an existing corpus restarts its batch
            # numbering at zero, so without a prefix its ids collide with the run it is
            # topping up and every id-keyed join silently takes the wrong row.
            prefix = str(ctx.cfg.get("id_prefix", ""))
            return [{
                "scenario_id": f"{prefix}{t.trait_id}_b{bi:02d}_s{j:03d}",
                "trait_id": t.trait_id, "trait_name": t.name, "trait_text": t.text,
                "domain": s.get("domain", ""), "situation": s["situation"],
                "shortcut": s.get("shortcut", ""),
                **axes,
                **{k: str(s[k]).strip() for k in req_fields},
                **{k: str(s.get(k, "")).strip() for k in opt_fields},
                **prov.get(t.trait_id, {}),
            } for j, s in enumerate(parsed) if isinstance(s, dict) and "situation" in s]

        def _prompts_for(spec: tuple) -> tuple[str, str]:
            ti, _bi, n = spec
            t = traits[ti]
            axes = axes_of(spec)
            fields = {"trait_name": t.name, "trait_text": t.text, **axes}
            extra_vars = {f"{name}_text": str(rotate[name].get("text", {})
                                              .get(axes[name], ""))
                          for name in rotate}
            if lib_spec:
                extra_vars[lib_spec.get("var", "library")] = library_block(spec, axes)
            return (_render(sys_t, fields, ctx, n=n, avoid=avoid_block,
                            overrepresented=over_block, **extra_vars),
                    _render(user_t, fields, ctx, n=n, avoid=avoid_block,
                            overrepresented=over_block, **extra_vars))

        def generate_spec(spec: tuple) -> list[dict]:
            ti = spec[0]
            sysr, usr = _prompts_for(spec)
            parsed, _ = call_json(ctx.client, ctx.usage, m["model"], sysr, usr,
                                  m["temperature"], m["max_tokens"], stage=mk,
                                  extra=m.get("extra_body"))
            assert isinstance(parsed, list), \
                f"{traits[ti].trait_id}: expected a JSON array, got {type(parsed)}"
            return _rows_from(spec, parsed)

        # Opt-in async batching of a wave's calls. A wave is still one steering step:
        # every request in it shares this wave's avoid/overrepresented blocks, the batch
        # completes, then the reject gate and the next wave's steer run exactly as
        # interactively — so batching preserves the diversity feedback and only serialises
        # the batch turnaround (each wave is one dependent round-trip). Attempt-0 only, but
        # the make-up rounds ARE the retry: a filtered/failed request just leaves its
        # trait short and is re-queued next round.
        batch_on = bool(ctx.cfg.get("batch")) and sc.get("batch", True) is not False

        def run_wave(wave: list[tuple], tag: str) -> list[dict]:
            if not (batch_on and len(wave) >= BATCH_MIN_ITEMS):
                nested = resilient(lambda k, w=wave: generate_spec(w[k]), len(wave),
                                   ctx.workers, sc["name"], max_fail_pct=max_fail)
                return [r for group in nested for r in group]
            from src.endpoints.openrouter import build_request_body, result_from_payload
            from .stage_runtime import run_batch as _run_batch
            _note_batched(ctx, sc["name"])
            reqs, spec_by_cid = {}, {}
            for spec in wave:
                ti, bi, _n = spec
                sysr, usr = _prompts_for(spec)
                cid = f"{traits[ti].trait_id}_b{bi:02d}"
                reqs[cid] = build_request_body(
                    m["model"], [{"role": "system", "content": sysr},
                                 {"role": "user", "content": usr}],
                    m["temperature"], m["max_tokens"], m.get("extra_body"))
                spec_by_cid[cid] = spec
            fresh: list[dict] = []

            def collect(cid: str, payload: dict) -> None:
                try:
                    res = result_from_payload(m["model"], payload)
                    ctx.usage.add(m["model"], res, mk, usd_scale=0.5)
                    parsed = _parse_json(res.content)
                    if isinstance(parsed, list):
                        fresh.extend(_rows_from(spec_by_cid[cid], parsed))
                except Exception:  # noqa: BLE001 - a failed request just re-queues via shortfall
                    pass

            _run_batch(m["model"], reqs, f"{sc['name']}:{tag}",
                       ctx.run_dir / f".batch_{sc['name']}_{tag}.json", collect)
            return fresh

        def report_axes(rows: list[dict]) -> list[dict]:
            """Print and record what the rotated axes ACTUALLY came out at.

            The deal is proportional by construction but the corpus is not: a batch can
            come back short or be rejected wholesale by the embedding gate, and the
            recipe's quotas are enforced downstream at selection. A realised split
            nobody printed is a quota nobody checked.
            """
            if not rotate:
                return rows
            realised = {}
            for name in rotate:
                counts: dict[str, int] = {}
                for r in rows:
                    counts[str(r.get(name, ""))] = counts.get(str(r.get(name, "")), 0) + 1
                realised[name] = dict(sorted(counts.items()))
                print(f"    rotate {name}: {realised[name]}")
            ctx.manifest_extra.setdefault("rotated_axes", {})[sc["name"]] = realised
            return rows

        # Unchanged fast path: no `diversity:` block means one wave, no ban list and no
        # embedding gate. Batches when --batch is set, interactive otherwise.
        if not div:
            avoid_block = over_block = ""
            return report_axes(run_wave(list(batches), "all"))

        target: dict[int, int] = {}
        next_bi: dict[int, int] = {}
        for ti, bi, n in batches:
            target[ti] = target.get(ti, 0) + n
            next_bi[ti] = max(next_bi.get(ti, 0), bi + 1)
        got: dict[int, int] = {ti: 0 for ti in target}
        per_call = max(int(ctx.cfg.get("scenarios_per_call", 8)), 1)

        kept: list[dict] = []
        kept_vecs = None
        queue = list(batches)
        rejected_total = 0

        for rnd in range(max_rounds + 1):
            if not queue:
                break
            for start in range(0, len(queue), wave_size):
                wave = queue[start:start + wave_size]
                avoid_block = render_avoid()
                over_block = render_overrepresented(kept)
                fresh = run_wave(wave, f"r{rnd}_w{start:03d}")
                if not fresh:
                    continue
                if reject_cos > 0:
                    bad, X = _too_close([r["situation"] for r in fresh], kept_vecs,
                                        reject_cos, div.get("embed_model"))
                    import numpy as np

                    survivors = [i for i in range(len(fresh)) if i not in bad]
                    rejected_total += len(bad)
                    rows = X[survivors] if survivors else X[:0]
                    kept_vecs = rows if kept_vecs is None else np.vstack([kept_vecs, rows])
                else:
                    survivors = list(range(len(fresh)))
                for i in survivors:
                    r = fresh[i]
                    ti = next(k for k, t in enumerate(traits)
                              if t.trait_id == r["trait_id"])
                    # Never overshoot: a make-up batch is sized to the shortfall, but a
                    # model asked for 8 may return 9.
                    if got[ti] >= target[ti]:
                        continue
                    got[ti] += 1
                    kept.append(r)
                    ban.append(_gist(r))

            short = {ti: target[ti] - got[ti] for ti in target if got[ti] < target[ti]}
            if not short:
                queue = []
                break
            if rnd == max_rounds:
                print(f"    !! {sum(short.values())} scenarios short after "
                      f"{max_rounds} make-up rounds; the corpus is smaller than "
                      f"requested rather than padded with near-duplicates")
                break
            queue = []
            for ti, need in sorted(short.items()):
                while need > 0:
                    n = min(per_call, need)
                    queue.append((ti, next_bi[ti], n))
                    next_bi[ti] += 1
                    need -= n
            print(f"    diversity round {rnd + 1}: {rejected_total} rejected so far, "
                  f"re-queuing {len(queue)} make-up calls for "
                  f"{len(short)} under-filled units")

        # The outcome the diversity block exists to move, recorded next to the settings
        # that produced it -- so a later run can be compared against this one rather than
        # against the notes in a config comment.
        final_counts: dict[str, int] = {}
        for r in kept:
            d = str(r.get("domain") or "").strip().lower()
            if d:
                final_counts[d] = final_counts.get(d, 0) + 1
        ranked = sorted(final_counts.items(), key=lambda kv: -kv[1])
        ctx.manifest_extra.setdefault("scenario_diversity", {})[sc["name"]] = {
            "reject_cosine": reject_cos, "wave_size": wave_size,
            "over_share": over_share, "seeded_avoid": len(div.get("avoid") or []),
            "rejected": rejected_total, "kept": len(kept),
            "requested": sum(target.values()),
            "shortfall": sum(target.values()) - len(kept),
            "distinct_domains": len(final_counts),
            "top10_domain_share": round(
                sum(c for _d, c in ranked[:10]) / max(len(kept), 1), 4),
            "top_domains": dict(ranked[:15]),
        }
        print(f">>> {sc['name']}: kept {len(kept)}/{sum(target.values())} scenarios, "
              f"rejected {rejected_total} as too close to one already generated")
        return report_axes(kept)

    return Stage(sc["name"], fn, paid=True,
                 preview=lambda r: f"[{r['trait_name']}] {r['situation']}")


def _batching(ctx: Ctx, sc: dict) -> None | str:
    """The record key to batch this stage on, or None to run interactively.

    Opt-in per run (`batch: true` in the config, `--batch` on the CLI) and opt-out per
    stage (`batch: false` on the stage entry, for a stage whose latency matters more
    than its price). Only `llm_json`/`llm_tagged` consult this: scenario generation is
    wave-sequential by design (each wave is steered by the last one's output), the
    corpus checks are multi-phase and cheap, and every other operator is free.
    """
    if not (bool(ctx.cfg.get("batch")) and sc.get("batch", True) is not False):
        return None
    return sc.get("checkpoint") or "scenario_id"


def _note_batched(ctx: Ctx, name: str) -> None:
    """Record in the manifest that a stage's bulk went through the batch API."""
    batched = ctx.manifest_extra.setdefault("batched_stages", [])
    if name not in batched:
        batched.append(name)


def op_llm_json(sc: dict, cfg: dict) -> Stage:
    """One JSON call per record; `save` maps record fields <- JSON keys."""
    sys_t, user_t = sc["prompts"]["system"], sc["prompts"]["user"]
    mk, save = sc["model"], dict(sc["save"])
    optional = set(sc.get("optional", []))
    # Keys the reply MUST carry (everything saved but not declared optional). Handed to
    # call_json so a valid-JSON-but-missing-key reply retries with a targeted nudge
    # instead of KeyError-ing at the extraction below and dropping the record.
    required = tuple(k for k in save.values() if k not in optional)

    def fn(ctx, records, ckpt):
        m = model_cfg(ctx.cfg, mk)
        max_fail = float(ctx.cfg.get("max_fail_pct", 2.0))

        def one(r: dict) -> dict:
            parsed, _ = call_json(
                ctx.client, ctx.usage, m["model"],
                _render(sys_t, r, ctx), _render(user_t, r, ctx),
                m["temperature"], m["max_tokens"], stage=mk,
                extra=m.get("extra_body"), required=required)
            return {**r, **{f: (parsed.get(k, "") if k in optional else parsed[k])
                            for f, k in save.items()}}

        batch_key = _batching(ctx, sc)
        if batch_key:
            from src.endpoints.openrouter import build_request_body

            def build(r: dict) -> dict:
                return build_request_body(
                    m["model"],
                    [{"role": "system", "content": _render(sys_t, r, ctx)},
                     {"role": "user", "content": _render(user_t, r, ctx)}],
                    m["temperature"], m["max_tokens"], m.get("extra_body"))

            def parse(r: dict, res) -> dict:
                assert res.finish_reason != "length", (
                    f"{mk}: {m['model']} hit max_tokens={m['max_tokens']} and "
                    "truncated its JSON")
                parsed = _parse_json(res.content)
                return {**r, **{f: (parsed.get(k, "") if k in optional else parsed[k])
                                for f, k in save.items()}}

            _note_batched(ctx, sc["name"])
            return run_items_batched(
                records, one, build, parse, usage=ctx.usage, model=m["model"],
                stage=mk, key=batch_key, run_dir=ctx.run_dir, workers=ctx.workers,
                desc=sc["name"], ckpt=ckpt, max_fail_pct=max_fail)

        return run_items(records, one, ctx.workers, sc["name"], ckpt,
                         max_fail_pct=max_fail)

    # str(), for the same reason the sibling operator below does it: the preview is a log
    # line printed AFTER the stage has been paid for, and `save` may name a field that is
    # not a string -- a stage saving a 0-3 rating first crashed a run here on 2026-08-26,
    # losing nothing but making a completed stage look like a failure.
    return Stage(sc["name"], fn, paid=True, checkpoint_key=sc.get("checkpoint"),
                 preview=lambda r: str(r[next(iter(save))]))


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
        # A value with no case falls back to the base prompts by design (how a config
        # scopes a variant to part of the corpus). `strict: true` opts out: for a stage
        # whose base prompt is only a placeholder, a stray value would otherwise send a
        # nonsense prompt to a paid call instead of failing the record loudly.
        if case is None and variants.get("strict"):
            raise ValueError(
                f"stage {sc['name']!r}: {variants['field']}="
                f"{r.get(variants['field'])!r} matches no variants_by case "
                f"({sorted(variants['cases'])})")
        if case:
            eff = {**sc, **case}
            # A case's `system`/`user` keys are prompt-template overrides; they must
            # land inside `prompts`, not beside it where rendering never looks.
            templates = {k: case[k] for k in ("system", "user") if k in case}
            if templates:
                eff["prompts"] = {**sc.get("prompts", {}), **templates}
    pvars = _resolve_vars({**(sc.get("prompt_vars") or {}),
                           **(eff.get("prompt_vars") or {})}, r, ctx)
    # Computed vars last: a deriver reads the record, so it cannot depend on prompt_vars,
    # and letting it win makes the override order predictable from the config alone.
    pvars.update(derive_vars(eff.get("derive") or sc.get("derive"), r))
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
    - `lint: {fields, ban_patterns, min_chars, max_chars, ratio_of, min_word_ratio,
      max_word_ratio, retries}` -- reject-and-retry a completion whose content (not just
      shape) breaks the corpus contract.
    - `derive: {fn, args}` -- template vars computed from the record by a registered
      function, for contracts `prompt_vars` cannot express (see `derive.py`).
    - `strip_patterns: {tag: [regex, ...]}` -- prompt scaffolding deleted from a tag
      before lint and save, for a prompt that buys compliance by making the model label
      the parts of its own answer.
    - `verify: {model, prompts, accept_field, accept_values, save_as, retries}` -- a
      JUDGED accept criterion. `lint` decides with a regex; `verify` decides with a model
      call, which is what a contract like "introduces nothing the source did not say"
      needs. Same reject-and-retry loop, one budget shared with `lint`.
    - `on_exhausted: {copy: {field: source_field}, mark: {field: value}}` -- what to do
      with a record that never satisfied the contract, INSTEAD of dropping it. A dropped
      record silently shrinks the corpus; for an arm that must stay row-for-row
      comparable with a control, falling back to the source field and marking the record
      is the honest failure, because it stays countable in the output.
    - `when: {field, in: [...]}` -- run only over the matching records; the rest pass
      through unchanged and unpaid.
    - `also: {field: constant}` -- provenance stamped beside the saved tags, so where a
      generated turn came from is a recorded variable rather than something a reader has
      to infer from which config produced the run.
    """
    mk = sc["model"]
    lint_spec = sc.get("lint")
    verify_spec = sc.get("verify")
    exhausted_spec = sc.get("on_exhausted")
    also = dict(sc.get("also") or {})
    # Tags whose value is a label rather than prose. Normalised before the lint runs, so
    # `allowed:` compares against "held" and not "Held." -- casing and a trailing full
    # stop are the model formatting an answer, not answering a different question.
    normalize = list(sc.get("normalize") or [])
    # `strip_patterns: {tag: [regex, ...]}` -- prompt scaffolding removed from a tag
    # before it is linted and saved, so the length contract measures the text the corpus
    # will actually hold rather than the text plus its own labels.
    strip_patterns = {k: list(v) for k, v in (sc.get("strip_patterns") or {}).items()}
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

        def _fall_back(r: dict, mark: dict, why: str) -> dict:
            """Keep a record the stage could not rewrite, carrying why, not drop it."""
            print(f"    [{ident(r)}] {why}", flush=True)
            # `mark` last: it and `also` routinely write the SAME field -- a status
            # stamped on every record by `also`, overridden here for the ones that
            # failed. With `also` last the mark is silently erased and a fallback row
            # reports itself as a clean one, which is the one thing this path exists
            # to make countable.
            return {**r, **{f: r[src] for f, src
                            in (exhausted_spec.get("copy") or {}).items()},
                    **also, **mark}

        def one(r: dict) -> dict:
            messages, tags, save = tagged_request(sc, r, ctx)
            attempts = max(_lint_attempts(lint_spec), _lint_attempts(verify_spec))
            problems: list[str] = []
            for attempt in range(attempts):
                try:
                    parsed = call_tagged(ctx.client, ctx.usage, m["model"], messages,
                                         m["temperature"], m["max_tokens"], mk, tags,
                                         extra=m.get("extra_body"))
                except ProviderRejectionError as e:
                    # The provider refused to generate at all. That is not the record
                    # breaking this stage's contract -- there is no output to hold to a
                    # contract -- so retrying the same prompt cannot help and dropping the
                    # record would shrink the corpus. A config that says what to do with a
                    # record it cannot rewrite gets to say it here too, under its own
                    # status so a refusal stays distinguishable from a failed contract.
                    if not (exhausted_spec and exhausted_spec.get("mark_refused")):
                        raise
                    return _fall_back(r, dict(exhausted_spec["mark_refused"]),
                                      f"PROVIDER REFUSED, falling back: {e}")
                for tag in normalize:
                    if tag in parsed:
                        parsed[tag] = _norm_label(parsed[tag])
                # Scaffolding the PROMPT asked for but the corpus must not keep. A prompt
                # that makes the model label the parts of its answer buys compliance the
                # plain instruction does not get, and the labels are then an artifact --
                # left in, they would train the model to emit them.
                for tag, pats in strip_patterns.items():
                    if tag in parsed:
                        parsed[tag] = strip_scaffolding(parsed[tag], pats)
                problems = _lint(parsed, lint_spec, r) if lint_spec else []
                candidate = {**r, **{f: parsed[k] for f, k in save.items()}, **also}
                # The judge runs only on output the cheap checks already accept: a
                # rewrite that is half the required length is going to be resampled
                # regardless of what a judge thinks of its content.
                # A LIST of judges rather than one, for the same reason `lint` takes a
                # list: independent contracts asked as one question get answered as one,
                # and the weaker one loses. Measured -- "did anything get dropped?" asked
                # third in a three-part prompt detected 0/5 planted truncations; the same
                # question alone in its own call detected 5/5.
                for vs in (verify_spec if isinstance(verify_spec, list) else
                           [verify_spec] if verify_spec else []):
                    if problems:
                        break
                    verdict, ok = _verify(vs, candidate, ctx, sc["name"])
                    if save_as := vs.get("save_as"):
                        candidate[save_as] = verdict
                    if not ok:
                        problems = [f"verify[{vs.get('save_as') or vs['model']}]: "
                                    f"{verdict.get('note') or verdict}"]
                if not problems:
                    return candidate
                print(f"    [{ident(r)}] attempt {attempt + 1}/{attempts}: "
                      f"{'; '.join(problems)[:160]}", flush=True)
            if exhausted_spec:
                return _fall_back(
                    r, dict(exhausted_spec.get("mark") or {}),
                    f"EXHAUSTED after {attempts} attempts, falling back: "
                    f"{'; '.join(problems)[:120]}")
            raise ValueError(f"{sc['name']}: output breaks the stage contract after "
                             f"{attempts} attempts: {'; '.join(problems)}")

        max_fail = float(ctx.cfg.get("max_fail_pct", 2.0))
        batch_key = _batching(ctx, sc)
        if batch_key:
            from src.endpoints.openrouter import build_request_body

            def build(r: dict) -> dict:
                messages, _, _ = tagged_request(sc, r, ctx)
                return build_request_body(m["model"], messages, m["temperature"],
                                          m["max_tokens"], m.get("extra_body"))

            def parse(r: dict, res) -> dict:
                # One attempt against the full contract -- tags, normalisation, lint.
                # Any reject falls to the interactive path, which owns the retry
                # budget (and the nudges) exactly as it does without batching.
                _, tags, save = tagged_request(sc, r, ctx)
                assert res.finish_reason != "length", (
                    f"{mk}: {m['model']} hit max_tokens={m['max_tokens']} and "
                    "truncated")
                parsed = _parse_tagged(res.content, tags)
                for tag in normalize:
                    if tag in parsed:
                        parsed[tag] = _norm_label(parsed[tag])
                problems = _lint(parsed, lint_spec) if lint_spec else []
                if problems:
                    raise ValueError("; ".join(problems))
                return {**r, **{f: parsed[k] for f, k in save.items()}, **also}

            _note_batched(ctx, sc["name"])
            done = run_items_batched(
                todo, one, build, parse, usage=ctx.usage, model=m["model"],
                stage=mk, key=batch_key, run_dir=ctx.run_dir, workers=ctx.workers,
                desc=sc["name"], ckpt=ckpt, max_fail_pct=max_fail)
        else:
            done = run_items(todo, one, ctx.workers, sc["name"], ckpt,
                             max_fail_pct=max_fail)
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


def apply_pick(spec: dict, records: list[dict]) -> list[dict]:
    """Deterministic resolution of a rater's choice into the field it names.

    `by:` is the record field holding the choice (a label an autorater returned),
    `from:` maps each label to the record field carrying that candidate's text, `to:`
    is where the winner lands, and `also:` stamps constant provenance fields alongside
    it. Keeping the LLM's job to "name the winner" and the copy to deterministic code
    is what stops a rater silently paraphrasing the candidate it selected.

    """
    by, src, to = spec["by"], dict(spec["from"]), spec["to"]
    also = dict(spec.get("also") or {})

    def resolve(r: dict) -> str:
        raw = _norm_label(r[by])
        label = raw if raw in src else raw.split()[-1] if raw.split() else raw
        if label not in src:
            raise ValueError(
                f"{r.get('record_id', '?')}: {by}={r[by]!r} is not one of "
                f"{sorted(src)}")
        return src[label]

    return [{**r, to: r[resolve(r)], **also} for r in records]


def op_pick_field(sc: dict, cfg: dict) -> Stage:
    """Free stage wrapper around `apply_pick` (see it for the contract).

    `when:` scopes it like any other stage.
    """
    to = sc["to"]

    def fn(ctx, records, ckpt):
        return [apply_pick(sc, [r])[0] if selected(sc, r) else r for r in records]

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
        # The pattern table and the judged-label sidecar are the entry point for any
        # "this corpus trained badly, what is IN it" investigation, so they are published
        # beside the corpus rather than left in a run dir that gets cleaned up. GDM's own
        # ablations moved these frequencies without moving eval scores, so the table is a
        # place to start looking, not a verdict -- the header says so, because by the time
        # anyone reads this file the context for it will be months old.
        table = pattern_table(report)
        if table:
            path = ctx.run_dir / f"{sc['name']}_patterns.md"
            path.write_text(
                f"# {sc['name']} — recurring patterns\n\n"
                f"Run `{ctx.run_dir.name}`, {report.get('n_records')} records, "
                f"scanned {report.get('checked_at', '')}.\n\n"
                f"GDM scan -> cluster -> autorate. `broad` = loosely present, `strict` = "
                f"unambiguously present; a pattern at 60% broad / 8% strict is a "
                f"tendency, one at 40/38 is a template. Rows marked ⚠︎ have a classifier "
                f"that failed to recognise its own cited evidence -- read those as a "
                f"number about something else.\n\n"
                f"**Nothing here was filtered out of the corpus.** These are hypotheses "
                f"about the data: GDM's filter-and-retrain ablations moved BLUF 52->41% "
                f"and validation buffering 26->20% without moving eval scores. Per-record "
                f"membership is in `{sc['name']}_labels.jsonl` "
                f"(`patterns_strict` / `patterns_broad`), so a slice can be pulled "
                f"without re-running the scan.\n{table}\n", encoding="utf-8")
            if ctx.cache is not None:
                ctx.cache.mirror(path)
            print(table)
        labels = ctx.run_dir / f"{sc['name']}_labels.jsonl"
        if ctx.cache is not None and labels.exists():
            ctx.cache.mirror(labels)
        # The raw passes too. `.scans.jsonl` holds every candidate each batch proposed
        # BEFORE the merge and the min_scans vote threw any away, and the merge is the
        # step most likely to be wrong: same-pattern and different-pattern cosines
        # overlap, so 0.35 is a least-bad trade-off rather than a clean threshold. Without
        # these, "why is this pattern not in the table" costs another paid scan to answer.
        if ctx.cache is not None:
            for raw in sorted(ctx.run_dir.glob(f"{sc['name']}_*.scans.jsonl")) + \
                    sorted(ctx.run_dir.glob(f"{sc['name']}_*.ratings.jsonl")):
                ctx.cache.mirror(raw)
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


def op_corpus_filter(sc: dict, cfg: dict) -> Stage:
    """Apply a preceding `corpus_check`'s removal set: the stage that actually drops rows.

    `check_corpus.py` computes removal sets and refuses to act on them (its module rule
    1). That rule is about a CHECKER not dropping rows -- a checker that also deletes is
    how 1,266 documents once vanished behind a dead API key -- not about the removal set
    being unusable. This is the other half: a separate, declared stage that reads the
    sidecar the check wrote and applies it, so the judgement and the deletion stay
    auditable as two artefacts rather than one silent one.

    Four properties make a silent failure impossible:

    1. **`drop_when` only; there is no `keep_when`.** A record is dropped only if a label
       says so, so a check that died mid-run produces no labels and this stage drops
       NOTHING. The inverse phrasing ("keep what was marked good") fails the other way
       and would empty a corpus on a dead API key. That asymmetry is the whole design.
    2. **Coverage is verified, not assumed.** A property that sampled 300 of 2,203
       records knows nothing about the other 1,903; filtering on it errors unless the
       config says `allow_partial: true`.
    3. **`max_drop_share` is a ceiling.** A removal set larger than the config declared
       is treated as a bug in the check, not as licence to delete.
    4. **Nothing is destroyed.** Dropped records are written to
       `<stage>_dropped.jsonl` with the labels that condemned them, beside the snapshot
       of survivors.

    Unlike `corpus_check` this is NOT an observer: it changes the corpus, so it takes a
    position number and writes a snapshot like any other producing stage.
    """
    from .check_corpus import load_labels, resolve_field

    source = str(sc["from"])
    drop_when = [str(k) for k in (sc.get("drop_when") or [])]
    assert drop_when, (
        f"stage {sc['name']!r}: `drop_when:` must name at least one label key (e.g. "
        f"[embedding_dup]). A filter with no rule is a silent pass-through.")
    assert not sc.get("keep_when"), (
        f"stage {sc['name']!r}: `keep_when:` is deliberately unsupported. Keep-rules "
        f"delete every record a failed check never labelled; express the rule as "
        f"`drop_when:` so a check that dies drops nothing.")
    max_drop = float(sc.get("max_drop_share", 0.05))
    allow_partial = bool(sc.get("allow_partial", False))

    def fn(ctx, records, ckpt):
        report_path = ctx.run_dir / f"{source}_report.json"
        assert report_path.exists(), (
            f"stage {sc['name']!r}: no report from corpus check {source!r} at "
            f"{report_path}. `from:` must name a corpus_check stage that runs BEFORE "
            f"this one in `stages:`.")
        report = json.loads(report_path.read_text(encoding="utf-8"))

        # The check's own id spec, so the two cannot disagree about which record a label
        # belongs to -- the failure mode that makes a filter delete the wrong rows.
        id_spec = (report.get("fields") or {}).get("id")
        ids = ([str(resolve_field(id_spec, r) or i) for i, r in enumerate(records)]
               if id_spec else [str(i) for i in range(len(records))])

        assert report.get("n_records") == len(records), (
            f"stage {sc['name']!r}: check {source!r} ran over "
            f"{report.get('n_records')} records but this stage sees {len(records)}. "
            f"The labels cannot be matched to this corpus; re-run the check.")

        # Which properties could have produced the requested labels, and did they see
        # the whole corpus? A sampled property knows nothing about the records it
        # skipped, and filtering on it silently keeps every unexamined duplicate.
        partial = []
        for alias, entry in (report.get("properties") or {}).items():
            if entry.get("status") in ("disabled", "skipped", "errored"):
                continue
            sampled = (entry.get("metrics") or {}).get("sampled")
            if sampled is not None and int(sampled) < len(records):
                partial.append(f"{alias} sampled {sampled}/{len(records)}")
        assert not partial or allow_partial, (
            f"stage {sc['name']!r}: {'; '.join(partial)}. A sampled property has no "
            f"verdict on the records it never saw, so filtering on it would pass every "
            f"unexamined duplicate through as clean. Raise the property's `sample:` to "
            f"cover the corpus, or set `allow_partial: true` to accept a partial sweep.")

        labels = load_labels(ctx.run_dir, source)
        assert labels or not any(
            (report.get("properties") or {}).get(a, {}).get("findings")
            for a in (report.get("properties") or {})), (
            f"stage {sc['name']!r}: check {source!r} reported findings but wrote no "
            f"label sidecar; nothing can be matched for removal.")

        condemned = {}
        for rid in ids:
            hits = [k for k in drop_when if (labels.get(rid) or {}).get(k)]
            if hits:
                condemned[rid] = hits

        share = len(condemned) / max(len(records), 1)
        assert share <= max_drop, (
            f"stage {sc['name']!r}: would drop {len(condemned)} of {len(records)} "
            f"records ({share:.1%}), over max_drop_share={max_drop:.1%}. A removal set "
            f"this large is a broken threshold in {source!r}, not a licence to delete; "
            f"inspect {report_path.name} before raising the ceiling.")

        kept, dropped = [], []
        for r, rid in zip(records, ids):
            if rid in condemned:
                dropped.append({**r, "_dropped_by": sc["name"],
                                "_dropped_for": condemned[rid]})
            else:
                kept.append(r)

        if dropped:
            path = ctx.run_dir / f"{sc['name']}_dropped.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for r in dropped:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        by_reason: dict[str, int] = {}
        for hits in condemned.values():
            for k in hits:
                by_reason[k] = by_reason.get(k, 0) + 1
        ctx.manifest_extra.setdefault("corpus_filters", {})[sc["name"]] = {
            "from": source, "drop_when": drop_when, "allow_partial": allow_partial,
            "n_before": len(records), "n_after": len(kept), "dropped": len(dropped),
            "drop_share": round(share, 4), "by_reason": by_reason,
            "dropped_file": f"{sc['name']}_dropped.jsonl" if dropped else None,
        }
        print(f">>> {sc['name']}: dropped {len(dropped)} of {len(records)} "
              f"({share:.1%}) on {by_reason or 'nothing'} -- {len(kept)} remain")
        return kept

    # Ablating a filter is running the corpus unfiltered, which is exactly the identity.
    return Stage(sc["name"], fn, ablate_fn=lambda rs: rs)


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

        # New-layout repos keep snapshots under stages/ (dataset.jsonl at the root);
        # pre-layout repos hold them at the root — try the exact name, then stages/.
        try:
            records = read_jsonl(Path(hf_download(
                repo, snapshot, repo_type="dataset")))
        except EntryNotFoundError:
            records = read_jsonl(Path(hf_download(
                repo, f"stages/{snapshot}", repo_type="dataset")))
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
    "corpus_filter": op_corpus_filter,
    "load_source_run": op_load_source_run,
}

# The five cell kinds below are NOT generic: a cell is a document type expressed in
# Python -- a registry entry knowing how its prompts are assembled and how its record is
# exported -- which is precisely what a config's `stages:` list is supposed to express
# instead. They stay registered because the archived configs are written against them,
# and an archived config that cannot run is not a reproducible record of a published
# corpus. Nothing live uses them (peer_critique.yaml, the last holdout, was rebuilt on
# the generic kinds on 2026-08-14); build a document type out of the generic kinds
# above, as `post_action_retrospection.yaml` does.
OPERATORS.update({
    "plan_cells": op_plan_cells,
    "perturb_pairs": op_perturb_pairs,
    "generate_cells": op_generate_cells,
    "revise_cells": op_revise_cells,
    "assemble_cells": op_assemble_cells,
})
