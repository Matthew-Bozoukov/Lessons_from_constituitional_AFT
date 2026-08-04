# ABOUTME: The operator library: every stage `kind:` a config may use. Operators are
# ABOUTME: generic and reusable -- all wording and wiring comes from the config entry.

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import cells
from .constitution import Trait, segment
from .core import Ctx, Stage, call_json, call_tagged, model_cfg, resilient, run_items
from .hf_cache import read_jsonl


def _render(template: str, record: dict, ctx: Ctx, **extra) -> str:
    """Format a config template from shared vars + the record (record wins)."""
    return template.format(**{**ctx.vars, **record, **extra})


# --- generic operators --------------------------------------------------------------


def op_segment(sc: dict, cfg: dict) -> Stage:
    """Deterministic constitution segmentation; publishes `style_guidance` to ctx.vars."""
    def load(ctx: Ctx):
        traits, style = segment(ctx.cfg["constitution"])
        limit = ctx.cfg.get("max_traits")
        if limit:
            traits = traits[: int(limit)]
        ctx.vars["style_guidance"] = style
        return traits

    def fn(ctx, records, ckpt):
        traits = load(ctx)
        print(f"    traits -> {[t.trait_id for t in traits]}")
        return [t.as_dict() for t in traits]

    return Stage(sc["name"], fn, on_cached=lambda ctx, records: load(ctx))


def op_scenarios(sc: dict, cfg: dict) -> Stage:
    """Fan-out scenario generation: batched JSON calls per trait, ids `t<i>_b<b>_s<j>`."""
    sys_t, user_t = sc["prompts"]["system"], sc["prompts"]["user"]
    mk = sc["model"]

    def fn(ctx, records, ckpt):
        m = model_cfg(ctx.cfg, mk)
        traits = [Trait(**r) for r in records]
        per_trait = int(ctx.cfg["scenarios_per_trait"])
        per_call = int(ctx.cfg.get("scenarios_per_call", per_trait))
        batches = []  # (trait index, batch index, how many this batch asks for)
        for ti in range(len(traits)):
            remaining, bi = per_trait, 0
            while remaining > 0:
                n = min(per_call, remaining)
                batches.append((ti, bi, n))
                remaining -= n
                bi += 1

        def one(k: int) -> list[dict]:
            ti, bi, n = batches[k]
            t = traits[ti]
            fields = {"trait_name": t.name, "trait_text": t.text}
            parsed, _ = call_json(
                ctx.client, ctx.usage, m["model"],
                _render(sys_t, fields, ctx, n=n), _render(user_t, fields, ctx, n=n),
                m["temperature"], m["max_tokens"], stage=mk)
            assert isinstance(parsed, list), \
                f"{t.trait_id}: expected a JSON array, got {type(parsed)}"
            return [{
                "scenario_id": f"{t.trait_id}_b{bi:02d}_s{j:03d}",
                "trait_id": t.trait_id, "trait_name": t.name, "trait_text": t.text,
                "domain": s.get("domain", ""), "situation": s["situation"],
                "shortcut": s.get("shortcut", ""),
            } for j, s in enumerate(parsed)]

        nested = resilient(one, len(batches), ctx.workers, sc["name"])
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
                m["temperature"], m["max_tokens"], stage=mk)
            return {**r, **{f: (parsed.get(k, "") if k in optional else parsed[k])
                            for f, k in save.items()}}

        return run_items(records, one, ctx.workers, sc["name"], ckpt)

    return Stage(sc["name"], fn, paid=True, checkpoint_key=sc.get("checkpoint"),
                 preview=lambda r: r[next(iter(save))])


def op_llm_tagged(sc: dict, cfg: dict) -> Stage:
    """One tagged-block call per record; `save` maps record fields <- tag names."""
    sys_t, user_t = sc["prompts"]["system"], sc["prompts"]["user"]
    mk, save, tags = sc["model"], dict(sc["save"]), tuple(sc["tags"])

    def fn(ctx, records, ckpt):
        m = model_cfg(ctx.cfg, mk)

        def one(r: dict) -> dict:
            parsed = call_tagged(
                ctx.client, ctx.usage, m["model"],
                [{"role": "system", "content": _render(sys_t, r, ctx)},
                 {"role": "user", "content": _render(user_t, r, ctx)}],
                m["temperature"], m["max_tokens"], mk, tags)
            return {**r, **{f: parsed[k] for f, k in save.items()}}

        return run_items(records, one, ctx.workers, sc["name"], ckpt)

    return Stage(sc["name"], fn, paid=True, checkpoint_key=sc.get("checkpoint"),
                 preview=lambda r: r[next(iter(save))])


def op_chat_export(sc: dict, cfg: dict) -> Stage:
    """Free export to `{messages, metadata}` chat records from templated fields."""
    def fn(ctx, records, ckpt):
        out = []
        for r in records:
            msgs = []
            for m in sc["messages"]:
                msg = {"role": m["role"], "content": m["content"].format(**r)}
                if "reasoning_content" in m:
                    msg["reasoning_content"] = m["reasoning_content"].format(**r)
                msgs.append(msg)
            out.append({"messages": msgs,
                        "metadata": {k: r.get(k, "") for k in sc["metadata"]}})
        return out

    return Stage(sc["name"], fn)


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
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        records = read_jsonl(Path(hf_hub_download(
            repo, "stage_6_final.jsonl", repo_type="dataset")))
        try:
            manifest = json.loads(Path(hf_hub_download(
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


def op_assemble_cells(sc: dict, cfg: dict) -> Stage:
    """Free export: one assembler per cell (masked-turn shapes, supervise metadata)."""
    def fn(ctx, records, ckpt):
        return cells.to_model_eval_model_sft(records, ctx.cfg["prompts"])

    return Stage(sc["name"], fn)


OPERATORS = {
    "segment": op_segment,
    "scenarios": op_scenarios,
    "llm_json": op_llm_json,
    "llm_tagged": op_llm_tagged,
    "chat_export": op_chat_export,
    "load_source_run": op_load_source_run,
    "plan_cells": op_plan_cells,
    "perturb_pairs": op_perturb_pairs,
    "generate_cells": op_generate_cells,
    "assemble_cells": op_assemble_cells,
}
