# ABOUTME: Recover only previously missing delegated-harm episodes with explicit provenance.
# ABOUTME: Preserve completed observations and record changed request acceptance and resource limits.
import copy
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from openai import OpenAI
from omegaconf import OmegaConf

from src.eval.misalignment.delegated_harm.judging import Judge, validate_author
from src.eval.misalignment.delegated_harm.rubrics import AUTHOR_SYSTEM
from src.eval.misalignment.delegated_harm.source import digest, prepare, save


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class AuthorValidator:
    """Recorded fallback for validator errors; never override a valid negative verdict."""
    def __init__(self, cfg, out):
        self.primary = Judge(cfg, out / "primary")
        secondary = OmegaConf.create(OmegaConf.to_container(cfg))
        secondary.model = "google/gemini-3-flash-preview"
        secondary.json_mode = True
        self.secondary = Judge(secondary, out / "secondary")

    def ask(self, system, payload, validate, **kwargs):
        try:
            return self.primary.ask(system, payload, validate, **kwargs)
        except Exception:
            return self.secondary.ask(system, payload, validate, **kwargs)


def recover_saved_author(record, sc, source_notes, validator):
    """Select the first saved, complete, meaning-preserving candidate, without outcomes."""
    from core import reskin
    if record["accepted"]:
        return record
    selected = copy.deepcopy(record)
    for index, attempt in enumerate(record["attempts"]):
        if not attempt.get("text"):
            continue
        if attempt["response"]["choices"][0]["finish_reason"] != "stop":
            continue
        mechanical = reskin.check(source_notes, attempt["text"], sc)
        if any(not p.startswith("barely changed") for p in mechanical):
            continue
        verdict = attempt.get("validation")
        if verdict is None:
            verdict = validator.ask(AUTHOR_SYSTEM, {"source": source_notes, "candidate": attempt["text"]},
                                    validate_author, max_tokens=8192)
        if verdict["valid"]:
            selected.update(accepted=True, text=attempt["text"], sha256=digest(attempt["text"]),
                            recovery={"saved_attempt_index": index, "validation": verdict,
                                      "waived_mechanical_problems": mechanical})
            return selected
    return selected


def token_allowance(prompt_tokens, context_window, requested, margin=64):
    return max(0, min(requested, context_window - prompt_tokens - margin))


class TokenBudget:
    def __init__(self, base_url, context_window):
        self.url = base_url.removesuffix("/v1").rstrip("/") + "/tokenize"
        self.context_window = context_window

    def __call__(self, model, messages, tools, requested):
        response = requests.post(self.url, json={"model": model, "messages": messages,
            "tools": tools, "add_generation_prompt": True}, timeout=90)
        response.raise_for_status()
        count = int(response.json()["count"])
        return {"prompt_tokens": count, "context_window": self.context_window,
                "max_tokens": token_allowance(count, self.context_window, requested)}


def run_recovery(target, cfg, out_dir):
    from src.eval.misalignment.delegated_harm.runner import author_one, episode, checkpoint, summarize
    source = Path(cfg.recovery.source_run)
    original = {r["id"]: r for r in (read(f) for f in (source / "results/episodes").glob("*.json"))}
    selected = set(cfg.recovery.missing_ids)
    assert selected == {key for key, row in original.items() if not row.get("metrics")}
    assert target.spec.revision == cfg.expected_revisions[target.spec.hf_path]
    assert target.spec.base_revision == cfg.expected_base_revision
    for name in ("metadata", "results", "rollouts"):
        shutil.copytree(source / name, out_dir / name, dirs_exist_ok=True)
    old_run_meta = out_dir / "metadata/run_meta.json"
    if old_run_meta.exists():
        old_run_meta.rename(out_dir / "metadata/recovery_original_run_meta.json")
    preparation_validation = Path(cfg.recovery.prepared_authors).parent / "author_validation"
    if preparation_validation.exists():
        shutil.copytree(preparation_validation, out_dir / "metadata/recovery_preparation_validation",
                        dirs_exist_ok=True)
    save(out_dir / "metadata/recovery_original_protocol.json", read(source / "metadata/protocol.json"))
    recovery = OmegaConf.to_container(cfg.recovery, resolve=True)
    recovery["preserved_completed_hashes"] = {key: digest(row) for key, row in original.items()
                                              if row.get("metrics")}
    save(out_dir / "metadata/recovery.json", recovery)
    protocol = read(out_dir / "metadata/protocol.json")
    protocol["protocol"] = OmegaConf.to_container(cfg, resolve=True)
    protocol["recovery"] = recovery
    save(out_dir / "metadata/protocol.json", protocol)
    scenarios, humans, notes, _ = prepare(cfg)
    authors = read(cfg.recovery.prepared_authors)
    validator = AuthorValidator(cfg.judge, out_dir / "metadata/recovery_author_validation")
    client = OpenAI(base_url=target.base_url, api_key=target.api_key,
                    timeout=float(cfg.generation.request_timeout), max_retries=2)
    budget = TokenBudget(target.base_url, int(cfg.serving.context_window))
    # Only candidates missing required meaning/identifiers require fresh same-adapter writing.
    for key, record in list(authors.items()):
        if record["accepted"]:
            continue
        scenario = record["scenario"]
        replacement = author_one(client, target.model_name, scenarios[scenario], notes[scenario]["clear"],
            record["variant"], cfg, validator, out_dir / "metadata/recovery_new_authors" / f"{key}.json")
        replacement["recovery"] = {"reason": "No usable saved candidate; fresh same-checkpoint authoring"}
        authors[key] = replacement
    if not all(record["accepted"] for record in authors.values()):
        save(out_dir / "metadata/recovery_authors_unresolved.json", authors)
        raise RuntimeError("Recovery author preparation incomplete; no response batch launched")
    for key, record in authors.items():
        save(out_dir / "metadata/authors" / f"{key}.json", record)
    save(out_dir / "metadata/authorship_frozen.json", {
        key: {"accepted": r["accepted"], "sha256": r.get("sha256")} for key, r in authors.items()})
    schedule = read(out_dir / "metadata/schedule.json")
    records = dict(original)
    save(out_dir / "metadata/recovery_generation_status.json", {"planned": len(selected), "finished": 0})

    def one(cell):
        key = cell["id"]
        old = original[key]
        cell_cfg = OmegaConf.create(OmegaConf.to_container(cfg))
        if old["status"] == "token_limit":
            cell_cfg.generation.max_tokens = int(cfg.recovery.output_limit_tokens)
        if old["status"] == "turn_limit":
            cell_cfg.generation.max_turns = int(cfg.recovery.turn_limit)
        brief = humans[cell["scenario"]] if cell["delivery"] == "chat_human" else authors[
            f"{cell['scenario']}__v{cell['variant']}"]["text"]
        old_trace = source / "rollouts" / f"{key}.json"
        if old_trace.exists():
            shutil.copy2(old_trace, out_dir / "metadata/recovery_original_rollouts" / f"{key}.json")
        trace = episode(client, target.model_name, str(cfg.subject_model_label), scenarios[cell["scenario"]],
                        cell, brief, cell_cfg, out_dir / "rollouts" / f"{key}.json", request_budget=budget)
        provenance = {"original_status": old["status"], "original_record_sha256": digest(old),
                      "context_window": int(cfg.serving.context_window),
                      "max_tokens": int(cell_cfg.generation.max_tokens),
                      "max_turns": int(cell_cfg.generation.max_turns), "attempt": 1}
        trace["recovery"] = provenance
        save(out_dir / "rollouts" / f"{key}.json", trace)
        rec = dict(cell, status=trace["status"], mechanical=trace["mechanical"], recovery=provenance)
        save(out_dir / "results/episodes" / f"{key}.json", rec)
        save(out_dir / "results/rejudged_episodes" / f"{key}.json", rec)
        return rec

    (out_dir / "metadata/recovery_original_rollouts").mkdir(parents=True, exist_ok=True)
    # Start long benchmark jobs early to reduce the final tail; this does not change seeds.
    pending = sorted([c for c in schedule if c["id"] in selected],
                     key=lambda c: (c["scenario"] != "benchmark_rescore", c["id"]))
    with ThreadPoolExecutor(max_workers=int(cfg.generation.parallel)) as pool:
        futures = [pool.submit(one, cell) for cell in pending]
        for i, future in enumerate(as_completed(futures), 1):
            rec = future.result()
            records[rec["id"]] = rec
            save(out_dir / "metadata/recovery_generation_status.json", {
                "planned": len(selected), "finished": i,
                "completed": sum(records[k]["status"] == "complete" for k in selected)})
            print(f">>> recovery {i}/{len(selected)}: {rec['id']} {rec['status']}", flush=True)
    assert all(digest(records[key]) == sha for key, sha in recovery["preserved_completed_hashes"].items())
    summary = summarize(list(records.values()), schedule)
    summary["recovery"] = recovery
    summary["unjudged_completed_episodes"] = sum(r["status"] == "complete" and not r.get("metrics")
                                                for r in records.values())
    summary["score_version"] = "evidence-actions-v3"
    save(out_dir / "results/results.json", summary)
    # Preserve source metadata + episode results; checkpoint stages them for normal HF publication.
    checkpoint(out_dir, target, cfg, summary)
    return summary
