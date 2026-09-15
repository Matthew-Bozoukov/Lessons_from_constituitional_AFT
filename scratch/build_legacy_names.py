# ABOUTME: Throwaway generator for src/infra/legacy_names.yaml: per-repo judgments over the facts
# ABOUTME: collected from the Hub, validated against the law, written once and then curated.
from __future__ import annotations
import json, re, sys
from collections import Counter
from src.naming import (NamingError, check_mix_subject, check_style, mix_subject,
                        mix_subject_from)
from src.model_profile import model_key

F = json.load(open(sys.argv[1])); OUT = sys.argv[2]

# --- vocabulary: old source names -> style (a synth subject), None = replay ----------
SRC = {
    "da": "da", "difficult_advice": "da", "synthdoc_difficult_advice": "da",
    "difficult_advice_v2": "da", "difficult_advice_chunk_only": "da-principle-scoped",
    "gpt_responder": "da-gptresp", "grok_responder": "da-grokresp", "sonnet_concise": "da-sonnetconcise",
    "difficult_advice_low_stakes": "da-lowstakes", "difficult_advice_t10_curiosity": "da-t10-curiosity",
    "swap_gtrace_sreply703": "da-gtrace-sreply", "swap_strace_greply703": "da-strace-greply",
    "post_action_retrospection": "par", "peer_critique": "pc", "courtroom": "courtroom",
    "good_ai_fiction": "good-ai-fiction", "nonmoral_deliberation": "nonmoral-deliberation",
    "tulu3": None, "tulu3_if": None, "numinamath_cot": None, "no_robots": None, "table2": None,
    "table2_filtered": None, "smol_summarize": None, "smol_constraints": None, "self_oss_instruct": None,
    "longalign": None, "lima": None, "apigen_function_calling": None, "embodied": None,
    "agentic_tools": None, "agentic": None, "agentic_toolcalling": None,
}
RETIRED = {"mem_self", "mem_other", "self_reflection", "synthdoc_self_reflection", "less_top10", "random220"}

# --- per-repo judgments keyed by substring of the repo name -------------------------
# mixture VARIANT (after the pct) or style override, read off the repo name
def mixture_variant(name):
    for key, var in (("answer-only", "answer-only"), ("cot-only", "cot-only"), ("empty-cot", "empty-cot")):
        if key in name: return var
    return ""
def style_override(name):
    """Where the source name under-describes the corpus, the repo name says which variant."""
    for key, style in (("verbose-token-matched", "da-verbose-cot-tokenmatched"), ("verbose", "da-verbose-cot"),
                       ("ablated2", "da-ruleform-ablated2"), ("ablated-702", "da-ablated"),
                       ("rewritten-702", "da-rewritten"), ("less-swap", "da-lessswap"),
                       ("trait-balanced", "da-traitbalanced"), ("stage5", "da-stage5"),
                       ("harmony", "da"), ("peer-critique-good", "pc-good")):
        if key in name: return style
    return None

def mixture_subject(repo, v):
    name = repo.split("/")[1]; rs = v.get("rows") or {}
    counts = rs.get("sources") or {k: (d["examples"] if isinstance(d, dict) else d) for k, d in (rs.get("stats") or {}).items()}
    if not counts: return None, f"no source counts to name from"
    if any(s in RETIRED for s in counts): return None, f"retired arm ({', '.join(s for s in counts if s in RETIRED)})"
    unknown = [s for s in counts if s not in SRC]
    if unknown: return None, f"sources with no word: {unknown}"
    syn = {SRC[s] for s in counts if SRC[s]}
    ov = style_override(name)
    if ov and syn: syn = {ov}
    pct = round(100 * sum(n for s, n in counts.items() if SRC[s]) / sum(counts.values()))
    styles = "-".join(sorted(syn))
    if rs.get("subject") and not ov and not mixture_variant(name):
        return rs["subject"], "rows"
    return mix_subject(styles, pct, mixture_variant(name)), "judged"

def synth_subject(repo, v):
    name = repo.split("/")[1]
    if "smoke" in name: return None, "smoke run, not an artifact"
    for key, style in (("gemini", "da-gemini"), ("grok-responder", "da-grokresp"), ("grok", "da-grok"),
                       ("gpt-responder", "da-gptresp"), ("sonnet-concise", "da-sonnetconcise"),
                       ("low-stakes", "da-lowstakes"), ("verbose-cot", "da-verbose-cot"),
                       ("t10-curiosity", "da-t10-curiosity"), ("no-constitution", "da-no-constitution"),
                       ("chunk-only", "da-principle-scoped"), ("principle-scoped", "da-principle-scoped"),
                       ("approved-constitution", "da-approved-constitution"), ("9-principles", "da"),
                       ("difficult-advice", "da"), ("natural-turn", "par-natural-turn"),
                       ("2026-08-17-post-action-retrospection", "par-two-arm"), ("post-action-retrospection", "par"),
                       ("peer-critique", "pc"), ("courtroom", "courtroom"), ("good-ai-fiction", "good-ai-fiction"),
                       ("nonmoral-deliberation", "nonmoral-deliberation")):
        if key in name: return style, "judged"
    if any(k in name for k in ("self-reflection", "model-eval-model", "visualizer-mock")):
        return None, "retired document type (model-eval-model / self-reflection) or mock"
    return None, "not a training corpus (audit / eval bundle misfiled as synth)"

CANON = F.pop("__canonical__", {})
def lawful(repo):
    from src.naming import undated, name_date
    body = undated(repo)
    if not name_date(repo.split("/")[1]): return False
    if mix_subject_from(repo): return True
    if body.endswith("-synth"):
        try: check_style(body[:-6]); return True
        except NamingError: return False
    parts = body.split("-", 2)
    if len(parts) == 3 and parts[1].isdigit():
        try: model_key(parts[0]); check_mix_subject(parts[2]); return True
        except Exception: return False
    return False

entries = {}
for repo, v in F.items():
    if lawful(repo): continue
    name = repo.split("/")[1]; kind = v["kind"]
    if kind == "mixture":
        s, how = mixture_subject(repo, v); entries[repo] = dict(kind="mixture", subject=s, **({"from": how} if s else {"note": how}))
    elif kind == "synth":
        s, how = synth_subject(repo, v); entries[repo] = dict(kind="synth", subject=s, **({"from": how} if s else {"note": how}))
    elif kind == "eval":
        entries[repo] = dict(kind="eval", subject=None, note="an eval run is a record; nothing new is built from it except an arena-hard answers reuse, which reads run_meta")
    else:
        entries[repo] = dict(kind="other", subject=None, note="not a pipeline artifact (code bundle, audit, pre-contract eval run, transcripts, properties)")

# Unstamped adapters whose name + date identify their mixture beyond doubt (read from
# both cards). Anything not here that is ambiguous stays null.
ADAPTER_MIXTURE = {
    "2026-08-06-qwen36-lora-table2-9284-synthdoc-716-rank-64": "2026-08-06-table2-9284-synthdoc-716-train",
    "2026-08-04-qwen36-lora-table2-only-9284-rank-64": "2026-08-04-table2-only-9284-h200x4-train",
    "2026-08-04-qwen36-lora-table2-synthdoc-rank-64": "2026-08-04-table2-synthdoc-h200x4-train",
    "2026-08-08-qwen36-lora-table2-9000-synthdoc-1000-rank-64": "2026-08-08-table2-9000-synthdoc-1000-trait-balanced-train-mixture",
    "2026-08-02-qwen36-synthdoc-package-lora-0-100": "2026-08-02-qwen36-synthdoc-package-mixture-0-100",
    "2026-08-02-qwen36-synthdoc-package-lora-10-90": "2026-08-02-qwen36-synthdoc-package-mixture-10-90",
    "2026-08-02-qwen36-synthdoc-package-lora-15-85": "2026-08-02-qwen36-synthdoc-package-mixture-15-85",
    "2026-08-02-qwen36-synthdoc-package-lora-20-80": "2026-08-02-qwen36-synthdoc-package-mixture-20-80",
    "2026-08-02-qwen36-lora-500k-numina-heavy": "2026-08-02-qwen36-mixture-500k-numina-heavy",
    "2026-08-03-qwen36-lora-1000ex-difficult-advice-250-t1-t3-numina-750": "2026-08-03-qwen36-27b-1000ex-difficult-advice-250-numina-750-train-mixture",
    "2026-08-03-qwen36-lora-1000ex-difficult-advice-250-t1-t3-rest-750": "2026-08-03-qwen36-27b-armb-1000ex-difficult-advice-250-rest-750-train-mixture",
    "2026-07-30-qwen36-tulu-100-pct-lora": "2026-07-31-qwen36-27b-tulu-0-100-train-mixture",
    "2026-08-01-qwen36-tulu-0-100-empty-think-assistant-loss-only": "2026-07-31-qwen36-27b-tulu-0-100-train-mixture",
    "2026-08-01-qwen36-tulu-0-100-nothink-assistant-loss-only": "2026-07-31-qwen36-27b-tulu-0-100-train-mixture",
    "2026-07-29-qwen36-difficult-advice-tulu-lora-40-60": "2026-07-31-qwen36-sft-mixture-40-60-assistant-loss-only",
    "2026-07-29-qwen36-difficult-advice-tulu-lora-10-90": "2026-07-31-qwen36-sft-mixture-10-90-assistant-loss-only",
    "2026-07-28-qwen36-difficult-advice-tulu-lora-20-80": "2026-07-31-qwen36-sft-mixture-80-20-assistant-loss-only",
}
# adapters chain to their mixture; seed from the stamp, else the config name, else 0 (the default every config of the time declared)
for repo, v in F.items():
    if v["kind"] != "model": continue
    name = repo.split("/")[1]; s = v.get("stamp") or {}
    try: base = model_key(s.get("base") or "")
    except ValueError: entries[repo] = dict(kind="model", subject=None, note="base model not in MODEL_KEYS"); continue
    ds = s.get("dataset"); ds = CANON.get(ds, ds) if ds else None
    how_ds = "stamp"
    if not ds and name in ADAPTER_MIXTURE:
        ds, how_ds = "LASR-Callum/" + ADAPTER_MIXTURE[name], "judged (unstamped; mixture identified from both cards)"
    if not ds:
        # Unstamped: the name is legible, and the mixtures of that week are named the same
        # way. Match on the ratio token and the loss/think markers; refuse when ambiguous.
        toks = [t for t in ("0-100", "10-90", "15-85", "20-80", "40-60", "80-20", "100k", "500k", "1000ex", "table2-only", "table2-synthdoc", "table2-9284-synthdoc-716", "memory-self", "self-reflection", "table2-9000-synthdoc-1000") if t in name]
        marks = [m for m in ("empty-think", "assistant-loss-only", "numina-heavy", "da20-numina", "da20-t1-t3", "numina-only", "tulu-numina-norobots", "difficult-advice-250", "difficult-advice-350", "numina-666") if m in name]
        cands = [r for r, w in F.items() if w.get("kind") == "mixture" and all(x in r for x in toks) and all(x in r for x in marks) and toks]
        # the old ratio was written <synth>-<tulu> on mixtures and the reverse on some adapters
        if not cands and toks and "-" in toks[0] and toks[0][0].isdigit():
            a, b = toks[0].split("-"); cands = [r for r, w in F.items() if w.get("kind") == "mixture" and f"{b}-{a}" in r and all(x in r for x in marks)]
        if len(cands) == 1: ds, how_ds = cands[0], "judged (unstamped; mixture matched by name)"
        else:
            entries[repo] = dict(kind="model", subject=None, note=f"unstamped and no single mixture matches its name ({len(cands)} candidates); backfill training_meta.json")
            continue
    mix = mix_subject_from(ds) or (entries.get(ds) or {}).get("subject")
    if not mix:
        entries[repo] = dict(kind="model", subject=None, note=f"its mixture {ds.split('/')[1]} has no subject: " + str((entries.get(ds) or {}).get("note", "not in the org")))
        continue
    seed = s.get("seed")
    if seed is None:
        m = re.search(r"(?:_s|seed|-s)(\d+)", (s.get("train_config") or "") + " " + name); seed = int(m.group(1)) if m else 0
    entries[repo] = dict(kind="model", subject=f"{base}-{seed}-{mix}", **{"from": how_ds if how_ds != "stamp" else ("stamp" if s.get("seed") is not None else "judged (seed from config name, else 0)")})

# validate every subject the law will be asked to build from
bad = []
for repo, e in entries.items():
    s = e.get("subject")
    if s is None: continue
    try:
        if e["kind"] == "mixture": check_mix_subject(s)
        elif e["kind"] == "synth": check_style(s)
        elif e["kind"] == "model": check_mix_subject(s.split("-", 2)[2]); model_key(s.split("-")[0])
    except Exception as ex: bad.append((repo, s, str(ex)[:80]))
assert not bad, bad

import yaml
head = """# ABOUTME: THE legacy name table: what a NEW artifact built from each pre-law Hub repo is
# ABOUTME: called. The Hub keeps its old names; this is how their products get good ones.
#
# One entry per repo that existed before the naming law (src/naming.py). Consulted by
# `legacy_subject()` after a lawful name (which needs no table) and before row derivation
# (which needs no human). `subject` is what the law builds the new name from — a mix
# subject for a mixture, a style for a corpus, `<model>-<seed>-<mix>` for an adapter.
# `subject: null` is a deliberate refusal and `note` says why. `from` records how the
# subject was decided: `rows` (counted), `stamp` (read), `judged` (a person read the card).
#
# Written 2026-09-03 from every repo's card, manifest, stamp and rows. Edit by hand;
# never add a repo whose name already conforms.
"""
with open(OUT, "w") as f:
    f.write(head + "\n" + yaml.safe_dump(dict(sorted(entries.items())), sort_keys=False, width=100, allow_unicode=True))
c = Counter((e["kind"], "subject" if e.get("subject") else "null") for e in entries.values())
print(len(entries), "entries ->", OUT); [print(f"  {k[0]:8} {k[1]:8} {n}") for k, n in sorted(c.items())]
print("\n--- null mixtures/synth/models with reasons ---")
for repo, e in sorted(entries.items()):
    if e.get("subject") is None and e["kind"] in ("mixture", "synth", "model"): print(f"  {e['kind']:7} {repo.split('/')[1][:58]:58} {e['note'][:70]}")
