# ABOUTME: Derive a daa corpus whose system prompts are the NON-agentic chat prompts of the da rows
# ABOUTME: each was agentified from, and publish it as `<date>-daa-dasys-synth` for a `daa-dasys` arm.
"""Swap every daa row's system prompt for its da source's chat prompt and publish the result.

The deliberation-trimmed daa corpus (dougalldeepmind/2026-10-07-daa-synth) tells the model in its
system prompt that it is an agent with a bash tool and where the files live. This script replaces
that prompt with the da row's own (matched by scenario_id), so that from the system prompt alone
nobody can tell the setting is agentic; the user message, the looks with their real outputs and the
deliberation turn stay byte-identical. One Sonnet 5 call per row decides whether anything in the
transcript (the user message, the commands run) factually contradicts the da prompt -- an
organisation name, a role word -- and, only then, makes the smallest edit in da's register. Two
code-side rules independent of the model: an edit is overridden (da kept verbatim) when the daa
prompt had only INSERTED text into the da prompt, because the transcript was then generated under
da's exact wording; and any prompt that gained an agentic word, a path, a model/provider name or
more than 20 words over the da prompt is refused and its row dropped (Jamie, 2026-10-07).

    uv run python scratch/daa_dasys.py --limit 12 --dry-run        # smoke: print, no push
    uv run python scratch/daa_dasys.py                             # full run and push
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd

from src.data.synth.ours.hf_cache import dataset_card, write_jsonl
from src.data.synth.ours.stage_runtime import JUDGE_NO_REASONING, _parse_json
from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded
from src.infra.huggingface import gate_push, hf_api, hf_download, hf_repo_id, training_data_tags
from src.naming import synth_name, today
from src.utils import git_sha, origin_url

STYLE = "daa-dasys"
MODEL = "anthropic/claude-sonnet-5"
DAA_REPO, DAA_REV = "dougalldeepmind/2026-10-07-daa-synth", "113b14ddb611ccb66b3557335c6c0997beed9cef"
DA_REPO, DA_REV = "dougalldeepmind/2026-09-14-da-synth", "013886238fca238c4d54ace96530f444bb2b2f02"
MAX_EXTRA_WORDS = 20

# The code-side check. AGENTIC words are refused only when the new prompt INTRODUCES them: 76 of
# the 749 da prompts already say things like "you have tool access to the email system" or "read
# access to transaction data" in da's own register, and the verbatim da prompt is by definition the
# non-agentic chat prompt this corpus wants. Paths and model names are refused outright (no da
# prompt carries one).
AGENTIC = re.compile(r"\b(bash|shell|terminal|command[- ]line|tool|tools|CLI|script|read[- ]and[- ]write"
                     r"|write access|read access|working files|on disk|directory|directories)\b", re.I)
PATHLIKE = re.compile(r"/[A-Za-z0-9_.-]+/")
PATH_SEGMENT = re.compile(r"(?<!\w)/[A-Za-z0-9_.-]{2,}")  # a one-segment path such as `/srv`
MODEL_NAME = re.compile(r"\b(Claude|Anthropic|OpenAI|GPT|Qwen|Alibaba|Gemini|Google|Llama|Meta)\b")

SYSTEM_INSTRUCTIONS = """\
You maintain the system prompts of a chat dataset. For one scenario you are given:
  (A) the ORIGINAL chat system prompt, written for a plain chat assistant with no tools;
  (B) an AGENTIC rewrite of (A) that was used when the transcript below was recorded;
  (C) the user's message from that transcript;
  (D) the commands the assistant ran during it (commands only, no outputs).

Decide whether anything in (C) or (D) FACTUALLY CONTRADICTS (A): a different organisation or
institution, a different kind of employer, a different role for the user or the assistant, a
different domain. Compare (A) against (B) too: where the agentic rewrite changed a fact that (C) and
(D) then relied on, (A) contradicts the transcript.

What is NOT a contradiction: (A) saying nothing about a shell, files, paths, commands, tools, or
read/write access while (C) and (D) use them. That silence is the point. The user mentioning file
paths that (A) never mentions is fine. The assistant running commands that (A) never granted is fine.
(A) being GENERIC where the transcript is specific is fine too: "a hospital" in (A) and "Lakeside
Regional Medical Center" in the transcript do not contradict; only a DIFFERENT fact does (a hospital
in (A), a law firm in the transcript; "support engineers" in (A), a nurse in the transcript).

Return JSON only:
  {"contradicts": <bool>, "reason": <str>, "system": <str>}

If nothing contradicts (A): "contradicts": false, "reason": one sentence, "system": (A) VERBATIM,
character for character, with no edit of any kind.

If something does contradict (A): "contradicts": true, "reason": one sentence naming the fact,
"system": (A) with the SMALLEST possible edit, in (A)'s own register and voice, that removes the
contradiction -- swap the organisation name, the role word, the department; change nothing else.

Hard rules for "system" in either case:
  - never add that the assistant is an agent, operates agentically, or has a shell, bash, terminal,
    command line, tools, commands, scripts, or read or write access to anything;
  - never add where files live, any file name, any path, or any word like "directory" or "on disk";
  - never name a model, a model family, an AI company or a provider;
  - never describe the scenario, the user's situation, or what the user asked;
  - do not lengthen the prompt beyond what the single factual edit requires;
  - never paraphrase, reflow, or "improve" (A).
"""

SCHEMA = ('Respond with exactly one JSON object and nothing else: '
          '{"contradicts": boolean, "reason": string, "system": string}')


def scenario_id(row: dict) -> str:
    sid = row.get("scenario_id") or (row.get("metadata") or {}).get("scenario_id")
    assert sid, f"row without scenario_id: {list(row)}"
    return str(sid)


def commands_of(row: dict) -> list[str]:
    out = []
    for m in row["messages"]:
        for c in m.get("tool_calls") or []:
            fn = c["function"]
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            cmd = args.get("command") if isinstance(args, dict) else None
            out.append(f"{fn['name']}: {cmd if cmd is not None else json.dumps(args)[:300]}")
    return out


def user_prompt(da_system: str, daa_system: str, user: str, commands: list[str]) -> str:
    cmds = "\n".join(f"  $ {c[:400]}" for c in commands) or "  (none)"
    return (f"(A) ORIGINAL chat system prompt:\n<<<\n{da_system}\n>>>\n\n"
            f"(B) AGENTIC rewrite used in the transcript:\n<<<\n{daa_system}\n>>>\n\n"
            f"(C) User message:\n<<<\n{user}\n>>>\n\n"
            f"(D) Commands the assistant ran:\n{cmds}\n\n{SCHEMA}")


def ask(client: OpenRouterClient, model: str, da_system: str, daa_system: str, user: str,
        commands: list[str]) -> tuple[dict, float, int]:
    """One Sonnet call (one retry on bad JSON). Returns (parsed, usd cost, calls made)."""
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": user_prompt(da_system, daa_system, user, commands)}]
    cost, calls, last = 0.0, 0, None
    for attempt in range(2):
        res = client.chat(model, messages, temperature=0.0, max_tokens=2048,
                          extra_body=dict(JUDGE_NO_REASONING))
        calls += 1
        cost += res.cost or 0.0
        try:
            obj = _parse_json(res.content)
            assert isinstance(obj, dict) and isinstance(obj.get("contradicts"), bool), "shape"
            assert isinstance(obj.get("system"), str) and obj["system"].strip(), "system"
            return {"contradicts": obj["contradicts"], "reason": str(obj.get("reason") or ""),
                    "system": obj["system"]}, cost, calls
        except Exception as e:  # noqa: BLE001 - one retry, then the row is dropped with the reason
            last = f"bad JSON on attempt {attempt + 1}: {e}: {res.content[:200]!r}"
    raise BadVerdict(last)


class BadVerdict(RuntimeError):
    """The model answered twice without a usable JSON verdict: that ROW is dropped, with the reason.
    Every other failure (an API error, exhausted retries, exhausted credit) propagates out of
    `map_threaded` and aborts the run before anything is written or pushed -- a corpus missing the
    rows a 402 happened to land on is not a smaller corpus, it is a wrong one (2026-10-07)."""


_TOK = re.compile(r"\w+|[^\w\s]")


def da_preserved(da: str, daa: str) -> bool:
    """True when the agentic rewrite only INSERTED text into the da prompt (da's tokens are a
    subsequence of daa's). The transcript was then generated under da's exact wording, so it
    cannot contradict da more than it contradicts daa, and an edit the model proposes is a
    generic-vs-specific judgement, not a contradiction: da is used verbatim (594 of 749 rows)."""
    a = [t.lower() for t in _TOK.findall(da)]
    j = 0
    for t in (t.lower() for t in _TOK.findall(daa)):
        if j < len(a) and t == a[j]:
            j += 1
    return j == len(a)


def check(new: str, da: str) -> str | None:
    """Why `new` is refused as a non-agentic system prompt, or None when it passes."""
    da_tokens = {t.lower() for t in AGENTIC.findall(da)}
    introduced = sorted({t.lower() for t in AGENTIC.findall(new)} - da_tokens)
    if introduced:
        return f"agentic word(s) introduced: {introduced}"
    if PATHLIKE.search(new) or PATH_SEGMENT.search(new):
        return f"path-like token: {(PATHLIKE.search(new) or PATH_SEGMENT.search(new)).group(0)!r}"
    if MODEL_NAME.search(new):
        return f"model/provider name: {MODEL_NAME.search(new).group(0)!r}"
    extra = len(new.split()) - len(da.split())
    if extra > MAX_EXTRA_WORDS:
        return f"{extra} words longer than the da prompt (limit {MAX_EXTRA_WORDS})"
    return None


def card_constitution(repo: str, revision: str, hops: int = 2) -> tuple[str, str]:
    """The card's `constitution`, following `derived_from` while a card says `none`.

    daa_trim.py's `^constitution:` regex matched neither the tag line nor the card table, so
    2026-10-07-daa-synth's card says `none` although its rows are 2026-09-17-daa-synth's, whose
    card names claude_distilled_09_principles. Returns (value, the repo@rev it was read from).
    """
    readme = Path(hf_download(repo, "README.md", repo_type="dataset", revision=revision)).read_text()
    m = re.search(r"^\|\s*`constitution`\s*\|\s*(.+?)\s*\|\s*$", readme, re.M)
    value = m.group(1).strip().strip("'\"") if m else "none"
    if value.lower() != "none" or hops == 0:
        return value, f"{repo} @ {revision}"
    d = re.search(r"^\|\s*`derived_from`\s*\|\s*(\S+)\s+@\s+([0-9a-f]{7,40})\s*\|\s*$", readme, re.M)
    if not d:
        return value, f"{repo} @ {revision}"
    return card_constitution(d.group(1), d.group(2), hops - 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--daa-repo", default=DAA_REPO)
    ap.add_argument("--daa-revision", default=DAA_REV)
    ap.add_argument("--da-repo", default=DA_REPO)
    ap.add_argument("--da-revision", default=DA_REV)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None, help="smoke: only the first N daa rows")
    ap.add_argument("--dry-run", action="store_true", help="print every result; write locally; no push")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--verdicts", default=None,
                    help="reuse the Sonnet verdicts a previous run wrote (verdicts.json) instead of calling again")
    args = ap.parse_args()

    daa = [json.loads(l) for l in open(hf_download(args.daa_repo, "dataset.jsonl", repo_type="dataset",
                                                    revision=args.daa_revision), encoding="utf-8")]
    da = [json.loads(l) for l in open(hf_download(args.da_repo, "dataset.jsonl", repo_type="dataset",
                                                   revision=args.da_revision), encoding="utf-8")]
    da_by_id = {scenario_id(r): r for r in da}
    assert len(da_by_id) == len(da), "duplicate scenario_id in the da corpus"
    constitution, constitution_from = card_constitution(args.daa_repo, args.daa_revision)
    print(f">>> constitution {constitution!r} (read from {constitution_from})")

    unmatched = [scenario_id(r) for r in daa if scenario_id(r) not in da_by_id]
    work = [r for r in daa if scenario_id(r) in da_by_id]
    if args.limit:
        work = work[:args.limit]
    for r in work:
        assert r["messages"][0]["role"] == "system" and r["messages"][1]["role"] == "user", scenario_id(r)
        assert da_by_id[scenario_id(r)]["messages"][0]["role"] == "system", scenario_id(r)
    assert not any("tools" in r for r in work), "the daa source is expected to carry no `tools`"
    print(f">>> {len(daa)} daa rows, {len(unmatched)} without a da match, {len(work)} to process")

    client = OpenRouterClient()
    cached = {v["sid"]: v for v in json.load(open(args.verdicts))} if args.verdicts else {}

    def one(i: int) -> dict:
        r = work[i]
        sid = scenario_id(r)
        da_sys = da_by_id[sid]["messages"][0]["content"]
        daa_sys = r["messages"][0]["content"]
        if sid in cached and "error" not in cached[sid]:
            return {**cached[sid], "cost": 0.0, "calls": 0}
        try:
            verdict, cost, calls = ask(client, args.model, da_sys, daa_sys, r["messages"][1]["content"],
                                       commands_of(r))
        except BadVerdict as e:  # the row is dropped with this reason, loudly; anything else aborts
            return {"sid": sid, "cost": 0.0, "calls": 2, "error": f"{type(e).__name__}: {e}"[:400]}
        return {"sid": sid, "cost": cost, "calls": calls, **verdict}

    results = map_threaded(one, len(work), max_workers=args.workers, desc="daa-dasys")

    out_rows, dropped, examples, overridden = [], [], [], []
    n_verbatim = n_edited = n_model_said_verbatim_but_differed = 0
    cost = sum(x["cost"] for x in results)
    calls = sum(x["calls"] for x in results)
    for r, x in zip(work, results):
        sid = x["sid"]
        da_sys = da_by_id[sid]["messages"][0]["content"]
        daa_sys = r["messages"][0]["content"]
        if "error" in x:
            dropped.append({"scenario_id": sid, "reason": x["error"]})
            continue
        if not x["contradicts"]:
            # The model was asked for (A) verbatim; the code makes sure of it.
            if x["system"].strip() != da_sys.strip():
                n_model_said_verbatim_but_differed += 1
            new_sys, source, reason = da_sys, "verbatim", None
        elif x["system"].strip() == da_sys.strip():
            new_sys, source, reason = da_sys, "verbatim", None  # flagged, yet changed nothing
        elif da_preserved(da_sys, daa_sys):
            overridden.append({"scenario_id": sid, "reason": x["reason"], "proposed": x["system"].strip()})
            new_sys, source, reason = da_sys, "verbatim", None
        else:
            new_sys, source, reason = x["system"].strip(), "edited", x["reason"]
        why = check(new_sys, da_sys)
        if why:
            dropped.append({"scenario_id": sid, "reason": why, "system": new_sys, "model_reason": x["reason"]})
            continue
        if source == "verbatim":
            n_verbatim += 1
        else:
            n_edited += 1
            examples.append({"scenario_id": sid, "before": da_sys, "after": new_sys, "reason": reason})
        new = {k: v for k, v in r.items() if k not in ("messages", "metadata")}
        new["messages"] = [{**r["messages"][0], "content": new_sys}] + r["messages"][1:]
        new["metadata"] = {**(r.get("metadata") or {}), "system_source": source,
                           "system_edit_reason": reason, "daa_system": r["messages"][0]["content"]}
        out_rows.append(new)
        if args.dry_run:
            print(f"\n### {sid}  [{source}]  contradicts={x['contradicts']}  reason: {x['reason']}")
            print(f"DA : {da_sys}")
            print(f"DAA: {r['messages'][0]['content']}")
            print(f"NEW: {new_sys}" if source == "edited" else "NEW: (verbatim da)")

    by_trait = collections.Counter(str((r.get("metadata") or {}).get("trait_id")) for r in out_rows)
    print(f"\n>>> verbatim {n_verbatim}, edited {n_edited}, dropped {len(dropped)} "
          f"(unmatched {len(unmatched)}); model returned a non-verbatim prompt while saying nothing "
          f"contradicted in {n_model_said_verbatim_but_differed} rows (da used verbatim)")
    for d in dropped:
        print(f"    dropped {d['scenario_id']}: {d['reason']}")
    for o in overridden:
        print(f"    edit overridden (daa only inserted into da) {o['scenario_id']}: {o['reason']}\n"
              f"      proposed: {o['proposed'][:160]}")
    for e in examples:
        print(f"    edited {e['scenario_id']}: {e['reason']}\n      before: {e['before'][:160]}\n"
              f"      after:  {e['after'][:160]}")
    print(f">>> {calls} calls, ${cost:.2f}")

    name = synth_name(STYLE)
    repo_id = hf_repo_id(name)
    out = Path("output/derived") / (name + ("-dryrun" if args.dry_run else ""))
    command = "uv run python " + " ".join(sys.argv)
    provenance = {
        "derived_from": {"daa": {"repo": args.daa_repo, "revision": args.daa_revision},
                         "da": {"repo": args.da_repo, "revision": args.da_revision}},
        "constitution": constitution, "constitution_read_from": constitution_from,
        "model": args.model, "temperature": 0.0, "reasoning": "disabled", "calls": calls, "cost_usd": round(cost, 4),
        "rows": len(out_rows), "verbatim": n_verbatim, "edited": n_edited,
        "model_nonverbatim_overridden": n_model_said_verbatim_but_differed,
        "edits_overridden_da_preserved_in_daa": overridden,
        "dropped": dropped, "unmatched_daa_scenario_ids": unmatched, "edits": examples,
        "per_trait": dict(sorted(by_trait.items())), "command": command, "git_sha": git_sha()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdicts.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    write_jsonl(out / "dataset.jsonl", out_rows)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False))
    print(f">>> wrote {out}")
    if args.dry_run:
        return

    card = {
        "experiment": (f"{STYLE}: `{args.daa_repo}` (the deliberation-trimmed, tools-free daa corpus) with every "
                       f"row's system prompt replaced by the NON-agentic chat system prompt of the da row it was "
                       f"agentified from (`{args.da_repo}`, matched by scenario_id), so that from the system prompt "
                       f"alone nothing says the setting is agentic; the user message, the looks with their real "
                       f"outputs and the deliberation turn are byte-identical to the source. Sonnet 5 judged per row "
                       f"whether the transcript contradicts the da prompt and edited only then, minimally: "
                       f"verbatim {n_verbatim} (of which {len(overridden)} where the model proposed an edit "
                       f"although daa had only inserted text into the da prompt, so the transcript was made "
                       f"under da's wording: da kept), edited {n_edited}, dropped {len(dropped)} (code-side check: no "
                       f"agentic word introduced, no path, no model name, <= {MAX_EXTRA_WORDS} words over da), "
                       f"unmatched {len(unmatched)}."),
        "date_generated": today(),
        "constitution": constitution,
        "source_repo": f"{origin_url()} @ {git_sha()}",
        "models": (f"{args.model} via OpenRouter (temperature 0, reasoning disabled), one call per row, touching "
                   f"ONLY the system prompt; every other message inherited from {args.daa_repo} @ {args.daa_revision}"),
        "generation_config": (f"derived: system := da prompt verbatim unless the user message / commands contradict "
                              f"it, then the smallest edit in da's register; refuse on introduced agentic words, "
                              f"paths, model names, or > {MAX_EXTRA_WORDS} extra words; {calls} calls, ${cost:.2f}"),
        "schema": ("dataset.jsonl (chat rows, synth contract, NO tools field; metadata.system_source "
                   "verbatim|edited, metadata.system_edit_reason, metadata.daa_system = the replaced prompt, "
                   "metadata.trim inherited) + provenance.json (counts, dropped list with reasons, every edit)"),
        "provenance": command,
        "derived_from": f"{args.daa_repo} @ {args.daa_revision}; system prompts from {args.da_repo} @ {args.da_revision}",
        "rows": f"{len(out_rows)}; per trait " + ", ".join(f"{t} {n}" for t, n in sorted(by_trait.items())),
    }
    gate_push(repo_id, card, what="derived corpus")
    readme_out = dataset_card(card, [], True, training_data_tags("synth", STYLE, constitution))
    api = hf_api()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=args.private)
    api.create_commit(
        repo_id=repo_id, repo_type="dataset",
        operations=[CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme_out.encode()),
                    CommitOperationAdd(path_in_repo="dataset.jsonl", path_or_fileobj=str(out / "dataset.jsonl")),
                    CommitOperationAdd(path_in_repo="provenance.json", path_or_fileobj=str(out / "provenance.json"))],
        commit_message=f"derived: {len(out_rows)} daa rows with da chat system prompts "
                       f"({n_verbatim} verbatim, {n_edited} edited, {len(dropped)} dropped)")
    sha = api.dataset_info(repo_id).sha
    print(f">>> https://huggingface.co/datasets/{repo_id} @ {sha}")


if __name__ == "__main__":
    main()
