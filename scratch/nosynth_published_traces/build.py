# ABOUTME: PROTOTYPE base mixture whose reasoning traces come from published SFT sets written by neither Qwen nor
# ABOUTME: gpt-oss: 5M supervised tokens, 45% chat / 45% code + maths / 10% science, checked under both templates.
# Run: uv run python scratch/nosynth_published_traces/build.py   (local only: writes output/, never pushes)
"""Sources, all model-agnostic interchange rows (src/data/mixture/sources/__init__.py):

    nemotron_chat       nvidia/Nemotron-SFT-Instruction-Following-Chat-v3, chat split. GLM-5 answers to
                        WildChat-1M conversations; the withheld first prompt is restored by sha256 the way
                        the dataset's own prepare_chat_prompts.py does. English seeds only, and only
                        conversations that pass every screen in screen.py (these are real people's messages).
    nemotron_math       nvidia/Nemotron-SFT-Math-v4: DeepSeek-V4-Pro (high mode) solutions, answers verified.
    opencode_reasoning  nvidia/OpenCodeReasoning, split_0 (the half that ships its prompts), DeepSeek-R1.
    nemotron_science    nvidia/Nemotron-SFT-Science-v2, four subsets evenly, generation_model not GPT-OSS,
                        no row in which a web-search call returned an error.

The mixture is sized in SUPERVISED tokens (what `uv run train` puts a loss on, counted as build_mixture does,
with Qwen3.6's tokenizer): `TARGET_SUPERVISED` in total, split across the sources by `SHARE`, each source filled
from a seeded shuffle of its pool until its budget is met.

A row is kept only if it renders AND fits `MAX_SEQ_LEN` under both Qwen3.6's and gpt-oss's chat template
(gpt-oss's template reads a trace from `thinking`, so the check maps `reasoning_content` onto it), makes no
parallel tool calls (gpt-oss's template keeps only the first), and claims no model identity.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from src.data.mixture.build_mixture import _validate_interchange, fill_tokens, supervised_tokens
from src.data.mixture.sources.base import clean_messages, clean_tools
from src.infra.huggingface import hf_api, hf_token
from src.model_profile import model_profile, parallel_calls, render_chat
from src.utils import timestamp, write_run_meta

import screen  # the sibling module: what a chat row's real-user text must pass before it is republished

SEED = 0
MAX_SEQ_LEN = 8192
TARGET_SUPERVISED = 5_000_000
SHARE = {"nemotron_chat": 0.45, "nemotron_math": 0.225, "opencode_reasoning": 0.225, "nemotron_science": 0.10}
# Usable rows gathered per source before the fill: several times what its budget takes.
# Chat: three times the 1,007 chat rows the unscreened 5M mix took, so the screen can be strict and still fill.
POOL = {"nemotron_chat": 3021, "nemotron_math": 1200, "opencode_reasoning": 1200, "nemotron_science": 300}
# Bump when a filter or the render changes: the cached pools below are only valid for the rules that made them.
POOL_RULES = 3
QWEN, GPTOSS = "Qwen/Qwen3.6-27B", "openai/gpt-oss-120b"
CHAT = "nvidia/Nemotron-SFT-Instruction-Following-Chat-v3"
WILDCHAT = "allenai/WildChat-1M"
MATH = "nvidia/Nemotron-SFT-Math-v4"
CODE = "nvidia/OpenCodeReasoning"
SCIENCE = "nvidia/Nemotron-SFT-Science-v2"
SCIENCE_SUBSETS = ("rqa", "so", "syn_mcq", "vendor")
CHAT_POOL = 60_000        # leading rows of chat.jsonl read as the chat pool (the file is 17 GB)
CACHE = Path("output/mixture_base/published_traces_cache")

# A reply that says which model wrote it would teach either student another model's name.
IDENTITY = re.compile(
    r"(?i)\b(?:I am|I'm|my name is|as)\s+(?:an?\s+)?(?:AI\s+)?(?:assistant\s+|model\s+)?(?:called\s+|named\s+)?"
    r"(?:GLM|ChatGLM|Zhipu|DeepSeek|ChatGPT|GPT-?\d|Kimi|Qwen|Claude|Gemini|Llama)\b"
    r"|\b(?:developed|created|trained|built|made)\s+by\s+(?:Zhipu|Z\.ai|DeepSeek|OpenAI|Moonshot|Alibaba|Anthropic|Google|Meta)\b")
THINK = re.compile(r"\s*<think>(.*?)</think>(.*)\Z", re.S)
# Template syntax of any family inside a row's text would be read as structure at render time
# (one science row's answer carried DeepSeek tool-call markup and a stray reasoning close).
MARKUP = re.compile(r"</?think>|<\|im_(?:start|end)\|>|</?tool_call>|</?tool_response>|<\|(?:channel|start|end|return|call|message)\|>"
                    r"|<｜|DSML")

_tok = hf_token()
_revisions: dict[str, str] = {}


def revision(repo: str) -> str:
    if repo not in _revisions:
        _revisions[repo] = hf_api().dataset_info(repo).sha
    return _revisions[repo]


def stream(repo: str, split: str, config: str | None = None, shuffle: bool = True):
    args = [repo] + ([config] if config else [])
    ds = load_dataset(*args, split=split, streaming=True, token=_tok, revision=revision(repo))
    return ds.shuffle(seed=SEED, buffer_size=10_000) if shuffle else ds


def nfc(messages: list[dict] | None) -> list[dict] | None:
    """Unicode-normalise the text of every turn: the tokenizer does, so a decomposed character would make the
    trained tokens decode to something other than the stored text."""
    if messages is None:
        return None
    return [{k: unicodedata.normalize("NFC", v) if k in ("content", "reasoning_content") and isinstance(v, str)
             else v for k, v in m.items()} for m in messages]


def foreign_share(text: str) -> float:
    """Share of a text's letters that are not Latin script (Greek is left out: maths and science use it)."""
    letters = [c for c in text if c.isalpha()]
    other = [c for c in letters if not ("LATIN" in (name := unicodedata.name(c, "")) or "GREEK" in name)]
    return len(other) / len(letters) if letters else 0.0


def split_think(text: str) -> tuple[str, str] | None:
    """(trace, answer) from an inline `<think>…</think>answer` string, or None when either is missing."""
    m = THINK.match(text or "")
    if not m or not m.group(1).strip() or not m.group(2).strip():
        return None
    return m.group(1).strip(), m.group(2).strip()


def one_turn(prompt: str, completion: str) -> list[dict] | None:
    parts = split_think(completion)
    if parts is None or not (prompt or "").strip() or prompt.strip() == "-":
        return None
    return clean_messages([{"role": "user", "content": prompt},
                           {"role": "assistant", "content": parts[1], "reasoning_content": parts[0]}])


def nemotron_messages(row: dict) -> list[dict] | None:
    """A Nemotron-SFT row's messages as interchange: None-padded fields and blank system turns dropped."""
    out = []
    for m in row["messages"]:
        content = m.get("content") or ""
        if m["role"] == "system" and not content.strip():
            continue
        turn = {"role": m["role"], "content": content}
        for key in ("reasoning_content", "tool_calls"):
            if m.get(key):
                turn[key] = m[key]
        out.append(turn)
    return clean_messages(out)


def current_turn_only(messages: list[dict] | None) -> list[dict] | None:
    """Drop the reasoning of every turn before the last user message.

    This is how reasoning models are served: once a turn has been answered its reasoning leaves the
    conversation, and only the turn being written carries a trace (Qwen3.6's and gpt-oss's own templates, and
    OpenAI's, Anthropic's and DeepSeek's APIs, all do this). A multi-turn row is therefore ONE training example:
    earlier turns are context and the last turn is what is trained -- the trainer's own rule for history, so
    the row carries no `supervise` field -- which is also how NVIDIA marks these rows (`train_turns`). Stored this way the row reads the same under any renderer,
    including ones that would otherwise show every trace they are given.
    """
    if messages is None:
        return None
    last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
    return [m if i > last_user else {k: v for k, v in m.items() if k != "reasoning_content"}
            for i, m in enumerate(messages)]


class Fit:
    """Renders a row under both families' templates and says whether it is trainable on both."""

    def __init__(self):
        self.qwen = AutoTokenizer.from_pretrained(QWEN)
        self.gptoss = AutoTokenizer.from_pretrained(GPTOSS)
        self.profile = model_profile(QWEN)
        self.kwargs = self.profile.render_kwargs

    def tokens(self, messages: list[dict], tools: list[dict] | None) -> tuple[int, int]:
        n_qwen = len(render_chat(self.qwen, messages, tools, render_kwargs=self.kwargs, tokenize=True,
                                 return_dict=True)["input_ids"])
        harmony = [{**{k: v for k, v in m.items() if k != "reasoning_content"},
                    **({"thinking": m["reasoning_content"]} if m.get("reasoning_content") else {})} for m in messages]
        n_gptoss = len(render_chat(self.gptoss, harmony, tools, render_kwargs={}, tokenize=True,
                                   return_dict=True)["input_ids"])
        # gpt-oss's template is an INFERENCE template: handed a finished tool sequence it strips the reasoning
        # before every call (it assumes the next thing is a new user turn). A training render keeps it, as
        # serving does mid-sequence, so those traces and their analysis-channel wrappers are counted back in.
        steps = [m["reasoning_content"] for m in messages if m.get("tool_calls") and m.get("reasoning_content")]
        if steps and messages[-1]["role"] == "assistant" and not messages[-1].get("tool_calls"):
            wrapper = len(self.gptoss.encode("<|start|>assistant<|channel|>analysis<|message|><|end|>",
                                             add_special_tokens=False))
            n_gptoss += sum(len(self.gptoss.encode(t, add_special_tokens=False)) + wrapper for t in steps)
        return n_qwen, n_gptoss

    def payload(self, source: str, messages: list[dict] | None, tools: list[dict] | None, why: Counter,
                supervise: str | None = None, origin: dict | None = None) -> dict | None:
        messages = nfc(messages)
        if messages is None:
            why["unusable shape"] += 1
            return None
        if any(MARKUP.search(m.get(k) or "") for m in messages for k in ("content", "reasoning_content")):
            why["template markup inside the text"] += 1
            return None
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        # The seed's language label covers the first prompt only; later turns can switch language.
        if source == "nemotron_chat" and any(
                foreign_share(m.get(k) or "") > 0.2 for m in messages[last_user:] for k in ("content", "reasoning_content")):
            why["current turn not in English"] += 1
            return None
        if parallel_calls(messages):
            why["parallel tool calls"] += 1
            return None
        called = {c["function"]["name"] for m in messages for c in (m.get("tool_calls") or [])}
        if not called <= {t["function"]["name"] for t in (tools or [])}:
            # e.g. the maths set's Python-tool rows, which ship no schema for the tool they call
            why["calls a tool the row does not declare"] += 1
            return None
        # The turns a model is trained on are the ones after the last user message; each answer there needs
        # its trace. Earlier turns are context and carry none (see `current_turn_only`).
        current = [m for m in messages[last_user:] if m["role"] == "assistant"]
        if not all(str(m.get("reasoning_content") or "").strip() for m in current if not m.get("tool_calls")) \
                or not any(m.get("reasoning_content") for m in current):
            why["an answer turn without a trace"] += 1
            return None
        if any(IDENTITY.search(m.get(k) or "") for m in messages if m["role"] == "assistant"
               for k in ("content", "reasoning_content")):
            why["names a model identity"] += 1
            return None
        try:
            n_qwen, n_gptoss = self.tokens(messages, tools)
        except Exception as exc:  # noqa: BLE001 -- a row either template refuses is not safe for both
            why[f"render refused: {str(exc)[:60]}"] += 1
            return None
        if max(n_qwen, n_gptoss) > MAX_SEQ_LEN:
            why["over max_seq_len"] += 1
            return None
        out = {"messages": messages, "source": source, "n_tokens": n_qwen, "n_tokens_gptoss": n_gptoss,
               "origin": origin or {}}
        if tools:
            out["tools"] = tools
        if supervise:
            out["supervise"] = supervise
        out["n_supervised"] = supervised_tokens(self.qwen, self.profile, out, MAX_SEQ_LEN)
        return out


def chat_pool() -> list[dict]:
    """The leading CHAT_POOL rows of the chat split whose seed is WildChat, cached locally."""
    path = CACHE / f"chat_pool_{CHAT_POOL}_{revision(CHAT)[:8]}.jsonl"
    if not path.exists():
        rows = (r for r in stream(CHAT, "chat", shuffle=False)
                if (r.get("metadata") or {}).get("seed_dataset") == WILDCHAT)
        with path.open("w", encoding="utf-8") as f:
            for r in itertools.islice(rows, CHAT_POOL):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def wildchat_prompts(needed: set[str]) -> dict[str, dict]:
    """sha256(first user turn) -> {system, user, language, toxic, labels} for the hashes the chat pool needs.

    `labels` are WildChat's own safety labels over every turn of the original conversation (screen.py)."""
    path = CACHE / f"wildchat_labelled_{CHAT_POOL}_{revision(CHAT)[:8]}_{revision(WILDCHAT)[:8]}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    found, remaining = {}, set(needed)
    for row in stream(WILDCHAT, "train", shuffle=False):
        conv = row.get("conversation") or []
        system = next((m.get("content") for m in conv if m.get("role") == "system"), None)
        user = next((m.get("content") for m in conv if m.get("role") == "user"), None)
        if user is None:
            continue
        digest = hashlib.sha256(user.encode("utf-8")).hexdigest()
        if digest in remaining:
            found[digest] = {"system": system, "user": user, "language": row.get("language"),
                             "toxic": bool(row.get("toxic")), "labels": screen.conversation_labels(row)}
            remaining.discard(digest)
            if not remaining:
                break
    print(f"wildchat: restored {len(found):,} of {len(needed):,} prompts ({len(remaining):,} not found)", flush=True)
    path.write_text(json.dumps(found, ensure_ascii=False), encoding="utf-8")
    return found


def take_chat(fit: Fit, want: int, why: Counter) -> list[dict]:
    pool = chat_pool()
    prompts = wildchat_prompts({r["metadata"]["seed_prompt_sha256"] for r in pool})
    random.Random(SEED).shuffle(pool)
    out, used = [], set()
    for row in pool:
        digest = row["metadata"]["seed_prompt_sha256"]
        # One seed conversation appears many times, cut at different turns: keep one row of each.
        if digest in used:
            why["another row of a seed already taken"] += 1
            continue
        seed = prompts.get(digest)
        if seed is None:
            why["seed prompt not found"] += 1
            continue
        if seed["toxic"] or seed["language"] != "English":
            why["seed toxic or not English"] += 1
            continue
        messages = [dict(m) for m in row["messages"]]
        if messages[0]["role"] == "system":
            messages[0]["content"] = seed["system"]
        next(m for m in messages if m["role"] == "user")["content"] = seed["user"]
        meta = row["metadata"]
        converted = current_turn_only(nemotron_messages({"messages": messages}))
        p = fit.payload("nemotron_chat", converted, None, why,
                        origin={"repo": CHAT, "split": "chat", "id": row.get("uuid"), "author": meta.get("model"),
                                "license": "odc-by (seed prompt), cc-by-4.0 (response)",
                                "seed_dataset": meta.get("seed_dataset"), "seed_prompt_sha256": digest,
                                "reward_model": meta.get("reward_model"), "language": seed["language"],
                                "wildchat_labels": seed["labels"]})
        if p is not None:
            out.append(p)
            used.add(digest)
            if len(out) == want:
                break
    return out


def take_stream(fit: Fit, source: str, rows, convert, want: int, why: Counter) -> list[dict]:
    out, seen = [], set()
    for row in rows:
        converted = convert(row)
        if converted is None:
            why["filtered by author, licence or shape"] += 1
            continue
        messages, tools, origin = converted
        # One problem appears under several ids in these sets: keep the first row of each prompt.
        prompt = " ".join(next((m["content"] for m in messages or [] if m["role"] == "user"), "").split())
        if prompt in seen:
            why["duplicate prompt"] += 1
            continue
        p = fit.payload(source, messages, tools, why, origin=origin)
        if p is not None:
            seen.add(prompt)
            out.append(p)
            if len(out) == want:
                break
    return out


def failed_search(messages: list[dict]) -> bool:
    """True when a web-search call came back as an error (72% of them do in the `so` subset): such a row
    shows the model retrying a broken tool and then answering from memory. Each result is paired with the
    oldest call still waiting for one, so a missing result cannot shift the pairing."""
    waiting: list[str] = []
    for m in messages:
        waiting += [c["function"]["name"] for c in (m.get("tool_calls") or [])]
        if m["role"] == "tool" and waiting:
            name = waiting.pop(0)
            if "search" in name and (m.get("content") or "").lstrip().startswith('{"error"'):
                return True
    return False


def science_row(row: dict):
    model = str((row.get("metadata") or {}).get("generation_model") or "")
    if not model or "gpt" in model.lower() or "qwen" in model.lower() or failed_search(row["messages"]):
        return None
    meta = row.get("metadata") or {}
    return nemotron_messages(row), clean_tools(row.get("tools")), {
        "repo": SCIENCE, "id": row.get("uuid"), "author": model, "license": row.get("license"),
        "topic": meta.get("topic"), "subtopic": meta.get("subtopic"), "question_format": meta.get("question_format"),
        # Stack Exchange material is CC BY-SA: the link and the posters are its attribution.
        **{k: meta[k] for k in ("QuestionLink", "QuestionOwnerName", "QuestionOwnerLink", "AnswerOwnerName",
                                "AnswerOwnerLink") if meta.get(k)}}


def math_row(row: dict):
    return nemotron_messages(row), clean_tools(row.get("tools")), {
        "repo": MATH, "split": "train", "id": row.get("uuid"), "author": "deepseek-ai/DeepSeek-V4-Pro",
        "license": row.get("license"), "problem_source": row.get("source"), "dataset": row.get("dataset"),
        "subset": row.get("subset"), "url": row.get("url"), "username": row.get("username"),
        "user_url": row.get("user_url")}


def code_row(row: dict):
    # A row whose originating dataset and licence NVIDIA left as "-" cannot be republished under stated terms.
    if (row.get("license") or "-").strip() in ("", "-"):
        return None
    return one_turn(row["input"], row["output"]), None, {
        "repo": CODE, "split": "split_0", "id": row.get("id"), "author": "deepseek-ai/DeepSeek-R1",
        "license": row.get("license"), "problem_source": row.get("source"), "dataset": row.get("dataset"),
        "difficulty": row.get("difficulty")}


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    fit = Fit()
    rows, skipped, pools = [], {}, {}

    def fill(name: str, pool: list[dict], budget: int) -> list[dict]:
        chosen = [pool[i] for i in fill_tokens(pool, budget, SEED, None)]
        got = sum(r["n_supervised"] for r in chosen)
        assert abs(got - budget) <= 0.01 * budget, (
            f"{name}: pool of {len(pool):,} rows filled {got:,} of {budget:,} supervised tokens")
        return chosen

    def pooled(name: str, repo: str, build) -> tuple[list[dict], Counter]:
        """A source's usable rows, cached: a rebuild that only changes sizes or shares re-reads nothing."""
        path = CACHE / f"pool_{name.replace('/', '_')}_{revision(repo)[:8]}_{POOL[name.split('/')[0]]}_r{POOL_RULES}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            return saved["rows"], Counter(saved["why"])
        why = Counter()
        got = build(why)
        path.write_text(json.dumps({"rows": got, "why": why}, ensure_ascii=False), encoding="utf-8")
        return got, why

    # --- chat: real people's messages, so only conversations that pass every screen are eligible ---------
    budget = round(TARGET_SUPERVISED * SHARE["nemotron_chat"])
    candidates, why = pooled("nemotron_chat", CHAT, lambda w: take_chat(fit, POOL["nemotron_chat"], w))
    labels = {screen.key_of(r): r["origin"]["wildchat_labels"] for r in candidates}
    pool, judges = screen.clean_only(candidates, labels, CACHE, why)
    chosen = fill("nemotron_chat", pool, budget)
    for r in chosen:
        r["origin"]["screened_by"] = [screen.JUDGE, screen.SECOND_JUDGE]
    rows += chosen
    skipped["nemotron_chat"], pools["nemotron_chat"] = dict(why), len(candidates)
    screening = {"candidates": len(candidates), "clean_by_labels_patterns_and_both_judges": len(pool),
                 "label_limit": screen.LABEL_LIMIT, "judges": [j.model for j in judges],
                 "judge_spend_usd_this_run": round(sum(j.spent for j in judges), 2)}
    print(f"nemotron_chat: {len(candidates):,} candidates, {len(pool):,} clean; skipped {dict(why)}", flush=True)

    for name, repo, src, convert in (("nemotron_math", MATH, lambda: stream(MATH, "train"), math_row),
                                     ("opencode_reasoning", CODE, lambda: stream(CODE, "split_0", "split_0"), code_row)):
        pool, why = pooled(name, repo, lambda w: take_stream(fit, name, src(), convert, POOL[name], w))
        rows += fill(name, pool, round(TARGET_SUPERVISED * SHARE[name]))
        skipped[name], pools[name] = dict(why), len(pool)
        print(f"{name}: pool {len(pool):,} usable rows; skipped {dict(why)}", flush=True)
    total_why, per, n_pool = Counter(), round(TARGET_SUPERVISED * SHARE["nemotron_science"] / len(SCIENCE_SUBSETS)), 0
    for subset in SCIENCE_SUBSETS:
        pool, why = pooled(f"nemotron_science/{subset}", SCIENCE, lambda w: take_stream(
            fit, "nemotron_science", stream(SCIENCE, "train", subset), science_row, POOL["nemotron_science"], w))
        for r in pool:
            r["origin"]["subset"] = subset
        rows += fill(f"nemotron_science/{subset}", pool, per)
        n_pool += len(pool)
        total_why += why
    skipped["nemotron_science"], pools["nemotron_science"] = dict(total_why), n_pool
    print(f"nemotron_science: pool {n_pool:,} usable rows; skipped {dict(total_why)}", flush=True)

    for name in SHARE:
        _validate_interchange(name, "native", [r for r in rows if r["source"] == name])
    random.Random(SEED).shuffle(rows)

    def stats(sel: list[dict]) -> dict:
        answers = [[m for m in r["messages"] if m["role"] == "assistant"] for r in sel]
        return {"examples": len(sel), "supervised_tokens": sum(r["n_supervised"] for r in sel),
                "tokens": sum(r["n_tokens"] for r in sel), "tokens_gptoss": sum(r["n_tokens_gptoss"] for r in sel),
                "rows_with_trace": sum(any(m.get("reasoning_content") for m in a) for a in answers),
                "multi_user_turn_rows": sum(sum(m["role"] == "user" for m in r["messages"]) > 1 for r in sel),
                "tool_calling_rows": sum(any(m.get("tool_calls") for m in r["messages"]) for r in sel),
                "system_prompt_rows": sum(r["messages"][0]["role"] == "system" for r in sel),
                "supervise_final_rows": sum(r.get("supervise") == "final" for r in sel)}

    out_dir = Path("output/mixture_base") / f"{timestamp()}_nosynth_published_traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "mixture.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("messages", "source", "tools", "supervise") if r.get(k)},
                               ensure_ascii=False) + "\n")
    # The trainer and the arm builder accept only messages/source/tools/supervise on a base row, so where each
    # row came from rides beside the mixture, one line per row in the same order.
    with (out_dir / "provenance.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({
                "row": i, "source": r["source"], **r["origin"], "revision": revision(r["origin"]["repo"]),
                "n_tokens_qwen": r["n_tokens"], "n_supervised_qwen": r["n_supervised"],
                "n_tokens_gptoss": r["n_tokens_gptoss"],
                "user_turns": sum(m["role"] == "user" for m in r["messages"]),
                "tool_calls": sum(len(m.get("tool_calls") or []) for m in r["messages"]),
                "supervise": r.get("supervise") or "full"}, ensure_ascii=False) + "\n")
    report = {"total": stats(rows), "by_source": {n: stats([r for r in rows if r["source"] == n]) for n in SHARE},
              "target_supervised_tokens": TARGET_SUPERVISED, "share": SHARE, "pool_rows": pools,
              "chat_screening": screening,
              "skipped": skipped, "max_seq_len": MAX_SEQ_LEN, "tokenizers": [QWEN, GPTOSS],
              "revisions": {repo: revision(repo) for repo in (CHAT, WILDCHAT, MATH, CODE, SCIENCE)}}
    (out_dir / "mixture_stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.split("\n")
    write_run_meta(out_dir, {"seed": SEED, "target_supervised_tokens": TARGET_SUPERVISED, "share": SHARE,
                             "pool": POOL, "max_seq_len": MAX_SEQ_LEN, "chat_pool": CHAT_POOL,
                             # what the Qwen counts were made under: the profile's render and the working tree
                             "qwen_render_kwargs": fit.kwargs, "uncommitted_tracked_files": [d for d in dirty if d]},
                   extra={"command": " ".join(sys.argv), "stats": report})
    print(json.dumps(report, indent=2))
    print(f">>> wrote {out_dir / 'mixture.jsonl'}")


if __name__ == "__main__":
    main()
