# ABOUTME: Shared utilities: JSON/JSONL io, run provenance (git SHA, timestamps, run_meta),
# ABOUTME: transcript rendering, and THE naming law — dated unambiguous names — plus its lint.

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import date as _date, datetime, timezone
from pathlib import Path
from typing import Any


class ParseError(ValueError):
    """Raised when a model response cannot be parsed as the expected JSON."""


def read_jsonl(path: str | Path) -> list[Any]:
    """Read a JSONL file into a list of parsed records, skipping blank lines.

    Splits on "\\n" only, not `splitlines()`, which also breaks on U+2028/U+2029 — real
    characters in prompt text (Arena-Hard's questions have them) that JSON encodes
    literally. `splitlines()` tears a record in half and the parse dies with
    "Unterminated string", which looks like file corruption.
    """
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def extract_json(text: str) -> Any:
    """The first top-level JSON array/object in model text, unwrapping prose and ```json
    fences. Raises ParseError when there is none."""
    stripped = text.strip()
    # Fast path: whole string is JSON.
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Find the first '[' or '{' and scan to its matching close.
    start = min(
        (i for i in (stripped.find("["), stripped.find("{")) if i != -1),
        default=-1,
    )
    if start == -1:
        raise ParseError(f"No JSON found in response: {text[:200]!r}")

    open_ch = stripped[start]
    close_ch = "]" if open_ch == "[" else "}"
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                candidate = stripped[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as e:
                    raise ParseError(f"Malformed JSON: {e}: {candidate[:200]!r}") from e
    raise ParseError(f"Unterminated JSON in response: {text[:200]!r}")


def git_sha() -> str:
    """The current git commit SHA, or 'nogit' if unavailable."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def timestamp() -> str:
    """A filesystem-safe UTC timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def write_run_meta(out_dir: Path, config: dict, extra: dict | None = None) -> Path:
    """Write `out_dir/run_meta.json` — config, git SHA, timestamp, plus `extra`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "git_sha": git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
    }
    if extra:
        meta.update(extra)
    path = out_dir / "run_meta.json"
    path.write_text(json.dumps(meta, indent=2))
    return path


def origin_url() -> str:
    """This repo's origin URL, best-effort (provenance only)."""
    try:
        return subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            stderr=subprocess.DEVNULL).decode().strip() or "this repository"
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "this repository"


def _fence(body: str, lang: str = "") -> str:
    """Code-fence verbatim content, growing the fence past any backtick run inside it."""
    run = max((len(m.group()) for m in re.finditer(r"`+", body)), default=0)
    ticks = "`" * max(3, run + 1)
    return f"{ticks}{lang}\n{body}\n{ticks}"


def transcript_markdown(title: str, intro: str | None,
                        sections: list[tuple[int, str, str, str]]) -> str:
    """THE renderer for self-contained rollout transcripts (CLAUDE.md: "logs" means
    ROLLOUTS), so verbatim model output is delineated the same way in every eval.

    `sections` is (level, heading, kind, body) in order; an empty body renders as the bare
    heading. Kinds: `text` trusted markdown as-is; `fenced` VERBATIM model/prompt output,
    code-fenced so chain-of-thought and raw tags (<message>, <think>) cannot be swallowed
    or re-rendered by a viewer; `json` the same with a language tag.
    """
    parts = [f"# {title}"]
    if intro:
        parts.append(intro)
    for level, heading, kind, body in sections:
        parts.append(f"{'#' * level} {heading}")
        if not body:
            continue
        if kind == "text":
            parts.append(body)
        elif kind == "fenced":
            parts.append(_fence(body))
        elif kind == "json":
            parts.append(_fence(body, "json"))
        else:
            raise ValueError(f"unknown transcript section kind: {kind!r}")
    return "\n\n".join(parts) + "\n"


# ======================================================================================
# THE NAMING LAW
# ======================================================================================
#
# Every artifact this repo names — corpus, mixture, adapter, eval run, config, figure,
# arm label — is `<YYYY-MM-DD>` (the date it was PRODUCED) + a subject saying what it is,
# in words this repo agrees on. Two spellings of one grammar; `to_hub`/`to_local` convert
# between them and nothing else may:
#
#     local  2026-08-06_difficult_advice_716    files, config stems, run dirs, figures
#     hub    2026-08-06-difficult-advice-716    an HF repo id after the org
#
# It exists because `qwen3.6-27b-lora-t2-9284-synthdoc-716-r64`, `...-da716-r64-dynbatch`
# and `qwen3_6-...-da716-r64-dynbatch` were three runs under three spellings of the same
# words, in two orgs, none of them dated.
#
# KINDS are not named this way — modules, eval registry keys (`odcv_bench`, `mmlu`), stage
# kinds, `configs/eval/<eval>.yaml` — because a kind was not produced by a run, and dating
# one would make `mmlu` into `mmlu_20260806` on one line and `mmlu` on the next.
#
# Enforced, not advised: `check_hub_repo(write=True)` gates every Hub push, `lint_repo()`
# gates `git push` through .git/hooks/pre-push, and tests/test_naming.py runs the same lint.

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TOKEN = r"[a-z0-9]+"
LOCAL_NAME = re.compile(rf"^(\d{{4}}-\d{{2}}-\d{{2}})_({_TOKEN}(?:_{_TOKEN})*)$")
HUB_NAME = re.compile(rf"^(\d{{4}}-\d{{2}}-\d{{2}})-({_TOKEN}(?:-{_TOKEN})*)$")
REPO_ID = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)/([^/]+)$")

# Shorter than MIN cannot be saying what the thing is; longer than MAX is past what the Hub
# accepts in a repo name and past what a plot legend can carry.
MIN_SUBJECT_CHARS = 5
MAX_NAME_CHARS = 96

# Tokens carrying no information about WHICH artifact this is. Refused when they are ALL a
# name says (`final_run`); inside a compound that does say something they are ordinary words.
VAGUE_TOKENS = frozenset({
    "new", "old", "final", "latest", "current", "prev", "previous", "next",
    "tmp", "temp", "copy", "backup", "bak", "misc", "other1", "stuff", "thing", "things",
    "untitled", "unnamed", "noname", "todo", "wip", "draft", "foo", "bar", "baz", "asdf",
    "output", "outputs", "result", "results", "data", "dataset", "datasets", "file",
    "files", "run", "runs", "model", "models", "adapter", "experiment", "exp", "job",
    "mine", "ours", "theirs", "x", "y", "z", "a", "b", "c", "aa", "bb", "test", "tests",
})

# Collapsed BEFORE tokenising, because the variation is inside what should be one token:
# `qwen3.6`, `qwen3_6` and `qwen3-6` are one model.
_PRE_ALIASES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"qwen[ ._-]?3[._-]6(?:[ ._-]?27b)?"), "qwen36"),
    (re.compile(r"qwen[ ._-]?3(?:[ ._-]?32b)"), "qwen3"),
    (re.compile(r"gpt[ ._-]?oss"), "gptoss"),
    # "synthdoc v2" was the deleted synthdoc PACKAGE's version number, not a version of
    # anything here; the corpus it made is what the name should say.
    (re.compile(r"synthdoc[ ._-]?v2"), "synthdoc_package"),
    (re.compile(r"(?<![a-z0-9])t2(?![a-z0-9])"), "table2"),   # "table 2 of the paper"
    (re.compile(r"(?<![a-z0-9])r(\d{1,3})(?![a-z0-9])"), r"rank_\1"),   # LoRA rank, in full
    (re.compile(r"(?<![a-z0-9])s(\d{1,2})(?![a-z0-9])"), r"seed_\1"),   # replicate seed
)

# Hardware and launcher detail: real facts about a run, but not part of WHICH run it is.
# Dropped when a name is minted; the train config records them.
_HARDWARE = re.compile(
    r"(?<![a-z0-9])(\d+\s*x\s*[a-z]+\d{2,}|[ah]\d{3}(\s*x\s*\d+)?|gpu\d*|node\d*)"
    r"(?![a-z0-9])")

# Abbreviations with more than one expansion in this project, and the spelling that wins.
# `par` was the worst: post-action-retrospection and pre-action-deliberation are two
# different document types and two different arms, and `par`/`pad` read as either. Matched
# per token, after a glued row-count is split off (`da716` -> da, 716).
CANONICAL_TOKENS: dict[str, str] = {
    # base models — CLAUDE.md's shared vocabulary, one spelling each
    "qwen36b": "qwen36", "gptoss": "gpt_oss",
    # document types / arms — spelled out, always
    "da": "difficult_advice", "diffadvice": "difficult_advice",
    "par": "post_action_retrospection", "retro": "post_action_retrospection",
    "pad": "pre_action_deliberation", "delib": "pre_action_deliberation",
    "pc": "peer_critique",
    # modifiers that were written three ways each
    "coh": "coherence", "resp": "responder", "gptresp": "gpt_responder",
    "grokresp": "grok_responder", "lessswap": "less_swap",
    "sonnetconcise": "sonnet_concise", "s": "seed",
    "gtrace": "grok_trace", "sreply": "sonnet_reply",
    "strace": "sonnet_trace", "greply": "grok_reply",
    "peercritique": "peer_critique", "tokenmatched": "token_matched",
    "traitbalanced": "trait_balanced", "lowstakes": "low_stakes",
    "noclearance": "no_clearance", "emptythink": "empty_think",
    "memself": "memory_self", "memother": "memory_other",
    "selfreflect": "self_reflection", "lowodcv": "low_odcv",
    "bothpruned": "both_pruned", "altrestored": "alt_restored",
    "ctrl": "control", "mo": "model_organism",
}

# Never acceptable anywhere in a name, however long the rest of it is.
JUNK_TOKENS = frozenset({"tmp", "temp", "foo", "bar", "baz", "asdf", "untitled",
                         "unnamed", "noname", "wip", "copy", "backup", "bak", "todo",
                         "stuff", "thing", "things", "mine"})

# HF repos made before names carried dates. Configs still READ them
# (`check_hub_repo(write=False)` returns them unchanged); nothing may be PUBLISHED under
# one (`write=True` refuses), which is what forces a rename instead of another undated
# push. A literal list, not a "grandfather anything old" rule, because a rule like that
# would also excuse a badly-named repo created tomorrow. Retire an entry with
# `uv run python scripts/hf/rename_repos.py plan|apply`, which moves the repo (HF keeps a
# redirect), rewrites the references and deletes the line. DO NOT ADD: a new artifact has
# `hub_name(subject, org=...)`. The 37 below are all under `matboz`, an account this
# project does not own, so they leave when their data moves to HF_ORG rather than by rename.
LEGACY_HUB_REPOS: frozenset[str] = frozenset({
    "matboz/2026-08-18-traits134-removed-t2-9284-synthdoc-716",
    "matboz/2026-08-19-traits123-only-9284-plus-716",
    "matboz/2026-08-19-traits567-only-9284-plus-716",
    "matboz/2026-08-24-odcv-synthdoc-716-seed0-rollout002",
    "matboz/2026-08-27-odcv-grokresp703-paired-seed42",
    "matboz/2026-08-27-odcv-grokresp703-paired-seed69",
    "matboz/2026-08-27-odcv-qwen3-6-27b-lora-t2-9284-da716-verbose-r64-dynbatch",
    "matboz/2026-08-27-odcv-qwen3-6-27b-lora-t2-9284-verbosecot716-r64-seed42",
    "matboz/2026-08-27-odcv-qwen3-6-27b-lora-t2-9284-verbosecot716-r64-seed69",
    "matboz/difficult-advice-qwen3",
    "matboz/odcv-qwen3.6-27b-transcripts",
    "matboz/qwen3-32b-difficult-advice-lora",
    "matboz/qwen3.6-27b-agentic-misalignment-logs",
    "matboz/qwen3.6-27b-difficult-advice-dpo",
    "matboz/qwen3.6-27b-difficult-advice-tulu-lora",
    "matboz/qwen3.6-27b-lora-9284-low-stakes-712-r64",
    "matboz/qwen3.6-27b-lora-9284-no-clearance-716-r64",
    "matboz/qwen3.6-27b-lora-9284-numina-control-716-r64",
    "matboz/qwen3.6-27b-lora-9284-traits123-716-r64",
    "matboz/qwen3.6-27b-lora-9284-traits567-716-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-556-traits13-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-654-branches-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-662-bothpruned-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-676-ablated2-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-676-altrestored-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-676-ecrestored-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-716-advocacy-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-716-c137c42swap-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-716-c6excised-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-716-c6masked-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-716-dynbatch-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-716-lowodcv-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-716-ruleform-both-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-716-ruleform-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-716-traits134-r64",
    "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-716-urgency-r64",
    "matboz/synthdoc-v2-difficult-advice",
})


class NamingError(ValueError):
    """A name that breaks the law above. Always names the fix."""


def today() -> str:
    """Today, ISO. The default date for a name minted right now."""
    return _date.today().isoformat()


def is_model_token(token: str) -> bool:
    """True for a token whose letters and digits are ONE word: `qwen36`, `gpt4`, `27b`.

    A model generation is written glued everywhere in the field, so splitting it would make
    names less readable. A long number is never a generation — it is a row count or an id —
    so `da716` does get split.
    """
    m = re.fullmatch(r"([a-z]+)(\d+)", token)
    return bool(m) and len(m.group(2)) <= 2


def split_tokens(text: str) -> list[str]:
    """Lowercase `text` into naming tokens, splitting glued letter/number pairs.

    `DA716` -> ['da', '716'], `qwen3.6-27B` -> ['qwen36', '27', 'b']. Splitting is what
    lets the ambiguity check see `da716` and `da_716` as one name.
    """
    lowered = str(text).lower()
    for pattern, replacement in _PRE_ALIASES:
        lowered = pattern.sub(replacement, lowered)
    parts = [p for p in re.split(r"[^A-Za-z0-9.]+", lowered) if p]
    out: list[str] = []
    for part in parts:
        # A number plus a unit is one fact — `100k` rows, a `27b` model — so it stays
        # whole; everything else splits letters from digits.
        for chunk in re.findall(r"[0-9]+[a-z]{1,2}(?![a-z0-9])|[a-z]+[0-9]*|[0-9]+",
                                part.replace(".", "")):
            m = re.fullmatch(r"([a-z]+)(\d+)", chunk)
            out.extend([chunk] if (not m or is_model_token(chunk))
                       else [m.group(1), m.group(2)])
    return [t for t in out if t]


def canonical_tokens(tokens: list[str]) -> list[str]:
    """Expand every aliased token to the one spelling this repo uses (pure)."""
    out: list[str] = []
    for t in tokens:
        replacement = CANONICAL_TOKENS.get(t, t)
        out.extend(p for p in replacement.replace("-", "_").split("_") if p)
    return out


def subject_of(name: str) -> str:
    """The part after the date, separators normalised to `_`; '' if the name has no date."""
    for pattern in (LOCAL_NAME, HUB_NAME):
        m = pattern.match(str(name))
        if m:
            return m.group(2).replace("-", "_")
    return ""


def name_date(name: str) -> str:
    """The ISO date a name leads with, or '' when it leads with something else."""
    m = re.match(r"^(\d{4}-\d{2}-\d{2})[-_]", str(name))
    return m.group(1) if m else ""


def undated(subject: str) -> str:
    """`subject` with a leading ISO date removed, in either spelling (pure).

    For composing a NEW name out of an already-dated one. `local_name`/`hub_name` prepend
    today's date, while `canonical_key` deliberately KEEPS the source's so a model key
    still says which run it belongs to — so concatenating the two states the date twice.
    On a long adapter name that is not merely ugly, it is fatal: an eval run against
    `2026-08-21-qwen36-lora-table2-9284-difficult-advice-chunk-only-702-rank-64-dynbatch`
    minted `2026-09-01_2026_08_21_qwen36_...`, 101 characters against the Hub's 96, and
    every eval in the registry was refused before it could start (observed 2026-09-01).

    `name_date` is not enough here: it only recognises the hyphen spelling, and a model
    key carries the underscore one.
    """
    return re.sub(r"^\d{4}[-_]\d{2}[-_]\d{2}[-_]", "", str(subject))


def squash(name: str) -> str:
    """The ambiguity key: a name reduced to the words it actually says (pure).

    Date, separators, spelling variants and letter/digit glue all fall away, so
    `2026-08-06-da716`, `da_716` and `DA-716` collapse to one key. Two artifacts sharing a
    squash key are NOT distinct, whatever their files are called.
    """
    stripped = re.sub(r"^\d{4}-\d{2}-\d{2}[-_]?", "", str(name))
    return "_".join(canonical_tokens(split_tokens(stripped)))


def _check_subject(subject: str, *, what: str, original: str) -> None:
    """Shared subject rules: long enough, no vague tokens, no ambiguous abbreviations."""
    tokens = subject.replace("-", "_").split("_")
    if len(subject.replace("_", "").replace("-", "")) < MIN_SUBJECT_CHARS:
        raise NamingError(
            f"{what}: {original!r} says too little after the date — a name is what a "
            f"reader has left in six months. Use at least {MIN_SUBJECT_CHARS} characters "
            "of subject (model + arm + variant). Try: " + suggest(original))
    if len(original) > MAX_NAME_CHARS:
        raise NamingError(
            f"{what}: {original!r} is {len(original)} characters — over the {MAX_NAME_CHARS} "
            "the Hub allows in a repo name, and unreadable in a legend. Drop the facts the "
            "config already records (hardware, launcher, rollout counts); keep model + arm "
            "+ what makes this run different.")
    junk = [t for t in tokens if t in JUNK_TOKENS]
    if junk or all(t in VAGUE_TOKENS for t in tokens):
        raise NamingError(
            f"{what}: {original!r} is built out of {junk or tokens} — those words name "
            "every artifact in the repo equally well. Say which corpus/arm/organism this "
            f"is. Try: {suggest(original)}")
    versions = [t for t in tokens if re.fullmatch(r"v\d+|version\d*", t)]
    if versions:
        raise NamingError(
            f"{what}: {original!r} versions the name ({versions}) instead of describing "
            "the variant. `sonnet_v2` tells a reader nothing; say the generator model, "
            "the document type and WHAT THIS ONE CHANGES — e.g. "
            "`2026-08-26_sonnet45_difficult_advice_716_length_capped`. The date already "
            "orders the versions.")
    aliased = [t for t in tokens if t in CANONICAL_TOKENS]
    if aliased:
        fixes = ", ".join(f"{t} -> {CANONICAL_TOKENS[t]}" for t in aliased)
        raise NamingError(
            f"{what}: {original!r} uses abbreviations with more than one expansion in "
            f"this project ({fixes}). Spell them out (src/utils.py CANONICAL_TOKENS). "
            f"Try: {suggest(original)}")
    glued = [t for t in tokens
             if re.fullmatch(r"[a-z]+\d+", t) and not is_model_token(t)]
    if glued:
        fixed = ", ".join(re.sub(r"([a-z])(\d)", r"\1_\2", t) for t in glued)
        raise NamingError(
            f"{what}: {original!r} glues a word to a number ({glued}) — `da716` and "
            f"`da_716` then read as different names. Separate them: {fixed}. (A model "
            "generation stays glued: `qwen36`, `gpt4`.) "
            f"Try: {suggest(original)}")


def suggest(text: str, *, date: str | None = None, hub: bool = False) -> str:
    """Build a compliant name out of a messy one — what every error message offers.

    Keeps the name's own date when it has one (a rename must not silently re-date the
    artifact), drops hardware/launcher noise, expands aliases, splits glued numbers.
    """
    date = date or name_date(text) or today()
    stripped = _HARDWARE.sub(" ", re.sub(r"^\d{4}-\d{2}-\d{2}", "", str(text)).lower())
    tokens = [t for t in canonical_tokens(split_tokens(stripped)) if t not in JUNK_TOKENS]
    subject = "_".join(tokens) or "unnamed_artifact"
    return f"{date or today()}{'-' if hub else '_'}{subject.replace('_', '-') if hub else subject}"


def check_local_name(name: str, *, what: str = "name") -> str:
    """Validate a local name (`2026-08-06_difficult_advice_716`); return it, or raise."""
    text = str(name)
    m = LOCAL_NAME.match(text)
    if not m:
        raise NamingError(
            f"{what}: {text!r} is not `<YYYY-MM-DD>_<subject>` (lowercase words joined "
            f"by `_`). Every name in this repo carries the date the thing was produced. "
            f"Try: {suggest(text)}")
    _check_subject(m.group(2), what=what, original=text)
    return text


def check_hub_name(name: str, *, what: str = "hub name") -> str:
    """Validate the part of an HF repo id after the org; return it, or raise."""
    text = str(name)
    m = HUB_NAME.match(text)
    if not m:
        raise NamingError(
            f"{what}: {text!r} is not `<YYYY-MM-DD>-<subject>` (lowercase words joined "
            f"by `-`). CLAUDE.md: the date is the date the data was GENERATED, and the "
            f"subject says which experiment it came from. Try: {suggest(text, hub=True)}")
    _check_subject(m.group(2), what=what, original=text)
    return text


def check_hub_repo(repo_id: str, *, what: str = "HF repo", write: bool = True) -> str:
    """THE gate on the Hub: no artifact reaches it under an undated or ambiguous name.

    `write=True` for anything that CREATES or UPLOADS (every push here). `write=False` for
    a read reference, which may still point at a LEGACY_HUB_REPOS entry.
    """
    text = str(repo_id)
    m = REPO_ID.match(text)
    if not m:
        raise NamingError(f"{what}: {text!r} is not an HF repo id (`org/name`).")
    org, name = m.groups()
    if not write and text in LEGACY_HUB_REPOS:
        return text
    try:
        check_hub_name(name, what=f"{what} ({text})")
    except NamingError as e:
        if write and text in LEGACY_HUB_REPOS:
            raise NamingError(
                f"{e}\n\n{text!r} is one of the pre-dating repos in LEGACY_HUB_REPOS. "
                "It can still be READ; nothing new may be published under it. Rename it "
                "first: `uv run python scripts/hf/rename_repos.py plan`.") from None
        raise
    return f"{org}/{name}"


def to_hub(local: str) -> str:
    """`2026-08-06_difficult_advice_716` -> `2026-08-06-difficult-advice-716`."""
    date, _, subject = check_local_name(local).partition("_")
    return f"{date}-{subject.replace('_', '-')}"


def to_local(hub: str) -> str:
    """`2026-08-06-difficult-advice-716` -> `2026-08-06_difficult_advice_716`."""
    name = check_hub_name(str(hub).split("/")[-1])
    date, subject = name[:10], name[11:]
    return f"{date}_{subject.replace('-', '_')}"


def local_name(subject: str, *, date: str | None = None) -> str:
    """Mint a local name from a subject (aliases expanded, date prepended, validated)."""
    tokens = canonical_tokens(split_tokens(subject))
    return check_local_name(f"{date or today()}_{'_'.join(tokens)}")


def hub_name(subject: str, *, date: str | None = None, org: str | None = None) -> str:
    """Mint a hub name (or full repo id when `org` is given) from a subject."""
    name = to_hub(local_name(subject, date=date))
    return f"{org}/{name}" if org else name


def canonical_key(text: str) -> str:
    """The one filesystem/served-name-safe spelling of a model or arm (pure).

    Not a full name — it is DERIVED from one (an adapter repo, a provider model id) for use
    as a served-model name, an output folder or a Hub tag. A date in the source name is
    kept, so the key still says which run it belongs to.
    """
    raw = str(text).split("/")[-1]
    date = name_date(raw)
    body = raw[len(date) + 1:] if date else raw
    key = "_".join(canonical_tokens(split_tokens(body))) or "unnamed"
    return f"{date}_{key}" if date else key


def check_distinct(names: list[str] | tuple[str, ...], *, what: str = "names") -> None:
    """Fail when two of these names say the same thing on the same day.

    A name is the pair (date, subject), so two runs of one arm a fortnight apart are
    distinct — that is the point of dating them. Two spellings of one subject under one
    date are not. Used wherever names are presented together and a reader must tell them
    apart: the arms of a plot, the `uv run chat` menu, the configs in one stage folder.
    """
    seen: dict[tuple[str, str], list[str]] = {}
    for n in names:
        seen.setdefault((name_date(str(n)), squash(n)), []).append(str(n))
    clashes = {k: v for k, v in seen.items() if len(v) > 1}
    if clashes:
        detail = "; ".join(f"{v} all mean {subject!r}" + (f" on {date}" if date else "")
                           for (date, subject), v in sorted(clashes.items()))
        raise NamingError(
            f"{what}: these are not distinct names — {detail}. Give each the thing that "
            "actually differs (date, arm, seed, ratio), not a spelling variant.")


def run_dir(base: str | Path, subject: str, *, date: str | None = None) -> Path:
    """`output/<eval>/<YYYY-MM-DD>_<subject>/` — the one way a run directory is named."""
    return Path(base) / local_name(subject, date=date)


def figure_path(out_dir: str | Path, subject: str, *, date: str | None = None,
                ext: str = "png") -> Path:
    """THE figure filename: `<out_dir>/<YYYY-MM-DD>_<subject>.<ext>`, validated.

    A plot outlives the conversation that produced it; `bars_overall.png` in a shared
    folder is a figure nobody can date or attribute. Every savefig here takes its path
    from this.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{local_name(subject, date=date)}.{ext.lstrip('.')}"


def label(name: str, *, date_in_label: bool = True) -> str:
    """The arm label for a plot legend: `difficult advice 716 (2026-08-06)`.

    Takes any dated name (local, hub, or a full repo id). The date is kept because a legend
    entry without one is the ambiguity back on the figure.
    """
    text = str(name).split("/")[-1]
    date, subject = name_date(text), subject_of(text)
    if not date:
        raise NamingError(
            f"plot label: {name!r} carries no date, so the figure would not say which run "
            f"it shows. Name the arm first (try {suggest(text)}).")
    words = " ".join(subject.split("_"))
    return f"{words} ({date})" if date_in_label else words


# --------------------------------------------------------------------------------------
# The lint that blocks `git push`
# --------------------------------------------------------------------------------------

# `configs/eval/<eval>.yaml` is a KIND (the registry default for that eval), not an
# instance, so it carries no date. Every other config names a specific run and does.
_KIND_CONFIGS = {
    "configs/endpoints/providers.yaml",
    "configs/data/synth/good_ai_fiction/archetypes.yaml",
    "configs/data/synth/good_ai_fiction/taxonomy.yaml",
}

_FIGURE_LITERAL = re.compile(r'["\']([^"\'/\\]*\.(?:png|svg|pdf))["\']')
# `f"{name}.png"` names nothing itself — whatever fills the placeholder does, and that is
# checked where it is built. A literal with words in it is the one to catch.
_FIGURE_PLACEHOLDER = re.compile(r"\{[^}]*\}|\*")
# A repo id is checked only where it is STRUCTURAL — a YAML value or a whole Python string
# literal. A sentence in a README, an f-string prefix and a glob are not names, and
# flagging them teaches people to ignore the linter.
_YAML_REPO = re.compile(r'(?m)^\s*[a-z_]+:\s*["\']?'
                        r'([A-Za-z0-9][A-Za-z0-9-]{2,}/[A-Za-z0-9._-]+?)["\']?\s*(?:#.*)?$')
_PY_REPO = re.compile(r'(?<![f}])["\']([A-Za-z0-9][A-Za-z0-9-]{2,}/[A-Za-z0-9._-]+)["\']')


@dataclass(frozen=True)
class Finding:
    path: str
    detail: str

    def __str__(self) -> str:
        return f"{self.path}: {self.detail}"


def _tracked_files(root: Path) -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True)
    return [root / line for line in out.stdout.splitlines() if line]


def _eval_kind_configs(root: Path) -> set[str]:
    """The per-eval default configs named by the registry — kinds, so undated.

    The lazy import is the only edge from here to `src.eval`, and it must stay inside the
    function: run_eval imports this module. The fallback reads the same registry as text so
    the lint runs without the eval extras installed.
    """
    try:
        from src.eval import EVALS

        return {spec.config for spec in EVALS.values()}
    except Exception:  # noqa: BLE001 - the lint must run without the eval extras installed
        text = (root / "src/eval/__init__.py").read_text(encoding="utf-8")
        return set(re.findall(r'"(configs/eval/[a-z0-9_]+\.yaml)"', text))


def lint_repo(root: str | Path = ".") -> list[Finding]:
    """Every naming violation in the tracked tree, as findings.

    Three checks: config stems are `<YYYY-MM-DD>_<subject>` and distinct within their stage
    folder; HF repo ids written into configs and code are dated or LEGACY_HUB_REPOS; literal
    figure filenames in code are dated.
    """
    root = Path(root).resolve()
    findings: list[Finding] = []
    kinds = _KIND_CONFIGS | _eval_kind_configs(root)

    by_folder: dict[str, list[str]] = {}
    for path in _tracked_files(root):
        rel = path.relative_to(root).as_posix()
        if rel.startswith("configs/") and rel.endswith((".yaml", ".yml")):
            if rel in kinds or "/archive/" in rel:
                continue
            stem = path.stem
            try:
                check_local_name(stem, what="config name")
                by_folder.setdefault(path.parent.as_posix(), []).append(stem)
            except NamingError as e:
                findings.append(Finding(rel, str(e)))
        if path.suffix not in (".py", ".yaml", ".yml", ".sh", ".md") or not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if rel.startswith(("docs/", "notes/")) or "third_party" in rel:
            continue  # the append-only record and vendored code keep the names they ran under
        finder = _YAML_REPO if path.suffix in (".yaml", ".yml") else _PY_REPO
        for repo in sorted(set(finder.findall(text))) if path.suffix != ".md" else ():
            org = repo.split("/")[0]
            if repo.endswith("-"):
                continue          # a prefix a variable completes, not a name
            if org not in ("LASR-Callum", "matboz") or repo in LEGACY_HUB_REPOS:
                continue
            try:
                check_hub_repo(repo, what="HF repo reference", write=False)
            except NamingError as e:
                findings.append(Finding(rel, str(e)))
        if path.suffix == ".py":
            for fig in sorted(set(_FIGURE_LITERAL.findall(text))):
                if "*" in fig:
                    continue                       # a glob reads names, it does not make one
                bare = _FIGURE_PLACEHOLDER.sub("", fig).rsplit(".", 1)[0]
                if not re.search(r"[a-z]{3}", bare):
                    continue
                if not name_date(fig):
                    findings.append(Finding(rel, (
                        f"figure filename {fig!r} carries no date — build it with "
                        "src.utils.figure_path(out_dir, subject) so the plot says when "
                        "it was made and which arm it shows.")))
    for folder, stems in sorted(by_folder.items()):
        try:
            check_distinct(stems, what=f"config names in {folder}")
        except NamingError as e:
            findings.append(Finding(folder, str(e)))
    return findings


def cli(root: str = ".", quiet: bool = False) -> int:
    """The lint as a command; exit 1 on any violation.

    Not a pipeline stage and deliberately not a console alias: the law is enforced where
    names are made, not by anyone remembering to run it. This exists because
    `.git/hooks/pre-push` needs something to invoke — `uv run --quiet python -m src.utils`.
    """
    findings = lint_repo(root)
    if findings:
        print(f"!!! {len(findings)} naming violation(s) — see src/utils.py for the law\n")
        for f in findings:
            print(f"  {f}\n")
        print("Nothing may be pushed (to git or to Hugging Face) under these names.")
        return 1
    if not quiet:
        print(f"names OK — every tracked config, HF reference and figure name is dated "
              f"and unambiguous.\n{len(LEGACY_HUB_REPOS)} pre-dating Hub repos remain "
              "readable-only (LEGACY_HUB_REPOS); rename them with "
              "`uv run python scripts/hf/rename_repos.py plan`.")
    return 0


def main() -> None:
    import sys

    import fire

    sys.exit(fire.Fire(cli))


if __name__ == "__main__":
    main()
