# ABOUTME: THE naming law: every artifact this repo names — dataset, mixture, model organism,
# ABOUTME: eval run, config, figure, arm label — is `<YYYY-MM-DD>` + an unambiguous subject.

"""One grammar, one date, one meaning per name.

Why this module exists: names in a research repo are the only handle a result has six
months later. `qwen3.6-27b-lora-t2-9284-synthdoc-716-r64`, `qwen3_6-27b-lora-t2-9284-da716-r64-dynbatch`
and `qwen3.6-27b-lora-t2-9284-da716-r64-dynbatch` were three different runs under three
spellings of the same words, in two orgs, with no date on any of them. That is the failure
mode this module makes impossible.

The law, in two sentences:

1. **Every name starts with the date the thing was produced**, ISO `YYYY-MM-DD`.
2. **The rest of the name says what the thing is, in words this repo agrees on** —
   lowercase tokens, no abbreviation that has more than one expansion.

Two spellings of the one grammar, because the Hub forbids `_` in some places and this
repo's files read better with it:

    local (files, config stems, run dirs, figure stems, arm keys)  2026-08-06_difficult_advice_716
    hub   (the part of an HF repo id after the org)                2026-08-06-difficult-advice-716

`to_hub` / `to_local` convert between them; nothing else may.

WHAT IS NAMED THIS WAY (instances — a specific thing a run produced):
    synth corpora, mixtures, LoRA adapters (model organisms), eval runs, answer caches,
    experiment configs, output run dirs, figures, and the arm labels that appear on a plot.

WHAT IS NOT (kinds — vocabulary the code is written against):
    python modules, eval registry keys (`odcv_bench`, `mmlu`), stage kinds, source
    adapters, the per-eval default config `configs/eval/<eval>.yaml`. A kind has no date
    because it was not produced by a run; it is refused a date so `mmlu` never becomes
    `mmlu_20260806` on one line and `mmlu` on the next.

ENFORCEMENT (this module is not advice):
    * every HF write in this repo calls `check_hub_repo(..., write=True)` first, so an
      undated or ambiguous artifact cannot reach the Hub at all;
    * `lint_repo()` checks every tracked config, reference and figure name, and runs both
      from `uv run names` and from `.git/hooks/pre-push`, so undated code cannot be pushed;
    * `tests/test_naming.py` runs the same lint, so `uv run pytest` fails on it too.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path

# --------------------------------------------------------------------------------------
# The grammar
# --------------------------------------------------------------------------------------

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TOKEN = r"[a-z0-9]+"
LOCAL_NAME = re.compile(rf"^(\d{{4}}-\d{{2}}-\d{{2}})_({_TOKEN}(?:_{_TOKEN})*)$")
HUB_NAME = re.compile(rf"^(\d{{4}}-\d{{2}}-\d{{2}})-({_TOKEN}(?:-{_TOKEN})*)$")
REPO_ID = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)/([^/]+)$")

# A subject shorter than this cannot be saying what the thing is; longer than the max is
# past what the Hub accepts in a repo name (96) and past what a plot legend can carry.
MIN_SUBJECT_CHARS = 5
MAX_NAME_CHARS = 96

# Tokens that carry no information about which artifact this is. They are refused when
# they are ALL a name says (`final_run`, `new_data`) — inside a compound that does say
# something (`model_eval_model_self`, `base_model_control`) they are ordinary words.
VAGUE_TOKENS = frozenset({
    "new", "old", "final", "latest", "current", "prev", "previous", "next",
    "tmp", "temp", "copy", "backup", "bak", "misc", "other1", "stuff", "thing", "things",
    "untitled", "unnamed", "noname", "todo", "wip", "draft", "foo", "bar", "baz", "asdf",
    "output", "outputs", "result", "results", "data", "dataset", "datasets", "file",
    "files", "run", "runs", "model", "models", "adapter", "experiment", "exp", "job",
    "mine", "ours", "theirs", "x", "y", "z", "a", "b", "c", "aa", "bb", "test", "tests",
})

# Spellings that must be collapsed BEFORE tokenising, because the variation is inside
# what should be one token: `qwen3.6`, `qwen3_6` and `qwen3-6` are one model.
_PRE_ALIASES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"qwen[ ._-]?3[._-]6(?:[ ._-]?27b)?"), "qwen36"),
    (re.compile(r"qwen[ ._-]?3(?:[ ._-]?32b)"), "qwen3"),
    (re.compile(r"gpt[ ._-]?oss"), "gptoss"),
    # "synthdoc v2" was the deleted synthdoc PACKAGE's own version number, not a version
    # of anything in this repo; the corpus it made is what the name should say.
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
# `par` was the worst of them: post-action-retrospection and pre-action-deliberation are
# two different document types and two different arms, and `par`/`pad` read as either.
# Keys are matched per token, after a glued row-count is split off (`da716` -> da, 716).
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


# Never acceptable, anywhere in a name, however long the rest of it is.
JUNK_TOKENS = frozenset({"tmp", "temp", "foo", "bar", "baz", "asdf", "untitled",
                         "unnamed", "noname", "wip", "copy", "backup", "bak", "todo",
                         "stuff", "thing", "things", "mine"})


class NamingError(ValueError):
    """A name that breaks the law in this module's docstring. Always names the fix."""


def today() -> str:
    """Today, ISO. The default date for a name minted right now."""
    return _date.today().isoformat()


def is_model_token(token: str) -> bool:
    """True for a token whose letters and digits are ONE word: `qwen36`, `gpt4`, `27b`.

    A model generation is written glued everywhere in the field (`k2`, `gpt4`, `qwen36`),
    so splitting it would make names less readable, not more. A long number is never a
    generation — it is a row count or an id — so `da716` and `par716` do get split, and
    the abbreviations that hide in front of one (`t2`, `r64`, `s1`) are expanded before
    tokenising by _PRE_ALIASES.
    """
    m = re.fullmatch(r"([a-z]+)(\d+)", token)
    return bool(m) and len(m.group(2)) <= 2


def split_tokens(text: str) -> list[str]:
    """Lowercase `text` into naming tokens, splitting glued letter/number pairs.

    `DA716` -> ['da', '716'], `qwen3.6-27B` -> ['qwen36', '27', 'b'], `qwen36` -> ['qwen36'].
    Splitting is what lets the ambiguity check see `da716` and `da_716` as one name.
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


def squash(name: str) -> str:
    """The ambiguity key: a name reduced to the words it actually says (pure).

    Date, separators, spelling variants and letter/digit glue all fall away, so
    `2026-08-06-da716`, `da_716` and `DA-716` collapse to one key and can be reported as
    the same name wearing three costumes. Two artifacts sharing a squash key are NOT
    distinct, whatever their files are called.
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
            f"this project ({fixes}). Spell them out (src/naming.py CANONICAL_TOKENS). "
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

    Args:
        repo_id: Full `org/name`.
        what: What is being pushed, for the error message.
        write: True for anything that CREATES or UPLOADS (every push in this repo).
            False for a read reference, which may still point at a legacy repo listed in
            src/naming_legacy.py — those exist on the Hub already and are renamed by
            scripts/hf/rename_repos.py, not by breaking every config that reads them.

    Returns:
        `repo_id` unchanged.

    Raises:
        NamingError: with the compliant name to use instead.
    """
    text = str(repo_id)
    m = REPO_ID.match(text)
    if not m:
        raise NamingError(f"{what}: {text!r} is not an HF repo id (`org/name`).")
    org, name = m.groups()
    if not write:
        from src.naming_legacy import LEGACY_HUB_REPOS

        if text in LEGACY_HUB_REPOS:
            return text
    try:
        check_hub_name(name, what=f"{what} ({text})")
    except NamingError as e:
        from src.naming_legacy import LEGACY_HUB_REPOS

        if write and text in LEGACY_HUB_REPOS:
            raise NamingError(
                f"{e}\n\n{text!r} is one of the pre-dating repos in src/naming_legacy.py. "
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


# --------------------------------------------------------------------------------------
# Distinctness — "unambiguous AND distinct" is two checks, not one
# --------------------------------------------------------------------------------------


def canonical_key(text: str) -> str:
    """The one filesystem/served-name-safe spelling of a model or arm (pure).

    Not a full name — it is DERIVED from one (an adapter repo, a provider model id) for
    use as a served-model name, an output folder or a Hub tag. Aliases are expanded and
    separators normalised so one organism cannot file itself under two keys; a date in
    the source name is kept, because the key then still says which run it belongs to.
    """
    raw = str(text).split("/")[-1]
    date = name_date(raw)
    body = raw[len(date) + 1:] if date else raw
    key = "_".join(canonical_tokens(split_tokens(body))) or "unnamed"
    return f"{date}_{key}" if date else key


def check_distinct(names: list[str] | tuple[str, ...], *, what: str = "names") -> None:
    """Fail when two of these names say the same thing on the same day.

    A name is the pair (date, subject), so two runs of one arm a fortnight apart are
    distinct — that is the whole point of dating them. Two spellings of one subject
    under one date are not, and that is what this refuses. Used wherever a set of names
    is presented together and a reader must tell them apart: the arms of a plot, the
    organisms on the `uv run chat` menu, the configs in one stage folder.
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


# --------------------------------------------------------------------------------------
# Naming things that get written out: run dirs, figures, plot labels
# --------------------------------------------------------------------------------------


def run_dir(base: str | Path, subject: str, *, date: str | None = None) -> Path:
    """`output/<eval>/<YYYY-MM-DD>_<subject>/` — the one way a run directory is named."""
    return Path(base) / local_name(subject, date=date)


def figure_path(out_dir: str | Path, subject: str, *, date: str | None = None,
                ext: str = "png") -> Path:
    """THE figure filename: `<out_dir>/<YYYY-MM-DD>_<subject>.<ext>`, validated.

    A plot outlives the conversation that produced it; `bars_overall.png` in a shared
    folder is a figure nobody can date or attribute. Every savefig in this repo takes its
    path from here.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{local_name(subject, date=date)}.{ext.lstrip('.')}"


def label(name: str, *, date_in_label: bool = True) -> str:
    """The human-readable arm label for a plot legend/axis: `difficult advice 716 (2026-08-06)`.

    Takes any dated name (local, hub, or a full repo id) and renders the subject as words
    with the date kept — a legend entry without a date is the ambiguity back on the figure.
    """
    text = str(name).split("/")[-1]
    date, subject = name_date(text), subject_of(text)
    if not date:
        raise NamingError(
            f"plot label: {name!r} carries no date, so the figure would not say which run "
            f"it shows. Name the arm first (try {suggest(text)}).")
    words = " ".join(subject.split("_"))
    return f"{words} ({date})" if date_in_label else words


def labels_for(names: list[str] | tuple[str, ...]) -> list[str]:
    """`label` over a set of arms, refusing a legend whose entries are not distinct."""
    check_distinct(names, what="plot arms")
    return [label(n) for n in names]


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
# `f"{name}.png"` names nothing itself — whatever fills the placeholder does, and that
# value is checked where it is built. A literal with words in it is the one to catch.
_FIGURE_PLACEHOLDER = re.compile(r"\{[^}]*\}|\*")
# A repo id has to be lint-able without guessing at prose. It is checked where it is
# STRUCTURAL — a YAML value or a whole Python string literal — and nowhere else: a
# sentence in a README, an f-string prefix (`f"org/2026-08-17-{eval}"`) and a glob are
# not names, and flagging them teaches people to ignore the linter.
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
    """The per-eval default configs named by the registry — kinds, so undated."""
    try:
        from src.eval import EVALS

        return {spec.config for spec in EVALS.values()}
    except Exception:  # noqa: BLE001 - the lint must run without the eval extras installed
        text = (root / "src/eval/__init__.py").read_text(encoding="utf-8")
        return set(re.findall(r'"(configs/eval/[a-z0-9_]+\.yaml)"', text))


def lint_repo(root: str | Path = ".") -> list[Finding]:
    """Every naming violation in the tracked tree, as findings (pure-ish: reads files).

    Checked, in the order a reader meets them:
      1. experiment configs — stem must be `<YYYY-MM-DD>_<subject>`, and distinct within
         its stage folder;
      2. HF repo ids written INTO configs and code — dated hub names, or a legacy repo
         listed in src/naming_legacy.py (read-only debt, enumerated so it can shrink);
      3. figure filenames in code — literal `*.png|svg|pdf` names must be dated.
    """
    root = Path(root).resolve()
    findings: list[Finding] = []
    kinds = _KIND_CONFIGS | _eval_kind_configs(root)
    from src.naming_legacy import LEGACY_HUB_REPOS

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
                        "src.naming.figure_path(out_dir, subject) so the plot says when "
                        "it was made and which arm it shows.")))
    for folder, stems in sorted(by_folder.items()):
        try:
            check_distinct(stems, what=f"config names in {folder}")
        except NamingError as e:
            findings.append(Finding(folder, str(e)))
    return findings


def cli(root: str = ".", quiet: bool = False) -> int:
    """`uv run names [--root .]` — the lint; exit 1 on any violation.

    Also what `.git/hooks/pre-push` runs, so code that names an artifact ambiguously or
    without a date cannot be pushed.
    """
    findings = lint_repo(root)
    from src.naming_legacy import LEGACY_HUB_REPOS

    if findings:
        print(f"!!! {len(findings)} naming violation(s) — see src/naming.py for the law\n")
        for f in findings:
            print(f"  {f}\n")
        print("Nothing may be pushed (to git or to Hugging Face) under these names.")
        return 1
    if not quiet:
        print(f"names OK — every tracked config, HF reference and figure name is dated "
              f"and unambiguous.\n{len(LEGACY_HUB_REPOS)} pre-dating Hub repos remain "
              "readable-only (src/naming_legacy.py); rename them with "
              "`uv run python scripts/hf/rename_repos.py plan`.")
    return 0


def main() -> None:
    import sys

    import fire

    sys.exit(fire.Fire(cli))


if __name__ == "__main__":
    main()
