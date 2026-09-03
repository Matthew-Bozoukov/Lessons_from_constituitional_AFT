# ABOUTME: THE naming law — one name shape per pipeline stage, BUILT by code from the
# ABOUTME: pipeline's own facts. The only human input is the style-type a config is named.

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path

from src.model_profile import model_key

# ======================================================================================
# THE NAMING LAW
# ======================================================================================
#
# Every artifact says which stage made it and which arm it belongs to, so a corpus, the
# model trained on it and its eval runs line up by eye:
#
#     synth    <date>-<style>-synth                 2026-09-01-da-synth
#     mix      <date>-<styles>-<pct>-mix            2026-09-03-da-par-20-mix
#     model    <date>-<model>-<seed>-<mix subject>  2026-09-04-qwen36-8-da-par-20
#     eval     <date>-<eval>-<model name, undated>  2026-09-05-odcv-qwen36-8-da-par-20
#
# A MIX SUBJECT is the styles making up its synthetic share, hyphenated, then the
# PERCENTAGE OF ITS ROWS that are synthetic. The two imply each other: no synthetic rows
# means no styles, so the base mixture — the fixed blend of non-synthetic sources every
# other mixture is built out of — is `<date>-0-mix`, and the control trained on it is
# `<date>-qwen36-8-0`. Percentages are ROWS, not tokens; `mixture_stats.json` records
# both, because the same split reads very differently in the two units.
#
# Two spellings of one grammar; `to_hub`/`to_local` convert between them and nothing else
# may:
#
#     local  2026-09-04_qwen36_8_da_par_20   files, run dirs, figures
#     hub    2026-09-04-qwen36-8-da-par-20   an HF repo id after the org
#
# CONFIG STEMS are spelled the HUB way, with `-`, because a config stem IS a fragment of
# the repo its run will publish to: `configs/train/qwen36-da-par-20.yaml` produces
# `<date>-qwen36-<seed>-da-par-20`, so one is greppable from the other.
#
# ONE HUMAN INPUT, ONE PLACE. The style-type is the stem of the synth or mixture config
# that produced the data — the only naming decision anyone makes, made once, where the
# document type is defined. Every other part is derived: the date from the clock at
# launch, the model from `MODEL_KEYS` (src/model_profile.py), the eval from
# `EvalSpec.key`, the seed from the
# training config. None of them is typed into a name, so none of them can drift from the
# thing it describes.
#
# CONFIGS ARE UNDATED AND UNSEEDED. A config names an ARM; a run produces an ARTIFACT.
# Dating a config records when the arm was written, which is not when anything was
# produced, and the two drift (a config dated 2026-08-18 pushing to a repo dated
# 2026-08-16 is what retired the old law). The exact config behind an artifact travels
# WITH the artifact, in its HF repo metadata — so re-running an arm means fetching the
# config from there, not keeping a dated copy here.

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TOKEN = r"[a-z0-9]+"
LOCAL_NAME = re.compile(rf"^(\d{{4}}-\d{{2}}-\d{{2}})_({_TOKEN}(?:_{_TOKEN})*)$")
HUB_NAME = re.compile(rf"^(\d{{4}}-\d{{2}}-\d{{2}})-({_TOKEN}(?:-{_TOKEN})*)$")
REPO_ID = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)/([^/]+)$")
STYLE = re.compile(rf"^{_TOKEN}(?:-{_TOKEN})*$")

# Past this the Hub refuses a repo name, and a plot legend is unreadable. The style-type
# is the only part anyone controls, so it is the only part that can be shortened.
MAX_NAME_CHARS = 96

# Words the stage shapes spend themselves: a style-type may not claim one, or
# `<date>-<style>-synth` stops parsing back into its parts.
RESERVED = frozenset({"synth", "mix", "pooled", "seed"})


class NamingError(ValueError):
    """A name that breaks the law above. Always names the fix."""


# --------------------------------------------------------------------------------------
# The registries — the ONLY parts of a name that are edited by hand, and never per run
# --------------------------------------------------------------------------------------
#
# `model_key` is the third (src/model_profile.py, MODEL_KEYS), and it lives there rather
# than here: which token stands for a base model is a fact ABOUT that model, and
# model_profile.py is where this repo keeps those. Adding a base model is then one edit to
# one file, beside the GPU it needs and the template it renders.

def api_model_key(provider: str, model_id: str) -> str:
    """The name token for an off-the-shelf API model (`openrouter`, `moonshotai/kimi-k2`).

    Not a MODEL_KEYS entry: a public model is not an artifact of this project and its id
    is not ours to canonicalise, so the id itself is sanitised. The provider is kept
    because two providers may serve one model id.
    """
    return _sanitize(f"{provider} {str(model_id).split('/')[-1]}")


def eval_key(name: str) -> str:
    """The registered token for an eval — `EvalSpec.key`, the eval registry's own word.

    The lazy import is deliberate: `src.eval` must be importable without this module
    being imported first, and importing an eval's package must not drag in the registry.
    """
    from src.eval import EVALS

    if name not in EVALS:
        raise NamingError(f"no eval named {name!r}; known: {', '.join(sorted(EVALS))}")
    return EVALS[name].key


# --------------------------------------------------------------------------------------
# The grammar
# --------------------------------------------------------------------------------------

def today() -> str:
    """Today, ISO — the date a name minted right now carries."""
    return _date.today().isoformat()


def name_date(name: str) -> str:
    """The ISO date a name leads with, or '' when it leads with something else."""
    m = re.match(r"^(\d{4}-\d{2}-\d{2})[-_]", str(name))
    return m.group(1) if m else ""


def undated(name: str) -> str:
    """A name with its leading date removed — how a model enters its eval run's name.

    Takes a bare name or a full repo id; the org is dropped with the date. An eval of
    `LASR-Callum/2026-09-04-qwen36-difficult-advice-0` is named for
    `qwen36-difficult-advice-0`, so the run says which arm it measured without carrying a
    second date that means something else.
    """
    text = str(name).split("/")[-1]
    date = name_date(text)
    return text[len(date) + 1:] if date else text


def subject_of(name: str) -> str:
    """The part after the date, separators normalised to `_`; '' if the name has no date."""
    for pattern in (LOCAL_NAME, HUB_NAME):
        m = pattern.match(str(name).split("/")[-1])
        if m:
            return m.group(2).replace("-", "_")
    return ""


def _sanitize(text: str) -> str:
    """Any text reduced to `-`-joined lowercase tokens (pure)."""
    return "-".join(t for t in re.split(r"[^a-z0-9]+", str(text).lower()) if t)


def _mint(subject: str, date: str | None, *, what: str) -> str:
    """Assemble and validate `<date>-<subject>`; every builder below ends here."""
    name = f"{date or today()}-{_sanitize(subject)}"
    return check_hub_name(name, what=what)


def check_hub_name(name: str, *, what: str = "hub name") -> str:
    """Validate the part of an HF repo id after the org; return it, or raise."""
    text = str(name)
    if not HUB_NAME.match(text):
        raise NamingError(
            f"{what}: {text!r} is not `<YYYY-MM-DD>-<subject>` (lowercase words joined by "
            "`-`). Every artifact carries the date it was PRODUCED; the builders in "
            "src/naming.py are the only things that mint one.")
    if len(text) > MAX_NAME_CHARS:
        raise NamingError(
            f"{what}: {text!r} is {len(text)} characters — over the {MAX_NAME_CHARS} the "
            "Hub allows in a repo name. Every part but the style-type is fixed by the "
            "law, so the style-type is what has to get shorter: rename the synth/mixture "
            "config and rebuild, or set `run_name` for this one run.")
    return text


def check_local_name(name: str, *, what: str = "name") -> str:
    """Validate a local name (`2026-09-04_qwen36_da_0`); return it, or raise."""
    text = str(name)
    if not LOCAL_NAME.match(text):
        raise NamingError(
            f"{what}: {text!r} is not `<YYYY-MM-DD>_<subject>` (lowercase words joined by "
            "`_`). Build it with src/naming.py rather than writing it out.")
    if len(text) > MAX_NAME_CHARS:
        raise NamingError(f"{what}: {text!r} is over {MAX_NAME_CHARS} characters.")
    return text


def to_hub(local: str) -> str:
    """`2026-09-04_qwen36_da_0` -> `2026-09-04-qwen36-difficult-advice-0`."""
    date, _, subject = check_local_name(local).partition("_")
    return f"{date}-{subject.replace('_', '-')}"


def to_local(hub: str) -> str:
    """`2026-09-04-qwen36-difficult-advice-0` -> `2026-09-04_qwen36_da_0`."""
    name = check_hub_name(str(hub).split("/")[-1])
    return f"{name[:10]}_{name[11:].replace('-', '_')}"


def check_hub_repo(repo_id: str, *, what: str = "HF repo") -> str:
    """THE gate on the Hub: nothing is PUBLISHED under a name the law did not mint.

    Only writes are checked. A read may point anywhere — at a repo from before this law,
    at someone else's dataset — because reading a badly named repo does not make another
    one, and refusing the read would only force a copy.
    """
    text = str(repo_id)
    m = REPO_ID.match(text)
    if not m:
        raise NamingError(f"{what}: {text!r} is not an HF repo id (`org/name`).")
    org, name = m.groups()
    return f"{org}/{check_hub_name(name, what=f'{what} ({text})')}"


# --------------------------------------------------------------------------------------
# The style-type: the one part a human writes
# --------------------------------------------------------------------------------------

def check_style(style: str, *, what: str = "style-type", numbers_ok: bool = False) -> str:
    """Validate a style-type — the stem of a synth or mixture config; return it, or raise.

    This is the whole of the human's naming input, so it is the whole of what can be got
    wrong. It carries the document type and its ablation (`da`, `da-length-capped`,
    `da-par` for a mixture of two), and it carries nothing the pipeline already knows: no
    date, no seed, no percentage, no row count, no stage word, no version.

    The vocabulary is short and therefore load-bearing: `par` has meant both
    post-action-retrospection and pre-action-deliberation in this project's history, and
    nothing here can tell them apart. Which style-type a short code means is settled by
    the config that carries it, and by nothing else.
    """
    text = str(style)
    if not STYLE.match(text):
        raise NamingError(
            f"{what}: {text!r} is not lowercase words joined by `-`. It is a config stem "
            "and a fragment of the repo name its run publishes to, so it is spelled the "
            "one way both of those are.")
    if name_date(text) or re.search(r"\d{4}-\d{2}-\d{2}", text):
        raise NamingError(
            f"{what}: {text!r} carries a date. A config names an ARM and is not dated; "
            "the run stamps the date on what it produces.")
    tokens = text.split("-")
    if "seed" in tokens:
        raise NamingError(
            f"{what}: {text!r} names a seed. A seed is decided at launch (`seed=1`) and "
            "appears in the ARTIFACT's name, never the arm's.")
    reserved = [t for t in tokens if t in RESERVED]
    if reserved:
        raise NamingError(
            f"{what}: {text!r} uses {reserved}, which the stage shapes spend themselves "
            f"(`<date>-<style>-synth`). Reserved: {', '.join(sorted(RESERVED))}.")
    versions = [t for t in tokens if re.fullmatch(r"v\d+|version\d*", t)]
    if versions:
        raise NamingError(
            f"{what}: {text!r} versions the name ({versions}) instead of describing the "
            "variant. Say WHAT THIS ONE CHANGES — `da-length-capped`. The "
            "date on every artifact already orders the versions.")
    numbers = [t for t in tokens if t.isdigit()]
    if numbers and not numbers_ok:
        raise NamingError(
            f"{what}: {text!r} carries a bare number ({numbers}). A row count is never part "
            "of a style — it is a fact about one RUN of the config, recorded in the "
            "artifact it produced — and in a mixture the only number is the synthetic "
            "percentage, which the build supplies. Drop it: `da-gemini`, not `da-gemini-716`.")
    if len(text.replace("-", "")) < 2:
        raise NamingError(f"{what}: {text!r} says too little to identify a document type.")
    return text


# --------------------------------------------------------------------------------------
# The builders — one per stage, and the escape hatch for what is not a stage
# --------------------------------------------------------------------------------------

def synth_name(style: str, *, date: str | None = None) -> str:
    """`<date>-<style>-synth` — a generated corpus."""
    return _mint(f"{check_style(style)} synth", date, what="synth corpus")


def mix_subject(styles: str, synthetic_pct: int, variant: str = "") -> str:
    """`<styles>-<pct>[-<variant>]`, or `0` — what a mixture IS, and what its arms carry.

    Args:
        styles: The styles making up the synthetic share, hyphenated (`da`, `da-par`);
            empty for the base mixture, which has no synthetic share at all.
        synthetic_pct: Percentage of the mixture's ROWS that are synthetic.
        variant: How this mixture was BUILT, where that differs from the default and the
            styles do not say it — `cot-only` (only the reasoning of a synthetic row
            is supervised). Named by whoever makes the variant, not by this module; it
            reaches here from the config that declares it.

    Raises:
        NamingError: styles and a synthetic share disagree. They imply each other — a
            style is a kind of synthetic document, so styles at 0% would name documents
            that are not in the mixture, and a share with no styles would leave a reader
            no way to know what the synthetic rows are.
    """
    pct = int(synthetic_pct)
    if bool(styles) != (pct > 0):
        raise NamingError(
            f"mix subject: styles {styles!r} and a {pct}% synthetic share do not agree. "
            "A mixture with no synthetic rows has no styles and is named `0`; a mixture "
            "with synthetic rows names the styles they came from.")
    head = f"{check_style(styles)}-{pct}" if styles else str(pct)
    return f"{head}-{check_style(variant)}" if variant else head


def split_mix_subject(subject: str, *, what: str = "mix subject") -> tuple[str, int, str]:
    """(styles, pct, variant) from a mix subject; raises if it is not one.

    The percentage is the pivot: everything before it is the styles, everything after is
    the variant. It is found by BEING the numeric token rather than by position, because
    the variant may be several tokens long (`da-7-cot-only`) and the styles may be too
    (`da-par-20-cot-only`).
    """
    tokens = str(subject).split("-")
    numeric = [i for i, tok in enumerate(tokens) if tok.isdigit()]
    if len(numeric) != 1:
        raise NamingError(
            f"{what}: {subject!r} carries {len(numeric)} bare numbers; a mix subject "
            "carries exactly one, the synthetic percentage. A mixture is its styles, the "
            "share of its rows they make up, and any variant of how it was built "
            "(`da-7-cot-only`); the base blend is `0`. Neither a style nor a variant "
            "ever carries a number, which is what makes the percentage findable.")
    i = numeric[0]
    styles, pct, variant = "-".join(tokens[:i]), int(tokens[i]), "-".join(tokens[i + 1:])
    mix_subject(styles, pct, variant)          # raises with the specific reason
    return styles, pct, variant


def styles_from_sources(source_keys) -> str:
    """The styles part of a mixture's name, DERIVED from its synthetic sources.

    Sorted and hyphen-joined, so one set of corpora has exactly one name: `{par,
    da-gemini}` is `da-gemini-par` however the config lists them, and a stem that says
    otherwise is wrong rather than merely different. Plain string order — a variant sorts
    with its style (`da-gemini` before `par`) because it is part of that style's subject.
    """
    keys = sorted(str(k) for k in source_keys)
    for k in keys:
        check_style(k, what="synthetic source key")
    return "-".join(keys)


def check_mix_subject(subject: str, *, what: str = "mix subject") -> str:
    """Validate `<styles>-<pct>[-<variant>]`, or `0` for the base blend; return it."""
    split_mix_subject(subject, what=what)
    return str(subject)


def mix_name(styles: str, synthetic_pct: int, variant: str = "", *,
             date: str | None = None) -> str:
    """`<date>-<styles>-<pct>[-<variant>]-mix`, or `<date>-0-mix` for the base blend."""
    return _mint(f"{mix_subject(styles, synthetic_pct, variant)} mix", date,
                 what="training mixture")


def model_name(model: str, seed: int, mix: str, *, date: str | None = None) -> str:
    """`<date>-<model>-<seed>-<mix>-<pct>` — a model organism.

    The seed sits with the model because that is what it belongs to — one base model
    drawn twice — and the mixture's whole subject follows, so an arm says which recipe
    made it without anyone opening a config.

    Args:
        model: The base model id; resolved through MODEL_KEYS.
        seed: The training seed, which is what distinguishes replicates of one arm.
        mix: The mixture's subject (`mix_subject`, or read off its repo with
            `mix_subject_from`).
    """
    return _mint(f"{model_key(model)} {int(seed)} {check_mix_subject(mix)}", date,
                 what="model organism")


def eval_name(eval_name_: str, subject: str, *, date: str | None = None) -> str:
    """`<date>-<eval>-<subject>` — one eval run.

    For an ordinary arm the subject is the target's own name WITHOUT its date, so the run
    carries exactly one date — its own — and still says which arm it measured.

    A POOLED run passes whatever its `pool()` decided the subject is, because only the
    eval knows what its arms have in common. ODCV pools seed replicates of one recipe, so
    it names the shared prefix (`qwen36-da-20-pooled3`); Arena-Hard compares arms that
    share nothing but the baseline they were judged against, so it names that
    (`vs-<baseline>`). Neither rule generalises, which is why neither lives here.
    """
    return _mint(f"{eval_key(eval_name_)} {undated(subject)}", date,
                 what=f"{eval_name_} run")


def artifact_name(subject: str, *, date: str | None = None) -> str:
    """`<date>-<subject>` — the escape hatch for what no stage shape covers.

    Answer caches, probe sweeps, one-off harnesses: things a pipeline stage did not
    produce still get a date and a subject, because that is the whole law. They just get
    no field structure, and nothing tries to parse one out of them.
    """
    return _mint(subject, date, what="artifact")


# What the sources of a PRE-LAW mixture were, in the vocabulary the law uses now. A legacy
# artifact keeps its name on the Hub; a NEW artifact built from it is named from what the
# old rows ARE — `source` and `supervise` columns — never from the old name. This is the
# one place a human maps old words to new, edited once per source and never per run.
# Value: the style a synthetic source is, or None for replay data. A source not listed
# here refuses, naming the line to add.
SOURCE_STYLES: dict[str, str | None] = {
    # synthetic — the constitution-grounded document types
    "da": "da", "difficult_advice": "da", "synthdoc": "da",
    "post_action_retrospection": "par", "peer_critique": "pc",
    "pre_action_deliberation": "pad", "courtroom": "courtroom",
    # the same style under the names older mixtures gave it, and its synth variants
    "synthdoc_difficult_advice": "da", "difficult_advice_v2": "da",
    "difficult_advice_chunk_only": "da-principle-scoped",
    "gpt_responder": "da-gptresp", "grok_responder": "da-grokresp",
    "sonnet_concise": "da-sonnetconcise", "difficult_advice_low_stakes": "da-lowstakes",
    "difficult_advice_t10_curiosity": "da-t10-curiosity",
    "swap_gtrace_sreply703": "da-gtrace-sreply", "swap_strace_greply703": "da-strace-greply",
    "good_ai_fiction": "good-ai-fiction", "nonmoral_deliberation": "nonmoral-deliberation",
    # replay — the non-synthetic blend, however it was assembled at the time
    "tulu3": None, "tulu3_if": None, "numinamath_cot": None, "no_robots": None,
    "table2": None, "smol_summarize": None, "smol_constraints": None,
    "self_oss_instruct": None, "longalign": None, "lima": None,
    "apigen_function_calling": None, "embodied": None, "agentic_tools": None,
    "table2_filtered": None, "agentic": None, "agentic_toolcalling": None,
}

# `supervise` values that are a mixture VARIANT in the law's vocabulary. "all" and
# "final" are not variants: they are what every mixture does by default for single- and
# multi-turn rows respectively.
SUPERVISE_VARIANTS: dict[str, str] = {"cot": "cot-only", "answer": "answer-only"}


LEGACY_NAMES = Path(__file__).parent / "infra" / "legacy_names.yaml"


def legacy_subject(repo_id: str) -> str | None:
    """The subject a NEW artifact takes from a pre-law repo, from the curated table.

    `src/infra/legacy_names.yaml` is one entry per repo that existed before the law — written
    once, by reading each repo's card, manifest and stamp, so the Hub never has to be
    renamed. Consulted after a lawful name (which needs no table) and before row
    derivation (which needs no human). An entry whose `subject` is null is a deliberate
    refusal — a retired arm, a fixture, not an artifact — and its `note` says why.

    Returns:
        The subject, or None when the repo has no entry (the caller falls through).

    Raises:
        NamingError: the entry exists and says this repo must not name anything new.
    """
    if not LEGACY_NAMES.exists():
        return None
    import yaml

    entry = (yaml.safe_load(LEGACY_NAMES.read_text(encoding="utf-8")) or {}).get(str(repo_id))
    if entry is None:
        return None
    if entry.get("subject") is None:
        raise NamingError(
            f"{repo_id} must not name a new artifact: {entry.get('note') or 'listed as '
            'retired in src/infra/legacy_names.yaml'}. Build from a live input instead.")
    return str(entry["subject"])


def derive_artifact_name_from_legacy(rows) -> str:
    """The mix subject a NEW artifact takes from a PRE-LAW mixture — from its rows.

    Runs only when the input's own name does not conform (`mix_subject_from` returned
    ''); a lawful input names its products the default way. Nothing here reads or
    rewrites the old name: the styles come from the `source` column through
    SOURCE_STYLES, the percentage is counted, and the variant is read off `supervise`.

    Args:
        rows: The mixture's rows (dicts, or a datasets.Dataset) with a `source` column and
            optionally `supervise`.

    Returns:
        `<styles>-<pct>[-<variant>]`, or `0` — e.g. `da-7-cot-only` for the 2026-08-31
        cot-only mixture, whose Hub name says none of that.

    Raises:
        NamingError: a source SOURCE_STYLES does not know, or rows with no `source`.
    """
    sources = list(rows["source"]) if hasattr(rows, "column_names") else [r.get("source") for r in rows]
    if not sources or any(s is None for s in sources):
        raise NamingError(
            "cannot name an artifact from this mixture: its rows carry no `source` column, "
            "so nothing says which rows are synthetic. Pass `hf_repo=<name>` explicitly.")
    unknown = sorted({s for s in sources if s not in SOURCE_STYLES})
    if unknown:
        raise NamingError(
            f"cannot name an artifact from this mixture: its rows come from sources the "
            f"law has no word for: {unknown}. Add each to SOURCE_STYLES in src/naming.py "
            "— a style for a synthetic source, None for replay — and rerun.")
    styles = sorted({SOURCE_STYLES[s] for s in sources if SOURCE_STYLES[s]})
    pct = round(100 * sum(1 for s in sources if SOURCE_STYLES[s]) / len(sources))
    supervise = (set(rows["supervise"]) if hasattr(rows, "column_names") and "supervise" in rows.column_names
                 else {r.get("supervise") for r in rows})
    variants = sorted({SUPERVISE_VARIANTS[v] for v in supervise if v in SUPERVISE_VARIANTS})
    if len(variants) > 1:
        raise NamingError(
            f"cannot name an artifact from this mixture: its rows mix supervise variants "
            f"{variants}, and a mixture has one. Pass `hf_repo=<name>` explicitly.")
    return mix_subject("-".join(styles), pct, variants[0] if variants else "")


def mix_subject_from(repo_id: str) -> str:
    """A mixture repo's subject, or '' when it predates this law.

    `LASR-Callum/2026-09-03-da-par-20-mix` -> `da-par-20`. This is the thread that makes a
    model organism's name derivable rather than typed: the mixture already carries its
    styles and its synthetic share, so the training run reads them off the data it was
    pointed at instead of asking anyone to spell them a second time.
    """
    name = str(repo_id).split("/")[-1]
    body = undated(name)
    if not name_date(name) or not body.endswith("-mix"):
        return ""
    return body[: -len("-mix")]


# --------------------------------------------------------------------------------------
# Local names built from the same law
# --------------------------------------------------------------------------------------

def run_dir(base: str | Path, subject: str, *, date: str | None = None) -> Path:
    """`output/<eval>/<YYYY-MM-DD>_<subject>/` — the one way a run directory is named."""
    return Path(base) / to_local(artifact_name(subject, date=date))


def figure_path(out_dir: str | Path, subject: str, *, date: str | None = None,
                ext: str = "png") -> Path:
    """THE figure filename: `<out_dir>/<YYYY-MM-DD>_<subject>.<ext>`, validated.

    A plot outlives the conversation that produced it; `bars_overall.png` in a shared
    folder is a figure nobody can date or attribute. Every savefig here takes its path
    from this. Figures are the one artifact kept out of the Hub, so a free subject is
    right — but not a free date.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{to_local(artifact_name(subject, date=date))}.{ext.lstrip('.')}"


def label(name: str) -> str:
    """The arm label for a plot legend: `qwen36 difficult advice 0 (2026-09-04)`."""
    text = str(name).split("/")[-1]
    date, subject = name_date(text), subject_of(text)
    if not date:
        raise NamingError(
            f"plot label: {name!r} carries no date, so the figure would not say which run "
            "it shows.")
    return f"{' '.join(subject.split('_'))} ({date})"


def check_distinct(names, *, what: str = "names") -> None:
    """Fail when one name is used twice.

    Names are built, not typed, so a repeat is a real collision — two arms that differ in
    something the law does not put in a name (two evals of one arm on one day, two seeds
    that were both left at 0) — and it would publish one over the other.
    """
    seen: dict[str, int] = {}
    for n in names:
        seen[str(n)] = seen.get(str(n), 0) + 1
    clashes = sorted(n for n, c in seen.items() if c > 1)
    if clashes:
        raise NamingError(
            f"{what}: {clashes} would each be published twice. Two runs the law cannot "
            "tell apart differ in something it does not name — give them different seeds, "
            "different style-types, or different days.")


# --------------------------------------------------------------------------------------
# The lint that blocks `git push`
# --------------------------------------------------------------------------------------

_FIGURE_LITERAL = re.compile(r'["\']([^"\'/\\]*\.(?:png|svg|pdf))["\']')
_FIGURE_PLACEHOLDER = re.compile(r"\{[^}]*\}|\*")


@dataclass(frozen=True)
class Finding:
    path: str
    detail: str

    def __str__(self) -> str:
        return f"{self.path}: {self.detail}"


def _tracked_files(root: Path) -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True)
    return [root / line for line in out.stdout.splitlines() if line]


def _yaml(path: Path) -> dict:
    """A config's top-level fields, or {} if it cannot be read.

    The lint's two exact checks — the seed and the mixture subject — both compare a stem
    against what the file itself declares, which is the only thing that can tell a seed
    from a ratio or a rename from a relabel. A config this cannot parse is not the naming
    law's business, so it reads as empty rather than failing.
    """
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - an unparseable config is not the lint's business
        return {}


def _seeded(path: Path, stem: str, folder: str) -> str:
    """The violation when a config's stem ends in the seed the config itself declares.

    Exact rather than heuristic, because the shapes are indistinguishable: `..-10-90` is
    a mixture ratio and `..-1` is a replicate, and only the file knows which. Reading
    `seed:` settles it — and settles it for the case that matters, three files whose only
    difference is the number in their name.

    Mixtures are exempt, and not as a convenience: a mixture stem ENDS in its synthetic
    percentage by law, so `0.yaml` (the base blend, `seed: 0`) collides with this check by
    construction and every other mixture would too whenever its share happened to equal
    its seed.
    """
    if folder == "configs/data/mixture":
        return ""
    seed = _yaml(path).get("seed")
    if seed is not None and stem.split("-")[-1] == str(seed):
        return (f"config stem {stem!r} ends in the seed it declares (seed: {seed}). A "
                "config names an ARM; the seed is a launch argument (`seed=1`) and shows "
                "up in the ARTIFACT's name. Drop it from the stem and run the arm with "
                "each seed you want.")
    return ""


def _check_config(rel: str, stem: str, path: Path) -> str:
    """The violation in one config's stem, or ''."""
    folder = rel.rsplit("/", 1)[0]
    try:
        if folder == "configs/data/synth":
            # `<style>[-<variant>]`, used verbatim as the corpus's subject. The variant is
            # part of the stem rather than a field because a synth name has nothing spliced
            # into it — unlike a mixture, where the percentage lands between the two.
            check_style(stem, what="style-type (config stem)")
        elif folder == "configs/data/mixture":
            # `<styles>[-<variant>]`, and the percentage is spliced BETWEEN them at build
            # time (`da` + `cot-only` -> `da-7-cot-only`). So the variant cannot be
            # inferred from the stem — the config declares it, and the stem must end in it.
            variant = str(_yaml(path).get("variant") or "")
            if variant:
                if not stem.endswith(f"-{variant}"):
                    raise NamingError(
                        f"mixture config {stem!r} declares `variant: {variant}` but its "
                        f"stem does not end in it. The stem is `<styles>-<variant>`, and "
                        "the build splices the synthetic percentage between the two: "
                        f"rename it `<styles>-{variant}.yaml`.")
                stem = stem[: -len(variant) - 1]
                if not stem:
                    raise NamingError(
                        f"mixture config for variant {variant!r} names no styles. A "
                        "variant is how a mixture was built; the styles are what is in it.")
            # `0.yaml` is the base blend: no synthetic share, so no styles to name.
            if stem != "0":
                check_style(stem, what="styles (mixture config stem)")
            # A mixture built on `base:` names its synthetic sources and nothing else, so
            # the styles part is not chosen — it is the source keys, sorted. The config's
            # `sources:` block is the record; the stem has to agree with it.
            cfg = _yaml(path)
            if cfg.get("base"):
                want = styles_from_sources((cfg.get("sources") or {}).keys())
                if stem != want:
                    raise NamingError(
                        f"mixture config {stem!r} names styles its `sources:` do not: the "
                        f"synthetic sources are {sorted(cfg.get('sources') or {})}, so the "
                        f"stem is `{want}`" + (f"-{variant}" if variant else "") +
                        ".yaml. The styles part is derived — sorted source keys, "
                        "hyphenated — so one set of corpora has exactly one name.")
        elif folder == "configs/train":
            head, _, mix = stem.partition("-")
            if not mix:
                raise NamingError(
                    f"train config stem {stem!r} is not `<model>-<mix>-<pct>`. It is the "
                    "adapter's own repo name minus the date and the seed, so the model "
                    "half is fixed by MODEL_KEYS and the rest is the mixture's subject.")
            if head != model_key(head):
                raise NamingError(
                    f"train config stem {stem!r} starts with {head!r}, which is not the "
                    f"registered key for that model ({model_key(head)}).")
            # The mixture half is checked against the MIXTURE, not against the grammar,
            # and only when there is a lawful mixture to check it against. An arm trained
            # on a pre-law dataset has no mix subject to carry — the share of its rows
            # that are synthetic was never recorded — so requiring one would be asking the
            # config to invent a number. Those stems only have to be well-formed; the ones
            # that CAN be checked are held to the exact subject their data declares.
            declared = mix_subject_from(_yaml(path).get("data_repo") or "")
            if declared and mix != declared:
                raise NamingError(
                    f"train config stem {stem!r} says the mixture is {mix!r}, but its "
                    f"data_repo is the {declared!r} mixture. The stem is the adapter's own "
                    "repo name minus the date and the seed, so it has to agree with the "
                    f"data it trains on: rename it `{head}-{declared}.yaml`.")
            # `numbers_ok` ONLY for a pre-law stem: those carry the row counts and ratios
            # of a mixture that was never named under the law, and the only lawful thing
            # to do with a number that names nothing is to leave it where it is.
            check_style(mix, what="mixture subject (train config stem)",
                        numbers_ok=not declared)
        elif folder == "configs/eval":
            # An eval config is a KIND — the registry default for that eval — so its stem
            # is the eval's own name, spelled exactly as the registry spells it. Checked
            # against the registry rather than against the grammar, because the point is
            # that the two cannot drift: `configs/eval/<key>.yaml` is findable from the
            # key and the key is findable from the file.
            from src.eval import EVALS

            if stem not in EVALS:
                raise NamingError(
                    f"eval config {stem!r} is not a registered eval. Every config here is "
                    f"one eval's default and carries its full name ({', '.join(sorted(EVALS))}); "
                    "a config for a specific comparison is not a kind and does not live here.")
        else:
            # A probe or endpoint config names an EXPERIMENT, not a style, so the style
            # rules apply except the one that is about styles: a probe of the 716-row
            # arm may say so, because that number is a fact about the arm it probes.
            check_style(stem, what="config stem", numbers_ok=True)
    except ValueError as e:
        # NamingError for a bad stem, plain ValueError for an unregistered model
        # (src/model_profile.py): both are the same thing to a reader of the lint.
        return str(e)
    return _seeded(path, stem, folder)


def lint_repo(root: str | Path = ".") -> list[Finding]:
    """Every naming violation in the tracked tree, as findings.

    Two checks, because two things are hand-written: config stems (undated, unseeded, and
    shaped for their stage) and literal figure filenames in code (dated). Artifact names
    are not checked here — they are built by the functions above and validated as they are
    minted, which is the only moment a check can still change the outcome.
    """
    root = Path(root).resolve()
    findings: list[Finding] = []
    by_folder: dict[str, list[str]] = {}
    for path in _tracked_files(root):
        rel = path.relative_to(root).as_posix()
        if rel.startswith("configs/") and rel.endswith((".yaml", ".yml")):
            if "/archive/" not in rel:
                detail = _check_config(rel, path.stem, path)
                if detail:
                    findings.append(Finding(rel, detail))
                else:
                    by_folder.setdefault(path.parent.as_posix(), []).append(path.stem)
        if path.suffix != ".py" or not path.exists() or "third_party" in rel:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for fig in sorted(set(_FIGURE_LITERAL.findall(text))):
            if "*" in fig:
                continue                       # a glob reads names, it does not make one
            bare = _FIGURE_PLACEHOLDER.sub("", fig).rsplit(".", 1)[0]
            if re.search(r"[a-z]{3}", bare) and not name_date(fig):
                findings.append(Finding(rel, (
                    f"figure filename {fig!r} carries no date — build it with "
                    "src.naming.figure_path(out_dir, subject) so the plot says when it "
                    "was made and which arm it shows.")))
    for folder, stems in sorted(by_folder.items()):
        try:
            check_distinct(stems, what=f"config names in {folder}")
        except NamingError as e:
            findings.append(Finding(folder, str(e)))
    return findings


def cli(root: str = ".", quiet: bool = False) -> int:
    """The lint as a command; exit 1 on any violation.

    Not a pipeline stage: the law is enforced where names are MADE, not by anyone
    remembering to run this. It exists because `.git/hooks/pre-push` needs something to
    invoke — `uv run --quiet python -m src.naming`.
    """
    findings = lint_repo(root)
    if findings:
        print(f"!!! {len(findings)} naming violation(s) — see src/naming.py for the law\n")
        for f in findings:
            print(f"  {f}\n")
        print("Nothing may be pushed (to git or to Hugging Face) under these names.")
        return 1
    if not quiet:
        print("names OK — every config stem is undated, unseeded and shaped for its "
              "stage, and every figure name is dated.")
    return 0


def main() -> None:
    import sys

    import fire

    sys.exit(fire.Fire(cli))


if __name__ == "__main__":
    main()
