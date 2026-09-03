# ABOUTME: Discover the model organisms this project trained — LoRA adapters on the Hub that
# ABOUTME: carry training_meta.json — and group them by base model, thinking mode and experiment.

"""What `uv run chat` shows when asked to pick an organism.

An organism is an adapter repo with the two files the eval framework needs to serve it:
`adapter_config.json` (base model, rank) and `training_meta.json` (the thinking stamp,
the train config it came from, the dataset). Adapters without the stamp are counted, not
listed — the framework refuses to guess their mode, so neither does the menu.

Every organism has ONE name, and that name carries the date it was trained: `name` is
the adapter repo reduced to canonical tokens (src/utils.py) with the training date in
front, so two spellings of one arm cannot appear as two organisms and no arm can appear
on a menu, in a served-model name or on a plot without saying when it was made.

Grouping is derived from the train config's name, the one self-describing handle every
arm carries (CLAUDE.md naming rules: `<YYYY-MM-DD>_lora_<model>_<arm>[_variant]`):
`2026-08-06_lora_qwen36_table2_9284_difficult_advice_716_dynbatch` → experiment
`table2_9284`, variant `difficult_advice_716`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from src.huggingface import hf_download, hf_org
from src.utils import canonical_key, check_distinct, label as name_label, name_date


def default_orgs() -> tuple[str, ...]:
    """The org(s) to list organisms from: where this project pushes (.env HF_ORG).

    Read at call time rather than frozen into a constant, for the same reason pushes
    resolve it at push time — the destination org is written down in exactly one place.
    """
    return (hf_org(),)


# Trailing tokens that describe hardware, rank or the batching scheme, not WHICH
# experiment this is: `2xh200`, `h200x4`, `r64`, `rank_64`, `dynbatch`. The train config
# records all of them; the name does not have to.
_NOISE = re.compile(r"^(\d+x\w+|h\d+x\d+|r\d+|dynbatch)$")
_NOISE_PAIR = ("rank",)  # `rank_64`: two tokens, one fact


@dataclass(frozen=True)
class Organism:
    repo: str
    base_model: str
    mode: str  # think | nothink
    train_config: (
        str  # config file stem, e.g. lora_qwen36_t2_9284_da716_dynbatch_2xh200
    )
    dataset: str  # training dataset repo id, "" when the stamp predates that field
    trained: str  # ISO date
    lora_rank: int
    unservable: str = ""  # why the framework cannot serve it; "" = servable

    @property
    def name(self) -> str:
        """THE organism's name: `<YYYY-MM-DD>_<canonical subject>`, always dated.

        Taken from the adapter repo, canonicalised so one arm has one spelling, and dated
        from the repo name when it carries a date and from the training stamp when it does
        not (the pre-dating repos in src/utils.py). An organism therefore cannot be
        referred to, served, filed or plotted without its date.
        """
        key = canonical_key(self.repo)
        return key if name_date(key) else f"{self.trained}_{key}"

    @property
    def label(self) -> str:
        """The legend/menu label: `difficult advice 716 (2026-08-06)`."""
        return name_label(self.name)

    @property
    def key(self) -> str:
        """The served-model-name-safe id vllm_server uses for this repo (== `name`)."""
        return self.name

    @property
    def group(self) -> str:
        return experiment_group(self.train_config)[0]

    @property
    def variant(self) -> str:
        return experiment_group(self.train_config)[1]


def experiment_group(config_name: str) -> tuple[str, str]:
    """(experiment, variant) from a train config name (pure; unit-tested).

    Strips the leading date, the `lora_<model>_` prefix and trailing hardware/rank noise,
    then takes the first token — plus a following run id of three or more digits
    (`table2_9284`) — as the experiment and the rest as the variant. A name with nothing
    left is its own group. The date is not part of the group: it is carried by
    `Organism.name`, which is where a reader looks for it.
    """
    name = config_name.rsplit("/", 1)[-1].removesuffix(".yaml")
    name = name[11:] if name_date(name) else name
    tokens = name.split("_")
    if tokens[0] == "lora" and len(tokens) > 2:
        tokens = tokens[2:]
    while len(tokens) > 1:
        if _NOISE.fullmatch(tokens[-1]):
            tokens.pop()
        elif (len(tokens) > 2 and tokens[-1].isdigit()
              and tokens[-2] in _NOISE_PAIR):
            tokens = tokens[:-2]
        else:
            break
    group, rest = tokens[0], tokens[1:]
    if rest and rest[0].isdigit() and len(rest[0]) >= 3:
        group, rest = f"{group}_{rest[0]}", rest[1:]
    return group, "_".join(rest) or group


def organism_from_files(
    repo: str, adapter_config: dict, training_meta: dict, last_modified: str
) -> Organism:
    """Build an Organism from the repo's two metadata files (pure; unit-tested).

    Base model comes from adapter_config — the file `resolve_target` reads — with the
    stamp's `base_model` as fallback when adapter_config holds a local path (one early
    arm was trained from `/root/qwen36`). No HF id anywhere means it cannot be served.
    """
    candidates = [
        adapter_config.get("base_model_name_or_path", ""),
        str(training_meta.get("base_model", "")),
    ]
    base = next((c for c in candidates if "/" in c and not c.startswith("/")), "")
    unservable = ""
    if not base:
        unservable = (
            f"base stamped as a local path ({candidates[0] or candidates[1]!r})"
        )
    thinking = training_meta.get("thinking")
    if not isinstance(thinking, bool):
        unservable = unservable or "training_meta.json has no boolean `thinking`"
    stamp_ts = str(training_meta.get("timestamp") or "")
    trained = _iso_date(stamp_ts) or _iso_date(last_modified) or "?"
    return Organism(
        repo=repo,
        base_model=base or candidates[0] or candidates[1],
        mode="think" if thinking else "nothink",
        train_config=str(training_meta.get("train_config") or "")
        .rsplit("/", 1)[-1]
        .removesuffix(".yaml")
        or repo.split("/")[-1],
        dataset=str((training_meta.get("dataset") or {}).get("repo") or "")
        if isinstance(training_meta.get("dataset"), dict)
        else "",
        trained=trained,
        lora_rank=int(adapter_config.get("r", 32)),
        unservable=unservable,
    )


def _iso_date(value: str) -> str:
    """YYYY-MM-DD from a stamp timestamp (`20260814_101010`), an ISO string, or ''."""
    if not value:
        return ""
    m = re.match(r"(\d{4})-?(\d{2})-?(\d{2})", value)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""


def discover(orgs: tuple[str, ...] | None = None) -> tuple[list[Organism], int]:
    """Every stamped adapter under `orgs`, sorted for the menu, plus the unstamped count.

    `orgs=None` uses `default_orgs()` — the project's own org, from .env.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    found: list[Organism] = []
    unstamped = 0
    for org in orgs or default_orgs():
        for m in api.list_models(author=org, expand=["siblings", "lastModified"]):
            files = {s.rfilename for s in (m.siblings or [])}
            if "adapter_config.json" not in files:
                continue
            if "training_meta.json" not in files:
                unstamped += 1
                continue
            with open(hf_download(m.id, "adapter_config.json")) as f:
                adapter_config = json.load(f)
            with open(hf_download(m.id, "training_meta.json")) as f:
                training_meta = json.load(f)
            modified = (
                m.last_modified.isoformat()
                if isinstance(m.last_modified, datetime)
                else str(m.last_modified or "")
            )
            found.append(
                organism_from_files(m.id, adapter_config, training_meta, modified)
            )
    return sort_for_menu(found), unstamped


def organisms_from_ids(repos: Sequence[str]) -> list[Organism]:
    """The named adapters, read the same way `discover` reads the ones it finds.

    The explicit-`--target` counterpart to `discover`: same two metadata files, same
    `organism_from_files`, so a named organism and a picked one are indistinguishable
    downstream. An unservable one is an error here rather than a greyed-out menu row --
    the user asked for it by name.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    out: list[Organism] = []
    for repo in repos:
        try:
            info = api.model_info(repo, expand=["lastModified"])
        except Exception as e:  # noqa: BLE001 - name the repo, the id is usually the typo
            raise SystemExit(f"cannot read {repo!r} on the Hub: {type(e).__name__}: {e}") from None
        try:
            with open(hf_download(repo, "adapter_config.json")) as f:
                adapter_config = json.load(f)
            with open(hf_download(repo, "training_meta.json")) as f:
                training_meta = json.load(f)
        except Exception:  # noqa: BLE001
            raise SystemExit(
                f"{repo} is not a servable model organism: it needs adapter_config.json and "
                "training_meta.json (the thinking stamp train_lora.py writes). A full model "
                "or an unstamped adapter cannot be served this way -- backfill the stamp, "
                "never guess the mode.") from None
        modified = (info.last_modified.isoformat()
                    if isinstance(info.last_modified, datetime) else str(info.last_modified or ""))
        o = organism_from_files(repo, adapter_config, training_meta, modified)
        if o.unservable:
            raise SystemExit(f"{repo} cannot be served: {o.unservable}")
        out.append(o)
    return out


def sort_for_menu(organisms: list[Organism]) -> list[Organism]:
    return sorted(
        organisms, key=lambda o: (o.base_model, o.mode, o.group, o.trained, o.key)
    )


def render_menu(
    organisms: list[Organism], unstamped: int = 0
) -> tuple[str, list[Organism]]:
    """The numbered, grouped listing; returns it with the servable organisms in number order."""
    lines: list[str] = []
    numbered: list[Organism] = []
    width = max((len(o.variant) for o in organisms), default=8)
    last_head = last_group = None
    for o in organisms:
        head = f"{o.base_model} · {o.mode}"
        if head != last_head:
            lines.append(f"\n{head}")
            last_head, last_group = head, None
        if o.group != last_group:
            lines.append(f"  {o.group}")
            last_group = o.group
        if o.unservable:
            lines.append(
                f"    [ ×] {o.variant:{width}}  {o.trained}  {o.repo}  — {o.unservable}"
            )
            continue
        numbered.append(o)
        lines.append(
            f"    [{len(numbered):2d}] {o.variant:{width}}  {o.trained}  {o.repo}"
        )
    if unstamped:
        lines.append(
            f"\n(+{unstamped} adapters without training_meta.json are hidden: the "
            "framework cannot infer their thinking mode; backfill the stamp to list them)"
        )
    return "\n".join(lines).lstrip("\n"), numbered


def parse_pick(text: str, count: int) -> list[int]:
    """`1 3 5`, `1,3`, `2-4` → zero-based indices; `q`/empty → []. Out-of-range is an error."""
    text = text.strip()
    if not text or text.lower() in ("q", "quit"):
        return []
    picks: list[int] = []
    for token in re.split(r"[\s,]+", text):
        a, sep, b = token.partition("-")
        try:
            lo, hi = (int(a), int(b)) if sep else (int(a), int(a))
        except ValueError as e:
            raise ValueError(f"not a number: {token!r}") from e
        for n in range(lo, hi + 1):
            if not 1 <= n <= count:
                raise ValueError(f"{n} is not on the menu (1-{count})")
            if n - 1 not in picks:
                picks.append(n - 1)
    return picks


def check_one_server(picked: list[Organism]) -> None:
    """Every pick must share base model and mode: one pod serves one base in one mode."""
    heads = {(o.base_model, o.mode) for o in picked}
    if len(heads) > 1:
        raise ValueError(
            "one session serves one base model in one thinking mode; these picks span "
            + ", ".join(f"{b} · {m}" for b, m in sorted(heads))
            + ". Pick within one heading."
        )


def arm_names(picked: list[Organism]) -> dict[str, str]:
    """Served/REPL names for a set of organisms: `<YYYY-MM-DD>_<variant>`, always distinct.

    The variant alone was the old shorthand and it collided — two `da716` arms trained a
    fortnight apart answered to the same word in the REPL and on every plot built from
    it. The date makes the common case distinct; an experiment qualifier is added when
    two arms of the SAME day still collide, and `check_distinct` refuses to return a set
    that is still ambiguous.
    """
    counts: dict[str, int] = {}
    for o in picked:
        counts[f"{o.trained}_{o.variant}"] = counts.get(f"{o.trained}_{o.variant}", 0) + 1
    names: dict[str, str] = {}
    for o in picked:
        stem = f"{o.trained}_{o.variant}"
        name = stem if counts[stem] == 1 else f"{o.trained}_{o.group}_{o.variant}"
        names[o.repo] = name
    check_distinct(list(names.values()), what="arm names")
    return names
