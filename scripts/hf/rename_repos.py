# ABOUTME: Rename Hugging Face repos onto the naming law (src/utils.py): every artifact
# ABOUTME: `<YYYY-MM-DD>-<what it is>`, with the generator/base model and what it changes said.
"""THE Hub migration tool.

    uv run python scripts/hf/rename_repos.py plan                 # propose, write the map
    uv run python scripts/hf/rename_repos.py plan --org matboz    # someone else's namespace
    uv run python scripts/hf/rename_repos.py apply                # move the repos
    uv run python scripts/hf/rename_repos.py apply --only synth   # a slice at a time

A proposal is built from what the repo itself knows — its card (`experiment`,
`date_generated`, `models`), its Hub tags (`kind:`, `pipeline:`, `eval:`, `model:`) and
the local configs that point at it — then run through `src.utils.suggest`. Names the
metadata cannot justify are listed as NEEDS SEMANTICS rather than guessed: `sonnet-v2`
becomes `2026-08-26-sonnet45-difficult-advice-716-length-capped` only because a human
knows what the v2 changed, and that knowledge is written down in
`scripts/hf/rename_overrides.yaml`, not invented here.

`apply` moves the repo (HF keeps a redirect from the old id), then rewrites every local
reference and drops the entry from src/utils.py, so the debt list only shrinks.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import yaml

from src.huggingface import hf_api
from src.utils import (
    NamingError,
    canonical_tokens,
    check_hub_name,
    name_date,
    split_tokens,
    suggest,
)

ROOT = Path(__file__).resolve().parents[2]
MAP_PATH = ROOT / "scripts/hf/rename_map.json"
OVERRIDES = ROOT / "scripts/hf/rename_overrides.yaml"
# Names a machine cannot repair: a version number, or a subject the card does not explain.
NEEDS_HUMAN = re.compile(r"(?<![a-z0-9])(v\d+|new|old|final|latest|test)(?![a-z0-9])")


def _card(api, repo_id: str, repo_type: str) -> dict:
    """The repo's card fields (`| `key` | value |` rows) plus its tags, best effort."""
    try:
        info = api.repo_info(repo_id, repo_type=repo_type, files_metadata=False)
        tags = list(getattr(info, "tags", []) or [])
        created = str(getattr(info, "created_at", "") or "")[:10]
    except Exception:  # noqa: BLE001 - a missing/private repo must not stop the plan
        tags, created = [], ""
    fields: dict[str, str] = {}
    try:
        from src.huggingface import hf_download

        text = Path(hf_download(repo_id, "README.md", repo_type=repo_type)).read_text()
        fields = {m.group(1): m.group(2).strip()
                  for m in re.finditer(r"^\|\s*`([a-z_]+)`\s*\|\s*(.*?)\s*\|$", text, re.M)}
    except Exception:  # noqa: BLE001
        pass
    return {"tags": tags, "created": created, **fields}


def _model_token(card: dict) -> str:
    """The generator or base model this artifact came from, as naming tokens."""
    for source in (card.get("models", ""), " ".join(card.get("tags", []))):
        for pattern, token in ((r"claude[- ]?sonnet[- ]?4[._-]?5", "sonnet45"),
                               (r"gpt[- ]?5", "gpt5"), (r"gpt[- ]?oss", "gptoss"),
                               (r"grok[- ]?4", "grok4"), (r"grok", "grok"),
                               (r"gemini[- ]?[\d.]*", "gemini"),
                               (r"qwen3\.6|qwen3_6|qwen36", "qwen36"),
                               (r"kimi[- ]?k2", "kimik2")):
            if re.search(pattern, str(source), re.I):
                return token
    return ""


def _kind_token(card: dict) -> str:
    for tag in card.get("tags", []):
        if str(tag).startswith("kind:"):
            return str(tag).split(":", 1)[1]
        if str(tag) == "eval-run":
            return "eval"
    return ""


def config_dates() -> dict[str, str]:
    """repo id -> the date of the local config that produced or consumed it.

    The most reliable date for an adapter: the train config that made it is itself dated
    (`2026-08-16_lora_qwen36_table2_9284_courtroom_716_dynbatch.yaml`), and the Hub does
    not always report a createdAt.
    """
    out: dict[str, str] = {}
    for cfg in sorted(ROOT.glob("configs/**/*.yaml")):
        date = name_date(cfg.stem)
        if not date:
            continue
        for repo in re.findall(r"(?:hf_repo|data_repo|dataset|repo)\s*:\s*[\"']?"
                               r"([A-Za-z0-9-]+/[A-Za-z0-9._-]+)", cfg.read_text(encoding="utf-8")):
            out.setdefault(repo, date)
    return out


def propose(repo_id: str, card: dict, overrides: dict,
            dates: dict[str, str] | None = None,
            repo_type: str = "dataset") -> tuple[str, str]:
    """(proposed hub name, note). Note is 'NEEDS SEMANTICS' when a human must decide."""
    org, _, name = repo_id.partition("/")
    if repo_id in overrides:
        return str(overrides[repo_id]), "override"
    # A card writes its date several ways (`2026-08-18`, `20260818`, `20260818_101010`);
    # all of them mean the same day, and any of them beats guessing.
    date = ""
    for candidate in (name, str(card.get("date_generated", "")),
                      (dates or {}).get(repo_id, ""), str(card.get("created", ""))):
        iso = re.match(r"(\d{4})-?(\d{2})-?(\d{2})", str(candidate).strip())
        if iso:
            date = "-".join(iso.groups())
            break
    if not date:
        return "", ("NO DATE ANYWHERE — no date in the name, the card's date_generated, "
                    "the local config that points at it, or the Hub's createdAt")
    body = name[11:] if name_date(name) else name
    tokens = canonical_tokens(split_tokens(body))
    model, kind = _model_token(card), _kind_token(card)
    # The model belongs in the name of the thing it MADE (a synth corpus names its
    # generator) or the thing it IS (an adapter names its base). A mixture's card lists
    # the judge and every upstream model, so prefixing one of them there would name the
    # artifact after a model that did not produce it.
    if model and (kind == "synth" or repo_type == "model") and model not in tokens:
        tokens = [model] + tokens
    if kind and kind not in tokens and kind != "synth":
        tokens = tokens + [kind]
    proposed = suggest("_".join(tokens), date=date, hub=True)
    note = ""
    if NEEDS_HUMAN.search(proposed):
        note = ("NEEDS SEMANTICS — a version number is not a name; say what this variant "
                f"changes. Card says: {str(card.get('experiment', ''))[:120]!r}")
    else:
        try:
            check_hub_name(proposed)
        except NamingError as e:
            note = f"STILL INVALID — {e}"
    return proposed, note


def _repos(api, org: str) -> list[tuple[str, str]]:
    return ([(m.id, "model") for m in api.list_models(author=org)]
            + [(d.id, "dataset") for d in api.list_datasets(author=org)])


def plan(org: str = "LASR-Callum", only: str = "") -> None:
    """Propose a compliant name for every repo in `org` whose name breaks the law."""
    api = hf_api()
    overrides = yaml.safe_load(OVERRIDES.read_text()) if OVERRIDES.exists() else {}
    dates = config_dates()
    rows, needs = [], []
    for repo_id, repo_type in sorted(_repos(api, org)):
        if only and only not in repo_id:
            continue
        name = repo_id.split("/")[-1]
        try:
            check_hub_name(name)
            if repo_id not in (overrides or {}):
                continue                      # already lawful; leave it alone
        except NamingError:
            pass
        card = _card(api, repo_id, repo_type)
        proposed, note = propose(repo_id, card, overrides or {}, dates, repo_type)
        (needs if note and note != "override" else rows).append(
            {"repo": repo_id, "type": repo_type, "to": proposed, "note": note})
    MAP_PATH.write_text(json.dumps({"rename": rows, "needs_human": needs}, indent=2))
    for r in rows:
        print(f"{r['repo']}\n  -> {r['org' if False else 'to']}")
    print(f"\n{len(rows)} ready, {len(needs)} need a human decision "
          f"(listed in {MAP_PATH.relative_to(ROOT)})")
    for r in needs:
        print(f"  ? {r['repo']}\n      {r['note']}")


def apply(org: str = "LASR-Callum", only: str = "", limit: int = 0) -> None:
    """Move the repos in the plan, then rewrite local references and the legacy ledger."""
    api = hf_api()
    plan_data = json.loads(MAP_PATH.read_text())
    moved: dict[str, str] = {}
    for row in plan_data["rename"]:
        if only and only not in row["repo"]:
            continue
        if limit and len(moved) >= limit:
            break
        new_id = f"{row['repo'].split('/')[0]}/{row['to']}"
        if new_id == row["repo"]:
            continue
        try:
            api.move_repo(from_id=row["repo"], to_id=new_id, repo_type=row["type"])
            moved[row["repo"]] = new_id
            print(f"moved {row['repo']} -> {new_id}")
        except Exception as e:  # noqa: BLE001 - report and continue; one repo must not stop the migration
            print(f"!!! {row['repo']}: {e}")
    if moved:
        rewrite_references(moved)


def rewrite_references(moved: dict[str, str]) -> None:
    """Point every tracked config/script at the new id and shrink src/utils.py."""
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                           text=True).stdout.split()
    for rel in files:
        path = ROOT / rel
        if path.suffix not in (".py", ".yaml", ".yml", ".sh", ".md", ".json") or not path.exists():
            continue
        try:
            text = original = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for old, new in moved.items():
            text = text.replace(old, new)
        if text != original:
            path.write_text(text, encoding="utf-8")
            print(f"  ref {rel}")
    legacy = ROOT / "src/utils.py"
    text = legacy.read_text()
    for old in moved:
        text = re.sub(rf'^\s*"{re.escape(old)}",\n', "", text, flags=re.M)
    legacy.write_text(text)


if __name__ == "__main__":
    import fire

    fire.Fire({"plan": plan, "apply": apply})
