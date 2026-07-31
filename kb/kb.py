# ABOUTME: Self-contained paper knowledge base — add papers from a URL/arXiv id with AI summaries,
# ABOUTME: then build a single standalone index.html force-graph. No coupling to the rest of the repo.

from __future__ import annotations

import html
import json
import os
import re
import sys
import textwrap
import urllib.parse
import webbrowser
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import fire
import requests
import yaml
from dotenv import load_dotenv

KB_DIR = Path(__file__).resolve().parent
PAPERS_DIR = KB_DIR / "papers"
TEMPLATE = KB_DIR / "template.html"
CONFIG = KB_DIR / "config.yaml"
INDEX = KB_DIR / "index.html"

SECTIONS = [
    ("tldr", "TL;DR"),
    ("contribution", "Main contribution"),
    ("method", "How they did it"),
    ("results", "Key results"),
    ("limitations", "Limitations"),
    ("for_us", "Why it matters for us"),
]
# Per-person notes live in the same file as '## Notes — <Name>' sections, so they diff and
# review like code. Everything from the first such heading down is human-owned and is never
# touched by the summariser.
NOTE_HEAD = re.compile(r"^##\s+Notes\s+[—–-]\s+(.+?)\s*$", re.M)


def split_notes(body: str) -> tuple[str, list[dict]]:
    """Split a note body into (model-written part, [{who, md}, ...])."""
    heads = list(NOTE_HEAD.finditer(body))
    if not heads:
        return body, []
    notes = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
        notes.append({"who": h.group(1).strip(), "md": body[h.end():end].strip()})
    return body[:heads[0].start()], notes


def whoami(cfg: dict, who: str | None = None) -> str:
    """Whose notes these are: --who > KB_USER env > config `me` > git user.name."""
    if who:
        return who
    if os.environ.get("KB_USER"):
        return os.environ["KB_USER"]
    if cfg.get("me"):
        return str(cfg["me"])
    import subprocess
    name = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True,
                          cwd=KB_DIR).stdout.strip()
    return name or "unknown"


# --------------------------------------------------------------------------- config / notes io


def load_config() -> dict:
    """Read kb/config.yaml."""
    return yaml.safe_load(CONFIG.read_text()) or {}


def slugify(s: str, maxlen: int = 60) -> str:
    """Lowercase, hyphenated, filesystem- and link-safe id."""
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (s or "").strip().lower()).strip("-")
    return s[:maxlen].strip("-") or "untitled"


def parse_note(path: Path) -> tuple[dict, str]:
    """Split a paper markdown file into (frontmatter dict, body markdown)."""
    text = path.read_text()
    if not text.startswith("---"):
        raise ValueError(f"{path} has no YAML frontmatter")
    _, fm, body = text.split("---", 2)
    meta = yaml.safe_load(fm) or {}
    return meta, body.lstrip("\n")


def write_note(path: Path, meta: dict, body: str) -> None:
    """Write frontmatter + body back out, with a stable key order."""
    order = ["id", "title", "short", "authors", "year", "venue", "url", "category",
             "takeaway", "tags", "relevance", "status", "added", "related"]
    ordered = {k: meta[k] for k in order if k in meta}
    ordered.update({k: v for k, v in meta.items() if k not in ordered})
    fm = yaml.safe_dump(ordered, sort_keys=False, allow_unicode=True, width=100)
    path.write_text(f"---\n{fm}---\n\n{body.rstrip()}\n")


def load_papers() -> list[dict]:
    """Load every paper note, returning dicts of frontmatter + body + file path."""
    out = []
    for p in sorted(PAPERS_DIR.glob("*.md")):
        meta, body = parse_note(p)
        meta.setdefault("id", p.stem)
        meta["_body"] = body
        meta["_file"] = str(p.relative_to(KB_DIR.parent))
        out.append(meta)
    return out


# --------------------------------------------------------------------------- source fetching


ARXIV_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/|arxiv[:\s]*)(\d{4}\.\d{4,5})", re.I)
BARE_ARXIV_RE = re.compile(r"^\d{4}\.\d{4,5}$")


def _arxiv_candidates(raw: str) -> list[str]:
    """Fire parses a bare id as a float, so 2510.04340 arrives as '2510.0434'.

    Try the id as given, then zero-padded to the 4- and 5-digit suffix forms.
    """
    base, _, frac = raw.partition(".")
    return [raw] + [f"{base}.{frac.ljust(n, '0')}" for n in (4, 5) if 0 < len(frac) < n]


def fetch_arxiv(arxiv_id: str) -> dict:
    """Pull title/authors/year/abstract from the arXiv Atom API."""
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = None
    for cand in _arxiv_candidates(arxiv_id):
        r = requests.get("https://export.arxiv.org/api/query",
                         params={"id_list": cand}, timeout=30)
        r.raise_for_status()
        e = ET.fromstring(r.text).find("a:entry", ns)
        if e is not None and e.find("a:title", ns) is not None:
            entry, arxiv_id = e, cand
            break
    if entry is None:
        raise RuntimeError(f"arXiv returned no entry for {arxiv_id}")
    get = lambda tag: (entry.findtext(f"a:{tag}", "", ns) or "").strip()
    return {
        "title": re.sub(r"\s+", " ", get("title")),
        "authors": [a.findtext("a:name", "", ns) for a in entry.findall("a:author", ns)],
        "year": int(get("published")[:4]) if get("published") else None,
        "venue": "arXiv",
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "text": re.sub(r"\s+", " ", get("summary")),
    }


def fetch_page(url: str) -> dict:
    """Crude readable-text extraction for blog posts / non-arXiv sources."""
    r = requests.get(url, timeout=30, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120 Safari/537.36"})
    r.raise_for_status()
    h = r.text
    title = re.search(r"<title[^>]*>(.*?)</title>", h, re.S | re.I)
    h = re.sub(r"(?is)<(script|style|nav|footer|svg|noscript)[^>]*>.*?</\1>", " ", h)
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"(?s)<[^>]+>", " ", h))).strip()
    return {
        "title": html.unescape(title.group(1)).strip() if title else "",
        "authors": [], "year": None, "venue": urllib.parse.urlparse(url).netloc,
        "url": url, "text": text[:24000],
    }


def resolve_source(source: str | None, notes: str, fetch: bool) -> dict:
    """Turn whatever the user typed into {title, authors, year, venue, url, text}."""
    blank = {"title": "", "authors": [], "year": None, "venue": "", "url": "", "text": ""}
    source = None if source is None else str(source)   # fire may hand us a float/int
    if not source:
        src = blank
    elif BARE_ARXIV_RE.match(source.strip()):
        src = fetch_arxiv(source.strip())
    elif (m := ARXIV_RE.search(source)):
        src = fetch_arxiv(m.group(1))
    elif source.startswith("http"):
        src = fetch_page(source) if fetch else {**blank, "url": source}
    else:  # a plain title, for things with no fetchable URL at all
        src = {**blank, "title": source}
    if notes:
        src["text"] = (src["text"] + "\n\nUSER-SUPPLIED CONTEXT:\n" + notes).strip()
    return src


# --------------------------------------------------------------------------- summariser


SYS = """You maintain a research knowledge base for an AI-safety project. You read a paper or \
blog post and emit a compact, honest structured summary plus links to papers already in the base.

Rules:
- Papers are filed by WHAT THEY GIVE THE PROJECT, not by topic — pick the `category` that says \
how we would actually use this paper. Topic goes in `tags`.
- Be specific and concrete. Name the models, datasets, metrics and numbers the source actually \
reports. No hype, no filler, no restating the title.
- If the source text is thin (an abstract only, or a page that failed to load), say what you can \
and write "unclear from the abstract" rather than inventing method details or numbers.
- `related` may ONLY use ids from the provided list of existing papers. Each `why` is one clause \
saying what THIS paper takes from / extends / contradicts in that one. Prefer 1-4 real links over \
many weak ones. Omit rather than pad.
- `relevance` is 1-5 for how load-bearing this is for the project described, 5 = the project is \
directly built on it.
Return ONLY a JSON object, no prose, no code fence."""

SCHEMA = """{
  "id": "firstauthorlastname-year-short-slug (lowercase, hyphens)",
  "title": "full title",
  "short": "graph label: 'Betley 2025 - Emergent Misalignment' style, <= 46 chars",
  "authors": ["Last, First", "..."],
  "year": 2025,
  "venue": "arXiv | NeurIPS | Anthropic blog | internal doc | ...",
  "category": "exactly one of:\n%(cats)s",
  "takeaway": "ONE line, <=110 chars, imperative and concrete: what we take from this paper, e.g. \
'reuse their 5-scenario agentic honeypot suite + judge rubric' or 'the baseline result we replicate'",
  "tags": ["3-6 lowercase-hyphenated topic tags"],
  "relevance": 4,
  "tldr": "2-3 sentences: what it shows and why anyone should care",
  "contribution": "the actual new thing, in 2-4 sentences",
  "method": "models, data, training setup, evals - markdown bullets '- ' allowed",
  "results": "headline numbers and findings - markdown bullets allowed",
  "limitations": "what it does NOT show; threats to validity - bullets allowed",
  "for_us": "concrete implications for the project: what to reuse, replicate, or avoid",
  "related": [{"id": "existing-paper-id", "why": "one clause"}],
  "related_from": [{"id": "existing-paper-id", "why": "why that EXISTING paper points at THIS one"}]
}"""


def _client():
    """OpenRouter via the OpenAI SDK. Kept local so this folder has no repo-internal imports."""
    from openai import OpenAI
    load_dotenv(KB_DIR.parent / ".env")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not set (put it in .env at the repo root), "
                 "or pass --no_ai to add a stub entry you fill in by hand.")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def extract_json(s: str) -> dict:
    """Tolerate code fences and leading prose around the JSON object."""
    s = re.sub(r"^```(?:json)?|```$", "", s.strip(), flags=re.M).strip()
    start = s.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in model output:\n{s[:400]}")
    obj, _ = json.JSONDecoder().raw_decode(s, start)
    return obj


def summarize(src: dict, cfg: dict, existing: list[dict], model: str | None = None) -> dict:
    """Ask the model for the structured summary + proposed links."""
    cats = "\n".join(f"    - {k}: {(v or {}).get('desc', '')}"
                     for k, v in (cfg.get("categories") or {}).items())
    catalog = "\n".join(
        f"- {p['id']} :: {p.get('title', '')} :: {(p.get('_tldr') or '')[:150]}" for p in existing
    ) or "(the knowledge base is empty - `related` and `related_from` must be [])"
    user = f"""PROJECT CONTEXT
{cfg.get('project', '').strip()}

EXISTING PAPERS IN THE KNOWLEDGE BASE (ids you may link to)
{catalog}

SOURCE
title: {src.get('title') or '(unknown)'}
url: {src.get('url') or '(none)'}
authors: {', '.join(src.get('authors') or []) or '(unknown)'}
year: {src.get('year') or '(unknown)'}

TEXT
{src.get('text') or '(no text was retrievable - rely on the title/url and say so)'}

Emit JSON matching exactly this schema:
{SCHEMA % {"cats": cats}}"""
    resp = _client().chat.completions.create(
        model=model or cfg.get("model", "anthropic/claude-sonnet-4.5"),
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
        temperature=0.2, max_tokens=4000,
    )
    return extract_json(resp.choices[0].message.content or "")


# --------------------------------------------------------------------------- commands


def _note_body(d: dict) -> str:
    """Render the model's sections into the markdown body."""
    parts = []
    for key, heading in SECTIONS:
        val = (d.get(key) or "").strip()
        parts.append(f"## {heading}\n\n{val or '_todo_'}")
    return "\n\n".join(parts)


def add(source: str | None = None, notes: str = "", category: str | None = None,
        no_ai: bool = False, no_fetch: bool = False, model: str | None = None,
        force: bool = False, open_after: bool = False) -> None:
    """Add a paper.

    Args:
        source: arXiv id/url, any http url, or a bare title for unfetchable sources.
        notes: extra context pasted in (abstract, your own take, internal-doc contents).
        category: force a category instead of letting the model pick.
        no_ai: write a stub with no model call.
        no_fetch: skip downloading the url, use --notes as the only source text.
        model: override the summariser model for this call.
        force: overwrite an existing note with the same id.
        open_after: open index.html when done.
    """
    cfg = load_config()
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    src = resolve_source(source, notes, fetch=not no_fetch)
    print(f"→ source: {src.get('title') or source or '(manual)'}")

    existing = [{"id": p["id"], "title": p.get("title", ""),
                 "_tldr": _first_section(p["_body"], "TL;DR")} for p in load_papers()]

    if no_ai:
        d = {"id": slugify(src.get("title") or source or "untitled"),
             "title": src.get("title") or (source or "untitled"),
             "authors": src.get("authors") or [], "year": src.get("year"),
             "venue": src.get("venue", ""), "category": category or "foundation",
             "takeaway": "", "tags": [], "relevance": 3, "related": [], "related_from": []}
    else:
        print(f"→ summarising with {model or cfg.get('model')} …")
        d = summarize(src, cfg, existing, model=model)

    known = {p["id"] for p in existing}
    meta = {
        "id": slugify(d.get("id") or d.get("title", "")),
        "title": d.get("title") or src.get("title", ""),
        "short": d.get("short") or "",
        "authors": d.get("authors") or src.get("authors") or [],
        "year": d.get("year") or src.get("year"),
        "venue": d.get("venue") or src.get("venue", ""),
        "url": src.get("url", ""),
        "category": category or d.get("category") or "foundation",
        "takeaway": d.get("takeaway") or "",
        "tags": d.get("tags") or [],
        "relevance": int(d.get("relevance") or 3),
        "status": "unread",
        "added": datetime.now(timezone.utc).date().isoformat(),
        "related": [r for r in (d.get("related") or []) if r.get("id") in known],
    }
    path = PAPERS_DIR / f"{meta['id']}.md"
    if path.exists() and not force:
        sys.exit(f"{path} already exists (use --force to overwrite, or edit it by hand)")
    write_note(path, meta, _note_body(d))
    print(f"✓ wrote {path.relative_to(KB_DIR.parent)}  ({len(meta['related'])} outgoing links)")

    # Back-links: let existing notes point at the new one where the model saw a reason to.
    added_back = 0
    for r in (d.get("related_from") or []):
        rid = r.get("id")
        if rid not in known:
            continue
        p = PAPERS_DIR / f"{rid}.md"
        m, b = parse_note(p)
        rel = m.get("related") or []
        if any(x.get("id") == meta["id"] for x in rel):
            continue
        rel.append({"id": meta["id"], "why": r.get("why", "")})
        m["related"] = rel
        write_note(p, m, b)
        added_back += 1
    if added_back:
        print(f"✓ added {added_back} back-link(s) from existing notes")

    build(open_after=open_after)


def resummarize(paper_id: str, model: str | None = None) -> None:
    """Re-run the summariser on an existing note, preserving your own notes section."""
    cfg = load_config()
    path = PAPERS_DIR / f"{paper_id}.md"
    meta, body = parse_note(path)
    _, notes = split_notes(body)
    src = resolve_source(meta.get("url") or meta.get("title"), notes="", fetch=True)
    existing = [{"id": p["id"], "title": p.get("title", ""),
                 "_tldr": _first_section(p["_body"], "TL;DR")}
                for p in load_papers() if p["id"] != paper_id]
    d = summarize(src, cfg, existing, model=model)
    known = {p["id"] for p in existing}
    meta["related"] = [r for r in (d.get("related") or []) if r.get("id") in known]
    for k in ("tags", "short", "takeaway"):
        if d.get(k):
            meta[k] = d[k]
    new_body = _note_body(d)
    for n in notes:                      # human notes survive verbatim
        new_body += f"\n\n## Notes — {n['who']}\n\n{n['md']}"
    write_note(path, meta, new_body)
    print(f"✓ refreshed {path.name} (kept notes from {', '.join(n['who'] for n in notes) or 'nobody'})")
    build()


def note(paper_id: str, text: str, who: str | None = None) -> None:
    """Append a dated bullet to your own notes section on a paper.

    Args:
        paper_id: the note id (filename without .md); `kb.py ls` lists them.
        text: the note. Markdown works; [[other-paper-id]] renders as a link.
        who: whose notes (defaults to KB_USER env, config `me`, then git user.name).
    """
    cfg = load_config()
    path = PAPERS_DIR / f"{paper_id}.md"
    if not path.exists():
        sys.exit(f"no such paper: {paper_id} (run `uv run kb/kb.py ls`)")
    person = whoami(cfg, who)
    meta, body = parse_note(path)
    head, notes = split_notes(body)
    bullet = f"- **{datetime.now().date().isoformat()}** — {text.strip()}"
    mine = next((n for n in notes if n["who"].lower() == person.lower()), None)
    if mine:
        mine["md"] = (mine["md"] + "\n" + bullet).strip()
    else:
        notes.append({"who": person, "md": bullet})
    new_body = head.rstrip() + "".join(f"\n\n## Notes — {n['who']}\n\n{n['md']}" for n in notes)
    write_note(path, meta, new_body)
    print(f"✓ note added to {path.name} as {person}")
    build()


def link(from_id: str, to_id: str, why: str = "") -> None:
    """Manually add an edge from_id → to_id."""
    path = PAPERS_DIR / f"{from_id}.md"
    meta, body = parse_note(path)
    rel = meta.get("related") or []
    if any(r.get("id") == to_id for r in rel):
        print("edge already exists")
    else:
        rel.append({"id": to_id, "why": why})
        meta["related"] = rel
        write_note(path, meta, body)
        print(f"✓ {from_id} → {to_id}")
    build()


def ls() -> None:
    """List every paper with its category, relevance and link count."""
    for p in sorted(load_papers(), key=lambda x: (-int(x.get("relevance") or 0), x["id"])):
        print(f"{p.get('relevance', '?')}  {p.get('category', ''):13} "
              f"{len(p.get('related') or []):2}→  {p['id']:42} {p.get('title', '')[:60]}")


# --------------------------------------------------------------------------- build


INLINE = [
    (re.compile(r"`([^`]+)`"), r"<code>\1</code>"),
    (re.compile(r"\[\[([a-z0-9\-]+)\]\]"), r'<span class="to" data-go="\1">\1</span>'),
    (re.compile(r"\[([^\]]+)\]\(([^)]+)\)"), r'<a href="\2" target="_blank" rel="noopener">\1</a>'),
    (re.compile(r"\*\*([^*]+)\*\*"), r"<b>\1</b>"),
    (re.compile(r"(?<![\*\w])\*([^*]+)\*"), r"<i>\1</i>"),
]


def md_to_html(md: str) -> str:
    """Tiny markdown subset — headings, bullets, paragraphs, inline marks. No new deps."""
    out, bullets = [], False
    for raw in md.splitlines():
        line = html.escape(raw.rstrip())
        for pat, rep in INLINE:
            line = pat.sub(rep, line)
        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            if not bullets:
                out.append("<ul>")
                bullets = True
            out.append(f"<li>{stripped[2:]}</li>")
            continue
        if bullets:
            out.append("</ul>")
            bullets = False
        if not stripped:
            continue
        if m := re.match(r"^(#{2,4})\s+(.*)$", stripped):
            out.append(f"<h3>{m.group(2)}</h3>" if len(m.group(1)) == 2
                       else f"<h4>{m.group(2)}</h4>")
        else:
            out.append(f"<p>{stripped}</p>")
    if bullets:
        out.append("</ul>")
    return "\n".join(out)


def _first_section(body: str, heading: str) -> str:
    """Pull the text under '## heading' out of a note body."""
    m = re.search(rf"^##\s+{re.escape(heading)}\s*$(.*?)(?=^##\s|\Z)", body, re.M | re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def _byline(p: dict) -> str:
    authors = p.get("authors") or []
    who = authors[0].split(",")[0] if authors else ""
    if len(authors) > 1:
        who += " et al."
    bits = [b for b in (who, str(p.get("year") or ""), p.get("venue") or "") if b]
    return " · ".join(bits)


def _short(p: dict) -> str:
    if p.get("short"):
        return p["short"]
    authors = p.get("authors") or []
    who = authors[0].split(",")[0] if authors else ""
    title = (p.get("title") or p["id"]).split(":")[0]
    title = textwrap.shorten(title, width=44, placeholder="…")
    return f"{who} {p.get('year') or ''} — {title}".strip(" —")


def build(open_after: bool = False) -> None:
    """Regenerate kb/index.html from the markdown notes."""
    cfg = load_config()
    cats = cfg.get("categories") or {}
    papers, people = [], set()
    for p in load_papers():
        body = p["_body"]
        head, notes = split_notes(body)
        people.update(n["who"] for n in notes)
        papers.append({
            "notes": [{"who": n["who"], "html": md_to_html(n["md"])} for n in notes],
            "id": p["id"],
            "title": p.get("title") or p["id"],
            "short": _short(p),
            "byline": _byline(p),
            "url": p.get("url") or "",
            "category": p.get("category") or "foundation",
            "color": (cats.get(p.get("category")) or {}).get("color", "#8a8a92"),
            "takeaway": p.get("takeaway") or "",
            "tags": p.get("tags") or [],
            "relevance": int(p.get("relevance") or 3),
            "status": p.get("status") or "unread",
            "related": p.get("related") or [],
            "file": p["_file"],
            "body": md_to_html(head),
            "haystack": " ".join([
                p.get("title") or "", p.get("takeaway") or "", " ".join(p.get("tags") or []),
                " ".join(p.get("authors") or []), p.get("category") or "", body,
            ]).lower(),
        })
    data = {"papers": papers, "categories": cats, "people": sorted(people)}
    html_out = (re.sub(r"/\*__KB_DATA__\*/.*?/\*__KB_DATA_END__\*/",
                       lambda _: json.dumps(data, ensure_ascii=False),
                       TEMPLATE.read_text(), flags=re.S)
                .replace("__KB_TITLE__", html.escape(cfg.get("title", "knowledge base")))
                .replace("__KB_BUILT__", datetime.now().strftime("%Y-%m-%d %H:%M")))
    INDEX.write_text(html_out)
    n_links = sum(len(p["related"]) for p in papers)
    print(f"✓ built {INDEX.relative_to(KB_DIR.parent)} — {len(papers)} papers, {n_links} links")
    if open_after:
        open_kb()


def open_kb() -> None:
    """Build if needed, then open the graph in your browser."""
    if not INDEX.exists():
        build()
    webbrowser.open(INDEX.as_uri())


if __name__ == "__main__":
    fire.Fire({"add": add, "note": note, "build": build, "open": open_kb, "ls": ls,
               "link": link, "resummarize": resummarize})
