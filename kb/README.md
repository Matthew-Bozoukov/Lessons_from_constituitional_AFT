# kb — living paper map for the constitution-SFT project

One markdown file per paper, one command to add one, one HTML file to look at them. No Obsidian,
no server, no database. Everything (papers, links, everyone's notes) is plain text in git.

Self-contained: nothing in here imports from `src/`, and nothing in the repo imports from here.

```bash
uv run kb/kb.py open                     # look at the graph (builds first if needed)
```

## Add a paper

```bash
uv run kb/kb.py add "https://arxiv.org/abs/2605.24229"
```

Fetches the metadata, summarises with Sonnet 4.5 via OpenRouter (`OPENROUTER_API_KEY` from `.env`),
proposes links to papers already in the base, writes `kb/papers/<id>.md`, rebuilds `kb/index.html`.
Takes ~40s. Works with an arXiv id, an arXiv URL, or any blog/paper URL.

Sources with no fetchable text (internal docs, Google Docs, PDFs behind a login):

```bash
uv run kb/kb.py add "Title of the thing" --no_fetch --notes "$(pbpaste)"   # paste the text in
uv run kb/kb.py add "Title of the thing" --no_ai                          # blank stub to fill by hand
```

Useful flags: `--category eval-setup` to force the bucket, `--force` to overwrite,
`--model <openrouter-model>` to summarise with something else.

## Take notes

Notes are per person and live in the paper's own file, so they diff and review like code.

```bash
uv run kb/kb.py note jakkli-2026-constitution-audit "their tenet decomposition is the piece we want"
uv run kb/kb.py note <id> "..." --who Nika
```

`--who` defaults to `KB_USER` in the env, then `me:` in `config.yaml`, then your git user name.
Set yours once: `export KB_USER=Kunwar`.

Editing the markdown directly is equally fine — add a `## Notes — <Name>` section and write
whatever. `[[other-paper-id]]` in any note becomes a clickable link in the viewer.

## Everything else

```bash
uv run kb/kb.py ls                            # every paper, sorted by relevance
uv run kb/kb.py link <from-id> <to-id> --why "…"   # add an edge by hand
uv run kb/kb.py resummarize <id>              # re-run the summariser, keeps everyone's notes
uv run kb/kb.py build                         # regenerate index.html
```

## How papers are filed

`category` = **what this paper gives our project**, not what it is about (topic goes in `tags`).
That is the grouping in the sidebar and the node colour in the graph. Buckets live in
`config.yaml` — add a line to add a bucket:

| category | means |
|---|---|
| `foundation` | result we are directly replicating or extending |
| `eval-setup` | eval / harness / benchmark we want to reproduce or run |
| `data-recipe` | a way of generating training data we could borrow |
| `threat-model` | the failure mode we are trying to reduce |
| `contrast` | complicates or challenges our approach; negative results |
| `ours` | our own runs, writeups and results |

Each paper also carries a one-line `takeaway` — the "what we take from it" line shown in the
sidebar, the tooltip and the top of the reading pane.

## The viewer

`kb/index.html` is a single generated file with all the data inlined — open it directly, or commit
it and anyone can open it with no setup. Nodes are papers (size = relevance, hollow ring = unread),
arrows are "builds on / relates to" with the reason attached. Click a node or a sidebar row to
read; click the coloured chips to filter by bucket or by whose notes exist; `/` focuses search
(it searches summaries and notes too).

## Layout

```
kb/
  kb.py           the whole CLI
  config.yaml     title, categories+colours, summariser model, project context for the prompt
  template.html   viewer shell; kb.py injects the data into it
  papers/*.md     one file per paper — frontmatter + AI summary + per-person notes
  index.html      generated, committed, openable
```

## Keeping the summaries honest

The summariser is told to write "unclear from the abstract" rather than invent method details, and
`add` only ever gets the text it could actually fetch — an arXiv entry is the abstract only, so
those summaries are abstract-level. If a paper matters, read it and use `note` for what the
abstract could not tell you. Stubs (like the internal GDM post) are deliberately left `_todo_`
rather than filled with plausible-sounding text.
