<!-- ABOUTME: Rebuild and use the read-only viewer for both pinned synthetic releases. -->
<!-- ABOUTME: Browser checks cover filtering, exact text display, review-note export and narrow layouts; no model calls. -->

# Dataset reading room

Run `uv run python -m scratch.dataset_refresh.build_explorer --config configs/data/audits/dataset-explorer.yaml`
from the repository root after staging the two pinned source payloads and final
comparison census at the config's paths. Source hashes must match; mismatches and
non-verbatim inspection quotes fail. The output is a self-contained HTML browser
of all1432 synthetic conversations, plus a72-entry category/trait-spanning inline
preview under1MB. No text is summarized or rewritten in the conversation display.

The full viewer opens directly as a local HTML file. For an in-app browser, serve
the output directory with `uv run python -m http.server 8766 --bind 127.0.0.1 --directory output/2026-09-15_dataset_explorer`.
Search covers system, user, reasoning, final answer, exact assigned domains and
stable IDs. Categories use the existing census mappings. Marks/notes live only
in the browser session; export JSON before closing or reloading.

Eight specifically reviewed examples have qualitative inspection notes. Three
have residual textual/count discrepancies, one has an unverified technical
premise, and four illustrate style/comparability preferences. Keyword and length
filters are screens, not defect judgments. This purposive selection does not
estimate error prevalence. Published datasets remain unchanged.

Verified with bundled Chromium: full1432 and sample72 initialization; dataset,
category, principle, factual/style and text filters; selected conversation;
sorting/pagination; mark/note retention and exported JSON; empty state; no390px
horizontal overflow; sandboxed inline interactions; no JavaScript errors. Source
payload SHA checks passed. No model calls, training or evaluation were performed.
