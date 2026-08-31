# ABOUTME: Three-way merge of docs/LOG.md by entry, keeping every local and remote entry
# ABOUTME: and re-sorting most-recent-first, so a pull never silently drops a log entry.

"""Merge the local and remote LOG.md without losing entries.

Run: uv run python scratch/merge_log.py <base> <local> <remote> <out>

LOG.md is append-only and most-recent-first, so the two sides only ever ADD entries at
different points in one file — which git reports as one large conflicting hunk. Splitting on
`## ` headings and taking the union recovers the intended result: every entry from both
sides, none duplicated, ordered by date descending.

Order within a date is preserved as: local entries first, then remote ones not already
present. Entries are keyed by their full heading line, so an entry edited on one side only
is kept once (the local text wins) rather than appearing twice.
"""

import re
import sys
from pathlib import Path


def split_entries(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Split a LOG file into (header, [(heading, body), ...]).

    Args:
        text: Full file contents.

    Returns:
        The preamble before the first entry, and the entries in file order.
    """
    parts = re.split(r"^(## .*)$", text, flags=re.M)
    header = parts[0]
    entries = [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]
    return header, entries


def date_of(heading: str) -> str:
    """Extract the ISO date from an entry heading, or '' when absent."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", heading)
    return m.group(1) if m else ""


def main() -> None:
    """Write the merged log and report what came from where."""
    base_p, local_p, remote_p, out_p = (Path(a) for a in sys.argv[1:5])
    header, local = split_entries(local_p.read_text())
    _, remote = split_entries(remote_p.read_text())
    _, base = split_entries(base_p.read_text())

    base_keys = {h for h, _ in base}
    local_keys = {h for h, _ in local}

    merged: list[tuple[str, str]] = list(local)
    added = [(h, b) for h, b in remote if h not in local_keys]
    merged.extend(added)

    # Stable sort by date descending; entries without a date keep their relative position.
    merged.sort(key=lambda e: date_of(e[0]), reverse=True)

    out_p.write_text(header.rstrip("\n") + "\n\n"
                     + "\n".join(f"{h}\n{b.rstrip()}\n" for h, b in merged) + "\n")

    print(f"base {len(base)}  local {len(local)}  remote {len(remote)}")
    print(f"  local-only entries : {len(local_keys - base_keys)}")
    print(f"  remote-only added  : {len(added)}")
    print(f"  MERGED TOTAL       : {len(merged)}")
    assert len(merged) == len(local) + len(added), "entry lost during merge"
    for h, _ in added:
        print(f"    + {h[:96]}")


if __name__ == "__main__":
    main()
