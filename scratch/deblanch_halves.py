# ABOUTME: Delete the branching units from a restore_move run, then re-rate what survives.
# ABOUTME: Prompting the restorer not to branch does not work; deletion cannot be argued with.
"""Ablate outcome branches out of already-restored halves.

The aggressive restorer puts the alternative and the counterargument in at 100%, but
outcome_branches comes back in ~20-40% of halves and strengthening the prohibition in the
prompt changed nothing -- the same lesson the original ablation learned, where rewriting
alone left the target in 34-40% and only rewrite-then-delete reached 3.6%.

This enumerates every unit of each half and labels it TWICE: once for branching, once for
the moves that must survive. A unit that branches is deleted ONLY if it does not also carry
the alternative or the counterargument -- otherwise removing the branch would remove the
thing the restoration was for, and the protected unit is kept with its branch intact. The
result is re-rated on all nine moves plus coherence. It reuses the unit splitter,
deleter and specs from scratch/ablate_alternative.py, so the labelling contract is the one
already validated on the 676-row corpus.

Run: uv run python scratch/deblanch_halves.py --run output/restore_move_spot2/<dir>
"""
import json
import re
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.endpoints.openrouter import OpenRouterClient, map_threaded
from src.utils import extract_json, read_jsonl, timestamp, write_run_meta

sys.path.insert(0, str(Path(__file__).parent))
from ablate_alternative import COHERENCE, LABEL, SPECS, excise, units  # noqa: E402
from rate_inversion_moves import _rate  # noqa: E402


def _label_nums(body: str, n_units: int) -> set[int]:
    """Unit indices from a labeller reply, tolerating the shapes it actually returns.

    The prompt asks for {"offering": [...]}, but when the criteria name two moves the
    labeller sometimes splits them into a key per move -- and sometimes emits invalid JSON
    doing it ('{"offering": [7, "engaging": [2]}'). Every integer in the object is a unit
    that matched SOME criterion, which is exactly what the deletion wants, so all integer
    lists are unioned and a bare integer run is recovered from the raw text as a fallback.

    Args:
        body: The raw response content.
        n_units: Number of units offered, used to reject out-of-range labels.

    Returns:
        The set of matched unit indices.
    """
    nums: set[int] = set()
    try:
        obj = extract_json(body)
        stack = [obj]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                stack.extend(cur.values())
            elif isinstance(cur, list):
                stack.extend(cur)
            elif isinstance(cur, bool):
                continue
            elif isinstance(cur, int):
                nums.add(cur)
    except Exception:
        # Unparseable: take the integers out of the body directly rather than lose the half.
        nums = {int(x) for x in re.findall(r"\b\d{1,3}\b", body)}
    return {x for x in nums if 1 <= x <= n_units}


def main(run: str, move: str = "outcome_branches",
         protect: str = "offers_alternative,engages_counterargument",
         model: str = "google/gemini-3-flash-preview",
         rater: str = "google/gemini-3-flash-preview", workers: int = 16,
         out_dir: str = "output/deblanch"):
    """Delete branching units from every half of a restore_move run and re-rate.

    Args:
        run: A scratch/restore_move.py run directory containing halves.jsonl.
        move: Comma-separated moves whose units are deleted; defaults to the branches.
        protect: Comma-separated moves that veto a deletion. A unit carrying any of these is
            kept even when it also branches, so the deletion cannot undo the restoration.
            Pass "" to delete every branching unit regardless.
        model: OpenRouter model id for the unit labeller.
        rater: OpenRouter model id for the blind nine-move rater and coherence check.
        workers: Thread pool size.
        out_dir: Output directory root.
    """
    load_dotenv()
    targets = [m.strip() for m in (move if isinstance(move, str)
                                   else ",".join(move)).split(",") if m.strip()]
    for m in targets:
        assert m in SPECS, f"unknown move {m}; have {sorted(SPECS)}"
    keep = [m.strip() for m in (protect if isinstance(protect, str)
                                else ",".join(protect)).split(",") if m.strip()]
    for m in keep:
        assert m in SPECS, f"unknown protected move {m}; have {sorted(SPECS)}"
    crit = lambda ms: "\n".join(
        f"{i}. {SPECS[m][0]}: {SPECS[m][1]}\n   Do NOT count: {SPECS[m][2]}."
        for i, m in enumerate(ms, 1))

    halves = read_jsonl(Path(run) / "halves.jsonl")
    print(f"{len(halves)} halves from {run}; deleting {targets} units, "
          f"protecting {keep or 'nothing'}")
    client = OpenRouterClient()

    def label(i, ms):
        """Unit indices in half i that make any move in ms."""
        us = [u for u in units(halves[i]["after"]) if u.strip()]
        numbered = "\n".join(f"{j}. {u}" for j, u in enumerate(us, 1))
        content = LABEL.format(units=numbered, criteria=crit(ms))
        nums: set[int] = set()
        for attempt in (0, 1, 2):
            r = client.chat(model=model, temperature=0.0, max_tokens=900,
                            response_format={"type": "json_object"},
                            messages=[{"role": "user", "content": content + (
                                "\n\nSTRICT JSON ONLY: ONE key \"offering\", one flat list "
                                "of integers, no trailing commas." if attempt else "")}])
            nums = _label_nums(r.content or "", len(us))
            if nums or "[]" in (r.content or ""):
                break
        return nums, len(us)

    labelled = map_threaded(lambda i: label(i, targets), len(halves),
                            max_workers=workers, desc="labelling branches")
    protected = (map_threaded(lambda i: label(i, keep), len(halves),
                              max_workers=workers, desc="labelling protected")
                 if keep else [(set(), 0)] * len(halves))
    cut = []
    for h, (drop, total), (safe, _) in zip(halves, labelled, protected):
        spared = drop & safe
        text, removed = excise(h["after"], drop - safe)
        cut.append({**h, "deblanched": text, "units_total": total,
                    "units_branching": len(drop), "units_protected": len(safe),
                    "units_spared": len(spared), "units_dropped": removed,
                    "words_deblanched": len(text.split())})

    rated = map_threaded(
        lambda i: _rate(client, rater, cut[i]["deblanched"],
                        subject=cut[i]["half"]),
        len(cut), max_workers=workers, desc="re-rating")
    coh = map_threaded(
        lambda i: extract_json(client.chat(
            model=rater, temperature=0.0, max_tokens=300,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": COHERENCE.format(
                text=cut[i]["deblanched"])}]).content),
        len(cut), max_workers=workers, desc="coherence")
    for c, out, k in zip(cut, rated, coh):
        c["moves_deblanched"] = out["moves"]
        c["coherence"] = k["verdict"]
        c["coherence_note"] = k.get("note", "")

    d = Path(out_dir) / timestamp()
    d.mkdir(parents=True, exist_ok=True)
    (d / "halves.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cut))

    pct = lambda key, m, sub: 100 * sum(r[key][m] for r in sub) / len(sub)
    L = [f"# Deleting `{'+'.join(targets)}` units from `{run}`", "",
         f"{len(cut)} halves, labelled by `{model}`, re-rated blind by `{rater}`.", "",
         "| move | reasoning before -> after | reply before -> after |", "|---|--:|--:|"]
    for m in cut[0]["moves_after"]:
        cells = []
        for h in ("reasoning", "reply"):
            sub = [r for r in cut if r["half"] == h]
            cells.append(f"{pct('moves_after', m, sub):.0f}% -> "
                         f"{pct('moves_deblanched', m, sub):.0f}%" if sub else "-")
        mark = " **(deleted)**" if m in targets else ""
        L.append(f"| {m}{mark} | " + " | ".join(cells) + " |")
    broken = [c for c in cut if c["coherence"] != "coherent"]
    L += ["", f"units dropped: {sum(c['units_dropped'] for c in cut)} of "
              f"{sum(c['units_total'] for c in cut)}",
          f"branching units found: {sum(c['units_branching'] for c in cut)}; "
          f"SPARED because they also carry {'/'.join(keep) or 'nothing'}: "
          f"{sum(c['units_spared'] for c in cut)}",
          f"coherent after deletion: {len(cut) - len(broken)}/{len(cut)}"]
    if broken:
        L.append("")
        L.append("BROKEN after deletion — decide before using:")
        for c in broken:
            L.append(f"- {c['scenario_id']} {c['half']}: {c['coherence_note']}")
    (d / "results.md").write_text("\n".join(L) + "\n")
    write_run_meta(d, {"run": run, "move": targets, "protect": keep,
                       "model": model, "rater": rater},
                   {"n_halves": len(cut)})
    print("\n".join(L))
    print(f"\nwrote {d}/results.md")


if __name__ == "__main__":
    fire.Fire(main)
