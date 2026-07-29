#!/usr/bin/env python3
"""Anthropic cost ledger across all Petri eval logs.

Prices per million tokens (recorded 2026-07-29):
  claude-sonnet-5   in 2.00 (intro, to 2026-08-31)  out 10.00
  claude-opus-5     in 5.00                          out 25.00
  claude-haiku-4-5  in 1.00                          out  5.00
Cache writes bill at 1.25x input; cache reads at 0.10x input.
"""
import glob
import sys

from inspect_ai.log import read_eval_log

PRICES = {
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

dirs = sys.argv[1:] or ["logs/petri-pilot", "logs/petri-focused",
                        "logs/petri-focused-conc1-partial"]

grand = 0.0
grand_by_model = {}
for d in dirs:
    for f in sorted(glob.glob(f"{d}/*.eval")):
        try:
            log = read_eval_log(f)
        except Exception:
            continue
        usage = getattr(getattr(log, "stats", None), "model_usage", {}) or {}
        run_total = 0.0
        for model, u in usage.items():
            key = next((k for k in PRICES if k in model), None)
            if not key:
                continue
            pin, pout = PRICES[key]
            inp = getattr(u, "input_tokens", 0) or 0
            out = getattr(u, "output_tokens", 0) or 0
            cw = getattr(u, "cache_creation_input_tokens", 0) or 0
            cr = getattr(u, "cache_read_input_tokens", 0) or 0
            cost = (inp * pin + cw * pin * 1.25 + cr * pin * 0.10 + out * pout) / 1e6
            run_total += cost
            grand_by_model[key] = grand_by_model.get(key, 0.0) + cost
        n = len(log.samples or [])
        print(f"{f.split(chr(92))[-1][:44]:<46} samples={n:<4} ${run_total:7.2f}"
              + (f"  (${run_total/n:.2f}/audit)" if n else ""))
        grand += run_total

print()
for m, c in sorted(grand_by_model.items(), key=lambda x: -x[1]):
    print(f"  {m:<20} ${c:7.2f}")
print(f"\nTOTAL ANTHROPIC (Petri logs): ${grand:.2f}")
print("plus eval-checksum judge ~$0.22 measured separately")
print(f"\nagainst MAX_ANTHROPIC_SPEND_USD = $120.00  ->  "
      f"{(grand+0.22)/120*100:.0f}% used, ${120-grand-0.22:.2f} remaining")
