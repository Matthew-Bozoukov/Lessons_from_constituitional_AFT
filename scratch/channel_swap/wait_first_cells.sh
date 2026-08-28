#!/usr/bin/env bash
# ABOUTME: Wait until the first few ODCV scenarios of a pass have landed, then print their statuses,
# ABOUTME: so a silent no-transcript failure is caught minutes in, not at the end of the pass.
# Run: bash scratch/channel_swap/wait_first_cells.sh <rollouts log> [min_cells] [max_minutes]
set -u
LOG="${1:?rollouts log}"
MIN="${2:-4}"
MAX="${3:-25}"
for i in $(seq 1 "$MAX"); do
  if [ -f "$LOG" ] && [ "$(grep -c 'ETA' "$LOG")" -ge "$MIN" ]; then break; fi
  sleep 60
done
grep "ETA\|ISSUE\|excluding\|scenarios=" "$LOG" | head -n 10
