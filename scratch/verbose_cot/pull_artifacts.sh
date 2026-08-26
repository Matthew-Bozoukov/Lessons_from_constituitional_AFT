#!/usr/bin/env bash
# ABOUTME: Continuously pull ODCV transcripts off the ephemeral vast boxes to this machine.
# ABOUTME: Run: bash scratch/verbose_cot/pull_artifacts.sh [once]
#
# The boxes are credential-free on purpose, so odcv_box_run's push-to-HF-per-pass cannot
# authenticate and records a `publish_error` instead. That does not stop the RUN, but it
# leaves every transcript living only on a rented VM that evaporates when the run ends.
# docs/GOTCHAS.md is explicit that a box's disk is not storage and artifacts must come off
# CONTINUOUSLY rather than at the end, so this closes the gap from the side that HAS the
# credentials: pull here, publish from here.
#
# tar-over-ssh rather than rsync, because rsync is not installed on this machine and
# transcripts are small text — re-pulling them whole each cycle is cheaper than making
# rsync a prerequisite of the run finishing safely.
set -uo pipefail

KEY=~/.ssh/msm_audit
DEST=output/odcv_verbose/pulled

BOXES=(
  "rows-a|184.144.255.144|18720"
  "rows-b|199.68.217.31|43034"
  "tok-a|184.144.154.180|43306"
  "tok-b|184.144.255.144|57412"
)

pull_one () {
  local id="$1" ip="$2" port="$3" rc n
  mkdir -p "$DEST/$id"
  # A box with no run dir yet fails here, which is normal early on rather than an error.
  ssh -i "$KEY" -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20 \
      -p "$port" "root@$ip" \
      'cd /root/work/output/odcv_bench 2>/dev/null && tar cf - . 2>/dev/null' \
    | tar xf - -C "$DEST/$id" 2>/dev/null
  rc=${PIPESTATUS[0]}
  # Count TRANSCRIPTS, not directories: a scenario that finishes `ok` while writing no
  # messages_record.txt is the silent failure that once cost a full run, and a directory
  # count would hide precisely that.
  n=$(find "$DEST/$id" -name "messages_record.txt" -size +0 2>/dev/null | wc -l)
  echo "[$id] ssh=$rc transcripts=$n"
}

run_cycle () {
  for b in "${BOXES[@]}"; do
    IFS='|' read -r id ip port <<<"$b"
    pull_one "$id" "$ip" "$port"
  done
  echo "[total] $(find "$DEST" -name 'messages_record.txt' -size +0 2>/dev/null | wc -l) transcripts held locally"
}

if [ "${1:-loop}" = "once" ]; then run_cycle; exit 0; fi
while true; do run_cycle; sleep 300; done
