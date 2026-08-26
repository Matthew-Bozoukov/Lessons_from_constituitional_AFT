#!/usr/bin/env bash
# ABOUTME: Block until every ODCV box has finished its passes, then summarise what landed.
set -uo pipefail
KEY=~/.ssh/msm_audit
BOXES=("rows-a|184.144.255.144|18720" "rows-b|199.68.217.31|43034" "tok-a|184.144.154.180|43306" "tok-b|184.144.255.144|57412")
done_count=0
while [ "$done_count" -lt 4 ]; do
  done_count=0
  for b in "${BOXES[@]}"; do
    IFS='|' read -r id ip port <<<"$b"
    # A box is done when its supervisor has exited AND its log says so; either alone can
    # lie (a dead process with a stale log, or a log line from a previous attempt).
    r=$(timeout 30 ssh -i "$KEY" -o StrictHostKeyChecking=no -o BatchMode=yes \
          -o ConnectTimeout=15 -p "$port" "root@$ip" \
          'p=$(pgrep -cf odcv_box_run || true); c=$(grep -c "ALL PASSES COMPLETE" /root/odcv/run.log 2>/dev/null || echo 0); echo "$p:$c"' 2>/dev/null)
    [ "${r%%:*}" = "0" ] && [ "${r##*:}" != "0" ] && done_count=$((done_count+1))
  done
  [ "$done_count" -lt 4 ] && sleep 120
done
echo "ALL FOUR BOXES FINISHED"
for b in "${BOXES[@]}"; do
  IFS='|' read -r id ip port <<<"$b"
  echo "[$id] $(timeout 30 ssh -i "$KEY" -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -p "$port" "root@$ip" 'grep -ao "pass [0-9] done: .\{0,150\}" /root/odcv/run.log | tail -2' 2>/dev/null)"
done
