#!/usr/bin/env bash
# ABOUTME: Bootstrap the four verbose-CoT ODCV boxes in parallel: phase 1 prepares each box
# ABOUTME: and prints the key its tunnel will use; phase 2 brings the tunnel up once authorized.
#
#   bash scratch/verbose_cot/bootstrap_boxes.sh prep     # packages, repo, docker preflight
#   bash scratch/verbose_cot/bootstrap_boxes.sh authorize # push box keys onto their pods
#   bash scratch/verbose_cot/bootstrap_boxes.sh tunnel    # systemd tunnel -> that arm's pod
#
# Boxes are paired ACROSS physical hosts on purpose: 1 and 2 landed on the same machine
# (184.144.255.144), so putting both on one arm would let a single host failure take that
# whole arm out. Each arm therefore gets one box from that host and one from elsewhere.
set -uo pipefail

SHA=03e384013c52542ddb8586f781da8292d05ef4bc
KEY=~/.ssh/msm_audit
CFG_ROWS=configs/eval/odcv_bench_da716_verbose_rows_r64_incentivized_5x30.yaml
CFG_TOK=configs/eval/odcv_bench_da716_verbose_tokens_r64_incentivized_5x30.yaml

# label | ip | ssh port | arm | config | pod ip | pod ssh port
BOXES=(
  "rows-a|184.144.255.144|18720|rows|$CFG_ROWS|212.247.220.117|14774"
  "rows-b|199.68.217.31|43034|rows|$CFG_ROWS|212.247.220.117|14774"
  "tok-a|184.144.154.180|43306|tokens|$CFG_TOK|198.145.108.71|18431"
  "tok-b|184.144.255.144|57412|tokens|$CFG_TOK|198.145.108.71|18431"
)

sshx () { ssh -i "$KEY" -o StrictHostKeyChecking=no -o BatchMode=yes \
              -o ConnectTimeout=20 -p "$2" "root@$1" "${@:3}"; }

case "${1:?usage: prep|authorize|tunnel}" in

prep)
  for b in "${BOXES[@]}"; do
    IFS='|' read -r id ip port arm cfg pip pport <<<"$b"
    (
      scp -i "$KEY" -o StrictHostKeyChecking=no -P "$port" \
          scratch/odcv_box_bootstrap.sh "root@$ip:/root/" >/dev/null 2>&1 \
        || { echo "[$id] SCP FAILED"; exit 1; }
      # pod_ip=NONE: prepare and PREFLIGHT without a tunnel. The preflight is the gate --
      # a box whose docker cannot build produces a run that reads clean and is missing
      # ~21% of its cells.
      sshx "$ip" "$port" "bash /root/odcv_box_bootstrap.sh $SHA NONE 0 $cfg $id" \
           > "output/odcv_verbose/prep_$id.log" 2>&1
      rc=$?
      key="$(grep -a '^PUBKEY: ' "output/odcv_verbose/prep_$id.log" | tail -1 | cut -d' ' -f2-)"
      echo "[$id] exit=$rc $(grep -acE 'PREFLIGHT OK' "output/odcv_verbose/prep_$id.log" \
            >/dev/null && echo 'preflight OK' || echo 'PREFLIGHT NOT CONFIRMED')"
      [ -n "$key" ] && echo "$key" > "output/odcv_verbose/pubkey_$id"
    ) &
  done
  wait
  echo "--- prep done; logs in output/odcv_verbose/prep_*.log ---"
  ;;

authorize)
  # Each box's key goes ONLY onto the pod of its own arm.
  for b in "${BOXES[@]}"; do
    IFS='|' read -r id ip port arm cfg pip pport <<<"$b"
    f="output/odcv_verbose/pubkey_$id"
    [ -s "$f" ] || { echo "[$id] no pubkey captured — rerun prep"; continue; }
    ssh -i "$KEY" -o StrictHostKeyChecking=no -o BatchMode=yes -p "$pport" "root@$pip" \
        "mkdir -p ~/.ssh && grep -qxF '$(cat "$f")' ~/.ssh/authorized_keys 2>/dev/null \
         || echo '$(cat "$f")' >> ~/.ssh/authorized_keys; wc -l < ~/.ssh/authorized_keys" \
      && echo "[$id] key authorized on pod $pip" || echo "[$id] AUTHORIZE FAILED"
  done
  ;;

tunnel)
  for b in "${BOXES[@]}"; do
    IFS='|' read -r id ip port arm cfg pip pport <<<"$b"
    (
      sshx "$ip" "$port" "bash /root/odcv_box_bootstrap.sh $SHA $pip $pport $cfg $id" \
           > "output/odcv_verbose/tunnel_$id.log" 2>&1
      echo "[$id] exit=$? $(tail -2 "output/odcv_verbose/tunnel_$id.log" | tr '\n' ' ' | cut -c1-120)"
    ) &
  done
  wait
  ;;
esac
