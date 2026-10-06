# ABOUTME: Complete GPU-free qualification on a prepared native-Docker SWE-bench host.
# ABOUTME: Run: bash scratch/swebench_cpu_qualify.sh STAMP [CONFIG]; requires successful readiness.
set -euo pipefail
cd /srv/lasr/repo
stamp=${1:?unique qualification stamp required}
config=${2:-configs/eval/swebench_mini/lite.yaml}
case "$stamp" in *[!a-zA-Z0-9_-]*) exit 2;; esac
P=scratch/swebench_cpu_env/.venv/bin/python
A=src/eval/capabilities/swebench_mini/envs/agent/.venv/bin/python
while :; do
  state=$(systemctl show lasr-swebench-prepare.service -p ActiveState --value)
  case "$state" in active|activating|reloading) sleep 15;; *) break;; esac
done
[ "$(systemctl show lasr-swebench-prepare.service -p Result --value)" = success ]
$P -c 'import json; from pathlib import Path; assert json.loads(Path("/srv/lasr/runs/cpu-readiness/results/readiness.json").read_text())["status"] == "ready"'
$P -m pytest tests/test_swebench_fleet.py tests/test_swebench_session.py tests/test_swebench_admission.py tests/test_swebench_browser.py tests/test_swebench_cpu_watchdog.py tests/test_swebench_qualification.py tests/test_swebench_template_policy.py tests/test_eval_revision_pins.py tests/test_runpod_watchdog.py tests/test_runpod_pod.py tests/test_runpod_lifecycle.py tests/test_runpod_serve.py tests/test_huggingface.py scratch/test_swebench_lite.py scratch/swebench_plain_campaign/test_queue.py scratch/swebench_plain_campaign/test_stage_cpu.py -q > "/srv/lasr/runs/lifecycle-tests-$stamp.log" 2>&1
$A -m scratch.test_swebench_timeout_transport > "/srv/lasr/runs/transport-$stamp.log" 2>&1
$A -m unittest discover -s tests -p test_swebench_protocol.py -v > "/srv/lasr/runs/protocol-$stamp.log" 2>&1
$A -m scratch.swebench_protocol_template_check --config "$config" > "/srv/lasr/runs/template-$stamp.log" 2>&1
$P -m src.eval.capabilities.swebench_mini.fleet qualify-shell --config "$config"
$P -m scratch.swebench_cpu_load --output "/srv/lasr/runs/capacity-$stamp"
$P -m scratch.swebench_lite_smoke --config "$config" > "/srv/lasr/runs/smoke-$stamp.log" 2>&1
smoke_root=$($P -c 'import json,sys; from pathlib import Path; print(json.loads(Path(sys.argv[1]).read_text().splitlines()[-1])["root"])' "/srv/lasr/runs/smoke-$stamp.log")
$P -m scratch.swebench_qualify_recipe --config "$config" --load-dir "/srv/lasr/runs/capacity-$stamp" --smoke-root "$smoke_root" --test-log "/srv/lasr/runs/lifecycle-tests-$stamp.log" --transport-log "/srv/lasr/runs/transport-$stamp.log" --protocol-log "/srv/lasr/runs/protocol-$stamp.log" --template-proof output/swebench-protocol-template-latest.json
bash scratch/swebench_lite_install.sh
