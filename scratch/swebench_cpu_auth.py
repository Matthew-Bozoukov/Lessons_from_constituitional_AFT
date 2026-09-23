# ABOUTME: Authenticate the receipted Vast CPU VM to Docker Hub using the user's .env.
# ABOUTME: Tokens travel only over SSH stdin and are never printed or placed in command arguments.
import argparse
import ipaddress
import json
from pathlib import Path
import shlex
import subprocess

from dotenv import dotenv_values
from vastai import VastAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    args = parser.parse_args()
    values = dotenv_values(args.env)
    required = ["VAST_API_KEY", "DOCKERHUB_USERNAME", "DOCKERHUB_TOKEN"]
    missing = [k for k in required if not (values.get(k) or "").strip()]
    if missing:
        raise SystemExit("Missing .env entries: " + ", ".join(missing))
    receipt = json.loads(args.receipt.read_text())
    client = VastAI(api_key=values["VAST_API_KEY"].strip())
    instance = client.show_instance(int(receipt["instance_id"]))
    assert instance and instance["label"] == receipt["label"], "VM ownership mismatch"
    assert instance["actual_status"] == "running", "VM must already be running"
    address = str(ipaddress.ip_address(instance["public_ipaddr"]))
    port = int(instance["ports"]["22/tcp"][0]["HostPort"])
    assert 0 < port < 65536
    remote = """import json,os,pathlib,subprocess,sys
os.umask(0o077)
credentials=json.load(sys.stdin)
result=subprocess.run(['docker','login','--username',credentials['username'],'--password-stdin'],
    input=credentials['token']+'\\n',text=True,capture_output=True,timeout=90)
if result.returncode:
    print('Docker login failed; verify username/token and account access.',file=sys.stderr)
    sys.exit(result.returncode)
path=pathlib.Path.home()/'.docker/config.json'
path.chmod(0o600)
print('Docker login succeeded; credential file permissions are 0600.')
"""
    cmd = ["ssh", "-i", str(args.identity.expanduser()), "-o", "BatchMode=yes",
           "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=yes",
           "-p", str(port), f"root@{address}", "python3 -c " + shlex.quote(remote)]
    payload = {"username": values["DOCKERHUB_USERNAME"].strip(),
               "token": values["DOCKERHUB_TOKEN"].strip()}
    subprocess.run(cmd, input=json.dumps(payload), text=True, check=True, timeout=120)


if __name__ == "__main__":
    main()
