# ABOUTME: Minimal vast.ai REST client for renting a Docker-capable VM to drive laptop-bound Docker evals (oai_hf):
# ABOUTME: balance, offer search, create, show, destroy. Run: uv run python scratch/qwen38/vast.py <balance|offers|create ID|show ID|destroy ID|mine>
import json
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()
API = "https://console.vast.ai/api/v0"
H = {"Authorization": f"Bearer {os.environ['VAST_API_KEY']}"}
# Ubuntu VM image: Docker can create compose networks inside a VM (containers on vast cannot, like RunPod).
VM_IMAGE = os.environ.get(
    "VAST_VM_IMAGE", "docker.io/vastai/kvm:ubuntu_cli_22.04-2025-11-21"
)


def call(method, path, **kw):
    r = requests.request(method, f"{API}{path}", headers=H, timeout=60, **kw)
    if r.status_code >= 400:
        raise SystemExit(f"{r.status_code} {method} {path}: {r.text[:500]}")
    return r.json()


def main(cmd, *args):
    if cmd == "balance":
        u = call("GET", "/users/current/")
        print(f"credit ${u.get('credit', 0):.2f}  balance ${u.get('balance', 0):.2f}")
    elif cmd == "offers":
        q = {
            "cpu_cores_effective": {"gte": 16},
            "cpu_ram": {"gte": 48000},
            "disk_space": {"gte": 150},
            "rentable": {"eq": True},
            "verified": {"eq": True},
            "vms_enabled": {"eq": True},
            "reliability2": {"gte": 0.98},
            "inet_down": {"gte": 300},
            "type": "on-demand",
            "order": [["dph_total", "asc"]],
            "limit": 15,
        }
        for o in call("POST", "/bundles/", json=q)["offers"]:
            print(
                f"{o['id']:>10} ${o['dph_total']:.3f}/h  cpu {o['cpu_cores_effective']:.0f}  ram {o['cpu_ram'] / 1024:.0f}G  "
                f"disk {o['disk_space']:.0f}G  gpu {o['num_gpus']}x{o.get('gpu_name')}  down {o['inet_down']:.0f}Mb  "
                f"{o.get('geolocation')}  rel {o['reliability2']:.3f}"
            )
    elif cmd == "create":
        body = {
            "client_id": "me",
            "image": VM_IMAGE,
            "disk": 150,
            "runtype": "ssh_direct ssh_proxy",
            "label": "kn-oaihf-docker-driver",
        }
        print(json.dumps(call("PUT", f"/asks/{args[0]}/", json=body)))
    elif cmd == "show":
        i = call("GET", f"/instances/{args[0]}/")["instances"]
        keep = (
            "id",
            "actual_status",
            "intended_status",
            "status_msg",
            "ssh_host",
            "ssh_port",
            "public_ipaddr",
            "dph_total",
            "cpu_cores_effective",
            "cpu_ram",
            "start_date",
            "label",
            "ports",
        )
        print(json.dumps({k: i.get(k) for k in keep}, indent=1, default=str))
    elif cmd == "mine":
        for i in call("GET", "/instances/")["instances"]:
            print(
                i["id"],
                i.get("label"),
                i.get("actual_status"),
                f"${i.get('dph_total', 0):.3f}/h",
                i.get("ssh_host"),
                i.get("ssh_port"),
            )
    elif cmd == "destroy":
        print(json.dumps(call("DELETE", f"/instances/{args[0]}/")))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(*sys.argv[1:])
