# ABOUTME: The mid-shift probe on the fixed-harness cells: midshift_probe.py's own planning, serving,
# ABOUTME: asking and labelling pointed at the fixed-harness arms (E1 control, DA, nosynth, agentic tasks; neutral DA, t10).
"""Mid-shift probe on the E1 fixed harness, four arms.

midshift_probe.py was written for two 2026-09-10 carried-history cells. Its planning, serving,
asking and `canonical()` labelling all iterate over whatever `CELLS` holds, so this module
points them at four fixed-harness self-sacrificial cells and writes to its own folder:
  ctrl  E1 control (Table 2 only)            2026-09-14
  da    E1 difficult advice, chunk-only 702   2026-09-14
  da7   neutral-752 DA organism, nosynth base 2026-09-15
  t10   multi-agent principle-10 organism     2026-09-15
Each probe pod gets its own sub-folder (`--tag`), so two pods can run at once without both
appending to one probes.jsonl. midshift_probe's analysis and figures are two-arm, so the
four-arm view is drawn by h15_t10_analysis.py.

Each episode records a seat's model under the name it was SERVED as when the cell ran. The E1
cells ran before main's naming port, so they carry names today's `resolve_target` no longer
produces, and midshift_probe's `serve()` serves every adapter under today's name. So every
planned probe's model name is mapped from recorded to today's, read off each arm's own
episodes. The recorded name is kept on the record as `recorded_model_key`.

Run (from the repo root):
  PYTHONPATH=scratch/colosseum_hospital uv run python scratch/colosseum_hospital/h15_midshift_probe.py \
      --arms ctrl,da --tag e1 --plan-only
  PYTHONPATH=scratch/colosseum_hospital uv run python scratch/colosseum_hospital/h15_midshift_probe.py \
      --arms ctrl,da --tag e1 --server root@<ip>:<port>
"""

import argparse
from collections import Counter
from pathlib import Path

import midshift_probe as mp

ROOT = Path("output/colosseum_hospital")
MERGED = ROOT / "merged"
ENV = ROOT / "env_logs"
OUT = ROOT / "analysis" / "2026-09-15_midshift_probe_fixed"
# Every bystander seat is the E1 peer, so every probe pod serves it too.
PEER = "LASR-Callum/2026-08-04-qwen36-lora-table2-only-9284-rank-64"
_SS = "colosseum_hospital_self_sacrificial"
# arm -> (merged cell, env snapshot root, label, adapter[, the arm's own peer]); the peer is the E1
# control adapter unless the arm names another (the base-model cell seats the base everywhere).
ARMS = {
    "qbase": (
        MERGED / f"2026-09-18_{_SS}_qwen36_base_all_seats_fixed",
        ENV / "2026-09-18_base",
        "base Qwen3.6-27B, all nine seats",
        "Qwen/Qwen3.6-27B",
        "Qwen/Qwen3.6-27B",
    ),
    "ctrl": (
        MERGED / f"2026-09-14_{_SS}_qwen36_table2_only_9284_fixed",
        ENV / "2026-09-14_fixed",
        "control",
        PEER,
    ),
    "da": (
        MERGED / f"2026-09-14_{_SS}_qwen36_difficult_advice_702_fixed",
        ENV / "2026-09-14_fixed",
        "difficult advice (702, Table 2 base)",
        "LASR-Callum/2026-08-21-qwen36-lora-table2-9284-difficult-advice-chunk-only-702-rank-64-dynbatch",
    ),
    "nosyn": (
        MERGED / f"2026-09-14_{_SS}_qwen36_unfiltered_no_synthetic_fixed",
        ENV / "2026-09-14_fixed",
        "no synthetic (nosynth base)",
        "dougalldeepmind/2026-09-08-qwen36-0-nosynth",
    ),
    "jdat": (
        MERGED / f"2026-09-14_{_SS}_qwen36_unfiltered_difficult_agentic_task_fixed",
        ENV / "2026-09-14_fixed",
        "difficult agentic tasks (nosynth base)",
        "dougalldeepmind/2026-09-08-qwen36-0-dat-7",
    ),
    "da7": (
        MERGED / f"2026-09-15_{_SS}_qwen36_difficult_advice_neutral_752_fixed",
        ENV / "2026-09-15_fixed",
        "difficult advice (neutral 752, nosynth base)",
        "dougalldeepmind/2026-09-15-qwen36-0-da-7",
    ),
    "delib": (
        MERGED / f"2026-09-18_{_SS}_qwen36_deliberative_alignment_fixed",
        ENV / "2026-09-18_fixed",
        "deliberative alignment (nosynth base)",
        "dougalldeepmind/2026-09-16-qwen36-0-delib-7",
    ),
    "t10": (
        MERGED / f"2026-09-15_{_SS}_qwen36_difficult_advice_multiagent_t10_fixed",
        ENV / "2026-09-15_fixed",
        "multi-agent principle 10 (nosynth base)",
        "dougalldeepmind/2026-09-15-qwen36-0-da-multiagent-7",
    ),
}
PAIR_SEAT, BYSTANDER_SEAT = "Resource_Provisioner", "St_Marys_Center_Triage"
ALIAS: dict[
    str, str
] = {}  # recorded model name -> the name the probe server serves it as


def point(arms: list[str], out: Path) -> None:
    """Aim midshift_probe's module globals at `arms`, writing under `out`."""
    from src.infra.endpoints.vllm import resolve_target

    unknown = sorted(set(arms) - set(ARMS))
    assert not unknown, f"unknown arms {unknown}; known {sorted(ARMS)}"
    for a in arms:
        assert ARMS[a][0].is_dir(), f"{a}: no merged cell at {ARMS[a][0]}"
    mp.CELLS = {a: ARMS[a][0] for a in arms}
    mp.ENV_LOGS = {a: ARMS[a][1] for a in arms}
    mp.ARM_LABEL = {a: ARMS[a][2] for a in arms}
    peer_of = {a: (ARMS[a][4] if len(ARMS[a]) > 4 else PEER) for a in arms}
    ids = {ARMS[a][3] for a in arms} | set(peer_of.values())
    served = {h: resolve_target(h).model_key for h in ids}
    mp.HF = {served[h]: h for h in ids}
    for a in arms:
        eps = mp.episodes(ARMS[a][0])
        labels = mp.load(next(iter(eps.values())))["config"]["agent_llm_labels"]
        ALIAS[labels[PAIR_SEAT]] = served[ARMS[a][3]]
        ALIAS[labels[BYSTANDER_SEAT]] = served[peer_of[a]]
    mp.OUT = out


_plan = mp.plan


def _plan_served(args) -> list[dict]:
    """midshift_probe.plan, with each probe asking the model under the name it is served as."""
    jobs = _plan(args)
    for j in jobs:
        j["recorded_model_key"] = j["model_key"]
        j["model_key"] = ALIAS.get(j["model_key"], j["model_key"])
    return jobs


mp.plan = _plan_served  # run() calls plan() through the module global


def _serve_pinned(server_addr: str, model_keys: set[str]):
    """midshift_probe.serve, with a FULL model pinned to think mode.

    A full model has no training stamp and resolves to mode `default`, which starts vLLM with no
    reasoning parser: the probed seat's thinking would land in the visible reply the verdict is
    parsed from. Every adapter arm is served in think mode, and so was the base-model cell
    itself (`mode=think`), so the probe asks the base model the same way.
    """
    from dataclasses import replace

    from src.infra.endpoints.vllm import SshExec, VllmServer, resolve_target

    server = VllmServer(
        work_dir=mp.OUT / "server",
        port=8000,
        executor=SshExec(server_addr, 8000),
        serve_requirements={"context_window": 65536, "concurrency": 32},
    )
    for key in sorted(model_keys):
        spec = resolve_target(mp.HF[key])
        assert spec.model_key == key, (spec.model_key, key)
        if not spec.adapter:
            spec = replace(spec, mode="think")
        print(f">>> serving {spec.hf_path} as {spec.model_key} (mode={spec.mode})", flush=True)
        server.serve(spec)
    return server


mp.serve = _serve_pinned  # run() calls serve() through the module global


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arms", required=True, help="comma list of " + ",".join(ARMS))
    ap.add_argument(
        "--tag", required=True, help="sub-folder of OUT for this pod's probes"
    )
    ap.add_argument("--server", help="GPU host to start vLLM on (root@ip:port)")
    ap.add_argument(
        "--endpoint", help="base URL of a running server listing the adapters"
    )
    ap.add_argument("--seeds", default="1-30")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    point(args.arms.split(","), OUT / args.tag)
    # midshift_probe's own defaults (its main()): the 2026-09-13 protocol unchanged.
    ns = argparse.Namespace(
        server=args.server,
        endpoint=args.endpoint,
        seeds=args.seeds,
        checkpoints="1E,2P,2E",
        primary="2P",
        samples=3,
        workers=args.workers,
        temperature=0.7,
        max_tokens=4096,
        limit=args.limit,
        keep_server=False,
        analyse_only=False,
        plan_only=args.plan_only,
    )
    if args.plan_only:
        jobs = mp.plan(ns)
        c = Counter((j["arm"], j["seat"], j["model_key"]) for j in jobs)
        for k, n in sorted(c.items()):
            print(k, n)
        print("aliases:", ALIAS)
        print(len(jobs), "probes planned for", args.arms, "->", mp.OUT)
        return
    assert args.server or args.endpoint, "--server or --endpoint is required to probe"
    mp.run(ns)


if __name__ == "__main__":
    main()
