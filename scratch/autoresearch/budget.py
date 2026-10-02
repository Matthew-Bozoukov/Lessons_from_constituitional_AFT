# ABOUTME: Local spend guard for the 2026-10-01 da-autoresearch night: a hard $ cap across RunPod pods named
# ABOUTME: `jamie-ar-*` and this key's OpenRouter usage, plus a ledger capping newly generated DA rows.
"""    uv run python scratch/autoresearch/budget.py init                    # baseline, once
       uv run python scratch/autoresearch/budget.py status
       uv run python scratch/autoresearch/budget.py check --reserve 8       # exit 1 if spent + 8 > cap
       uv run python scratch/autoresearch/budget.py rows --add 20 --what "smoke: ..."   # exit 1 past 150 rows
       uv run python scratch/autoresearch/budget.py watchdog               # loop; tears MY pods down at the cap

RunPod spend = sum over pods whose name starts with `jamie-ar-` of costPerHr x lifetime, read from the API every
sync (a pod that has gone is charged up to the last sync that saw it, so sync at least once a minute while pods
are up: the watchdog does). OpenRouter spend = this key's `usage` now minus its usage at `init` (it over-counts
if anything else uses the key tonight, which errs on the safe side). The watchdog only ever terminates pods
named `jamie-ar-*`: at the cap, or when one is older than MAX_POD_HOURS (a stuck boot nobody is awake to judge;
its boot log is saved first). It also writes output/autoresearch/STOP, which every pipeline checks before renting.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import fire
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv(".env")
CAP_USD = 240.0
STOP_MARGIN_USD = 6.0        # the watchdog acts this far below the cap: one pod-hour plus judge tail
MAX_ROWS = 150
MAX_POD_HOURS = 3.0
PREFIX = "jamie-ar-"
DIR = Path("output/autoresearch")
LEDGER = DIR / "budget.json"
STOP = DIR / "STOP"


def _load() -> dict:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else {}


def _save(d: dict) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2))
    tmp.replace(LEDGER)


def _openrouter_usage() -> float:
    r = requests.get("https://openrouter.ai/api/v1/key",
                     headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]}, timeout=30)
    r.raise_for_status()
    return float(r.json()["data"]["usage"])


def _parse(ts: str) -> float:
    ts = ts.replace(" +0000 UTC", "+00:00").replace(" ", "T", 1)
    head, _, tz = ts.partition("+")
    if "." in head:
        a, b = head.split(".")
        head = a + "." + (b + "000000")[:6]
    return datetime.fromisoformat(head + "+" + (tz or "00:00")).timestamp()


def sync() -> dict:
    from src.infra.runpod import active_pods
    d = _load()
    pods = d.setdefault("pods", {})
    now = time.time()
    for p in active_pods():
        if not str(p.get("name", "")).startswith(PREFIX):
            continue
        e = pods.setdefault(p["id"], {"name": p["name"], "rate": float(p.get("costPerHr") or 0),
                                      "start": _parse(str(p.get("createdAt")))})
        e["rate"] = float(p.get("costPerHr") or e["rate"])
        e["last_seen"] = now
    try:
        d["openrouter_now"] = _openrouter_usage()
    except Exception as exc:  # a network blip must not kill the watchdog; the last reading stands
        d["openrouter_error"] = str(exc)[:200]
    d["synced"] = now
    _save(d)
    return d


def _spent(d: dict) -> tuple[float, float]:
    pods = sum(e["rate"] * max(0.0, e.get("last_seen", e["start"]) - e["start"]) / 3600 for e in d.get("pods", {}).values())
    return pods * 1.03, max(0.0, d.get("openrouter_now", d.get("openrouter_base", 0.0)) - d.get("openrouter_base", 0.0))


def init() -> str:
    assert not LEDGER.exists(), f"{LEDGER} exists: the baseline is taken once"
    _save({"cap_usd": CAP_USD, "started": time.time(), "openrouter_base": _openrouter_usage(), "pods": {}, "rows": []})
    return status()


def status() -> str:
    d = sync()
    pod, orr = _spent(d)
    rows = sum(r["n"] for r in d.get("rows", []))
    live = [e["name"] for e in d["pods"].values() if time.time() - e.get("last_seen", 0) < 150]
    return (f"spent ${pod + orr:.2f} of ${CAP_USD:.0f}  (runpod ${pod:.2f}, openrouter ${orr:.2f})  "
            f"remaining ${CAP_USD - pod - orr:.2f} | new rows {rows}/{MAX_ROWS} | live pods {live}"
            + (" | STOP is set" if STOP.exists() else ""))


def check(reserve: float = 0.0) -> None:
    d = sync()
    pod, orr = _spent(d)
    if STOP.exists() or pod + orr + reserve > CAP_USD:
        raise SystemExit(f"BUDGET REFUSES: spent ${pod + orr:.2f} + reserve ${reserve:.2f} > cap ${CAP_USD:.0f}"
                         + (" (STOP set)" if STOP.exists() else ""))
    print(f"budget ok: ${pod + orr:.2f} spent, ${reserve:.2f} reserved, cap ${CAP_USD:.0f}")


def rows(add: int, what: str) -> None:
    d = _load()
    used = sum(r["n"] for r in d.get("rows", []))
    if used + add > MAX_ROWS:
        raise SystemExit(f"ROW CAP REFUSES: {used} generated + {add} > {MAX_ROWS}")
    d.setdefault("rows", []).append({"n": int(add), "what": what, "at": time.time()})
    _save(d)
    print(f"rows ok: {used + add}/{MAX_ROWS}")


def watchdog(every: int = 60) -> None:
    from src.infra.runpod import down
    while True:
        try:
            d = sync()
            pod, orr = _spent(d)
            now = time.time()
            live = {i: e for i, e in d["pods"].items() if now - e.get("last_seen", 0) < every * 2.5}
            over = pod + orr >= CAP_USD - STOP_MARGIN_USD
            if over and not STOP.exists():
                STOP.write_text(f"{datetime.now(timezone.utc).isoformat()} spent ${pod + orr:.2f}\n")
                print(f"!!! CAP REACHED: ${pod + orr:.2f}; STOP written", flush=True)
            for i, e in live.items():
                old = (now - e["start"]) / 3600 > MAX_POD_HOURS
                if over or old:
                    try:
                        log = requests.get(f"https://{i}-8080.proxy.runpod.net/boot.log", timeout=15).text
                        (DIR / f"bootlog_{e['name']}_{i}.txt").write_text(log)
                    except Exception:
                        pass
                    print(f"!!! terminating {e['name']} ({i}): {'cap reached' if over else f'older than {MAX_POD_HOURS}h'}", flush=True)
                    print(down(i), flush=True)
        except SystemExit:
            raise
        except Exception as exc:
            print(f"watchdog error (continuing): {exc!r}"[:300], flush=True)
        time.sleep(every)


if __name__ == "__main__":
    fire.Fire({"init": init, "sync": lambda: status(), "status": status, "check": check, "rows": rows, "watchdog": watchdog})
