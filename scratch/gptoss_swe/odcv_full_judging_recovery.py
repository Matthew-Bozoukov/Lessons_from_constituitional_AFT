# ABOUTME: Recover only ODCV judging and packaging from saved full-run rollouts and paid caches.
# ABOUTME: Never starts inference; retains blocked verdict exclusions and original conservative reservations.
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from omegaconf import OmegaConf
from scratch.gptoss_swe import odcv_full as full
from src.eval.misalignment.odcv import odcv_judge as judge, progress_judge
from src.eval.misalignment.odcv.passes import package_run, submission_stats


def original_alive(arm):
    pid = full.read(full.ROOT / arm / "owner-started.json")["pid"]
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    code = ctypes.c_ulong()
    try:
        assert ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        return code.value == 259
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def request_index(mod, items, combined):
    captured = {}

    class Captured(BaseException):
        pass

    def create(**kw):
        captured["hash"] = hashlib.sha256(
            json.dumps(kw["messages"], ensure_ascii=False).encode()
        ).hexdigest()
        raise Captured()

    original = mod.OpenAI
    mod.OpenAI = lambda **kw: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    index = {}
    try:
        for variant, unit, path in items:
            copied, _ = judge.judge_copy(path, combined / "evaluations/judge_copies")
            try:
                mod.evaluation_routine(str(copied))
            except Captured:
                pass
            index[captured["hash"]] = variant + "/" + unit
    finally:
        mod.OpenAI = original
    return index


def preserve_blocked(arm, mod, items, combined, cache_path):
    root = full.ROOT / arm
    index = request_index(mod, items, combined)
    cache = full.read(cache_path, {})
    exclusions = []
    for i, entry in enumerate(full.read(root / "judge-budget.json", [])):
        diagnostic = entry.get("response_diagnostic")
        if not diagnostic or entry["request_sha256"] not in index:
            continue
        response = full.read(root / diagnostic)
        if response.get("choices") or not response.get("error"):
            continue
        key = index[entry["request_sha256"]]
        reason = "Provider returned no verdict: " + response["error"]["message"]
        if key not in cache:
            cache[key] = dict(
                score="N/A",
                reasoning=reason,
                provider_exclusion=True,
                ledger_entry=i,
                response_diagnostic=diagnostic,
            )
        exclusions.append(dict(key=key, ledger_entry=i, reason=reason))
    if exclusions:
        full.save(cache_path, cache)
    return exclusions


def recover_arm(arm):
    root = full.ROOT / arm
    full.claim(root / "judging-recovery-started.json")
    while original_alive(arm):
        time.sleep(10)
    old_status = full.read(root / "status.json")
    assert old_status["phase"] == "held", old_status
    assert full.read(root / "shim-stopped.json")["state"]["Running"] is False
    backup = root / "before-judging-recovery"
    backup.mkdir()
    for name in ("status.json", "owner.log", "judge-budget.json", "judge-budget.tmp"):
        if (root / name).exists():
            shutil.copyfile(root / name, backup / name)
    cfg = OmegaConf.load(root / "odcv.yaml")
    (combined,) = list((root / "odcv" / cfg.model_key).glob("combined*"))
    shutil.copytree(combined / "evaluations", backup / "evaluations")
    items = judge.transcripts(combined, cfg.model_key)
    assert len(items) == 240
    old_cache = full.read(
        combined / "evaluations/scores_gemini-3-flash-preview.json", {}
    )
    judge.openrouter_usage = lambda **kw: None
    progress_judge.openrouter_usage = lambda **kw: None
    full.phase(root, "recovering_judging")
    try:
        mod = judge._load_upstream_judge(Path(cfg.bench_dir))
        mod.OPENAI_MODEL = cfg.judges["gemini-3-flash-preview"]
        mr_cache = combined / "evaluations/scores_gemini-3-flash-preview.json"
        exclusions = preserve_blocked(arm, mod, items, combined, mr_cache)
        ledger = full.read(root / "judge-budget.json")
        unknown = [
            i
            for i, e in enumerate(ledger)
            if e["status"] == "reserved" and not e.get("response_diagnostic")
        ]
        if unknown:
            # Exact captured failure was before send at JudgeBudget.create's first save.
            assert arm == "da15" and unknown == [87]
            log = (backup / "owner.log").read_text(errors="replace")
            assert "line 62, in create" in log and "PermissionError" in log
            assert (
                ledger[87]["request_sha256"]
                == "12809ef3cf9b50a302cac16733832f7a5f0c19aa22db37c1298c7f06cb4ca61d"
            )
            full.save(
                root / "known-unsent-judge-receipt.json",
                dict(
                    entry=87,
                    reason="Only undiagnosed reservation; original traceback failed durable reservation save before SDK send.",
                    original_reservation_retained=True,
                    request_sha256=ledger[87]["request_sha256"],
                ),
            )

        def stage(fn, module, cache_path):
            try:
                return fn()
            except Exception:
                blocked = preserve_blocked(arm, module, items, combined, cache_path)
                cached = full.read(cache_path, {})
                assert blocked and all(v + "/" + u in cached for v, u, _ in items), (
                    "Unknown missing verdicts; hold without blind replay"
                )
                return fn()  # All items cached: aggregation only, no paid calls.

        stage(
            lambda: judge.main(str(combined), str(root / "odcv.yaml"), max_workers=4),
            mod,
            mr_cache,
        )
        for key, value in old_cache.items():
            assert full.read(mr_cache)[key] == value, "Existing paid verdict changed"
        results = full.read(combined / "results.json")
        results["progress"] = stage(
            lambda: progress_judge.main(
                str(combined), str(root / "odcv.yaml"), max_workers=4
            ),
            progress_judge,
            combined / "evaluations/progress_gemini-3-flash-preview.json",
        )
        results["submission"] = submission_stats(combined, cfg.model_key)
        audits = full.read(root / "completed-passes.json")
        for audit in audits:
            audit["kept"] = True
        results["passes"] = dict(
            requested=3, kept=3, dropped=0, n_transcripts=240, audits=audits
        )
        results["recovery"] = dict(
            source=full.source(),
            original_source=full.read(full.ROOT / "manifest.json")["source"],
            inference_reruns=0,
            existing_verdicts_preserved=len(old_cache),
            provider_exclusions=exclusions,
            conservative_reservations_retained=True,
        )
        full.save(root / "summary.json", results)
        package_run(root / "odcv", cfg.model_key, audits, combined)
        dropped = results["n_dropped_all_na"] + results["progress"]["n_dropped_all_na"]
        full.phase(
            root,
            "judged_with_exclusions" if dropped else "complete",
            unjudged_axis_cells=dropped,
        )
        return dict(arm=arm, dropped=dropped)
    except BaseException as exc:
        full.phase(root, "judging_recovery_held", error=repr(exc))
        raise


def run():
    full.claim(full.ROOT / "judging-recovery-owner-started.json")
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

    def child(arm):
        with (full.ROOT / arm / "judging-recovery.log").open(
            "x", encoding="utf-8"
        ) as log:
            p = subprocess.Popen(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    "-m",
                    "scratch.gptoss_swe.odcv_full_judging_recovery",
                    "arm",
                    "--arm",
                    arm,
                ],
                env=full.environment(),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            full.save(full.ROOT / arm / "judging-recovery-child.json", dict(pid=p.pid))
            return p.wait()

    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            exits = dict(zip(full.ARMS, pool.map(child, full.ARMS)))
        full.save(full.ROOT / "judging-recovery-exits.json", exits)
        # Publication is separately reconciled after all original owners have exited.
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "arm"])
    parser.add_argument("--arm", choices=list(full.ARMS))
    args = parser.parse_args()
    run() if args.action == "run" else recover_arm(args.arm)
