# ABOUTME: Reattaches the campaign owner to one verified bootstrapping pod without renting another.
# ABOUTME: Preserves its original lifetime deadline; run with uv run python -m scratch.da_supervision.archive.resume_boot.
import json, time
from pathlib import Path
from scratch.da_supervision import owner
from src.infra import runpod


def main():
    base = Path("output/da_supervision/2026-09-16/runs")
    old = base / "empty-dual-attempt3"
    out = base / "empty-dual-recovered"
    s = json.loads((old / "status.json").read_text())
    assert (
        s["owned_pod"] == "266cqfrfcyxxgs"
        and not s.get("training_started")
        and s["phase"] == "bootstrap"
    )
    deadline = s["created_epoch"] + s["plan"]["pod"]["max_hours"] * 3600
    plan = s["plan"]
    plan["pod"]["max_hours"] = (deadline - time.time() - 10) / 3600
    assert 0 < plan["pod"]["max_hours"] < 5
    out.mkdir(exist_ok=True)
    p = out / "remaining_plan.json"
    p.write_text(json.dumps(plan, indent=2))

    def existing_up(name, **kwargs):
        info = runpod.call("GET", "/pods/" + s["owned_pod"])
        assert (
            info["name"] == name
            and info["gpuCount"] == 2
            and info["desiredStatus"] == "RUNNING"
        )
        kwargs["on_provisioned"](s["owned_pod"])
        return (old / "provision.txt").read_text()

    owner.runpod.up = existing_up
    owner.run(p, "empty", out)


if __name__ == "__main__":
    main()
