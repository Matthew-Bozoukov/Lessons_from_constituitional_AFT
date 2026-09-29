# ABOUTME: Render one or more ODCV rollouts in condensed form (task once, then each assistant turn's text,
# ABOUTME: reasoning, commands, and truncated tool output). Run: show.py <run> <variant> <Scenario> <pass> [...]
import sys

sys.path.insert(0, "output/da_tools_odcv_read")
from parse import load_run, parse  # noqa: E402

args = sys.argv[1:]
lim_c = int(next((a.split("=")[1] for a in args if a.startswith("c=")), 1500))
lim_r = int(next((a.split("=")[1] for a in args if a.startswith("r=")), 1500))
lim_t = int(next((a.split("=")[1] for a in args if a.startswith("t=")), 400))
show_task = not any(a == "notask" for a in args)
args = [a for a in args if "=" not in a and a != "notask"]
cache = {}
for i in range(0, len(args), 4):
    run, v, s, k = args[i], args[i + 1], args[i + 2], int(args[i + 3])
    if run not in cache:
        cache[run] = load_run(run)
    r = cache[run][(v, s, k)]
    steps = parse(r["path"].read_text())
    print("#" * 110)
    print(f"# {run} {v}/{s} pass{k}  severity={r['sev']} progress={r['prog']}")
    print("#" * 110)
    for j, st in enumerate(steps):
        if st["role"] == "system":
            if j == 0 and show_task:
                print("[SYSTEM]", st["content"][:1200])
            elif j > 0:
                print("[SYSTEM-later]", st["content"][:300])
            continue
        if st["role"] == "user":
            if j <= 1 and show_task:
                print("[USER]", st["content"][:1500])
            elif j > 1:
                print("[USER-nudge]", st["content"][:90])
            continue
        if st["role"] == "tool":
            print("   [TOOL OUT]", st["content"][:lim_t].replace("\\n", "\n      "))
            continue
        print(f"\n[ASSISTANT step {j + 1}] text:", st["content"][:lim_c])
        if st["reason"]:
            print("   [REASON]", st["reason"][:lim_r])
        for c in st["calls"]:
            f = c.get("function", {})
            print("   [CALL]", f.get("name"), str(f.get("arguments"))[:700])
    print()
