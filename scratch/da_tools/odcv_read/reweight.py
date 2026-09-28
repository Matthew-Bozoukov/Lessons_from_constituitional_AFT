# ABOUTME: Reweighting check: apply pooled da-15 violation rates per tool-call band to da-tools' own call mix, to see
# ABOUTME: how much of the gap the call-count shift alone would explain.
import sys

sys.path.insert(0, "output/da_tools_odcv_read")
from parse import call_list, load_run, parse  # noqa: E402

bands = [(0, 5), (6, 8), (9, 11), (12, 15), (16, 999)]


def band(n):
    return next(i for i, (lo, hi) in enumerate(bands) if lo <= n <= hi)


rows = {run: load_run(run) for run in ("tools", "da0", "da1")}
cnt = {run: [0] * 5 for run in rows}
vio = {run: [0] * 5 for run in rows}
for run, rs in rows.items():
    for r in rs.values():
        b = band(len(call_list(parse(r["path"].read_text()))))
        cnt[run][b] += 1
        vio[run][b] += r["sev"] >= 3
pooled = [
    (vio["da0"][i] + vio["da1"][i]) / (cnt["da0"][i] + cnt["da1"][i]) for i in range(5)
]
exp = sum(cnt["tools"][i] * pooled[i] for i in range(5))
print(
    "da-15 pooled actual:",
    round(100 * (sum(vio["da0"]) + sum(vio["da1"])) / 480, 2),
    "%",
)
print("da-15 band rates on da-tools call mix:", round(100 * exp / 240, 2), "%")
print("da-tools actual:", round(100 * sum(vio["tools"]) / 240, 2), "%")
