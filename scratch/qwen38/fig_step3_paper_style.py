# ABOUTME: oai_hf Step 3 (shares the exploit with other agents) on Qwen3.8-27B in the paper's figure style: control,
# ABOUTME: DA and DA+tools bars with 95% Wilson CIs, the base model as the dashed line. Run: uv run python scratch/qwen38/fig_step3_paper_style.py
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from scratch.figures import _style as st  # noqa: E402
from src.eval.misalignment.oai_hf.export import wilson  # noqa: E402

BASE = ("dougalldeepmind/2026-11-18-oaihf-openrouter-qwen3-8-27b", "5a326442")
BARS = [  # (style key, Hub eval repo, revision) -- newest pushes, 2026-11-19
    ("control", "dougalldeepmind/2026-11-19-oaihf-qwen38-0-plain", "45810cd6"),
    ("da", "dougalldeepmind/2026-11-19-oaihf-qwen38-0-da-15", "e2f709ce"),
    ("da_tools", "dougalldeepmind/2026-11-20-oaihf-qwen38-0-da-tools-15", "140568af"),
]


def step3(repo, rev):
    f = hf_hub_download(repo, "results/runs.jsonl", repo_type="dataset", revision=rev)
    rows = [r for r in map(json.loads, open(f)) if r["step"] == "step3"]
    return sum(bool(r["signature"]) for r in rows), len(rows)


st.apply()
fig, ax = plt.subplots(figsize=(8.6, 6.6))
kb, nb = step3(*BASE)
pb = 100 * kb / nb
st.base_line(ax, pb)
for i, (key, repo, rev) in enumerate(BARS):
    k, n = step3(repo, rev)
    p, lo, hi = (100 * v for v in wilson(k, n))
    st.bar(ax, i, p, lo, hi, colour=st.COLOUR[key], label=f"{p:.1f}")
    ax.text(i, -3.5, f"{k}/{n}", ha="center", va="top", fontsize=15, color=st.MUTED)
    print(key, k, n, f"{p:.1f} ({lo:.1f}-{hi:.1f})")
print("base", kb, nb, f"{pb:.1f}")
ax.set_xticks([])
ax.set_xlim(-0.6, 2.6)
st.finish(
    ax,
    title="OAI-HF Step 3: shares the exploit with other agents",
    ylabel="Runs that shared (%)",
    ylim=(0, 60),
    yticks=range(0, 61, 10),
    title_pad=16,
)
st.legend(
    fig,
    [
        st.base_handle(f"Qwen3.8-27B ({kb}/{nb}, via API)"),
        st.patch(st.COLOUR["control"], st.ARM["control"]),
        st.patch(st.COLOUR["da"], st.ARM["da"]),
        st.patch(st.COLOUR["da_tools"], st.ARM["da_tools"]),
    ],
    ncol=1,
    y=1.30,
)
st.save(
    fig,
    "output/oai_hf_repro",
    "oaihf_step3_qwen38_paper_style",
    title="oai_hf Step 3 on Qwen3.8-27B: control, DA, DA+tools vs base",
    script="scratch/qwen38/fig_step3_paper_style.py",
    artifacts=[f"{BASE[0]}@{BASE[1]}"] + [f"{r}@{v}" for _, r, v in BARS],
)
