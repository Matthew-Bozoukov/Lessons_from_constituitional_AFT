# ABOUTME: AM native tools vs text tools in the paper's figure style: per scenario, control / DA / DA+tools bars,
# ABOUTME: light = text tools (plain AM), solid = native tools, 95% Wilson CIs. Run: uv run python scratch/am_native_tools/fig_paper_style.py
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

from scratch.figures import _style as st  # noqa: E402

ARMS = [  # (style key, plain-AM run, native-tools run or None while unjudged)
    ("control", "dougalldeepmind/2026-10-08-am-qwen36-0-plain",
     "dougalldeepmind/2026-11-23-am-native-tools-qwen36-0-plain"),
    ("da", "dougalldeepmind/2026-10-08-am-qwen36-0-da-15",
     "dougalldeepmind/2026-11-23-am-native-tools-qwen36-0-da-15"),
    ("da_tools", "dougalldeepmind/2026-10-13-am-qwen36-0-da-tools-15",
     "dougalldeepmind/2026-11-24-am-native-tools-qwen36-0-da-tools-15"),
]
PANELS = [("overall", "All 180 trials"), ("blackmail", "Blackmail"), ("leaking", "Leaking"),
          ("murder", "Murder")]
TINT = 0.35


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def counts(repo):
    s = json.load(open(hf_hub_download(repo, "results/results.json", repo_type="dataset")))
    s = s.get("summary", s)
    return {"overall": (s["overall"]["harmful"], s["overall"]["n"]),
            **{k: (v["harmful"], v["n"]) for k, v in s["by_scenario"].items()}}


data = {key: (counts(p), counts(n) if n else None) for key, p, n in ARMS}
st.apply()
fig, axes = plt.subplots(1, 4, figsize=(22, 6.6), sharey=True)
for ax, (panel, title) in zip(axes, PANELS):
    for i, (key, _, _) in enumerate(ARMS):
        for j, c in enumerate(data[key]):
            x = i + (j - 0.5) * 0.38
            if c is None:
                ax.text(x, 2, "not\njudged", ha="center", va="bottom", fontsize=13, color=st.MUTED)
                continue
            k, n = c[panel]
            lo, hi = (100 * v for v in wilson(k, n))
            colour = to_rgba(st.COLOUR[key], TINT) if j == 0 else st.COLOUR[key]
            st.bar(ax, x, 100 * k / n, lo, hi, colour=colour, width=0.36,
                   label=f"{100 * k / n:.0f}", label_size=14)
            print(panel, key, "text" if j == 0 else "native", k, n)
    ax.set_xticks([])
    ax.set_xlim(-0.55, 2.55)
    st.finish(ax, title=title, ylabel="Harmful trials (%)" if panel == "overall" else None,
              ylim=(0, 80), yticks=range(0, 81, 20))
st.legend(fig, [st.patch(st.COLOUR[k], st.ARM[k]) for k, _, _ in ARMS]
          + [st.patch(st.INK, "Tools as text in the system prompt (plain AM)", tint=TINT),
             st.patch(st.INK, "Tools through the API (native tool block)")], ncol=2, y=1.2)
st.save(fig, "output/figures", "am_native_tools_vs_text_tools_paper_style",
        title="Agentic misalignment: tools as prompt text vs native tools, Qwen3.6-27B",
        script="scratch/am_native_tools/fig_paper_style.py",
        artifacts=[r for _, p, n in ARMS for r in (p, n) if r])
