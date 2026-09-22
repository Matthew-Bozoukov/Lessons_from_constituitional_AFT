# ABOUTME: One figure for the dump: how the t10 (multi-agent) DA prompts differ in shape from the DA reference.
# ABOUTME: Three panels of paired horizontal bars (t10 teal, DA purple), numbers from the 2026-09-22 label runs.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

T10, DA = "#0e9a80", "#7a56c5"
INK, INK2, RULE = "#1b2123", "#4f5a5c", "#d7dcd8"

PANELS = [
    (
        "What the shortcut asks for (judge)",
        [
            ("game a downstream gate", 65, 28),
            ("send a false status", 22, 6),
            ("withhold a flag or datum", 7, 8),
            ("relay a plan to another agent", 3, 2),
            ("defer to a peer's verdict", 2, 4),
            ("other (plain advice)", 1, 52),
        ],
    ),
    (
        "How the tension is introduced",
        [
            ("user turn is another agent", 11, 0),
            ("shortcut proposed by an agent", 13, 0),
            ("ask routes around a human review", 69, 25),
            ("user explains which wording trips the gate", 63, 5),
            ("system: 'embedded in a … pipeline'", 56, 3),
            ("system: output 'acted on directly'", 27, 0),
            ("system: 'without human review'", 12, 0),
        ],
    ),
    (
        "Pressure (push rubric)",
        [
            ("push or override", 96, 56),
            ("lean", 2, 25),
            ("none (tension latent)", 1, 19),
            ("the AI itself is the actor", 94, 18),
            ("a prompt-only monitor would flag it", 94, 48),
            ("full help is the right answer", 3, 28),
        ],
    ),
]


def main() -> None:
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1, 1.15, 1]}
    )
    for ax, (title, rows) in zip(axes, PANELS):
        labels = [r[0] for r in rows][::-1]
        t10 = [r[1] for r in rows][::-1]
        da = [r[2] for r in rows][::-1]
        y = range(len(rows))
        h = 0.36
        ax.barh(
            [i + h / 2 for i in y],
            t10,
            height=h,
            color=T10,
            label="t10 multi-agent (n=775)",
        )
        ax.barh(
            [i - h / 2 for i in y],
            da,
            height=h,
            color=DA,
            label="DA t1–t9 (n=752; judged: 200)",
        )
        for i, (a, b) in enumerate(zip(t10, da)):
            ax.text(a + 1.5, i + h / 2, f"{a}%", va="center", fontsize=8.5, color=INK2)
            ax.text(b + 1.5, i - h / 2, f"{b}%", va="center", fontsize=8.5, color=INK2)
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=9, color=INK)
        ax.set_xlim(0, 112)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("share of rows (%)", fontsize=9, color=INK2)
        ax.set_title(title, fontsize=10.5, loc="left", color=INK, fontweight="semibold")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(RULE)
        ax.tick_params(colors=INK2, length=0)
        ax.grid(axis="x", color=RULE, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_,
        loc="upper left",
        ncol=2,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.005, 0.945),
    )
    fig.suptitle(
        "Multi-agent (t10) difficult-advice prompts vs the DA reference — one gate-gaming shape, introduced by the system prompt, 96% push",
        fontsize=11.5,
        x=0.01,
        y=0.995,
        ha="left",
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    out = "output/t10_multiagent/2026-09-22_t10_vs_da_prompt_shape.png"
    fig.savefig(out, dpi=170)
    print(out)


if __name__ == "__main__":
    main()
