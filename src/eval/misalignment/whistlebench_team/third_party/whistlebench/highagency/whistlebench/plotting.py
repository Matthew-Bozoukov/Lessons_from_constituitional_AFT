# type: ignore
import colorsys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Base colors per provider (+ some aliases)
PROVIDER_BASE = {
    "openai": "#1a7f37",  # green
    "xai": "#3a3a3a",
    "x-ai": "#3a3a3a",
    "x.ai": "#3a3a3a",  # dark gray for grok
    "anthropic": "#ff7f00",  # orange
    "google": "#c0392b",  # red
    "qwen": "#6f42c1",  # purple
    "meta-llama": "#4285F4",  # blue
}

CLEAN_MODEL_NAMES = {
    "gpt-4.1-2025-04-14": "GPT-4.1",
    "gpt-5-2025-08-07": "GPT-5",
    "claude-opus-4-20250514": "Claude Opus 4",
    "claude-opus-4-1-20250805": "Claude Opus 4.1",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "llama-4-maverick": "Llama 4 Maverick",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "grok-4": "Grok 4",
}

COMBINED_GROUPS = {
    "GovWhistleblowingSeparate": ["score_gov_whistleblowing_accuracy"],
    "MediaWhistleblowingSeparate": ["score_media_whistleblowing_accuracy"],
    "InternalWhistleblowingSeparate": ["score_internal_whistleblowing_accuracy"],
    "Whistleblowing": [
        "score_gov_whistleblowing_accuracy",
        "score_media_whistleblowing_accuracy",
        "score_internal_whistleblowing_accuracy",
    ],
    "Evaluation Awareness": [
        "score_evaluation_awareness_accuracy",
        "score_hypothetical_recognition_accuracy",
    ],
    "Refusal": [
        "score_refusal_accuracy",
    ],
}


def _lighten_hex(hex_color, factor):
    """Return a lighter shade of a hex color. factor∈[0,1] (higher = lighter)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    l = min(1.0, l + (1 - l) * factor)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r2 * 255):02x}{int(g2 * 255):02x}{int(b2 * 255):02x}"


def _shade_for_model(provider, model, provider_models):
    """Shade per model within a provider (same base hue, lighter tones)."""
    base = PROVIDER_BASE.get(provider, "#777777")
    models = sorted(provider_models.get(provider, [model]))
    total = len(models)
    if total <= 1:
        return base
    rank = models.index(model)
    factor = (rank) / max(1, (total - 1)) * 0.6  # up to ~60% lighter
    return _lighten_hex(base, factor)


def _sem_for_score_column(df_cols, score_col):
    """
    Map a score column (e.g. 'score_refusal_accuracy') to its SEM column.
    Your data uses 'sem_*'. We'll try a few sensible matches:
      - 'sem_' + score_col                     -> 'sem_score_refusal_accuracy'
      - 'sem_' + score_col.replace('score_', '') -> 'sem_refusal_accuracy'
    Return the first one that exists, else None.
    """
    candidates = [
        "sem_" + score_col,
        "sem_" + score_col.replace("score_", "", 1),
    ]
    for c in candidates:
        if c in df_cols:
            return c
    return None


def _build_palette(data: pd.DataFrame, provider_models: dict[str, list[str]]) -> dict[str, str]:
    provider_lookup = (
        data.groupby("model")["provider"].first().dropna().to_dict()
        if "provider" in data.columns
        else {}
    )
    palette: dict[str, str] = {}
    for model, provider in provider_lookup.items():
        palette[model] = _shade_for_model(provider, model, provider_models)
    # Fallback colors for any models without provider information
    for model in data["model"].unique():
        palette.setdefault(model, "#777777")
    return palette


def plot_combined_group_scores(
    df: pd.DataFrame,
    varying_param: str | None,
    fixed_filters: dict | None = None,
    annotate_n: bool = True,
    figsize=(3.5, 3.3),
    output_dir: str | Path | None = None,
    groups: dict[str, list[str]] = COMBINED_GROUPS,
):
    """
    Create one figure per group in COMBINED_GROUPS.
    Each figure has one subplot per score column listed in that group.
    The x-axis is shared across subplots.
    """
    data = df.copy()
    if fixed_filters:
        for key, value in fixed_filters.items():
            data = (
                data[data[key].isin(value)]
                if isinstance(value, (list, tuple, set))
                else data[data[key] == value]
            )

    # only require the column if varying_param is provided
    if varying_param is not None and varying_param not in data.columns:
        raise ValueError(f"Column '{varying_param}' not in DataFrame.")

    # x-levels only if varying_param is provided
    if varying_param is None:
        x_levels = [""]
    else:
        x_levels = list(dict.fromkeys(data[varying_param].tolist()))
        if not x_levels:
            raise ValueError("No categories available for the chosen varying_param.")

    if "model" not in data.columns:
        raise ValueError("DataFrame must include a 'model' column for plotting.")
    data["model"] = data["model"].map(CLEAN_MODEL_NAMES).fillna(data["model"])
    model_levels = sorted(data["model"].unique())

    provider_models = (
        data.groupby("provider")["model"].unique().apply(list).to_dict()
        if "provider" in data.columns
        else {}
    )
    palette = _build_palette(data, provider_models)
    hue_order = model_levels

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="white", context="talk")

    figures: list[tuple[str, plt.Figure]] = []

    for group_name, score_cols in groups.items():
        # Remove Media whistleblowing plot for parameter variation experiments
        if group_name == "Whistleblowing" and varying_param is not None:
            score_cols = [c for c in score_cols if c != "score_media_whistleblowing_accuracy"]

        # keep only scores that actually exist & have data
        score_cols = [c for c in score_cols if c in data.columns and data[c].notna().any()]
        if not score_cols:
            continue

        n_cols = len(score_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(figsize[0] * n_cols, figsize[1]), sharey=True)
        if n_cols == 1:
            axes = [axes]

        # choose/compute x-order & display mapping once so all subplots share it
        if varying_param is None:
            order = [""]
            display_map = {"": ""}
            display_order = [""]
            inverse_display_map = {"": ""}
        else:
            if varying_param == "boldness_prompt_detail":
                order = ["none", "low", "medium", "high"]
            elif varying_param == "agent_responsibility":
                order = ["ediscovery_review", "pii_detection", "summarization"]
            else:
                order = sorted(x_levels)
            display_map = {
                lvl: str(lvl)
                .replace("_", "\n")
                .title()
                .replace("Pii", "PII")
                .replace("Ediscovery", "E-Discovery")
                .replace("Summarization", "Summari-\nzation")
                for lvl in order
            }
            display_order = [display_map[lvl] for lvl in order]
            inverse_display_map = {v: k for k, v in display_map.items()}

        # draw each score column on its own subplot
        legend_handles_labels = None
        for ax, score_col in zip(axes, score_cols, strict=True):
            sem_col = _sem_for_score_column(data.columns, score_col)
            subset = data[data[score_col].notna()].copy()

            if varying_param is None:
                hue_levels = [m for m in hue_order if m in subset["model"].unique()]
                subset["_display"] = ""
            else:
                hue_levels = [m for m in hue_order if m in subset["model"].unique()]
                subset = subset.assign(_display=subset[varying_param].map(display_map))

            sns.barplot(
                data=subset,
                x="_display",
                y=score_col,
                hue="model",
                order=display_order,
                hue_order=hue_levels,
                palette=palette,
                errorbar=None,
                width=0.55,
                ax=ax,
            )

            # build lookup for error bars
            if varying_param is None:
                display_lookup = {("", row["model"]): row for _, row in subset.iterrows()}
            else:
                display_lookup = {
                    (row[varying_param], row["model"]): row for _, row in subset.iterrows()
                }

            for container, model in zip(ax.containers, hue_levels, strict=False):
                for bar, display_val in zip(container.patches, display_order, strict=False):
                    original_val = inverse_display_map.get(display_val, "")
                    key = (original_val if varying_param is not None else "", model)
                    row = display_lookup.get(key)
                    if row is None:
                        continue
                    if sem_col and pd.notna(row.get(sem_col)):
                        sem = float(row[sem_col])
                        ax.errorbar(
                            bar.get_x() + bar.get_width() / 2,
                            float(row[score_col]),
                            yerr=1.96 * sem,
                            fmt="none",
                            ecolor="black",
                            capsize=3,
                            linewidth=1,
                        )

            pretty_score_name = (
                score_col.replace("score_", "")
                .replace("_accuracy", "")
                .replace("gov", "government")
                .replace("_", " ")
                .title()
            )
            ax.set_title(pretty_score_name, fontsize=12)

            y_label_name = "Whistleblowing" if "Whistleblowing" in group_name else group_name
            ax.set_ylabel(f"{y_label_name} Rate", fontsize=12)
            ax.set_ylim(0, 1.0)
            ax.tick_params(axis="y", labelsize=10)
            ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.8)
            # sns.despine(ax=ax)

            # No x tick labels for all_models experiment
            if varying_param is None:
                ax.set_xlabel("Model", fontsize=12)
                ax.set_xticklabels([])
            else:
                pretty_param_name = (
                    varying_param.replace("_", " ")
                    .title()
                    .replace("Agent", "Task")
                    .replace("Environment And ", "")
                )
                ax.set_xlabel(pretty_param_name, fontsize=12)

            # hide legends on individual subplots; collect for one shared legend
            if legend_handles_labels is None:
                handles, labels = ax.get_legend_handles_labels()
                legend_handles_labels = (handles, labels)
            ax.legend_.remove() if ax.get_legend() else None

            # optional N annotation per subplot
            if annotate_n:
                epochs_values = (
                    sorted({int(v) for v in subset["epochs"].dropna().unique()})
                    if "epochs" in subset.columns
                    else []
                )
                if epochs_values:
                    sample_text = (
                        f"N = {epochs_values[0]}"
                        if len(epochs_values) == 1
                        else "N = " + ", ".join(str(v) for v in epochs_values)
                    )
                    ax.text(
                        0.02,
                        0.98,
                        sample_text,
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=11,
                        color="#444444",
                    )

        for ax in axes:
            plt.setp(ax.xaxis.get_majorticklabels(), fontsize=10, rotation=0)
            plt.setp(ax.yaxis.get_majorticklabels(), fontsize=10)

        # one shared legend for the figure
        # (not for refusal, which will co-occur with eval awareness plots)
        if legend_handles_labels and group_name != "Refusal":
            handles, labels = legend_handles_labels
            axes[-1].legend(
                handles,
                labels,
                title="Models",
                loc="upper right",
                bbox_to_anchor=(0.985, 0.98),
                # bbox_transform=fig.transFigure,
                borderaxespad=0,
                frameon=True,
                fontsize=9,
                title_fontsize=10,
            )

        fig.tight_layout()

        if output_dir is not None:
            safe_group = group_name.lower()
            fig.savefig(output_dir / f"{safe_group}.pdf", dpi=300, bbox_inches="tight")

        figures.append((group_name, fig))

    return figures


def plot_experiment_scores(experiment_dir_name: str, varying_param: str):
    baseline_df = pd.read_csv("final_runs/0_baseline/aggregated_analysis.csv")
    experiment_df = pd.read_csv(f"final_runs/{experiment_dir_name}/aggregated_analysis.csv")
    agg = pd.concat([baseline_df, experiment_df], axis=0)

    _ = plot_combined_group_scores(
        agg,
        varying_param=varying_param,
        fixed_filters={},
        output_dir=f"images/experiment_{experiment_dir_name}",
        groups=COMBINED_GROUPS,
    )


def plot_all_models_scores():
    agg = pd.read_csv("final_runs/all_models/aggregated_analysis.csv")
    _ = plot_combined_group_scores(
        agg,
        varying_param=None,
        fixed_filters={},
        output_dir="images/experiment_all_models",
        groups=COMBINED_GROUPS,
    )


if __name__ == "__main__":
    plot_all_models_scores()
    plot_experiment_scores("1_responsibility", "agent_responsibility")
    plot_experiment_scores("2_boldness", "boldness_prompt_detail")
    plot_experiment_scores(
        "3_detailed_instructions",
        "include_environment_and_workflow_details",
    )
    plot_experiment_scores("4_distractor_docs", "include_distractor_docs")
    plot_experiment_scores("5_distractor_tools", "include_distractor_tools")
