# ABOUTME: Builds a wandb Report comparing the 4xH200 legacy run and the 2xH200 dynamic-
# ABOUTME: batching run of the same arm, which live in different projects.

"""One view over two projects.

The 2026-08-08 run logged to `huggingface` and today's dynamic-batching run to
`teaching-claude-why`. wandb only puts runs from different projects side by side inside a
Report, so this creates one with a runset per project and the panels that matter for the
comparison: loss (are they gradient-equivalent?) and runtime (is it faster?).

Run: uv run python scratch/wandb_compare_report.py
"""

import os

os.environ.setdefault(
    "WANDB_API_KEY",
    "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")

import wandb_workspaces.reports.v2 as wr  # noqa: E402

ENTITY = "mebozoukov-uc-san-diego"
LEGACY_PROJECT, LEGACY_RUN = "huggingface", "difficult-advice-20260808_095609"
NEW_PROJECT, NEW_RUN = "teaching-claude-why", "difficult-advice-20260811_132208"

report = wr.Report(
    project=NEW_PROJECT,
    entity=ENTITY,
    title="t2-9000-synthdoc-1000: 4xH200 legacy vs 2xH200 dynamic batching",
    description=(
        "Same arm, same data (LASR-Callum/2026-08-08-table2-9000-synthdoc-1000-"
        "traitbalanced-train), same global batch of 16, LR 1e-4 cosine, warmup 0.05, seed 0, "
        "625 steps. Differences: the 08-08 run used 4 ranks x grad_accum 4 with batch-1 "
        "forward passes; today's uses 2 ranks with token-budgeted dynamic batching "
        "(~1,589 passes/epoch vs 10,000) and max_length 8192 rather than 8000. "
        "Loss curves should coincide if the grouping is gradient-equivalent; runtime is the "
        "payoff (180.2 min on 4 GPUs vs ~166 min projected on 2)."
    ),
    blocks=[
        wr.H1("Loss — do the curves coincide?"),
        wr.PanelGrid(
            runsets=[
                wr.Runset(entity=ENTITY, project=LEGACY_PROJECT, name="4xH200 legacy (08-08)",
                          filters=f'Name == "{LEGACY_RUN}"'),
                wr.Runset(entity=ENTITY, project=NEW_PROJECT, name="2xH200 dynamic batching",
                          filters=f'Name == "{NEW_RUN}"'),
            ],
            panels=[
                wr.LinePlot(x="train/global_step", y=["train/loss"], title="training loss"),
                wr.LinePlot(x="train/global_step", y=["train/grad_norm"], title="grad norm"),
                wr.LinePlot(x="train/global_step", y=["train/learning_rate"],
                            title="learning rate"),
                wr.LinePlot(x="train/global_step", y=["train/epoch"], title="epoch"),
            ],
        ),
        wr.H1("Throughput"),
        wr.PanelGrid(
            runsets=[
                wr.Runset(entity=ENTITY, project=LEGACY_PROJECT, name="4xH200 legacy (08-08)",
                          filters=f'Name == "{LEGACY_RUN}"'),
                wr.Runset(entity=ENTITY, project=NEW_PROJECT, name="2xH200 dynamic batching",
                          filters=f'Name == "{NEW_RUN}"'),
            ],
            panels=[
                wr.ScalarChart(metric="train/train_runtime", title="total runtime (s)"),
                wr.ScalarChart(metric="train/train_samples_per_second", title="samples/s"),
            ],
        ),
    ],
)

report.save()
print("report URL:", report.url)
