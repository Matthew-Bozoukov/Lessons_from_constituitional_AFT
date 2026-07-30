"use client";

import { useMemo, useState } from "react";
import {
  CartesianGrid,
  ComposedChart,
  ErrorBar,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ResearchEntry, humanize } from "@/lib/content";

const stageOrder = ["base", "midtraining", "sft", "bounded-dpo", "rl"];
const colors = ["#68e4df", "#b797ff", "#a8e66b", "#ffb86b"];

type PlotPoint = {
  stageIndex: number;
  stage: string;
  value: number;
  seed?: number;
  runId?: string;
};

function mean(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function deviation(values: number[]) {
  if (values.length < 2) return 0;
  const average = mean(values);
  return Math.sqrt(
    values.reduce((sum, value) => sum + (value - average) ** 2, 0) /
      (values.length - 1),
  );
}

export function MetricExplorer({ entries }: { entries: ResearchEntry[] }) {
  const compatibleGroups = useMemo(() => {
    const groups = new Map<string, ResearchEntry[]>();
    for (const entry of entries) {
      if (entry.type !== "evals" || !entry.eval_suite) continue;
      const key = `${entry.eval_suite}|${entry.eval_version}|${entry.dataset_version}`;
      groups.set(key, [...(groups.get(key) || []), entry]);
    }
    return [...groups.entries()].sort((a, b) => b[1].length - a[1].length);
  }, [entries]);

  const defaultGroup = compatibleGroups[0]?.[0] || "";
  const [groupKey, setGroupKey] = useState(defaultGroup);
  const groupEntries =
    compatibleGroups.find(([key]) => key === groupKey)?.[1] || [];
  const metricNames = [...new Set(groupEntries.flatMap((entry) => Object.keys(entry.metrics)))];
  const defaultMetric = metricNames.includes("agentic_misalignment_rate")
    ? "agentic_misalignment_rate"
    : metricNames[0] || "";
  const [selectedMetric, setSelectedMetric] = useState(defaultMetric);
  const activeMetric = metricNames.includes(selectedMetric)
    ? selectedMetric
    : defaultMetric;

  const stages = [...new Set(groupEntries.map((entry) => entry.training_stage || "unknown"))].sort(
    (a, b) => {
      const ai = stageOrder.indexOf(a);
      const bi = stageOrder.indexOf(b);
      return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi) || a.localeCompare(b);
    },
  );

  const individual: PlotPoint[] = groupEntries
    .filter((entry) => entry.metrics[activeMetric])
    .map((entry) => ({
      stageIndex: stages.indexOf(entry.training_stage || "unknown"),
      stage: entry.training_stage || "unknown",
      value: entry.metrics[activeMetric].value,
      seed: entry.seed,
      runId: entry.run_id,
    }));

  const aggregates = stages
    .map((stage, stageIndex) => {
      const values = individual
        .filter((point) => point.stage === stage)
        .map((point) => point.value);
      return values.length
        ? { stage, stageIndex, value: mean(values), error: deviation(values), count: values.length }
        : null;
    })
    .filter(Boolean) as Array<{
      stage: string;
      stageIndex: number;
      value: number;
      error: number;
      count: number;
    }>;

  if (!compatibleGroups.length) {
    return <div className="empty-state">No compatible structured eval group yet.</div>;
  }

  const [suite, version, dataset] = groupKey.split("|");
  const selectedDefinition = groupEntries.find((entry) => entry.metrics[activeMetric])
    ?.metrics[activeMetric];
  const proportion = selectedDefinition?.unit === "proportion";
  const color = colors[metricNames.indexOf(activeMetric) % colors.length] || colors[0];

  return (
    <div className="metric-explorer">
      <div className="explorer-controls">
        <div>
          <span className="eyebrow">Compatible comparison</span>
          <h2>Metric explorer</h2>
        </div>
        <div className="control-group">
          <label>
            Evaluation group
            <select value={groupKey} onChange={(event) => setGroupKey(event.target.value)}>
              {compatibleGroups.map(([key, grouped]) => {
                const [groupSuite, groupVersion] = key.split("|");
                return (
                  <option value={key} key={key}>
                    {groupSuite} · {groupVersion} · {grouped.length} runs
                  </option>
                );
              })}
            </select>
          </label>
          <label>
            Metric
            <select value={activeMetric} onChange={(event) => setSelectedMetric(event.target.value)}>
              {metricNames.map((metric) => (
                <option value={metric} key={metric}>
                  {humanize(metric)}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>

      <div className="compatibility-strip">
        <code>{suite}</code>
        <span>{version}</span>
        <span>{dataset}</span>
        <span>{individual.length} run points</span>
        <span>
          {selectedDefinition?.lower_is_better === true
            ? "lower is better"
            : selectedDefinition?.lower_is_better === false
              ? "higher is better"
              : "direction not registered"}
        </span>
      </div>

      <div className="chart-frame" aria-label={`${humanize(activeMetric)} across training stages`}>
        <ResponsiveContainer width="100%" height={350}>
          <ComposedChart margin={{ top: 24, right: 24, bottom: 12, left: 0 }}>
            <CartesianGrid stroke="#273038" vertical={false} />
            <XAxis
              dataKey="stageIndex"
              type="number"
              domain={[-0.2, Math.max(stages.length - 0.8, 0.8)]}
              ticks={stages.map((_, index) => index)}
              tickFormatter={(value) => humanize(stages[value] || "")}
              stroke="#73808b"
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              type="number"
              domain={proportion ? [0, 1] : ["auto", "auto"]}
              tickFormatter={(value) => (proportion ? `${Math.round(value * 100)}%` : value)}
              stroke="#73808b"
              tickLine={false}
              axisLine={false}
              width={54}
            />
            <Tooltip
              contentStyle={{
                background: "#14191e",
                border: "1px solid #34404a",
                borderRadius: 10,
                color: "#e9eef3",
              }}
              formatter={(value) => [
                proportion ? `${(Number(value) * 100).toFixed(1)}%` : Number(value).toFixed(3),
                humanize(activeMetric),
              ]}
              labelFormatter={(value) => humanize(stages[Number(value)] || "")}
            />
            <Line
              data={aggregates}
              dataKey="value"
              type="linear"
              stroke={color}
              strokeWidth={2.5}
              dot={{ fill: color, r: 6, strokeWidth: 0 }}
              isAnimationActive={false}
            >
              <ErrorBar dataKey="error" width={7} stroke={color} strokeWidth={1.5} />
            </Line>
            <Scatter
              data={individual}
              dataKey="value"
              fill={color}
              opacity={0.38}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <p className="chart-note">
        Faint points are individual seeds. The solid line is the stage mean; error bars show
        sample standard deviation when multiple seeds exist. Unknown metrics are discovered
        directly from frontmatter.
      </p>
    </div>
  );
}

