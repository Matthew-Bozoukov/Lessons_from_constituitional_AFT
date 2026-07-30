import { ArrowDown, ArrowUp, Minus } from "lucide-react";
import { Metric, formatMetric, humanize } from "@/lib/content";

export function MetricTiles({
  metrics,
  limit,
}: {
  metrics: Record<string, Metric>;
  limit?: number;
}) {
  const visible = Object.entries(metrics).slice(0, limit);
  if (!visible.length) return null;

  return (
    <div className="metric-tiles">
      {visible.map(([name, metric]) => (
        <div className="metric-tile" key={name}>
          <div className="metric-label">
            <span>{humanize(name)}</span>
            {metric.lower_is_better === true ? (
              <ArrowDown size={13} aria-label="Lower is better" />
            ) : metric.lower_is_better === false ? (
              <ArrowUp size={13} aria-label="Higher is better" />
            ) : (
              <Minus size={13} aria-label="No preferred direction set" />
            )}
          </div>
          <strong>{formatMetric(metric)}</strong>
          <small>{metric.unit || "unitless"}</small>
        </div>
      ))}
    </div>
  );
}

