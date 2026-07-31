import Link from "next/link";
import {
  ArrowUpRight,
  Beaker,
  Database,
  FileText,
  FlaskConical,
  ScanSearch,
} from "lucide-react";
import { ResearchEntry, humanize } from "@/lib/content";
import { MockBadge } from "./MockDataBanner";

const icons = {
  logs: FlaskConical,
  evals: Beaker,
  findings: FileText,
  datasets: Database,
  "petri-runs": ScanSearch,
};

export function EntryCard({
  entry,
  compact = false,
}: {
  entry: ResearchEntry;
  compact?: boolean;
}) {
  const Icon = icons[entry.type];
  const className = [compact ? "entry-card compact" : "entry-card", entry.mock && "is-mock"]
    .filter(Boolean)
    .join(" ");
  return (
    <Link className={className} href={`/entry/${entry.slug}`}>
      <div className="entry-card-top">
        <span className={`type-chip ${entry.type}`}>
          <Icon size={13} />
          {humanize(entry.type)}
        </span>
        {entry.mock && <MockBadge />}
        <span className={`status status-${entry.status}`}>{humanize(entry.status)}</span>
      </div>
      <div>
        <h3>{entry.title}</h3>
        {entry.summary && <p>{entry.summary}</p>}
      </div>
      <div className="entry-card-bottom">
        <div className="entry-meta">
          <time dateTime={entry.date}>{entry.date}</time>
          {entry.model_id && <code>{entry.model_id}</code>}
          {entry.training_stage && <code>{entry.training_stage}</code>}
        </div>
        <ArrowUpRight size={16} aria-hidden="true" />
      </div>
    </Link>
  );
}
