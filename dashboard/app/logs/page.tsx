import type { Metadata } from "next";
import { FileDown, Terminal } from "lucide-react";
import { EntryCard } from "../components/EntryCard";
import { entriesOfType } from "@/lib/content";
import { CorpusMix } from "../components/CorpusMix";
import { MockDataBanner } from "../components/MockDataBanner";
import { allMock, anyMock } from "@/lib/content";

export const metadata: Metadata = { title: "Experiment logs" };

export default function LogsPage() {
  const logs = entriesOfType("logs");
  const assets = logs.flatMap((entry) => entry.assets);

  return (
    <main className="page-container inner-page">
      {anyMock(logs) && (
        <MockDataBanner scope={allMock(logs) ? "all" : "some"} />
      )}
      <header className="page-heading">
        <div>
          <span className="eyebrow">Chronological source record</span>
          <h1>Experiment logs</h1>
          <p>
            What was run, what it cost, what broke, and what it showed —
            written up in order, most recent first. These are the narrative
            record: the evals and datasets pages hold the numbers, this holds
            the reasoning that connects them.
          </p>
          <CorpusMix entries={logs} noun="log entries" />
        </div>
        <div className="heading-stat">
          <Terminal size={19} />
          <strong>{logs.length}</strong>
          <span>readable records</span>
        </div>
        <div className="heading-stat">
          <FileDown size={19} />
          <strong>{assets.length}</strong>
          <span>linked artifacts</span>
        </div>
      </header>
      <div className="collection-list">
        {logs.map((entry) => <EntryCard entry={entry} key={entry.id} />)}
      </div>
    </main>
  );
}

