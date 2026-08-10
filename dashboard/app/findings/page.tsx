import type { Metadata } from "next";
import { AlertTriangle, CheckCircle2, Lightbulb } from "lucide-react";
import { EntryCard } from "../components/EntryCard";
import { entriesOfType } from "@/lib/content";
import { CorpusMix } from "../components/CorpusMix";
import { MockDataBanner } from "../components/MockDataBanner";
import { allMock, anyMock } from "@/lib/content";

export const metadata: Metadata = { title: "Findings" };

export default function FindingsPage() {
  const findings = entriesOfType("findings");
  return (
    <main className="page-container inner-page">
      {anyMock(findings) && (
        <MockDataBanner scope={allMock(findings) ? "all" : "some"} />
      )}
      <header className="page-heading">
        <div>
          <span className="eyebrow">Interpretation layer</span>
          <h1>Findings</h1>
          <p>
            The claims this project is prepared to make, kept deliberately
            separate from the runs that produced them. A finding here states
            what was concluded, what supports it, and what would overturn it —
            including the ones where the answer was that nothing happened.
          </p>
          <CorpusMix entries={findings} noun="findings" />
        </div>
      </header>
      <div className="principle-strip">
        <span><Lightbulb size={16} /> State the claim</span>
        <span><CheckCircle2 size={16} /> Link compatible evidence</span>
        <span><AlertTriangle size={16} /> Preserve counterevidence</span>
      </div>
      <div className="collection-list">
        {findings.map((entry) => <EntryCard entry={entry} key={entry.id} />)}
      </div>
    </main>
  );
}

