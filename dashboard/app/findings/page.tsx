import type { Metadata } from "next";
import { AlertTriangle, CheckCircle2, Lightbulb } from "lucide-react";
import { EntryCard } from "../components/EntryCard";
import { entriesOfType } from "@/lib/content";
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
            Claims are curated separately from raw logs and eval outputs. Each
            should retain its uncertainty, supporting evidence, counterevidence,
            and unresolved vulnerability checks.
          </p>
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

