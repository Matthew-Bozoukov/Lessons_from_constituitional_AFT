import type { ResearchEntry } from "@/lib/content";
import { entryMix } from "@/lib/content";

/**
 * One line saying how much of a listing a human actually wrote.
 *
 * Listings render a write-up, a transcribed metric and a bare artifact link as
 * identical rows, so "30 evaluation runs" reads as thirty results when six are
 * analysed, nine are numbers copied from a published bundle that nobody has
 * interpreted, and fifteen are a link and a title. That difference is exactly
 * what a reader needs in order to know how much weight a row carries, and it
 * was nowhere on the page.
 */
export function CorpusMix({ entries, noun }: { entries: ResearchEntry[]; noun: string }) {
  const mix = entryMix(entries);
  if (!mix.total) return null;

  const parts = [
    mix.written > 0 && `${mix.written} written up`,
    mix.auto > 0 && `${mix.auto} carrying measured numbers with no write-up yet`,
    mix.stub > 0 && `${mix.stub} linked to their artifact only`,
  ].filter(Boolean) as string[];

  return (
    <p className="corpus-mix">
      <strong>{mix.total}</strong> {noun} —{" "}
      {parts.length > 1
        ? `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}`
        : parts[0]}
      .
    </p>
  );
}

/** The per-row marker, so the mix above is checkable against any given row. */
export function KindChip({ kind }: { kind: "written" | "auto" | "stub" }) {
  if (kind === "written") return null;
  return (
    <span className={`kind-chip ${kind}`}>
      {kind === "auto" ? "no write-up" : "link only"}
    </span>
  );
}
