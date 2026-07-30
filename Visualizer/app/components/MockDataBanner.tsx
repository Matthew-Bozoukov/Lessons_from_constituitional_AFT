import { TriangleAlert } from "lucide-react";

/**
 * The unmissable mock-data warning.
 *
 * A fabricated fixture that reads as a research result is the worst failure this
 * site can have, so the warning does not sit in a footnote or a tag chip. It is
 * a full-width yellow bar with dark text, placed above the content it describes,
 * and it says what the data is not: not a measurement, not citable.
 *
 * `scope` distinguishes "this page is entirely fixtures" from "some of what is
 * listed below is a fixture", because those call for different reader caution.
 */
export function MockDataBanner({
  scope = "entry",
  detail,
}: {
  scope?: "entry" | "all" | "some";
  detail?: string;
}) {
  const headline =
    scope === "all"
      ? "Everything on this page is mock data"
      : scope === "some"
        ? "Some entries on this page are mock data"
        : "This is mock data";

  const body =
    detail ??
    (scope === "some"
      ? "Entries marked MOCK below are fabricated interface fixtures. They are not measurements of any model and support no claim. Unmarked entries are real."
      : "Fabricated interface fixture. Every number, transcript and model id below was invented to exercise this interface. It is not a measurement of any model, it supports no claim, and it must not be cited or trained on.");

  return (
    <aside className="mock-banner" role="alert" aria-live="polite">
      <TriangleAlert className="mock-banner-icon" aria-hidden="true" />
      <div className="mock-banner-copy">
        <strong>{headline}</strong>
        <span>{body}</span>
      </div>
    </aside>
  );
}

/** The compact marker for a listing card or a row in a table. */
export function MockBadge() {
  return (
    <span className="mock-badge" title="Fabricated interface fixture, not a research result">
      <TriangleAlert size={11} aria-hidden="true" />
      Mock
    </span>
  );
}
