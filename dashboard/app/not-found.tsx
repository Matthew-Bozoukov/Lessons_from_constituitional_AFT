import Link from "next/link";

export default function NotFound() {
  return (
    <main className="page-container not-found">
      <code>404 / corpus-miss</code>
      <h1>Research record not found</h1>
      <p>The entry may have moved or the generated index needs to be refreshed.</p>
      <Link href="/" className="button primary">Return to overview</Link>
    </main>
  );
}

