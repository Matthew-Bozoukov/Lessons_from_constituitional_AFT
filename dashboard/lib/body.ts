// Reads an entry's Markdown body from its generated sidecar.
//
// This used to be `fs.readFile(process.cwd() + "/lib/generated/bodies/...")`,
// which silently returned "" for EVERY entry: the RSC server runs with
// `process.cwd() === "/bundle"`, a virtual filesystem that contains only what
// the bundler put there. `lib/generated/bodies` is generated after the graph is
// resolved, so it was never in there. The catch-and-return-"" that was supposed
// to degrade a single missing body gracefully instead swallowed a total failure,
// and every writeup on the site rendered as an empty <div class="markdown-body">
// under a fully-populated header - which looks like "this entry has no prose"
// rather than "the loader is broken".
//
// `import.meta.glob` resolves at build time through the bundler, so the sidecars
// travel with the code. Left lazy (no `eager`) so each body stays its own chunk
// and opening one entry does not download all nineteen - the property the
// sidecar split existed for in the first place.

const bodies = import.meta.glob<string>("./generated/bodies/*.md", {
  query: "?raw",
  import: "default",
});

/**
 * The rendered Markdown for one entry, or an empty string when the sidecar is
 * missing. A missing body must not fail a build: an entry with no prose still
 * has a title, metrics and provenance worth rendering, which is the same way
 * every other missing payload degrades here.
 */
export async function entryBody(slug: string): Promise<string> {
  const load = bodies[`./generated/bodies/${slug}.md`];
  if (!load) return "";
  try {
    return await load();
  } catch {
    return "";
  }
}
