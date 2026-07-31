// Reads an entry's Markdown body from its generated sidecar.
//
// Kept OUT of `lib/content.ts` on purpose: that module is imported by client
// components, and pulling `node:fs` into their graph would break the browser
// bundle. Nothing here may ever be imported from a `"use client"` file.
//
// Every route in this site prerenders, so these reads happen at build time and
// the body ends up inlined in the static HTML - the same place it was before,
// just no longer duplicated into every other page's payload.

import { promises as fs } from "node:fs";
import path from "node:path";

const bodyRoot = path.join(process.cwd(), "lib", "generated", "bodies");

/**
 * The rendered Markdown for one entry, or an empty string when the sidecar is
 * missing. A missing body must not fail a build: an entry with no prose still
 * has a title, metrics and provenance worth rendering, which is the same way
 * every other missing payload degrades here.
 */
export async function entryBody(slug: string): Promise<string> {
  try {
    return await fs.readFile(path.join(bodyRoot, `${slug}.md`), "utf8");
  } catch {
    return "";
  }
}
