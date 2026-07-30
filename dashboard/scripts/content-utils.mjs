import { promises as fs } from "node:fs";
import path from "node:path";

export const projectRoot = path.resolve(import.meta.dirname, "..");
export const contentRoot = path.join(projectRoot, "content");
export const supportedTypes = [
  "logs",
  "evals",
  "findings",
  "datasets",
  "petri-runs",
];

export async function walk(directory) {
  const results = [];
  let entries = [];
  try {
    entries = await fs.readdir(directory, { withFileTypes: true });
  } catch (error) {
    if (error.code === "ENOENT") return results;
    throw error;
  }

  for (const entry of entries) {
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) results.push(...(await walk(fullPath)));
    else results.push(fullPath);
  }
  return results;
}

export function slugify(value) {
  return value
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 80);
}

export function contentTypeFor(filePath) {
  const relative = path.relative(contentRoot, filePath);
  const type = relative.split(path.sep)[0];
  return supportedTypes.includes(type) ? type : null;
}

export function titleFromMarkdown(markdown, fallback) {
  const heading = markdown.match(/^#\s+(.+)$/m);
  return heading?.[1]?.trim() || fallback;
}

export function toPosix(value) {
  return value.split(path.sep).join("/");
}
