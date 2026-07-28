import { promises as fs } from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import {
  contentRoot,
  contentTypeFor,
  projectRoot,
  slugify,
  titleFromMarkdown,
  toPosix,
  walk,
} from "./content-utils.mjs";

const outputDirectory = path.join(projectRoot, "lib", "generated");
const outputFile = path.join(outputDirectory, "content-index.json");
const publicAssetRoot = path.join(projectRoot, "public", "content-assets");

function normalize(value) {
  return JSON.parse(JSON.stringify(value));
}

function rewriteAssetLinks(body, type, slug) {
  const prefix = `/content-assets/${type}/${slug}/`;
  return body
    .replace(/(\]\()\.\/(assets|artifacts)\//g, `$1${prefix}$2/`)
    .replace(/(<img[^>]+src=["'])\.\/(assets|artifacts)\//g, `$1${prefix}$2/`);
}

async function copyEntryAssets(entryDirectory, type, slug) {
  const files = await walk(entryDirectory);
  const copied = [];
  for (const file of files) {
    if (file.endsWith(".md")) continue;
    const relative = path.relative(entryDirectory, file);
    if (relative.startsWith("..")) continue;
    const destination = path.join(publicAssetRoot, type, slug, relative);
    await fs.mkdir(path.dirname(destination), { recursive: true });
    await fs.copyFile(file, destination);
    const stat = await fs.stat(file);
    copied.push({
      name: path.basename(file),
      path: `/content-assets/${type}/${slug}/${toPosix(relative)}`,
      size_bytes: stat.size,
      kind: path.extname(file).slice(1).toLowerCase() || "file",
    });
  }
  return copied;
}

const markdownFiles = (await walk(contentRoot)).filter((file) =>
  file.endsWith(".md"),
);
const entries = [];
const seenIds = new Set();

for (const file of markdownFiles) {
  const type = contentTypeFor(file);
  if (!type) continue;

  const raw = await fs.readFile(file, "utf8");
  const parsed = matter(raw);
  const entryDirectory = path.dirname(file);
  const directorySlug = path.basename(entryDirectory);
  const slug = slugify(parsed.data.slug || directorySlug || path.basename(file, ".md"));
  const id = String(parsed.data.id || `${type}:${slug}`);
  if (seenIds.has(id)) throw new Error(`Duplicate content id: ${id}`);
  seenIds.add(id);

  const stat = await fs.stat(file);
  const assets = await copyEntryAssets(entryDirectory, type, slug);
  const rawDate = parsed.data.date || stat.mtime;
  const date =
    rawDate instanceof Date
      ? rawDate.toISOString().slice(0, 10)
      : String(rawDate).slice(0, 10);
  const title = String(
    parsed.data.title ||
      titleFromMarkdown(parsed.content, path.basename(file, ".md")),
  );

  entries.push({
    ...normalize(parsed.data),
    id,
    slug,
    type,
    title,
    date,
    summary: String(parsed.data.summary || ""),
    status: String(parsed.data.status || "unknown"),
    tags: Array.isArray(parsed.data.tags) ? parsed.data.tags.map(String) : [],
    models: Array.isArray(parsed.data.models)
      ? parsed.data.models.map(String)
      : parsed.data.model_id
        ? [String(parsed.data.model_id)]
        : [],
    metrics:
      parsed.data.metrics && typeof parsed.data.metrics === "object"
        ? normalize(parsed.data.metrics)
        : {},
    body: rewriteAssetLinks(parsed.content, type, slug),
    source_path: toPosix(path.relative(projectRoot, file)),
    assets,
  });
}

entries.sort((a, b) => b.date.localeCompare(a.date) || a.title.localeCompare(b.title));

await fs.mkdir(outputDirectory, { recursive: true });
await fs.writeFile(
  outputFile,
  `${JSON.stringify(
    {
      generated: true,
      generated_at: new Date().toISOString(),
      entry_count: entries.length,
      entries,
    },
    null,
    2,
  )}\n`,
  "utf8",
);

console.log(`Indexed ${entries.length} research entries.`);
