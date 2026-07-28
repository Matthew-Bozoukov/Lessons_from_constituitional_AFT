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
const publicDatasetRoot = path.join(projectRoot, "public", "generated-datasets");

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

async function readJsonl(file) {
  const raw = await fs.readFile(file, "utf8");
  return raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch {
        throw new Error(`Invalid JSONL at ${file}:${index + 1}`);
      }
    });
}

function messagesFor(record) {
  const candidate =
    record.messages || record.conversation || record.turns || record.dialogue;
  if (Array.isArray(candidate)) return candidate;
  if (record.prompt !== undefined || record.response !== undefined) {
    return [
      { role: "user", content: String(record.prompt || "") },
      { role: "assistant", content: String(record.response || "") },
    ];
  }
  return [];
}

async function buildDatasetManifest(entryDirectory, slug) {
  const files = await walk(entryDirectory);
  const dataFile = files.find(
    (file) => file.endsWith(".jsonl") && file.includes(`${path.sep}data${path.sep}`),
  );
  if (!dataFile) return null;

  const records = await readJsonl(dataFile);
  const chunkSize = 50;
  const chunks = [];
  const destinationRoot = path.join(publicDatasetRoot, slug);
  await fs.mkdir(destinationRoot, { recursive: true });

  for (let index = 0; index < records.length; index += chunkSize) {
    const chunkNumber = Math.floor(index / chunkSize);
    const name = `chunk-${String(chunkNumber).padStart(3, "0")}.json`;
    await fs.writeFile(
      path.join(destinationRoot, name),
      `${JSON.stringify(records.slice(index, index + chunkSize))}\n`,
      "utf8",
    );
    chunks.push(`/generated-datasets/${slug}/${name}`);
  }

  const turns = records.map((record) => messagesFor(record).length);
  const roleCounts = {};
  const splits = {};
  const categories = {};
  for (const record of records) {
    for (const message of messagesFor(record)) {
      const role = String(message.role || "unknown");
      roleCounts[role] = (roleCounts[role] || 0) + 1;
    }
    const metadata = record.metadata || {};
    const split = String(metadata.split || record.split || "unspecified");
    const category = String(metadata.category || record.category || "uncategorized");
    splits[split] = (splits[split] || 0) + 1;
    categories[category] = (categories[category] || 0) + 1;
  }

  return {
    source_file: `/content-assets/datasets/${slug}/${toPosix(path.relative(entryDirectory, dataFile))}`,
    format: "jsonl",
    record_count: records.length,
    chunk_size: chunkSize,
    chunks,
    stats: {
      average_turns:
        turns.length > 0
          ? Number((turns.reduce((sum, value) => sum + value, 0) / turns.length).toFixed(1))
          : 0,
      role_counts: roleCounts,
      splits,
      categories,
    },
  };
}

async function buildPetriManifest(entryDirectory) {
  const files = await walk(entryDirectory);
  const scenarioFile = files.find((file) => file.endsWith("scenarios.jsonl"));
  const transcriptFile = files.find((file) => file.endsWith("transcripts.jsonl"));
  const scoreFile = files.find((file) => file.endsWith("scores.json"));
  if (!scenarioFile && !transcriptFile && !scoreFile) return null;
  return {
    scenarios: scenarioFile ? await readJsonl(scenarioFile) : [],
    transcripts: transcriptFile ? await readJsonl(transcriptFile) : [],
    scores: scoreFile
      ? JSON.parse(await fs.readFile(scoreFile, "utf8"))
      : {},
  };
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
  const dataset =
    type === "datasets"
      ? await buildDatasetManifest(entryDirectory, slug)
      : undefined;
  const petri =
    type === "petri-runs"
      ? await buildPetriManifest(entryDirectory)
      : undefined;
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
    ...(dataset ? { dataset } : {}),
    ...(petri ? { petri } : {}),
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
