import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";
import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import matter from "gray-matter";
import {
  contentRoot,
  slugify,
  supportedTypes,
  titleFromMarkdown,
} from "./content-utils.mjs";

const args = process.argv.slice(2);
const sourceArgument = args.find((arg) => !arg.startsWith("--"));
const typeFlag = args.find((arg) => arg.startsWith("--type="));
const type = typeFlag?.split("=")[1] || "logs";
if (!sourceArgument) throw new Error("Usage: npm run import:content -- <file> [--type=logs]");
if (!supportedTypes.includes(type)) throw new Error(`Unsupported type: ${type}`);

const source = path.resolve(sourceArgument);
const original = await fs.readFile(source, "utf8");
const parsed = matter(original);
const stat = await fs.stat(source);
const title = String(
  parsed.data.title ||
    titleFromMarkdown(parsed.content, path.basename(source, path.extname(source))),
);
const date = String(parsed.data.date || stat.mtime.toISOString().slice(0, 10));
const slug = `${date}-${slugify(title)}`;
const destination = path.join(contentRoot, type, slug, "index.md");
const proposed = {
  ...parsed.data,
  title,
  date,
  status: parsed.data.status || "imported",
  generated: {
    by: "research-log-importer",
    source_path: source,
    generated_at: new Date().toISOString(),
    source_hash: `sha256:${createHash("sha256").update(original).digest("hex")}`,
  },
};

console.log("\nProposed metadata:\n", proposed);
if (!args.includes("--yes")) {
  const prompt = readline.createInterface({ input, output });
  const answer = (await prompt.question("\nCreate prepared copy? (y/N) ")).trim();
  prompt.close();
  if (answer.toLowerCase() !== "y") {
    console.log("No files written.");
    process.exit(0);
  }
}

await fs.mkdir(path.dirname(destination), { recursive: true });
await fs.writeFile(destination, matter.stringify(parsed.content, proposed), {
  flag: "wx",
});
console.log(`Prepared copy created at ${path.relative(process.cwd(), destination)}`);

