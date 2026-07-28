import { promises as fs } from "node:fs";
import matter from "gray-matter";
import { contentRoot, contentTypeFor, walk } from "./content-utils.mjs";

const files = (await walk(contentRoot)).filter((file) => file.endsWith(".md"));
let errors = 0;
let warnings = 0;

function report(level, file, message) {
  if (level === "ERROR") errors += 1;
  else warnings += 1;
  console.log(`${level} ${file}: ${message}`);
}

for (const file of files) {
  const type = contentTypeFor(file);
  const parsed = matter(await fs.readFile(file, "utf8"));
  if (!type) report("ERROR", file, "file is outside a supported collection");
  if (!parsed.data.title) report("ERROR", file, "missing title");
  if (!parsed.data.date) report("WARN", file, "missing date; file modification date will be used");
  if (type === "evals") {
    for (const field of ["model_id", "checkpoint_id", "eval_suite", "eval_version", "dataset_version"]) {
      if (!parsed.data[field]) report("WARN", file, `missing ${field}`);
    }
  }
  if (parsed.data.metrics && typeof parsed.data.metrics === "object") {
    for (const [name, metric] of Object.entries(parsed.data.metrics)) {
      if (!metric || typeof metric !== "object" || typeof metric.value !== "number") {
        report("ERROR", file, `metric ${name} must contain a numeric value`);
      }
    }
  }
}

console.log(`\nValidated ${files.length} files: ${errors} error(s), ${warnings} warning(s).`);
process.exitCode = errors ? 1 : 0;

