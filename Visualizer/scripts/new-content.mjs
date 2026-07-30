import { promises as fs } from "node:fs";
import path from "node:path";
import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { contentRoot, slugify, supportedTypes } from "./content-utils.mjs";

const prompt = readline.createInterface({ input, output });
const requestedType = (await prompt.question(
  "Type (logs, evals, findings): ",
)).trim();
if (!supportedTypes.includes(requestedType)) {
  prompt.close();
  throw new Error(`Unsupported type: ${requestedType}`);
}

const title = (await prompt.question("Title: ")).trim();
const model = (await prompt.question("Model id (optional): ")).trim();
const date = new Date().toISOString().slice(0, 10);
const slug = `${date}-${slugify(title)}`;
const directory = path.join(contentRoot, requestedType, slug);
const file = path.join(directory, "index.md");

const extra =
  requestedType === "evals"
    ? `eval_suite: replace-me\neval_version: v1\ndataset_version: replace-me\nmetrics: {}\n`
    : "";
const modelLine = model ? `model_id: ${model}\nmodels:\n  - ${model}\n` : "";
const markdown = `---
title: "${title.replaceAll('"', '\\"')}"
date: ${date}
status: draft
${modelLine}${extra}tags: []
---

# ${title}

> [!NOTE]
> Replace this generated scaffold with the research record.
`;

await fs.mkdir(directory, { recursive: true });
await fs.writeFile(file, markdown, { flag: "wx" });
prompt.close();
console.log(`Created ${path.relative(process.cwd(), file)}`);

