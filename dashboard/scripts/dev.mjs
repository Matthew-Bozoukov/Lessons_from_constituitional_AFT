import { spawn } from "node:child_process";
import { watch } from "node:fs";
import path from "node:path";
import { contentRoot } from "./content-utils.mjs";
import { projectRoot } from "./content-utils.mjs";

await import("./index-content.mjs");

let timer;
try {
  watch(contentRoot, { recursive: true }, () => {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      const url = new URL(`./index-content.mjs?time=${Date.now()}`, import.meta.url);
      try {
        await import(url.href);
      } catch (error) {
        console.error("Content indexing failed:", error.message);
      }
    }, 120);
  });
} catch {
  console.warn("Content watching is unavailable; restart after content changes.");
}

const cli = path.join(projectRoot, "node_modules", "vinext", "dist", "cli.js");
const child = spawn(process.execPath, [cli, "dev"], {
  stdio: "inherit",
  env: {
    ...process.env,
    WRANGLER_LOG_PATH: ".wrangler/wrangler.log",
  },
});

child.on("exit", (code) => process.exit(code ?? 1));
