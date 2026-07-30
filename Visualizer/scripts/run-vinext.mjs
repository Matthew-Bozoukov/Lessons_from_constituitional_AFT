import { spawn } from "node:child_process";
import path from "node:path";
import { projectRoot } from "./content-utils.mjs";

const mode = process.argv[2] || "dev";
const cli = path.join(projectRoot, "node_modules", "vinext", "dist", "cli.js");
const child = spawn(process.execPath, [cli, mode], {
  stdio: "inherit",
  env: {
    ...process.env,
    WRANGLER_LOG_PATH: ".wrangler/wrangler.log",
  },
});

child.on("exit", (code) => process.exit(code ?? 1));
