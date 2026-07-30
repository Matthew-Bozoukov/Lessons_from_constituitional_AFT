import { spawn } from "node:child_process";
import path from "node:path";
import "./index-content.mjs";
import { projectRoot } from "./content-utils.mjs";

const cli = path.join(projectRoot, "node_modules", "vinext", "dist", "cli.js");
const child = spawn(process.execPath, [cli, "build"], {
  stdio: "inherit",
  env: {
    ...process.env,
    WRANGLER_LOG_PATH: ".wrangler/wrangler.log",
  },
});

child.on("exit", (code) => process.exit(code ?? 1));
