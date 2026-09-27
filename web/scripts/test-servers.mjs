import { spawn } from "node:child_process";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const root = path.resolve(web, "..");
mkdirSync(path.join(root, "artifacts"), { recursive: true });
const database = path
  .join(root, "artifacts", `playwright-${Date.now()}.db`)
  .replaceAll("\\", "/");
const python = path.join(
  root,
  ".venv",
  process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
);
const api = spawn(
  python,
  [
    "-m",
    "uvicorn",
    "packages.api.main:app",
    "--host",
    "127.0.0.1",
    "--port",
    "8001",
    "--no-access-log",
  ],
  {
    cwd: root,
    stdio: "inherit",
    env: {
      ...process.env,
      DATABASE_URL: `sqlite:///${database}`,
      SESSION_HASH_KEY: "playwright-test-key-not-for-production",
      CORS_ORIGINS: "http://127.0.0.1:3010",
      STACKS_DATA_DIR: path.join(web, "public", "data"),
    },
  },
);
const app = spawn(
  process.execPath,
  [
    path.join(web, "node_modules", "next", "dist", "bin", "next"),
    "dev",
    "--port",
    "3010",
    "--hostname",
    "127.0.0.1",
  ],
  {
    cwd: web,
    stdio: "inherit",
    env: {
      ...process.env,
      NEXT_PUBLIC_API_URL: "http://127.0.0.1:8001",
      NEXT_PUBLIC_BASE_PATH: "",
    },
  },
);
for (const child of [api, app])
  child.on("error", (error) => {
    console.error(error);
    process.exitCode = 1;
    stop();
  });
function stop() {
  api.kill();
  app.kill();
}
process.on("SIGTERM", stop);
process.on("SIGINT", stop);
process.on("exit", stop);
