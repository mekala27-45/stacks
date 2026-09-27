import { build } from "../web/node_modules/esbuild/lib/main.js";
import { mkdir, copyFile, readdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const server = path.join(root, "dist", "server"), assets = path.join(root, "dist", "client");
await mkdir(server, { recursive: true });
await mkdir(path.join(assets, "data"), { recursive: true });
await build({ entryPoints: [path.join(root, "packages", "edge", "worker.ts")], bundle: true,
  outfile: path.join(server, "index.js"), platform: "browser", format: "esm", target: "es2022", sourcemap: true });
for (const name of await readdir(path.join(root, "web", "public", "data"))) {
  if (name === "catalog.json" || name === "readers.json" || name.startsWith("edge-") || /\.(f32|u32)$/.test(name)) {
    await copyFile(path.join(root, "web", "public", "data", name), path.join(assets, "data", name));
  }
}
await writeFile(path.join(assets, "index.html"), '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>stacks model service</title><body><h1>stacks model service</h1><p>Evaluated recommendation artifacts, persistent demonstration sessions, and server-sent shelf updates.</p><p><a href="https://mekala27-45.github.io/stacks/">Open the bookstore</a> · <a href="/health">Service health</a> · <a href="https://github.com/mekala27-45/stacks">Source and methods</a></p></body></html>');
await writeFile(path.join(server, "wrangler.json"), JSON.stringify({ name: "stacks-recommender-api", main: "index.js",
  compatibility_date: "2026-09-26", no_bundle: true,
  assets: { directory: "../client", binding: "ASSETS", run_worker_first: ["/v1/*", "/health"] },
  d1_databases: [{ binding: "DB", database_name: "stacks-demo", database_id: "00000000-0000-4000-8000-000000000000" }],
  vars: { CORS_ORIGINS: "https://mekala27-45.github.io,http://127.0.0.1:3010,http://localhost:3010" },
}, null, 2));
console.log("Built portable model API and frozen model assets in dist.");
