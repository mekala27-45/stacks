import { cp, mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Keep the complete research repository on GitHub; Sites receives only serving inputs.
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const destination = path.resolve(process.argv[2] ?? path.join(root, ".cache", "site-source"));
for (const directory of ["scripts", "packages/edge", "web/public/data", ".openai"]) {
  await mkdir(path.join(destination, directory), { recursive: true });
}
for (const filename of [".gitattributes", ".openai/hosting.json", "packages/edge/worker.ts", "packages/edge/schema.sql"]) {
  await cp(path.join(root, filename), path.join(destination, filename));
}
await cp(path.join(root, "drizzle"), path.join(destination, "drizzle"), { recursive: true });
for (const filename of await readdir(path.join(root, "web/public/data"))) {
  if (["catalog.json", "readers.json", "LICENSE-goodbooks.txt"].includes(filename) || filename.startsWith("edge-") || /\.(f32|u32)$/.test(filename)) {
    await cp(path.join(root, "web/public/data", filename), path.join(destination, "web/public/data", filename));
  }
}
const build = (await readFile(path.join(root, "scripts/build-edge.mjs"), "utf8"))
  .replace('"../web/node_modules/esbuild/lib/main.js"', '"esbuild"');
await writeFile(path.join(destination, "scripts/build-edge.mjs"), build);
await writeFile(path.join(destination, "package.json"), JSON.stringify({ private: true, type: "module", scripts: { build: "node scripts/build-edge.mjs" }, devDependencies: { esbuild: "0.28.2" } }, null, 2) + "\n");
await writeFile(path.join(destination, ".gitignore"), "node_modules/\ndist/\nartifacts/\n.wrangler/\n");
if (existsSync(path.join(destination, ".git"))) {
  // Reapply changed attributes even when Git's stat cache considers the file clean.
  execFileSync("git", ["add", "--renormalize", "--", "web/public/data/edge-profiles.json"], { cwd: destination });
}
const commit = execFileSync("git", ["rev-parse", "HEAD"], { cwd: root, encoding: "utf8" }).trim();
await writeFile(path.join(destination, "SOURCE.json"), JSON.stringify({ repository: "https://github.com/mekala27-45/stacks", commit, purpose: "Portable serving source and frozen artifacts. Full training data, notebooks, tests and reports are retained in the linked repository." }, null, 2) + "\n");
console.log(`Prepared portable serving source at ${destination}`);
