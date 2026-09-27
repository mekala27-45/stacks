import { defineConfig, devices } from "@playwright/test";
export default defineConfig({
  testDir: "./e2e",
  timeout: 60000,
  expect: { timeout: 20000 },
  workers: 1,
  use: { baseURL: "http://127.0.0.1:3010", trace: "retain-on-failure" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "node scripts/test-servers.mjs",
    url: "http://127.0.0.1:3010",
    reuseExistingServer: false,
    timeout: 120000,
  },
});
