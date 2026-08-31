import { defineConfig, devices } from "@playwright/test";

const python = process.env.APAR_PYTHON ?? "python3";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: true,
  retries: 0,
  reporter: [["line"], ["html", { open: "never" }]],
  use: {
    baseURL: "http://127.0.0.1:4173",
    colorScheme: "dark",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  projects: [
    { name: "desktop-chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile-chromium", use: { ...devices["Pixel 7"] } },
  ],
  webServer: {
    command: `${python} ../scripts/run_apar_console.py start --skip-build --port 4173`,
    cwd: ".",
    reuseExistingServer: false,
    timeout: 30_000,
    url: "http://127.0.0.1:4173/api/health",
  },
});
