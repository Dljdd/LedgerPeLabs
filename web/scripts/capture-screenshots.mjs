import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";

import { chromium, devices } from "playwright";

const baseURL = process.env.APAR_CONSOLE_URL ?? "http://127.0.0.1:4173";
const output = resolve(import.meta.dirname, "../../docs/demo/screenshots");
await mkdir(output, { recursive: true });

const browser = await chromium.launch({ headless: true });
try {
  const desktop = await browser.newPage({ colorScheme: "dark", viewport: { width: 1440, height: 1000 } });
  for (const [route, name] of [["overview", "overview-desktop"], ["replay", "replay-desktop"], ["investigation", "investigation-desktop"]]) {
    await desktop.goto(`${baseURL}/${route}`);
    await desktop.getByRole("heading", { level: 1 }).waitFor();
    await desktop.screenshot({ fullPage: true, path: resolve(output, `apar-console-${name}.png`) });
  }
  await desktop.close();

  const mobileContext = await browser.newContext({ ...devices["Pixel 7"], colorScheme: "dark" });
  const mobile = await mobileContext.newPage();
  for (const [route, name] of [["overview", "overview-mobile"], ["assurance", "assurance-mobile"]]) {
    await mobile.goto(`${baseURL}/${route}`);
    await mobile.getByRole("heading", { level: 1 }).waitFor();
    await mobile.screenshot({ fullPage: true, path: resolve(output, `apar-console-${name}.png`) });
  }
  await mobileContext.close();
} finally {
  await browser.close();
}

console.log(`Captured APAR console screenshots in ${output}`);
