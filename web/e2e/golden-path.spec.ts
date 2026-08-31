import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

const routes = [
  ["overview", "Adaptive payment assurance"],
  ["scenario", "Bounded campaign replay"],
  ["replay", "Verified decision replay"],
  ["investigation", "Campaign-level evidence"],
  ["defenses", "Four arms. One honest champion."],
  ["assurance", "Evidence before assertion."],
] as const;

test("every judge route renders without viewport overflow", async ({ page }) => {
  for (const [route, heading] of routes) {
    await page.goto(`/${route}`);
    await expect(page.getByRole("heading", { name: new RegExp(heading, "i") })).toBeVisible();
    const sizing = await page.evaluate(() => ({
      overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      offenders: [...document.querySelectorAll<HTMLElement>("body *")]
        .filter((element) => element.getBoundingClientRect().right > document.documentElement.clientWidth + 1)
        .slice(0, 8)
        .map((element) => ({ className: element.className, right: Math.round(element.getBoundingClientRect().right), tag: element.tagName })),
    }));
    expect(sizing.overflow, `${route} horizontal overflow: ${JSON.stringify(sizing.offenders)}`).toBeLessThanOrEqual(1);
  }
});

test("five-minute golden path moves from threat to assurance", async ({ page }) => {
  await page.goto("/overview");
  await expect(page.getByRole("img", { name: /12 ordered calibrated decisions/i })).toBeVisible();
  await page.getByRole("link", { name: /inspect scenario controls/i }).click();
  await expect(page.getByRole("heading", { name: "Bounded campaign replay" })).toBeVisible();

  await page.getByRole("link", { name: /start verified replay/i }).click();
  await expect(page.locator(".live-arm").getByText(/ensemble_with_graph/)).toBeVisible();
  await expect(page.getByText(/Event 01 \/ 12/)).toBeVisible();
  await expect(page.getByRole("img", { name: /14 genuine scenario entities and 10 ordered payment edges/i })).toBeVisible();
  await expect(page.getByText(/no payment-to-trace record mapping asserted/i)).toBeVisible();
  await expect(page.getByRole("img", { name: /calibrated risk 100.0%.*bound action thresholds/i })).toBeVisible();
  await page.getByRole("button", { name: /play both streams/i }).click();
  await expect(page.getByRole("button", { name: /pause both streams/i })).toBeVisible();
  await expect(page.getByText("SCENARIO PAYMENT 02")).toBeVisible();
  await expect(page.getByText(/Event 02 \/ 12/)).toBeVisible();
  await page.getByRole("button", { name: /pause both streams/i }).click();

  await page.getByRole("navigation", { name: "Primary" }).getByRole("link", { name: /investigation/i }).click();
  await expect(page.getByRole("img", { name: /14 linked entities and 10 directional payment edges/i })).toBeVisible();
  await expect(page.getByText("Arrow = payment direction")).toBeVisible();

  await page.getByRole("navigation", { name: "Primary" }).getByRole("link", { name: /defenses/i }).click();
  await expect(page.getByText("Recovered diagnostic evidence — non-authoritative")).toBeVisible();
  await expect(page.getByText(/Graph ensemble usable/i)).toBeVisible();

  await page.getByRole("navigation", { name: "Primary" }).getByRole("link", { name: /assurance/i }).click();
  await expect(page.getByRole("region", { name: /agentic integrity proof/i })).toBeVisible();
  await expect(page.getByText(/no Kaggle locked-successor\/seed-2404 chain was run/i)).toBeVisible();
});

test("overview, replay, and assurance pass automated accessibility checks", async ({ page }) => {
  for (const route of ["overview", "replay", "assurance"]) {
    await page.goto(`/${route}`);
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
    const results = await new AxeBuilder({ page }).analyze();
    expect(results.violations, `${route}: ${JSON.stringify(results.violations)}`).toEqual([]);
  }
});

test("primary journey is keyboard reachable", async ({ page }) => {
  await page.goto("/overview");
  await expect(page.getByRole("heading", { name: /adaptive payment assurance/i })).toBeVisible();
  await page.keyboard.press("Tab");
  await expect(page.getByRole("link", { name: /skip to main content/i })).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.locator("#main-content")).toBeFocused();

  const overviewEvent = page.getByRole("button", { name: /event 4, 70.5%, review hold/i });
  await overviewEvent.focus();
  await page.keyboard.press("Enter");
  await expect(overviewEvent).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("status", { name: /focused trace event/i })).toContainText("Event 04");

  const replayLink = page.getByRole("navigation", { name: "Primary" }).getByRole("link", { name: /replay/i });
  await replayLink.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("heading", { name: "Verified decision replay" })).toBeVisible();
  await page.getByRole("button", { name: "Play both streams" }).focus();
  await page.keyboard.press("Enter");
  await expect(page.getByText(/Event 02 \/ 12/)).toBeVisible();
  await page.getByRole("button", { name: "Pause both streams" }).focus();
  await page.keyboard.press("Enter");

  const assuranceLink = page.getByRole("navigation", { name: "Primary" }).getByRole("link", { name: /assurance/i });
  await assuranceLink.focus();
  await page.keyboard.press("Enter");
  const checkpoint = page.getByRole("button", { name: "Inspect Stage 30 source checkpoint lineage" });
  await checkpoint.focus();
  await page.keyboard.press("Enter");
  await expect(checkpoint).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("status", { name: "Selected lineage artifact" })).toContainText("Stage 30 source checkpoint");
});

test("reduced motion campaign advances only on explicit steps", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/replay");

  const stepCampaign = page.getByRole("button", { name: "Step both streams" });
  await expect(stepCampaign).toBeVisible();
  await stepCampaign.click();
  await expect(page.getByText("SCENARIO PAYMENT 02")).toBeVisible();
  await expect(page.getByText("Event 02 / 12")).toBeVisible();
  await page.waitForTimeout(1100);
  await expect(page.getByText("SCENARIO PAYMENT 02")).toBeVisible();

  const animationName = await page.locator(".replay-value-packet").evaluate((element) => getComputedStyle(element).animationName);
  expect(animationName).toBe("none");
});
