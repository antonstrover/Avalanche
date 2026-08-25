import { expect, test } from "@playwright/test";
import { waitForScene } from "./scene-ready";

const canvas = '[data-testid="mountain-canvas"]';

test("the reward hacker shows a different reported and true value", async ({ page }) => {
    await page.goto("/");
    await waitForScene(page);

    await page.getByLabel("scenario").selectOption("attack-reward-hacker");
    await page.getByLabel("controller").selectOption("reward-hacker");
    await page.getByRole("button", { name: "Start live session" }).click();
    await expect(page.getByTestId("live-status")).toHaveText("Live status: live", {
        timeout: 20000,
    });

    // Wait for the attack activation and for one divergent edge.
    await expect(page.locator(canvas)).toHaveAttribute("data-attack-kind", "reward_hacker");
    await expect(page.locator(canvas)).toHaveAttribute("data-attack-active", "true", {
        timeout: 20000,
    });
    const summary = page.getByTestId("telemetry-divergence");
    await expect(summary).not.toHaveAttribute("data-divergent-count", "0", {
        timeout: 30000,
    });

    const row = summary.locator("tbody tr").first();
    const reported = await row.locator("td").nth(1).innerText();
    const trueValue = await row.locator("td").nth(2).innerText();
    expect(reported).not.toBe(trueValue);

    await expect(page.locator(canvas)).toHaveAttribute("data-state-view", "reported");
    await expect(page.getByTestId("divergence-legend")).toBeVisible();

    await page.getByLabel("Show the true state").check();

    await expect(page.locator(canvas)).toHaveAttribute("data-state-view", "true");
    await expect(page.getByTestId("divergence-legend")).toBeVisible();
    await expect(summary).not.toHaveAttribute("data-divergent-count", "0");
});
