import { expect, test } from "@playwright/test";
import { waitForScene } from "./scene-ready";

async function measuredFrames(page: Parameters<typeof waitForScene>[0], seconds: number) {
    return page.evaluate(
        (duration) =>
            new Promise<number>((resolve) => {
                const start = performance.now();
                let count = 0;
                const draw = (now: number) => {
                    count += 1;
                    if (now - start >= duration * 1000) resolve(count);
                    else requestAnimationFrame(draw);
                };
                requestAnimationFrame(draw);
            }),
        seconds,
    );
}

test("a live session draws 5000 skiers smoothly", async ({ page }) => {
    await page.goto("/");
    await waitForScene(page);
    await page.getByRole("button", { name: "Start live session" }).click();
    await expect(page.getByTestId("live-status")).toHaveText("Live status: live", {
        timeout: 15000,
    });
    await expect(page.getByTestId("live-skier-count")).toHaveText("Live skiers: 5000");
    await page.waitForTimeout(2000);

    const frames = await measuredFrames(page, 5);
    expect(frames / 5).toBeGreaterThanOrEqual(30);
    await expect(page.getByTestId("live-status")).toHaveText("Live status: live");
});

test("a live session shows an honest proposal", async ({ page }) => {
    await page.goto("/");
    await waitForScene(page);
    await page.getByRole("button", { name: "Start live session" }).click();
    await expect(page.getByTestId("proposal-controller")).toHaveText("honest", {
        timeout: 15000,
    });
    await expect(page.getByTestId("proposal-explanation")).not.toBeEmpty();
});

test("a live failure appears on the timeline", async ({ page }) => {
    await page.goto("/");
    await waitForScene(page);
    await page.getByRole("button", { name: "Start failure demo" }).click();
    await expect(page.getByTestId("live-status")).toHaveText("Live status: live", {
        timeout: 15000,
    });
    const marker = page.locator('[data-event-type="failure_started"]');
    await expect(marker).toContainText("lift stoppage", { timeout: 15000 });
    await expect(marker).toContainText("praz_plaza->plan_bois");
});

test("a monitor rule appears in the decision inspector", async ({ page }) => {
    await page.goto("/");
    await waitForScene(page);
    await page.getByRole("button", { name: "Start monitor demo" }).click();
    await expect(page.getByTestId("decision-type")).toHaveText("BLOCK", {
        timeout: 15000,
    });
    await expect(page.getByText("EVACUATION_ROUTE_CLOSURE", { exact: true })).toBeVisible();
});

test("a decision reference selects its mountain item", async ({ page }) => {
    await page.goto("/");
    await waitForScene(page);
    await page.getByRole("button", { name: "Start monitor demo" }).click();
    const reference = page.locator('[data-testid="related-infrastructure"] button').first();
    await expect(reference).toBeVisible({ timeout: 15000 });

    await reference.click();

    await expect(reference).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByTestId("selection")).not.toHaveText("nothing selected");
});

test("an escalation is approved before the time limit", async ({ page }) => {
    await page.goto("/");
    await waitForScene(page);
    await page.getByRole("button", { name: "Start approval demo" }).click();
    await expect(page.getByTestId("approval-panel")).toBeVisible({ timeout: 15000 });
    await expect(page.getByTestId("approval-deadline")).toContainText("seconds remain");
    await page.getByRole("button", { name: "Approve" }).click();
    await expect(page.getByTestId("approval-panel")).toBeHidden({ timeout: 15000 });
    await expect(page.getByTestId("proposal-controller")).toHaveText("rule-demo");
});
