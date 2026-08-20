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
