import { expect, test } from "@playwright/test";
import { waitForScene } from "./scene-ready";

test("the scene draws the hazards and the weather", async ({ page }) => {
    // The reduced motion setting holds the snow still, so the image stays stable.
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto("/");

    await waitForScene(page);

    await expect(page).toHaveScreenshot("mountain-conditions.png");
});
