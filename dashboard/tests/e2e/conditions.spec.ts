import { expect, test } from "@playwright/test";

test("the scene draws the hazards and the weather", async ({ page }) => {
    // The reduced motion setting holds the snow still, so the image stays stable.
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto("/");

    const canvas = page.locator('[data-testid="mountain-canvas"] canvas');
    await expect(canvas).toBeVisible();
    await expect(page.getByTestId("mountain-canvas")).toHaveAttribute("data-drawn", "true");

    await expect(page).toHaveScreenshot("mountain-conditions.png");
});
