import { expect, test } from "@playwright/test";

// The camera stays at the preset, so a point on the canvas hits the same piste.
const PISTE_POINT = { x: 440, y: 220 };

test("the scene selects a piste on a click", async ({ page }) => {
    await page.goto("/");

    const canvas = page.locator('[data-testid="mountain-canvas"] canvas');
    await expect(canvas).toBeVisible();
    await expect(page.getByTestId("selection")).toHaveText("nothing selected");

    await canvas.click({ position: PISTE_POINT });
    await expect(page.getByTestId("selection")).toHaveText(/^piste: \d+$/);
});
