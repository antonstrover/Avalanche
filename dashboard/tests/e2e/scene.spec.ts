import { expect, test } from "@playwright/test";
import { waitForScene } from "./scene-ready";

// The camera stays at the preset, so a point on the canvas hits the same piste.
const PISTE_POINT = { x: 440, y: 220 };

test("the scene selects a piste on a click", async ({ page }) => {
    await page.goto("/");

    const canvas = await waitForScene(page);
    await expect(page.getByTestId("selection")).toHaveText("nothing selected");

    await canvas.click({ position: PISTE_POINT });
    await expect(page.getByTestId("selection")).toHaveText(/^piste: \d+$/);
});
