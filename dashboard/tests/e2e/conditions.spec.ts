import { expect, test } from "@playwright/test";
import { waitForScene } from "./scene-ready";

test("the scene draws live conditions", async ({ page }) => {
    // The reduced motion setting holds the snow still, so the image stays stable.
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto("/");

    await waitForScene(page);
    await expect(page.getByRole("heading", { name: "Event timeline" })).toBeVisible();
    await expect(page.getByText("No material events yet.")).toBeVisible();
});
