import { expect, type Page, type Locator } from "@playwright/test";

// The renderer measures its container after the first layout. It draws at the
// default canvas size until then. A test must wait for the full size, or it
// clicks the wrong point.
export async function waitForScene(page: Page): Promise<Locator> {
    const canvas = page.locator('[data-testid="mountain-canvas"] canvas');
    await expect(canvas).toBeVisible();
    await expect(page.getByTestId("mountain-canvas")).toHaveAttribute("data-drawn", "true");
    await expect
        .poll(async () => (await canvas.boundingBox())?.width ?? 0)
        .toBeGreaterThan(1000);
    return canvas;
}
