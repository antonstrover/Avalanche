import { defineConfig, devices } from "@playwright/test";

// The browser tests run on a local development server. They do not run in CI.
export default defineConfig({
    testDir: "./tests/e2e",
    reporter: "list",
    outputDir: "./test-results",
    use: {
        ...devices["Desktop Chrome"],
        baseURL: "http://localhost:5173",
        // The scene click needs a fixed window size and a fixed camera preset.
        viewport: { width: 1280, height: 800 },
    },
    webServer: {
        command: "npm run dev -- --port 5173 --strictPort",
        url: "http://localhost:5173",
        reuseExistingServer: !process.env.CI,
        timeout: 60000,
    },
});
