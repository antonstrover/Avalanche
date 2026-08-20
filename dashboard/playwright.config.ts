import { defineConfig, devices } from "@playwright/test";

const apiPort = process.env.PLAYWRIGHT_API_PORT ?? "8000";
const dashboardPort = process.env.PLAYWRIGHT_DASHBOARD_PORT ?? "5173";
const apiCommand =
    process.env.PLAYWRIGHT_API_COMMAND ??
    `uv run uvicorn avalanche.api.app:app --port ${apiPort}`;

// The browser tests run on a local development server. They do not run in CI.
export default defineConfig({
    testDir: "./tests/e2e",
    reporter: "list",
    outputDir: "./test-results",
    use: {
        ...devices["Desktop Chrome"],
        baseURL: `http://localhost:${dashboardPort}`,
        // The scene click needs a fixed window size and a fixed camera preset.
        viewport: { width: 1280, height: 800 },
    },
    webServer: [
        {
            command: apiCommand,
            url: `http://127.0.0.1:${apiPort}/health`,
            reuseExistingServer: !process.env.CI,
            timeout: 60000,
        },
        {
            command: `AVALANCHE_API_PORT=${apiPort} npm run dev -- --port ${dashboardPort} --strictPort`,
            url: `http://localhost:${dashboardPort}`,
            reuseExistingServer: !process.env.CI,
            timeout: 60000,
        },
    ],
});
