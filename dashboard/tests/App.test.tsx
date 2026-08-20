import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../src/App";

describe("App shell", () => {
    beforeEach(() => {
        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue({
                json: () => Promise.resolve({ status: "ok" }),
            }),
        );
    });

    afterEach(() => {
        vi.unstubAllGlobals();
    });

    it("renders the fetched health status", async () => {
        render(<App />);

        await waitFor(() => {
            expect(screen.getByTestId("health-status")).toHaveTextContent(
                "API status: ok",
            );
        });
    });
});
